import pandas as pd
import numpy as np

class KSTCalculator:
    def __init__(self, roc_periods=[10, 15, 20, 30], 
                 sma_periods=[10, 10, 10, 15],
                 signal_period=9):
        """
        Initialize KST Calculator with default parameters
        
        Parameters:
        roc_periods (list): Periods for Rate of Change calculations
        sma_periods (list): Periods for smoothing the ROC values
        signal_period (int): Period for signal line calculation
        """
        self.roc_periods = roc_periods
        self.sma_periods = sma_periods
        self.signal_period = signal_period
        self.weights = [1, 2, 3, 4]  # Standard weights for KST components
        self.kst_data = None
        
    def calculate_roc(self, data, period, column='close'):
        """
        Calculate Rate of Change
        
        Parameters:
        data (pd.DataFrame/Series): Price data
        period (int): ROC period
        column (str): Column name if DataFrame is provided
        
        Returns:
        pd.Series: ROC values
        """
        if isinstance(data, pd.DataFrame):
            prices = data[column]
        else:
            prices = data
            
        return prices.pct_change(period) * 100
    
    def calculate_kst(self, data, column='close'):
        """
        Calculate KST Oscillator and Signal Line
        
        Parameters:
        data (pd.DataFrame/Series): Price data
        column (str): Column name if DataFrame is provided
        
        Returns:
        pd.DataFrame: DataFrame with KST, Signal, and component values
        """
        if isinstance(data, pd.DataFrame):
            prices = data[column]
        else:
            prices = data
            
        components = {}
        
        # Calculate each ROC component
        for i, (roc_period, sma_period) in enumerate(zip(self.roc_periods, self.sma_periods)):
            # Calculate ROC
            roc = self.calculate_roc(prices, roc_period)
            
            # Smooth ROC with SMA
            smooth_roc = roc.rolling(window=sma_period).mean()
            
            # Apply weight
            components[f'RCMA{i+1}'] = smooth_roc * self.weights[i]
        
        # Calculate KST
        kst = pd.DataFrame(components).sum(axis=1)
        
        # Calculate Signal Line
        signal = kst.rolling(window=self.signal_period).mean()
        
        # Calculate Histogram
        histogram = kst - signal
        
        # Store results
        self.kst_data = pd.DataFrame({
            'kst': kst,
            'signal': signal,
            'histogram': histogram,
            'price': prices,
            **components
        })
        
        return self.kst_data
    
    def get_crossover_signals(self):
        """
        Generate KST crossover signals
        
        Returns:
        pd.Series: Crossover signals (1 for bullish, -1 for bearish)
        """
        if self.kst_data is None:
            raise ValueError("KST must be calculated first")
            
        # Calculate crossover
        signals = pd.Series(0, index=self.kst_data.index)
        signals[self.kst_data['kst'] > self.kst_data['signal']] = 1
        signals[self.kst_data['kst'] < self.kst_data['signal']] = -1
        
        # Get actual crossover points
        crossover = signals.diff()
        
        return crossover
    
    def get_overbought_oversold(self, threshold=40):
        """
        Generate overbought/oversold signals
        
        Parameters:
        threshold (float): Overbought/oversold threshold
        
        Returns:
        pd.DataFrame: Overbought/oversold signals
        """
        if self.kst_data is None:
            raise ValueError("KST must be calculated first")
            
        signals = pd.DataFrame(index=self.kst_data.index)
        
        signals['overbought'] = self.kst_data['kst'] > threshold
        signals['oversold'] = self.kst_data['kst'] < -threshold
        
        return signals
    
    def detect_divergence(self, window=20):
        """
        Detect regular and hidden divergences
        
        Parameters:
        window (int): Lookback window for divergence detection
        
        Returns:
        pd.DataFrame: DataFrame with divergence signals
        """
        if self.kst_data is None:
            raise ValueError("KST must be calculated first")
            
        divergence = pd.DataFrame(index=self.kst_data.index, 
                                columns=['regular_bullish', 'regular_bearish',
                                       'hidden_bullish', 'hidden_bearish'])
        divergence = divergence.fillna(0)
        
        for i in range(window, len(self.kst_data)):
            window_data = self.kst_data.iloc[i-window:i+1]
            
            # Get local extremes
            price_high = window_data['price'].max()
            price_low = window_data['price'].min()
            kst_high = window_data['kst'].max()
            kst_low = window_data['kst'].min()
            
            current_price = window_data['price'].iloc[-1]
            current_kst = window_data['kst'].iloc[-1]
            
            # Regular Bullish Divergence
            if (current_price == price_low and current_kst > kst_low):
                divergence.iloc[i]['regular_bullish'] = 1
                
            # Regular Bearish Divergence
            elif (current_price == price_high and current_kst < kst_high):
                divergence.iloc[i]['regular_bearish'] = 1
                
            # Hidden Bullish Divergence
            elif (current_price > price_low and current_kst == kst_low):
                divergence.iloc[i]['hidden_bullish'] = 1
                
            # Hidden Bearish Divergence
            elif (current_price < price_high and current_kst == kst_high):
                divergence.iloc[i]['hidden_bearish'] = 1
                
        return divergence
    
    def get_kst_strength(self, lookback=5):
        """
        Calculate KST trend strength
        
        Parameters:
        lookback (int): Periods to look back
        
        Returns:
        dict: KST strength indicators
        """
        if self.kst_data is None:
            raise ValueError("KST must be calculated first")
            
        recent_data = self.kst_data.iloc[-lookback:]
        
        # Calculate slopes
        kst_slope = np.polyfit(range(lookback), recent_data['kst'].values, 1)[0]
        signal_slope = np.polyfit(range(lookback), recent_data['signal'].values, 1)[0]
        hist_slope = np.polyfit(range(lookback), recent_data['histogram'].values, 1)[0]
        
        # Get current values
        current_kst = self.kst_data['kst'].iloc[-1]
        current_signal = self.kst_data['signal'].iloc[-1]
        current_hist = self.kst_data['histogram'].iloc[-1]
        
        return {
            'kst_slope': kst_slope,
            'signal_slope': signal_slope,
            'histogram_slope': hist_slope,
            'kst_value': current_kst,
            'signal_value': current_signal,
            'histogram_value': current_hist,
            'trend': 'bullish' if current_kst > current_signal else 'bearish',
            'strength': abs(current_hist),
            'momentum': 'increasing' if hist_slope > 0 else 'decreasing'
        }
    
    def get_trading_signals(self, threshold=40, divergence_window=20):
        """
        Get comprehensive trading signals
        
        Returns:
        dict: Dictionary with various KST signals and indicators
        """
        crossovers = self.get_crossover_signals()
        overbought_oversold = self.get_overbought_oversold(threshold)
        divergences = self.detect_divergence(divergence_window)
        strength = self.get_kst_strength()
        
        latest_signals = {
            'crossover': crossovers.iloc[-1],
            'overbought': overbought_oversold['overbought'].iloc[-1],
            'oversold': overbought_oversold['oversold'].iloc[-1],
            'divergence': {
                'regular_bullish': divergences['regular_bullish'].iloc[-1],
                'regular_bearish': divergences['regular_bearish'].iloc[-1],
                'hidden_bullish': divergences['hidden_bullish'].iloc[-1],
                'hidden_bearish': divergences['hidden_bearish'].iloc[-1]
            },
            'strength_indicators': strength
        }
        
        return latest_signals
