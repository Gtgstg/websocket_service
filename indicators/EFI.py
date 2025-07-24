"""
Elder Force Index (EFI) Indicator Module
Provides classes and functions for calculating the Elder Force Index and related signals.
"""
import pandas as pd
import numpy as np
from typing import Union, Optional
import warnings

class ElderForceIndex:
    """
    Elder Force Index (EFI) Technical Indicator
    
    The Elder Force Index combines price and volume to measure the strength
    behind market moves. It's calculated as:
    EFI = Volume × (Close - Previous Close)
    
    A smoothed version using exponential moving average is often used:
    EFI_smoothed = EMA(EFI, period)
    """
    
    def __init__(self, period: int = 13, smoothing_method: str = 'ema'):
        """
        Initialize Elder Force Index calculator
        
        Args:
            period: Period for smoothing (default 13)
            smoothing_method: 'ema', 'sma', or 'none' for raw EFI
        """
        self.period = period
        self.smoothing_method = smoothing_method.lower()
        
        if self.smoothing_method not in ['ema', 'sma', 'none']:
            raise ValueError("smoothing_method must be 'ema', 'sma', or 'none'")
    
    def calculate(self, high: pd.Series, low: pd.Series, close: pd.Series, 
                  volume: pd.Series) -> pd.Series:
        """
        Calculate Elder Force Index
        
        Args:
            high: High prices
            low: Low prices  
            close: Close prices
            volume: Volume data
            
        Returns:
            Elder Force Index values
        """
        # Validate inputs
        if len(close) != len(volume):
            raise ValueError("Close and volume series must have same length")
        
        if len(close) < 2:
            raise ValueError("Need at least 2 data points")
        
        # Calculate raw Elder Force Index
        price_change = close.diff()
        raw_efi = volume * price_change
        
        # Apply smoothing if requested
        if self.smoothing_method == 'ema':
            efi = self._exponential_moving_average(raw_efi, self.period)
        elif self.smoothing_method == 'sma':
            efi = raw_efi.rolling(window=self.period).mean()
        else:  # 'none'
            efi = raw_efi
            
        return efi
    
    def _exponential_moving_average(self, data: pd.Series, period: int) -> pd.Series:
        """Calculate exponential moving average"""
        return data.ewm(span=period, adjust=False).mean()
    
    def generate_signals(self, efi: pd.Series, 
                        bullish_threshold: float = 0,
                        bearish_threshold: float = 0,
                        use_zero_line: bool = True,
                        use_divergence: bool = False,
                        price_data: Optional[pd.Series] = None) -> pd.DataFrame:
        """
        Generate trading signals based on Elder Force Index
        
        Args:
            efi: Elder Force Index values
            bullish_threshold: Threshold for bullish signals
            bearish_threshold: Threshold for bearish signals  
            use_zero_line: Use zero line crossovers
            use_divergence: Detect price-EFI divergences (requires price_data)
            price_data: Price data for divergence analysis
            
        Returns:
            DataFrame with signals
        """
        signals = pd.DataFrame(index=efi.index)
        signals['efi'] = efi
        signals['signal'] = 0
        signals['position'] = 0
        
        # Zero line crossover signals
        if use_zero_line:
            signals.loc[efi > bullish_threshold, 'signal'] = 1  # Bullish
            signals.loc[efi < bearish_threshold, 'signal'] = -1  # Bearish
        
        # Generate position signals (1 for long, -1 for short, 0 for neutral)
        signals['position'] = signals['signal'].replace(0, np.nan).ffill().fillna(0)
        
        # Divergence signals (if requested and price data provided)
        if use_divergence and price_data is not None:
            div_signals = self._detect_divergence(efi, price_data)
            signals = signals.join(div_signals, rsuffix='_div')
        
        return signals
    
    def _detect_divergence(self, efi: pd.Series, price: pd.Series, 
                          window: int = 20) -> pd.DataFrame:
        """
        Detect bullish and bearish divergences between price and EFI
        
        Args:
            efi: Elder Force Index values
            price: Price data
            window: Lookback window for divergence detection
            
        Returns:
            DataFrame with divergence signals
        """
        divergence = pd.DataFrame(index=efi.index)
        divergence['bullish_div'] = False
        divergence['bearish_div'] = False
        
        for i in range(window, len(efi)):
            # Get recent data
            recent_efi = efi.iloc[i-window:i+1]
            recent_price = price.iloc[i-window:i+1]
            
            # Find local minima and maxima
            efi_min_idx = recent_efi.idxmin()
            efi_max_idx = recent_efi.idxmax()
            price_min_idx = recent_price.idxmin()
            price_max_idx = recent_price.idxmax()
            
            # Bullish divergence: price makes lower low, EFI makes higher low
            if (recent_price.iloc[-1] < recent_price.loc[price_min_idx] and 
                recent_efi.iloc[-1] > recent_efi.loc[efi_min_idx]):
                divergence.iloc[i, divergence.columns.get_loc('bullish_div')] = True
            
            # Bearish divergence: price makes higher high, EFI makes lower high  
            if (recent_price.iloc[-1] > recent_price.loc[price_max_idx] and
                recent_efi.iloc[-1] < recent_efi.loc[efi_max_idx]):
                divergence.iloc[i, divergence.columns.get_loc('bearish_div')] = True
        
        return divergence


class ElderForceIndexStrategy:
    """
    Complete trading strategy using Elder Force Index
    """
    
    def __init__(self, efi_period: int = 13, 
                 efi_smoothing: str = 'ema',
                 volume_ma_period: int = 20):
        """
        Initialize strategy
        
        Args:
            efi_period: Period for EFI calculation
            efi_smoothing: Smoothing method for EFI
            volume_ma_period: Period for volume moving average filter
        """
        self.efi_calc = ElderForceIndex(efi_period, efi_smoothing)
        self.volume_ma_period = volume_ma_period
    
    def backtest(self, data: pd.DataFrame, 
                 initial_capital: float = 100000,
                 commission: float = 0.001) -> pd.DataFrame:
        """
        Backtest the Elder Force Index strategy
        
        Args:
            data: DataFrame with OHLCV data
            initial_capital: Starting capital
            commission: Commission rate (0.001 = 0.1%)
            
        Returns:
            DataFrame with backtest results
        """
        # Validate required columns
        required_cols = ['high', 'low', 'close', 'volume']
        if not all(col in data.columns for col in required_cols):
            raise ValueError(f"Data must contain columns: {required_cols}")
        
        results = data.copy()
        
        # Calculate Elder Force Index
        efi = self.efi_calc.calculate(
            data['high'], data['low'], data['close'], data['volume']
        )
        
        # Calculate volume filter
        volume_ma = data['volume'].rolling(window=self.volume_ma_period).mean()
        volume_filter = data['volume'] > volume_ma
        
        # Generate signals
        signals_df = self.efi_calc.generate_signals(efi)
        
        # Apply volume filter
        signals_df.loc[~volume_filter, 'signal'] = 0
        signals_df['position'] = signals_df['signal'].replace(0, np.nan).ffill().fillna(0)
        
        # Calculate returns
        results['efi'] = efi
        results['signal'] = signals_df['signal']
        results['position'] = signals_df['position']
        results['volume_filter'] = volume_filter
        
        # Calculate strategy returns
        results['returns'] = data['close'].pct_change()
        results['strategy_returns'] = results['position'].shift(1) * results['returns']
        
        # Apply commission
        position_changes = results['position'].diff().fillna(0)
        results['commission_cost'] = abs(position_changes) * commission
        results['strategy_returns'] -= results['commission_cost']
        
        # Calculate cumulative returns
        results['cumulative_returns'] = (1 + results['returns']).cumprod()
        results['strategy_cumulative'] = (1 + results['strategy_returns']).cumprod()
        
        # Calculate portfolio value
        results['portfolio_value'] = initial_capital * results['strategy_cumulative']
        
        return results
    
    def get_performance_metrics(self, results: pd.DataFrame) -> dict:
        """
        Calculate performance metrics
        
        Args:
            results: Backtest results DataFrame
            
        Returns:
            Dictionary of performance metrics
        """
        strategy_returns = results['strategy_returns'].dropna()
        
        if len(strategy_returns) == 0:
            return {}
        
        # Basic metrics
        total_return = results['strategy_cumulative'].iloc[-1] - 1
        annualized_return = (1 + total_return) ** (252 / len(strategy_returns)) - 1
        volatility = strategy_returns.std() * np.sqrt(252)
        sharpe_ratio = annualized_return / volatility if volatility > 0 else 0
        
        # Drawdown
        cumulative = results['strategy_cumulative']
        rolling_max = cumulative.expanding().max()
        drawdown = (cumulative - rolling_max) / rolling_max
        max_drawdown = drawdown.min()
        
        # Win rate
        winning_trades = strategy_returns[strategy_returns > 0]
        win_rate = len(winning_trades) / len(strategy_returns) if len(strategy_returns) > 0 else 0
        
        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'total_trades': len(strategy_returns[strategy_returns != 0])
        }


# Example usage and testing
if __name__ == "__main__":
    # Generate sample data
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=252, freq='D')
    
    # Create realistic OHLCV data
    base_price = 100
    returns = np.random.normal(0.001, 0.02, 252)
    prices = base_price * (1 + returns).cumprod()
    
    sample_data = pd.DataFrame({
        'date': dates,
        'open': prices * (1 + np.random.normal(0, 0.001, 252)),
        'high': prices * (1 + abs(np.random.normal(0, 0.01, 252))),
        'low': prices * (1 - abs(np.random.normal(0, 0.01, 252))),
        'close': prices,
        'volume': np.random.randint(1000000, 5000000, 252)
    }, index=dates)
    
    # Initialize strategy
    strategy = ElderForceIndexStrategy(efi_period=13)
    
    # Run backtest
    results = strategy.backtest(sample_data)
    
    # Get performance metrics
    metrics = strategy.get_performance_metrics(results)
    
    print("Elder Force Index Strategy Performance:")
    print("-" * 40)
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")
    
    # Display recent signals
    print("\nRecent Signals:")
    print(results[['close', 'efi', 'signal', 'position']].tail(10))