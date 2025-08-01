import numpy as np
import pandas as pd
from typing import Union, Tuple, Optional
import warnings


class MoneyFlowIndex:
    """
    Money Flow Index (MFI) Indicator
    
    The MFI is a momentum oscillator that uses both price and volume to identify 
    overbought and oversold conditions. It's often called the "Volume-Weighted RSI".
    
    Calculation Steps:
    1. Calculate Typical Price: (High + Low + Close) / 3
    2. Calculate Raw Money Flow: Typical Price × Volume
    3. Identify Positive/Negative Money Flow based on Typical Price direction
    4. Calculate Money Flow Ratio: Sum(Positive MF) / Sum(Negative MF) over period
    5. Calculate MFI: 100 - (100 / (1 + Money Flow Ratio))
    
    MFI oscillates between 0 and 100:
    - Values above 80 indicate overbought conditions
    - Values below 20 indicate oversold conditions
    """
    
    def __init__(self, period: int = 14):
        """
        Initialize Money Flow Index indicator
        
        Args:
            period (int): Number of periods for MFI calculation (default: 14)
        """
        if period <= 0:
            raise ValueError("Period must be greater than 0")
        
        self.period = period
        self._reset_state()
    
    def _reset_state(self):
        """Reset internal state for fresh calculations"""
        self._typical_prices = []
        self._money_flows = []
        self._positive_flows = []
        self._negative_flows = []
    
    def calculate(self, 
                 high: Union[np.ndarray, pd.Series, list], 
                 low: Union[np.ndarray, pd.Series, list], 
                 close: Union[np.ndarray, pd.Series, list], 
                 volume: Union[np.ndarray, pd.Series, list]) -> np.ndarray:
        """
        Calculate Money Flow Index values
        
        Args:
            high: High prices
            low: Low prices  
            close: Close prices
            volume: Volume data
            
        Returns:
            np.ndarray: MFI values
            
        Raises:
            ValueError: If input arrays have different lengths or insufficient data
        """
        # Convert inputs to numpy arrays
        high = np.asarray(high, dtype=float)
        low = np.asarray(low, dtype=float)
        close = np.asarray(close, dtype=float)
        volume = np.asarray(volume, dtype=float)
        
        # Validate inputs
        self._validate_inputs(high, low, close, volume)
        
        # Calculate MFI
        return self._calculate_mfi(high, low, close, volume)
    
    def calculate_single(self, 
                        high: float, 
                        low: float, 
                        close: float, 
                        volume: float) -> Optional[float]:
        """
        Calculate MFI for a single data point (streaming/real-time use)
        
        Args:
            high: High price
            low: Low price
            close: Close price
            volume: Volume
            
        Returns:
            Optional[float]: MFI value if enough data available, None otherwise
        """
        # Calculate typical price
        typical_price = (high + low + close) / 3
        self._typical_prices.append(typical_price)
        
        # Calculate raw money flow
        money_flow = typical_price * volume
        self._money_flows.append(money_flow)
        
        # Determine positive/negative money flow
        if len(self._typical_prices) > 1:
            if typical_price > self._typical_prices[-2]:
                self._positive_flows.append(money_flow)
                self._negative_flows.append(0.0)
            elif typical_price < self._typical_prices[-2]:
                self._positive_flows.append(0.0)
                self._negative_flows.append(money_flow)
            else:
                # No change in typical price
                self._positive_flows.append(0.0)
                self._negative_flows.append(0.0)
        else:
            # First data point - no comparison possible
            self._positive_flows.append(0.0)
            self._negative_flows.append(0.0)
        
        # Keep only required periods
        if len(self._typical_prices) > self.period + 1:
            self._typical_prices.pop(0)
            self._money_flows.pop(0)
            self._positive_flows.pop(0)
            self._negative_flows.pop(0)
        
        # Calculate MFI if we have enough data
        if len(self._positive_flows) >= self.period:
            positive_mf_sum = sum(self._positive_flows[-self.period:])
            negative_mf_sum = sum(self._negative_flows[-self.period:])
            
            if negative_mf_sum == 0:
                return 100.0  # All positive money flow
            
            money_flow_ratio = positive_mf_sum / negative_mf_sum
            mfi = 100 - (100 / (1 + money_flow_ratio))
            return mfi
        
        return None
    
    def _validate_inputs(self, high: np.ndarray, low: np.ndarray, 
                        close: np.ndarray, volume: np.ndarray):
        """Validate input arrays"""
        arrays = [high, low, close, volume]
        lengths = [len(arr) for arr in arrays]
        
        # Check equal lengths
        if not all(length == lengths[0] for length in lengths):
            raise ValueError("All input arrays must have equal length")
        
        # Check minimum data requirement
        if lengths[0] < self.period + 1:
            raise ValueError(f"Insufficient data: need at least {self.period + 1} periods")
        
        # Check for negative volumes
        if np.any(volume < 0):
            warnings.warn("Negative volume values detected", UserWarning)
        
        # Check for invalid prices (high < low)
        if np.any(high < low):
            raise ValueError("High prices cannot be lower than low prices")
    
    def _calculate_mfi(self, high: np.ndarray, low: np.ndarray, 
                      close: np.ndarray, volume: np.ndarray) -> np.ndarray:
        """Internal MFI calculation"""
        # Calculate typical prices
        typical_prices = (high + low + close) / 3
        
        # Calculate raw money flow
        raw_money_flow = typical_prices * volume
        
        # Identify positive and negative money flows
        positive_money_flow = np.zeros_like(raw_money_flow)
        negative_money_flow = np.zeros_like(raw_money_flow)
        
        # Compare typical prices (skip first element as no previous comparison)
        for i in range(1, len(typical_prices)):
            if typical_prices[i] > typical_prices[i-1]:
                positive_money_flow[i] = raw_money_flow[i]
            elif typical_prices[i] < typical_prices[i-1]:
                negative_money_flow[i] = raw_money_flow[i]
            # If equal, both remain 0
        
        # Calculate MFI for each period
        mfi_values = []
        
        for i in range(self.period, len(typical_prices)):
            # Sum positive and negative money flows over the period
            start_idx = i - self.period + 1
            end_idx = i + 1
            
            positive_mf_sum = np.sum(positive_money_flow[start_idx:end_idx])
            negative_mf_sum = np.sum(negative_money_flow[start_idx:end_idx])
            
            # Calculate Money Flow Index
            if negative_mf_sum == 0:
                # All positive money flow
                mfi = 100.0
            else:
                money_flow_ratio = positive_mf_sum / negative_mf_sum
                mfi = 100 - (100 / (1 + money_flow_ratio))
            
            mfi_values.append(mfi)
        
        return np.array(mfi_values)
    
    def get_signals(self, mfi_values: np.ndarray, 
                   overbought_threshold: float = 80, 
                   oversold_threshold: float = 20) -> dict:
        """
        Generate trading signals based on MFI values
        
        Args:
            mfi_values: Array of MFI values
            overbought_threshold: MFI level indicating overbought condition (default: 80)
            oversold_threshold: MFI level indicating oversold condition (default: 20)
            
        Returns:
            dict: Dictionary containing signal arrays
        """
        signals = {
            'overbought': mfi_values > overbought_threshold,
            'oversold': mfi_values < oversold_threshold,
            'neutral': (mfi_values >= oversold_threshold) & (mfi_values <= overbought_threshold),
            'buy_signal': np.zeros_like(mfi_values, dtype=bool),
            'sell_signal': np.zeros_like(mfi_values, dtype=bool)
        }
        
        # Generate buy signals (oversold to neutral crossover)
        for i in range(1, len(mfi_values)):
            if mfi_values[i-1] < oversold_threshold and mfi_values[i] >= oversold_threshold:
                signals['buy_signal'][i] = True
            elif mfi_values[i-1] > overbought_threshold and mfi_values[i] <= overbought_threshold:
                signals['sell_signal'][i] = True
        
        return signals
    
    def get_divergences(self, prices: np.ndarray, mfi_values: np.ndarray, 
                       window: int = 5) -> dict:
        """
        Detect bullish and bearish divergences between price and MFI
        
        Args:
            prices: Price array (typically close prices)
            mfi_values: MFI values array
            window: Window size for peak/trough detection
            
        Returns:
            dict: Dictionary containing divergence signals
        """
        if len(prices) != len(mfi_values):
            raise ValueError("Prices and MFI arrays must have equal length")
        
        # Find peaks and troughs
        price_peaks = self._find_peaks(prices, window)
        price_troughs = self._find_troughs(prices, window)
        mfi_peaks = self._find_peaks(mfi_values, window)
        mfi_troughs = self._find_troughs(mfi_values, window)
        
        bullish_divergence = np.zeros_like(prices, dtype=bool)
        bearish_divergence = np.zeros_like(prices, dtype=bool)
        
        # Detect bullish divergence (price makes lower low, MFI makes higher low)
        for i in range(len(price_troughs)):
            if price_troughs[i] and i > 0:
                # Find previous trough
                prev_trough_idx = None
                for j in range(i-1, -1, -1):
                    if price_troughs[j]:
                        prev_trough_idx = j
                        break
                
                if prev_trough_idx is not None:
                    if (prices[i] < prices[prev_trough_idx] and 
                        mfi_troughs[i] and mfi_troughs[prev_trough_idx] and
                        mfi_values[i] > mfi_values[prev_trough_idx]):
                        bullish_divergence[i] = True
        
        # Detect bearish divergence (price makes higher high, MFI makes lower high)
        for i in range(len(price_peaks)):
            if price_peaks[i] and i > 0:
                # Find previous peak
                prev_peak_idx = None
                for j in range(i-1, -1, -1):
                    if price_peaks[j]:
                        prev_peak_idx = j
                        break
                
                if prev_peak_idx is not None:
                    if (prices[i] > prices[prev_peak_idx] and 
                        mfi_peaks[i] and mfi_peaks[prev_peak_idx] and
                        mfi_values[i] < mfi_values[prev_peak_idx]):
                        bearish_divergence[i] = True
        
        return {
            'bullish_divergence': bullish_divergence,
            'bearish_divergence': bearish_divergence
        }
    
    def _find_peaks(self, data: np.ndarray, window: int) -> np.ndarray:
        """Find peaks in data using simple window-based approach"""
        peaks = np.zeros_like(data, dtype=bool)
        
        for i in range(window, len(data) - window):
            is_peak = True
            for j in range(i - window, i + window + 1):
                if j != i and data[j] >= data[i]:
                    is_peak = False
                    break
            peaks[i] = is_peak
        
        return peaks
    
    def _find_troughs(self, data: np.ndarray, window: int) -> np.ndarray:
        """Find troughs in data using simple window-based approach"""
        troughs = np.zeros_like(data, dtype=bool)
        
        for i in range(window, len(data) - window):
            is_trough = True
            for j in range(i - window, i + window + 1):
                if j != i and data[j] <= data[i]:
                    is_trough = False
                    break
            troughs[i] = is_trough
        
        return troughs
    
    def reset(self):
        """Reset indicator state for fresh calculations"""
        self._reset_state()


# Example usage and demonstration
def demonstrate_mfi():
    """Demonstrate MFI indicator usage with sample data"""
    
    # Create sample market data
    np.random.seed(42)
    n_periods = 100
    
    # Generate sample price data with trend and noise
    base_price = 100
    trend = np.linspace(0, 20, n_periods)
    noise = np.random.normal(0, 2, n_periods)
    close_prices = base_price + trend + noise
    
    # Generate high/low prices
    high_prices = close_prices + np.random.uniform(0.5, 2.0, n_periods)
    low_prices = close_prices - np.random.uniform(0.5, 2.0, n_periods)
    
    # Generate volume data (higher volume during price movements)
    price_changes = np.abs(np.diff(np.concatenate([[close_prices[0]], close_prices])))
    volumes = 1000 + price_changes * 100 + np.random.uniform(0, 500, n_periods)
    
    # Initialize MFI indicator
    mfi = MoneyFlowIndex(period=14)
    
    # Calculate MFI values
    mfi_values = mfi.calculate(high_prices, low_prices, close_prices, volumes)
    
    # Generate signals
    signals = mfi.get_signals(mfi_values)
    
    # Detect divergences
    divergences = mfi.get_divergences(close_prices[-len(mfi_values):], mfi_values)
    
    print("Money Flow Index Demonstration")
    print("=" * 40)
    print(f"Generated {len(mfi_values)} MFI values from {n_periods} price periods")
    print(f"MFI Range: {mfi_values.min():.2f} - {mfi_values.max():.2f}")
    print(f"Average MFI: {mfi_values.mean():.2f}")
    print(f"Overbought signals: {signals['overbought'].sum()}")
    print(f"Oversold signals: {signals['oversold'].sum()}")
    print(f"Buy signals: {signals['buy_signal'].sum()}")
    print(f"Sell signals: {signals['sell_signal'].sum()}")
    print(f"Bullish divergences: {divergences['bullish_divergence'].sum()}")
    print(f"Bearish divergences: {divergences['bearish_divergence'].sum()}")
    
    # Show some recent values
    print(f"\nLast 10 MFI values:")
    for i, value in enumerate(mfi_values[-10:]):
        idx = len(mfi_values) - 10 + i
        status = "OVERBOUGHT" if value > 80 else "OVERSOLD" if value < 20 else "NEUTRAL"
        print(f"Period {idx}: {value:.2f} ({status})")
    
    return {
        'mfi_values': mfi_values,
        'signals': signals,
        'divergences': divergences,
        'prices': close_prices,
        'volumes': volumes
    }


# Utility functions for backtesting and analysis
def backtest_mfi_strategy(prices: np.ndarray, volumes: np.ndarray, 
                         high: np.ndarray, low: np.ndarray,
                         period: int = 14, 
                         overbought: float = 80, 
                         oversold: float = 20) -> dict:
    """
    Simple backtest of MFI-based trading strategy
    
    Args:
        prices: Close prices
        volumes: Volume data  
        high: High prices
        low: Low prices
        period: MFI calculation period
        overbought: Overbought threshold
        oversold: Oversold threshold
        
    Returns:
        dict: Backtest results
    """
    mfi = MoneyFlowIndex(period=period)
    mfi_values = mfi.calculate(high, low, prices, volumes)
    signals = mfi.get_signals(mfi_values, overbought, oversold)
    
    # Simple strategy: buy on oversold exit, sell on overbought exit
    position = 0  # 0: no position, 1: long, -1: short
    trades = []
    equity_curve = [10000]  # Starting capital
    
    for i in range(len(mfi_values)):
        current_price = prices[period + i]  # Adjust for MFI calculation offset
        
        if signals['buy_signal'][i] and position == 0:
            # Enter long position
            position = 1
            entry_price = current_price
            trades.append({'type': 'buy', 'price': entry_price, 'index': i})
        
        elif signals['sell_signal'][i] and position == 1:
            # Exit long position
            position = 0
            exit_price = current_price
            pnl = (exit_price - entry_price) / entry_price
            equity_curve.append(equity_curve[-1] * (1 + pnl))
            trades.append({'type': 'sell', 'price': exit_price, 'index': i, 'pnl': pnl})
        
        else:
            # No change in position
            equity_curve.append(equity_curve[-1])
    
    total_return = (equity_curve[-1] - equity_curve[0]) / equity_curve[0]
    
    return {
        'total_return': total_return,
        'final_equity': equity_curve[-1],
        'num_trades': len([t for t in trades if t['type'] == 'sell']),
        'trades': trades,
        'equity_curve': equity_curve
    }


if __name__ == "__main__":
    # Run demonstration
    demo_results = demonstrate_mfi()
    
    print("\n" + "=" * 50)
    print("MFI INDICATOR READY FOR TRADING BOT INTEGRATION")
    print("=" * 50)
    
    # Example of streaming usage
    print("\nStreaming MFI Example:")
    mfi_stream = MoneyFlowIndex(period=5)  # Shorter period for demo
    
    sample_data = [
        {'high': 101, 'low': 99, 'close': 100, 'volume': 1000},
        {'high': 102, 'low': 100, 'close': 101, 'volume': 1200},
        {'high': 103, 'low': 101, 'close': 102, 'volume': 800},
        {'high': 102, 'low': 100, 'close': 101, 'volume': 900},
        {'high': 101, 'low': 99, 'close': 100, 'volume': 1100},
        {'high': 100, 'low': 98, 'close': 99, 'volume': 1300},
    ]
    
    for i, data in enumerate(sample_data):
        mfi_value = mfi_stream.calculate_single(
            data['high'], data['low'], data['close'], data['volume']
        )
        if mfi_value is not None:
            print(f"Period {i+1}: MFI = {mfi_value:.2f}")
        else:
            print(f"Period {i+1}: Insufficient data for MFI calculation")