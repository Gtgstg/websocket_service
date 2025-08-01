"""
Awesome Oscillator (AO) Indicator Module
Provides classes and functions for AO calculation and signal generation.
"""
import numpy as np
import pandas as pd
from typing import Union, Tuple, Optional, Dict
from enum import Enum

class AOSignal(Enum):
    """Awesome Oscillator signal types"""
    ZERO_LINE_CROSS_UP = "zero_cross_up"
    ZERO_LINE_CROSS_DOWN = "zero_cross_down"
    TWIN_PEAKS_BULLISH = "twin_peaks_bullish"
    TWIN_PEAKS_BEARISH = "twin_peaks_bearish"
    SAUCER_BULLISH = "saucer_bullish"
    SAUCER_BEARISH = "saucer_bearish"
    MOMENTUM_INCREASE = "momentum_increase"
    MOMENTUM_DECREASE = "momentum_decrease"
    NO_SIGNAL = "no_signal"

class AwesomeOscillator:
    """
    Awesome Oscillator (AO) Indicator for Algorithmic Trading
    
    The Awesome Oscillator is a momentum indicator that measures the difference
    between a 5-period and 34-period simple moving average of the median price (H+L)/2.
    
    Key Features:
    - Zero line crossovers
    - Twin Peaks pattern detection
    - Saucer pattern detection
    - Momentum change detection
    - Multiple timeframe analysis
    """
    
    def __init__(self, fast_period: int = 5, slow_period: int = 34):
        """
        Initialize Awesome Oscillator
        
        Args:
            fast_period: Period for fast SMA (default: 5)
            slow_period: Period for slow SMA (default: 34)
        """
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.min_periods = max(fast_period, slow_period)
    
    def calculate(self, high: Union[pd.Series, np.ndarray], 
                  low: Union[pd.Series, np.ndarray]) -> Union[pd.Series, np.ndarray]:
        """
        Calculate Awesome Oscillator values
        
        Args:
            high: High prices
            low: Low prices
            
        Returns:
            Awesome Oscillator values
        """
        # Convert to pandas Series if numpy arrays
        if isinstance(high, np.ndarray):
            high = pd.Series(high)
        if isinstance(low, np.ndarray):
            low = pd.Series(low)
        
        # Calculate median price (HL2)
        median_price = (high + low) / 2
        
        # Calculate moving averages
        sma_fast = median_price.rolling(window=self.fast_period).mean()
        sma_slow = median_price.rolling(window=self.slow_period).mean()
        
        # Awesome Oscillator = Fast SMA - Slow SMA
        ao = sma_fast - sma_slow
        
        return ao
    
    def calculate_with_signals(self, high: Union[pd.Series, np.ndarray], 
                             low: Union[pd.Series, np.ndarray]) -> Dict:
        """
        Calculate AO with comprehensive signal analysis
        
        Args:
            high: High prices
            low: Low prices
            
        Returns:
            Dictionary containing AO values and signals
        """
        ao = self.calculate(high, low)
        
        # Generate signals
        signals = self._generate_signals(ao)
        
        # Calculate additional metrics
        ao_histogram = self._calculate_histogram(ao)
        zero_line_distance = self._calculate_zero_line_distance(ao)
        momentum_strength = self._calculate_momentum_strength(ao)
        
        return {
            'ao': ao,
            'signals': signals,
            'histogram': ao_histogram,
            'zero_line_distance': zero_line_distance,
            'momentum_strength': momentum_strength,
            'ao_above_zero': ao > 0,
            'ao_increasing': ao > ao.shift(1),
            'ao_color': self._get_ao_colors(ao)
        }
    
    def _generate_signals(self, ao: pd.Series) -> pd.Series:
        """Generate trading signals based on AO patterns"""
        signals = pd.Series(AOSignal.NO_SIGNAL.value, index=ao.index)
        
        # Zero line crossovers
        zero_cross_up = (ao > 0) & (ao.shift(1) <= 0)
        zero_cross_down = (ao < 0) & (ao.shift(1) >= 0)
        
        signals[zero_cross_up] = AOSignal.ZERO_LINE_CROSS_UP.value
        signals[zero_cross_down] = AOSignal.ZERO_LINE_CROSS_DOWN.value
        
        # Twin Peaks patterns
        twin_peaks_bull = self._detect_twin_peaks_bullish(ao)
        twin_peaks_bear = self._detect_twin_peaks_bearish(ao)
        
        signals[twin_peaks_bull] = AOSignal.TWIN_PEAKS_BULLISH.value
        signals[twin_peaks_bear] = AOSignal.TWIN_PEAKS_BEARISH.value
        
        # Saucer patterns
        saucer_bull = self._detect_saucer_bullish(ao)
        saucer_bear = self._detect_saucer_bearish(ao)
        
        signals[saucer_bull] = AOSignal.SAUCER_BULLISH.value
        signals[saucer_bear] = AOSignal.SAUCER_BEARISH.value
        
        return signals
    
    def _detect_twin_peaks_bullish(self, ao: pd.Series) -> pd.Series:
        """
        Detect Twin Peaks bullish pattern:
        - Two consecutive peaks below zero
        - Second peak higher than first peak
        - Confirmation bar above the valley between peaks
        """
        condition = pd.Series(False, index=ao.index)
        
        for i in range(10, len(ao) - 2):
            if ao.iloc[i] < 0:  # Current bar below zero
                # Look for valley and previous peak
                valley_idx = None
                peak1_idx = None
                
                # Find valley before current position
                for j in range(i-1, max(0, i-10), -1):
                    if valley_idx is None and ao.iloc[j] < ao.iloc[j-1] and ao.iloc[j] < ao.iloc[j+1]:
                        valley_idx = j
                    elif valley_idx is not None and ao.iloc[j] > ao.iloc[j-1] and ao.iloc[j] > ao.iloc[j+1]:
                        peak1_idx = j
                        break
                
                if valley_idx is not None and peak1_idx is not None:
                    # Check if current is a peak and higher than previous peak
                    if (ao.iloc[i] > ao.iloc[i-1] and ao.iloc[i] > ao.iloc[i+1] and
                        ao.iloc[i] > ao.iloc[peak1_idx] and
                        i + 1 < len(ao) and ao.iloc[i+1] > ao.iloc[valley_idx]):
                        condition.iloc[i+1] = True
        
        return condition
    
    def _detect_twin_peaks_bearish(self, ao: pd.Series) -> pd.Series:
        """
        Detect Twin Peaks bearish pattern:
        - Two consecutive peaks above zero
        - Second peak lower than first peak
        - Confirmation bar below the valley between peaks
        """
        condition = pd.Series(False, index=ao.index)
        
        for i in range(10, len(ao) - 2):
            if ao.iloc[i] > 0:  # Current bar above zero
                # Look for valley and previous peak
                valley_idx = None
                peak1_idx = None
                
                # Find valley before current position
                for j in range(i-1, max(0, i-10), -1):
                    if valley_idx is None and ao.iloc[j] > ao.iloc[j-1] and ao.iloc[j] > ao.iloc[j+1]:
                        valley_idx = j
                    elif valley_idx is not None and ao.iloc[j] < ao.iloc[j-1] and ao.iloc[j] < ao.iloc[j+1]:
                        peak1_idx = j
                        break
                
                if valley_idx is not None and peak1_idx is not None:
                    # Check if current is a peak and lower than previous peak
                    if (ao.iloc[i] < ao.iloc[i-1] and ao.iloc[i] < ao.iloc[i+1] and
                        ao.iloc[i] < ao.iloc[peak1_idx] and
                        i + 1 < len(ao) and ao.iloc[i+1] < ao.iloc[valley_idx]):
                        condition.iloc[i+1] = True
        
        return condition
    
    def _detect_saucer_bullish(self, ao: pd.Series) -> pd.Series:
        """
        Detect Saucer bullish pattern:
        - Three consecutive bars above zero
        - First bar red, second bar red, third bar green
        - Second bar lower than first bar
        """
        condition = pd.Series(False, index=ao.index)
        
        for i in range(2, len(ao)):
            if (ao.iloc[i] > 0 and ao.iloc[i-1] > 0 and ao.iloc[i-2] > 0 and  # All above zero
                ao.iloc[i-2] > ao.iloc[i-3] and  # First bar green (higher than previous)
                ao.iloc[i-1] < ao.iloc[i-2] and  # Second bar red (lower than first)
                ao.iloc[i] > ao.iloc[i-1]):      # Third bar green (higher than second)
                condition.iloc[i] = True
        
        return condition
    
    def _detect_saucer_bearish(self, ao: pd.Series) -> pd.Series:
        """
        Detect Saucer bearish pattern:
        - Three consecutive bars below zero
        - First bar green, second bar green, third bar red
        - Second bar higher than first bar
        """
        condition = pd.Series(False, index=ao.index)
        
        for i in range(2, len(ao)):
            if (ao.iloc[i] < 0 and ao.iloc[i-1] < 0 and ao.iloc[i-2] < 0 and  # All below zero
                ao.iloc[i-2] < ao.iloc[i-3] and  # First bar red (lower than previous)
                ao.iloc[i-1] > ao.iloc[i-2] and  # Second bar green (higher than first)
                ao.iloc[i] < ao.iloc[i-1]):      # Third bar red (lower than second)
                condition.iloc[i] = True
        
        return condition
    
    def _calculate_histogram(self, ao: pd.Series) -> pd.Series:
        """Calculate AO histogram (difference between consecutive AO values)"""
        return ao - ao.shift(1)
    
    def _calculate_zero_line_distance(self, ao: pd.Series) -> pd.Series:
        """Calculate distance from zero line (normalized)"""
        return np.abs(ao) / ao.rolling(window=50).std()
    
    def _calculate_momentum_strength(self, ao: pd.Series) -> pd.Series:
        """Calculate momentum strength based on AO slope and magnitude"""
        ao_change = ao - ao.shift(1)
        ao_magnitude = np.abs(ao)
        return ao_change * ao_magnitude
    
    def _get_ao_colors(self, ao: pd.Series) -> pd.Series:
        """Get AO bar colors (green for increasing, red for decreasing)"""
        colors = pd.Series('red', index=ao.index)
        colors[ao > ao.shift(1)] = 'green'
        return colors
    
    def get_trading_signals(self, high: Union[pd.Series, np.ndarray], 
                           low: Union[pd.Series, np.ndarray],
                           strength_threshold: float = 0.5) -> Dict:
        """
        Get comprehensive trading signals with entry/exit points
        
        Args:
            high: High prices
            low: Low prices
            strength_threshold: Minimum strength required for signal confirmation
            
        Returns:
            Dictionary with buy/sell signals and strength ratings
        """
        result = self.calculate_with_signals(high, low)
        ao = result['ao']
        signals = result['signals']
        momentum_strength = result['momentum_strength']
        
        # Generate buy/sell signals
        buy_signals = pd.Series(False, index=ao.index)
        sell_signals = pd.Series(False, index=ao.index)
        signal_strength = pd.Series(0.0, index=ao.index)
        
        # Strong buy signals
        strong_buy = (
            (signals == AOSignal.ZERO_LINE_CROSS_UP.value) |
            (signals == AOSignal.TWIN_PEAKS_BULLISH.value) |
            (signals == AOSignal.SAUCER_BULLISH.value)
        )
        
        # Strong sell signals
        strong_sell = (
            (signals == AOSignal.ZERO_LINE_CROSS_DOWN.value) |
            (signals == AOSignal.TWIN_PEAKS_BEARISH.value) |
            (signals == AOSignal.SAUCER_BEARISH.value)
        )
        
        # Apply strength filter
        buy_signals = strong_buy & (np.abs(momentum_strength) > strength_threshold)
        sell_signals = strong_sell & (np.abs(momentum_strength) > strength_threshold)
        
        # Calculate signal strength
        signal_strength[buy_signals] = momentum_strength[buy_signals]
        signal_strength[sell_signals] = -momentum_strength[sell_signals]
        
        return {
            'buy_signals': buy_signals,
            'sell_signals': sell_signals,
            'signal_strength': signal_strength,
            'ao_value': ao,
            'ao_above_zero': result['ao_above_zero'],
            'ao_increasing': result['ao_increasing'],
            'signal_type': signals
        }
    
    def calculate_divergence(self, high: Union[pd.Series, np.ndarray], 
                           low: Union[pd.Series, np.ndarray],
                           close: Union[pd.Series, np.ndarray],
                           lookback_period: int = 20) -> Dict:
        """
        Detect bullish and bearish divergences between price and AO
        
        Args:
            high: High prices
            low: Low prices
            close: Close prices
            lookback_period: Period to look back for divergence patterns
            
        Returns:
            Dictionary with divergence signals
        """
        ao = self.calculate(high, low)
        
        if isinstance(close, np.ndarray):
            close = pd.Series(close)
        
        bullish_div = pd.Series(False, index=ao.index)
        bearish_div = pd.Series(False, index=ao.index)
        
        for i in range(lookback_period, len(ao)):
            # Look for price and AO extremes in the lookback period
            price_window = close.iloc[i-lookback_period:i+1]
            ao_window = ao.iloc[i-lookback_period:i+1]
            
            # Bullish divergence: price makes lower low, AO makes higher low
            price_min_idx = price_window.idxmin()
            ao_min_idx = ao_window.idxmin()
            
            if (price_min_idx == close.index[i] and  # Current price is the lowest
                ao.iloc[i] > ao.iloc[ao_min_idx] and  # AO is higher than its minimum
                close.iloc[i] < close.iloc[i-lookback_period//2]):  # Price trend is down
                bullish_div.iloc[i] = True
            
            # Bearish divergence: price makes higher high, AO makes lower high
            price_max_idx = price_window.idxmax()
            ao_max_idx = ao_window.idxmax()
            
            if (price_max_idx == close.index[i] and  # Current price is the highest
                ao.iloc[i] < ao.iloc[ao_max_idx] and  # AO is lower than its maximum
                close.iloc[i] > close.iloc[i-lookback_period//2]):  # Price trend is up
                bearish_div.iloc[i] = True
        
        return {
            'bullish_divergence': bullish_div,
            'bearish_divergence': bearish_div,
            'ao': ao
        }

# Example usage and backtesting functions
def example_usage():
    """Example of how to use the AwesomeOscillator class"""
    
    # Generate sample OHLC data
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=200, freq='D')
    
    # Simulate price movement with trend and volatility
    returns = np.random.normal(0.001, 0.02, 200)
    close_prices = 100 * np.exp(np.cumsum(returns))
    
    # Generate high and low based on close with some randomness
    daily_range = np.abs(np.random.normal(0.01, 0.005, 200))
    high_prices = close_prices * (1 + daily_range)
    low_prices = close_prices * (1 - daily_range)
    
    # Create DataFrame
    df = pd.DataFrame({
        'high': high_prices,
        'low': low_prices,
        'close': close_prices
    }, index=dates)
    
    # Initialize Awesome Oscillator
    ao = AwesomeOscillator(fast_period=5, slow_period=34)
    
    # Calculate AO with signals
    ao_result = ao.calculate_with_signals(df['high'], df['low'])
    
    # Add results to DataFrame
    df['ao'] = ao_result['ao']
    df['ao_signals'] = ao_result['signals']
    df['ao_histogram'] = ao_result['histogram']
    df['ao_color'] = ao_result['ao_color']
    
    # Get trading signals
    trading_signals = ao.get_trading_signals(df['high'], df['low'])
    df['buy_signals'] = trading_signals['buy_signals']
    df['sell_signals'] = trading_signals['sell_signals']
    df['signal_strength'] = trading_signals['signal_strength']
    
    # Calculate divergences
    divergences = ao.calculate_divergence(df['high'], df['low'], df['close'])
    df['bullish_divergence'] = divergences['bullish_divergence']
    df['bearish_divergence'] = divergences['bearish_divergence']
    
    print("Awesome Oscillator Analysis:")
    print("=" * 50)
    print(f"Total Buy Signals: {df['buy_signals'].sum()}")
    print(f"Total Sell Signals: {df['sell_signals'].sum()}")
    print(f"Bullish Divergences: {df['bullish_divergence'].sum()}")
    print(f"Bearish Divergences: {df['bearish_divergence'].sum()}")
    print("\nRecent Data:")
    print(df[['close', 'ao', 'ao_signals', 'buy_signals', 'sell_signals']].tail(10))
    
    return df

if __name__ == "__main__":
    sample_data = example_usage()