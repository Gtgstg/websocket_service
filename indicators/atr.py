"""
ATR Indicator Module
Provides classes and functions for calculating the Average True Range (ATR) and related signals.
"""
import pandas as pd
import numpy as np
from typing import Union, List, Tuple

class ATRIndicator:
    """
    Average True Range (ATR) Indicator for Algorithmic Trading
    
    ATR measures market volatility by calculating the average of true ranges
    over a specified period. Higher ATR indicates higher volatility.
    """
    
    def __init__(self, period: int = 14):
        """
        Initialize ATR indicator
        
        Args:
            period (int): Period for ATR calculation (default: 14)
        """
        self.period = period
        self.atr_values = []
        
    def calculate_true_range(self, high: float, low: float, prev_close: float) -> float:
        """
        Calculate True Range for a single period
        
        True Range = max(
            high - low,
            abs(high - prev_close),
            abs(low - prev_close)
        )
        
        Args:
            high: Current period high
            low: Current period low
            prev_close: Previous period close
            
        Returns:
            float: True Range value
        """
        tr1 = high - low
        tr2 = abs(high - prev_close)
        tr3 = abs(low - prev_close)
        
        return max(tr1, tr2, tr3)
    
    def calculate_atr_series(self, highs: List[float], lows: List[float], 
                           closes: List[float]) -> List[float]:
        """
        Calculate ATR for a series of price data
        
        Args:
            highs: List of high prices
            lows: List of low prices
            closes: List of close prices
            
        Returns:
            List[float]: ATR values
        """
        if len(highs) != len(lows) or len(highs) != len(closes):
            raise ValueError("All price arrays must have the same length")
        
        if len(highs) < self.period + 1:
            raise ValueError(f"Need at least {self.period + 1} periods of data")
        
        true_ranges = []
        atr_values = []
        
        # Calculate True Range for each period
        for i in range(1, len(highs)):
            tr = self.calculate_true_range(highs[i], lows[i], closes[i-1])
            true_ranges.append(tr)
        
        # Calculate initial ATR (Simple Moving Average of first n True Ranges)
        initial_atr = sum(true_ranges[:self.period]) / self.period
        atr_values.append(initial_atr)
        
        # Calculate subsequent ATR values using Wilder's smoothing
        # ATR = (Previous ATR * (period - 1) + Current TR) / period
        for i in range(self.period, len(true_ranges)):
            current_atr = ((atr_values[-1] * (self.period - 1)) + true_ranges[i]) / self.period
            atr_values.append(current_atr)
        
        return atr_values
    
    def calculate_atr_pandas(self, df: pd.DataFrame, high_col: str = 'high',
                           low_col: str = 'low', close_col: str = 'close') -> pd.Series:
        """
        Calculate ATR using pandas DataFrame
        
        Args:
            df: DataFrame with OHLC data
            high_col: Column name for high prices
            low_col: Column name for low prices
            close_col: Column name for close prices
            
        Returns:
            pd.Series: ATR values
        """
        # Calculate True Range
        df = df.copy()
        df['prev_close'] = df[close_col].shift(1)
        
        df['tr1'] = df[high_col] - df[low_col]
        df['tr2'] = abs(df[high_col] - df['prev_close'])
        df['tr3'] = abs(df[low_col] - df['prev_close'])
        
        df['true_range'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
        
        # Calculate ATR using exponential weighted mean (Wilder's smoothing)
        alpha = 1.0 / self.period
        atr = df['true_range'].ewm(alpha=alpha, adjust=False).mean()
        
        return atr
    
    def update_atr(self, high: float, low: float, close: float, 
                   prev_close: float) -> float:
        """
        Update ATR with new price data (for real-time trading)
        
        Args:
            high: Current high
            low: Current low  
            close: Current close
            prev_close: Previous close
            
        Returns:
            float: Updated ATR value
        """
        tr = self.calculate_true_range(high, low, prev_close)
        
        if not self.atr_values:
            # First value - need to accumulate true ranges
            if not hasattr(self, 'tr_buffer'):
                self.tr_buffer = []
            
            self.tr_buffer.append(tr)
            
            if len(self.tr_buffer) == self.period:
                # Calculate initial ATR
                initial_atr = sum(self.tr_buffer) / self.period
                self.atr_values.append(initial_atr)
                return initial_atr
            else:
                return None
        else:
            # Update ATR using Wilder's smoothing
            current_atr = ((self.atr_values[-1] * (self.period - 1)) + tr) / self.period
            self.atr_values.append(current_atr)
            return current_atr
    
    def get_atr_stop_loss(self, price: float, atr: float, 
                         multiplier: float = 2.0, direction: str = 'long') -> float:
        """
        Calculate stop loss using ATR
        
        Args:
            price: Entry price
            atr: Current ATR value
            multiplier: ATR multiplier for stop distance
            direction: 'long' or 'short'
            
        Returns:
            float: Stop loss level
        """
        atr_distance = atr * multiplier
        
        if direction.lower() == 'long':
            return price - atr_distance
        elif direction.lower() == 'short':
            return price + atr_distance
        else:
            raise ValueError("Direction must be 'long' or 'short'")
    
    def get_atr_position_size(self, account_balance: float, risk_percent: float,
                            entry_price: float, atr: float, 
                            atr_multiplier: float = 2.0) -> int:
        """
        Calculate position size based on ATR risk management
        
        Args:
            account_balance: Total account balance
            risk_percent: Risk percentage (e.g., 0.02 for 2%)
            entry_price: Entry price
            atr: Current ATR value
            atr_multiplier: ATR multiplier for stop distance
            
        Returns:
            int: Position size (number of shares/units)
        """
        risk_amount = account_balance * risk_percent
        atr_distance = atr * atr_multiplier
        
        if atr_distance == 0:
            return 0
        
        position_size = int(risk_amount / atr_distance)
        return position_size
    
    def is_volatile_market(self, current_atr: float, 
                          atr_history: List[float], 
                          volatility_threshold: float = 1.5) -> bool:
        """
        Determine if market is currently volatile based on ATR
        
        Args:
            current_atr: Current ATR value
            atr_history: Historical ATR values
            volatility_threshold: Multiplier for average ATR
            
        Returns:
            bool: True if market is volatile
        """
        if not atr_history:
            return False
        
        avg_atr = sum(atr_history) / len(atr_history)
        return current_atr > (avg_atr * volatility_threshold)


# Example usage and testing
def example_usage():
    """Example of how to use the ATR indicator"""
    
    # Sample OHLC data
    sample_data = {
        'high': [100, 102, 101, 105, 103, 107, 106, 108, 110, 109, 112, 111, 113, 115, 114],
        'low': [98, 99, 99, 101, 100, 103, 104, 105, 107, 106, 109, 108, 110, 112, 111],
        'close': [99, 101, 100, 104, 102, 106, 105, 107, 109, 108, 111, 110, 112, 114, 113]
    }
    
    # Create DataFrame
    df = pd.DataFrame(sample_data)
    
    # Initialize ATR indicator
    atr = ATRIndicator(period=14)
    
    # Method 1: Calculate ATR for entire series
    print("Method 1: Calculate ATR for entire series")
    atr_series = atr.calculate_atr_pandas(df)
    print(f"ATR values: {atr_series.dropna().tolist()}")
    
    # Method 2: Real-time updates
    print("\nMethod 2: Real-time ATR updates")
    atr_realtime = ATRIndicator(period=5)  # Shorter period for demo
    
    for i in range(1, len(df)):
        atr_value = atr_realtime.update_atr(
            df.iloc[i]['high'], 
            df.iloc[i]['low'], 
            df.iloc[i]['close'],
            df.iloc[i-1]['close']
        )
        if atr_value is not None:
            print(f"Period {i}: ATR = {atr_value:.4f}")
    
    # Method 3: Trading applications
    print("\nMethod 3: Trading applications")
    current_atr = 2.5
    entry_price = 110
    
    # Stop loss calculation
    long_stop = atr.get_atr_stop_loss(entry_price, current_atr, 2.0, 'long')
    short_stop = atr.get_atr_stop_loss(entry_price, current_atr, 2.0, 'short')
    print(f"Long position stop loss: {long_stop:.2f}")
    print(f"Short position stop loss: {short_stop:.2f}")
    
    # Position sizing
    position_size = atr.get_atr_position_size(10000, 0.02, entry_price, current_atr, 2.0)
    print(f"Recommended position size: {position_size} shares")
    
    # Volatility check
    atr_history = [2.1, 2.3, 2.0, 2.4, 2.2]
    is_volatile = atr.is_volatile_market(current_atr, atr_history, 1.2)
    print(f"Is market volatile? {is_volatile}")


if __name__ == "__main__":
    example_usage()