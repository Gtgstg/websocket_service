import pandas as pd
import numpy as np
from typing import Union, Tuple, Optional
import warnings

class ChaikinMoneyFlow:
    """
    Chaikin Money Flow (CMF) Technical Indicator
    
    CMF measures buying and selling pressure over a specified period.
    Values above 0 suggest buying pressure, below 0 suggest selling pressure.
    
    Formula:
    1. Money Flow Multiplier = [(Close - Low) - (High - Close)] / (High - Low)
    2. Money Flow Volume = Money Flow Multiplier × Volume
    3. CMF = Sum(Money Flow Volume, n) / Sum(Volume, n)
    """
    
    def __init__(self, period: int = 20):
        """
        Initialize CMF indicator
        
        Args:
            period (int): Lookback period for CMF calculation (default: 20)
        """
        self.period = period
        self.history = []
    
    @staticmethod
    def calculate_cmf(high: Union[pd.Series, np.ndarray], 
                     low: Union[pd.Series, np.ndarray],
                     close: Union[pd.Series, np.ndarray], 
                     volume: Union[pd.Series, np.ndarray],
                     period: int = 20) -> Union[pd.Series, np.ndarray]:
        """
        Calculate Chaikin Money Flow
        
        Args:
            high: High prices
            low: Low prices  
            close: Close prices
            volume: Volume data
            period: Lookback period
            
        Returns:
            CMF values
        """
        # Convert to pandas Series if numpy arrays
        if isinstance(high, np.ndarray):
            high = pd.Series(high)
            low = pd.Series(low)
            close = pd.Series(close)
            volume = pd.Series(volume)
        
        # Calculate Money Flow Multiplier
        # Handle division by zero case
        hl_diff = high - low
        hl_diff = hl_diff.replace(0, np.nan)  # Replace 0 with NaN to avoid division by zero
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            mf_multiplier = ((close - low) - (high - close)) / hl_diff
        
        # Replace NaN values with 0 (when high == low, no price movement)
        mf_multiplier = mf_multiplier.fillna(0)
        
        # Calculate Money Flow Volume
        mf_volume = mf_multiplier * volume
        
        # Calculate CMF using rolling windows
        mf_volume_sum = mf_volume.rolling(window=period, min_periods=1).sum()
        volume_sum = volume.rolling(window=period, min_periods=1).sum()
        
        # Avoid division by zero
        volume_sum = volume_sum.replace(0, np.nan)
        cmf = mf_volume_sum / volume_sum
        cmf = cmf.fillna(0)
        
        return cmf
    
    def update(self, high: float, low: float, close: float, volume: float) -> float:
        """
        Update CMF with new price/volume data (for real-time trading)
        
        Args:
            high: Current period high
            low: Current period low
            close: Current period close
            volume: Current period volume
            
        Returns:
            Current CMF value
        """
        # Add new data point
        self.history.append({
            'high': high,
            'low': low,
            'close': close,
            'volume': volume
        })
        
        # Keep only required history
        if len(self.history) > self.period:
            self.history = self.history[-self.period:]
        
        # Calculate CMF on current history
        if len(self.history) < 1:
            return 0.0
        
        df = pd.DataFrame(self.history)
        cmf_values = self.calculate_cmf(
            df['high'], df['low'], df['close'], df['volume'], 
            min(len(self.history), self.period)
        )
        
        return float(cmf_values.iloc[-1])
    
    def get_signal(self, cmf_value: float, threshold: float = 0.1) -> str:
        """
        Generate trading signal based on CMF value
        
        Args:
            cmf_value: Current CMF value
            threshold: Signal threshold (default: 0.1)
            
        Returns:
            'BUY', 'SELL', or 'HOLD'
        """
        if cmf_value > threshold:
            return 'BUY'
        elif cmf_value < -threshold:
            return 'SELL'
        else:
            return 'HOLD'


class CMFTradingBot:
    """
    Example trading bot using Chaikin Money Flow indicator
    """
    
    def __init__(self, cmf_period: int = 20, signal_threshold: float = 0.1):
        """
        Initialize trading bot
        
        Args:
            cmf_period: CMF calculation period
            signal_threshold: Signal generation threshold
        """
        self.cmf = ChaikinMoneyFlow(period=cmf_period)
        self.threshold = signal_threshold
        self.position = 0  # 0: no position, 1: long, -1: short
        self.entry_price = 0
        self.trades = []
        
    def process_bar(self, high: float, low: float, close: float, volume: float, 
                   timestamp: Optional[str] = None) -> dict:
        """
        Process new price bar and generate trading decisions
        
        Args:
            high: Bar high price
            low: Bar low price
            close: Bar close price
            volume: Bar volume
            timestamp: Optional timestamp
            
        Returns:
            Dictionary with CMF value, signal, and action taken
        """
        # Update CMF
        cmf_value = self.cmf.update(high, low, close, volume)
        
        # Get signal
        signal = self.cmf.get_signal(cmf_value, self.threshold)
        
        # Trading logic
        action = 'HOLD'
        
        if signal == 'BUY' and self.position <= 0:
            if self.position == -1:
                # Close short position
                pnl = self.entry_price - close
                self.trades.append({
                    'type': 'COVER',
                    'price': close,
                    'pnl': pnl,
                    'timestamp': timestamp
                })
            
            # Open long position
            self.position = 1
            self.entry_price = close
            action = 'BUY'
            self.trades.append({
                'type': 'BUY',
                'price': close,
                'pnl': 0,
                'timestamp': timestamp
            })
            
        elif signal == 'SELL' and self.position >= 0:
            if self.position == 1:
                # Close long position
                pnl = close - self.entry_price
                self.trades.append({
                    'type': 'SELL',
                    'price': close,
                    'pnl': pnl,
                    'timestamp': timestamp
                })
            
            # Open short position
            self.position = -1
            self.entry_price = close
            action = 'SHORT'
            self.trades.append({
                'type': 'SHORT',
                'price': close,
                'pnl': 0,
                'timestamp': timestamp
            })
        
        return {
            'cmf': round(cmf_value, 4),
            'signal': signal,
            'action': action,
            'position': self.position,
            'timestamp': timestamp
        }
    
    def get_performance_stats(self) -> dict:
        """Get trading performance statistics"""
        if not self.trades:
            return {'total_trades': 0, 'total_pnl': 0, 'win_rate': 0}
        
        completed_trades = [t for t in self.trades if t['pnl'] != 0]
        total_pnl = sum(t['pnl'] for t in completed_trades)
        winning_trades = len([t for t in completed_trades if t['pnl'] > 0])
        win_rate = (winning_trades / len(completed_trades) * 100) if completed_trades else 0
        
        return {
            'total_trades': len(completed_trades),
            'total_pnl': round(total_pnl, 2),
            'win_rate': round(win_rate, 2),
            'avg_pnl': round(total_pnl / len(completed_trades), 2) if completed_trades else 0
        }


# Example usage and backtesting
def example_usage():
    """
    Example of how to use the CMF indicator and trading bot
    """
    # Sample OHLCV data (replace with your data source)
    sample_data = pd.DataFrame({
        'high': [100, 102, 101, 103, 105, 104, 106, 108, 107, 109, 108, 110, 109, 111, 113],
        'low': [98, 99, 99, 101, 103, 102, 104, 106, 105, 107, 106, 108, 107, 109, 111],
        'close': [99, 101, 100, 102, 104, 103, 105, 107, 106, 108, 107, 109, 108, 110, 112],
        'volume': [1000, 1200, 800, 1500, 1800, 1100, 1600, 2000, 1300, 1700, 1400, 1900, 1200, 2200, 2500]
    })
    
    # Method 1: Calculate CMF for entire dataset
    print("=== CMF Calculation Example ===")
    cmf_values = ChaikinMoneyFlow.calculate_cmf(
        sample_data['high'], 
        sample_data['low'], 
        sample_data['close'], 
        sample_data['volume'], 
        period=10
    )
    
    for i, cmf in enumerate(cmf_values[-5:]):  # Show last 5 values
        print(f"Bar {len(cmf_values)-5+i+1}: CMF = {cmf:.4f}")
    
    print("\n=== Trading Bot Example ===")
    # Method 2: Real-time trading bot simulation
    bot = CMFTradingBot(cmf_period=10, signal_threshold=0.05)
    
    for i, row in sample_data.iterrows():
        result = bot.process_bar(
            row['high'], row['low'], row['close'], row['volume'],
            timestamp=f"Bar_{i+1}"
        )
        print(f"Bar {i+1}: CMF={result['cmf']}, Signal={result['signal']}, Action={result['action']}")
    
    # Show performance stats
    print(f"\n=== Performance Stats ===")
    stats = bot.get_performance_stats()
    for key, value in stats.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    example_usage()