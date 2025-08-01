import pandas as pd
import numpy as np
from typing import Union, Optional, List, Tuple
from datetime import datetime, time
import math

class VWAP:
    """
    Volume-Weighted Average Price (VWAP) Indicator
    
    VWAP is the cumulative average price weighted by volume from a starting point.
    Commonly used as a benchmark for institutional trading and intraday strategies.
    
    Formula: VWAP = Σ(Typical Price × Volume) / Σ(Volume)
    Typical Price = (High + Low + Close) / 3
    """
    
    def __init__(self, reset_period: str = 'daily'):
        """
        Initialize VWAP indicator
        
        Args:
            reset_period: When to reset VWAP ('daily', 'weekly', 'session', 'never')
        """
        self.reset_period = reset_period
        self.cumulative_pv = 0.0  # Price × Volume cumulative sum
        self.cumulative_volume = 0.0  # Volume cumulative sum
        self.vwap_values = []
        self.last_reset_time = None
        self.session_start = None
        
    def _should_reset(self, timestamp: Optional[datetime] = None) -> bool:
        """Check if VWAP should be reset based on the reset period"""
        if self.reset_period == 'never':
            return False
            
        if timestamp is None:
            return False
            
        if self.last_reset_time is None:
            self.last_reset_time = timestamp
            return True
            
        if self.reset_period == 'daily':
            return timestamp.date() != self.last_reset_time.date()
        elif self.reset_period == 'weekly':
            return timestamp.isocalendar()[1] != self.last_reset_time.isocalendar()[1]
        elif self.reset_period == 'session':
            # Reset at market open (9:30 AM ET example)
            market_open = time(9, 30)
            if (timestamp.time() >= market_open and 
                self.last_reset_time.time() < market_open):
                return True
                
        return False
    
    def update(self, high: float, low: float, close: float, volume: float, 
               timestamp: Optional[datetime] = None) -> float:
        """
        Update VWAP with new OHLCV data
        
        Args:
            high: High price
            low: Low price  
            close: Close price
            volume: Volume
            timestamp: Optional timestamp for reset logic
            
        Returns:
            Current VWAP value
        """
        # Check if we should reset
        if self._should_reset(timestamp):
            self.reset()
            self.last_reset_time = timestamp
            
        # Calculate typical price
        typical_price = (high + low + close) / 3.0
        
        # Update cumulative values
        pv = typical_price * volume
        self.cumulative_pv += pv
        self.cumulative_volume += volume
        
        # Calculate VWAP
        if self.cumulative_volume == 0:
            vwap = typical_price
        else:
            vwap = self.cumulative_pv / self.cumulative_volume
            
        self.vwap_values.append(vwap)
        return vwap
    
    def reset(self):
        """Reset VWAP calculation"""
        self.cumulative_pv = 0.0
        self.cumulative_volume = 0.0
        
    @property
    def current_value(self) -> Optional[float]:
        """Get current VWAP value"""
        return self.vwap_values[-1] if self.vwap_values else None


class VWAPBands:
    """
    VWAP with standard deviation bands
    """
    
    def __init__(self, num_bands: int = 2, reset_period: str = 'daily'):
        """
        Initialize VWAP with bands
        
        Args:
            num_bands: Number of standard deviation bands (1, 2, 3)
            reset_period: When to reset VWAP
        """
        self.vwap = VWAP(reset_period)
        self.num_bands = num_bands
        self.price_data = []
        self.volume_data = []
        
    def update(self, high: float, low: float, close: float, volume: float, 
               timestamp: Optional[datetime] = None) -> dict:
        """
        Update VWAP and calculate bands
        
        Returns:
            Dictionary with VWAP and band values
        """
        # Update base VWAP
        vwap_val = self.vwap.update(high, low, close, volume, timestamp)
        
        # Store data for standard deviation calculation
        typical_price = (high + low + close) / 3.0
        self.price_data.append(typical_price)
        self.volume_data.append(volume)
        
        # Reset price data when VWAP resets
        if self.vwap.cumulative_volume == volume:  # First data point after reset
            self.price_data = [typical_price]
            self.volume_data = [volume]
        
        # Calculate volume-weighted standard deviation
        if len(self.price_data) < 2:
            std_dev = 0
        else:
            std_dev = self._calculate_vw_std_dev(vwap_val)
        
        # Calculate bands
        result = {'vwap': vwap_val}
        for i in range(1, self.num_bands + 1):
            result[f'upper_band_{i}'] = vwap_val + (i * std_dev)
            result[f'lower_band_{i}'] = vwap_val - (i * std_dev)
            
        return result
    
    def _calculate_vw_std_dev(self, vwap: float) -> float:
        """Calculate volume-weighted standard deviation"""
        if len(self.price_data) < 2:
            return 0.0
            
        total_volume = sum(self.volume_data)
        if total_volume == 0:
            return 0.0
            
        # Volume-weighted variance
        variance_sum = 0.0
        for price, volume in zip(self.price_data, self.volume_data):
            variance_sum += volume * (price - vwap) ** 2
            
        vw_variance = variance_sum / total_volume
        return math.sqrt(max(0, vw_variance))


def calculate_vwap_series(df: pd.DataFrame, 
                         reset_period: str = 'daily',
                         high_col: str = 'high',
                         low_col: str = 'low', 
                         close_col: str = 'close',
                         volume_col: str = 'volume',
                         timestamp_col: Optional[str] = None) -> pd.Series:
    """
    Calculate VWAP for entire DataFrame
    
    Args:
        df: DataFrame with OHLCV data
        reset_period: When to reset VWAP calculation
        high_col, low_col, close_col, volume_col: Column names
        timestamp_col: Optional timestamp column for reset logic
        
    Returns:
        Series of VWAP values
    """
    vwap_indicator = VWAP(reset_period)
    vwap_values = []
    
    for idx, row in df.iterrows():
        timestamp = None
        if timestamp_col and timestamp_col in df.columns:
            timestamp = pd.to_datetime(row[timestamp_col])
        elif isinstance(idx, pd.Timestamp):
            timestamp = idx
            
        vwap_val = vwap_indicator.update(
            row[high_col], row[low_col], row[close_col], 
            row[volume_col], timestamp
        )
        vwap_values.append(vwap_val)
        
    return pd.Series(vwap_values, index=df.index)


class VWAPSignalGenerator:
    """
    Signal generator using VWAP for trading decisions
    """
    
    def __init__(self, 
                 use_bands: bool = True, 
                 num_bands: int = 2,
                 reset_period: str = 'daily'):
        """
        Initialize VWAP signal generator
        
        Args:
            use_bands: Whether to use VWAP bands for signals
            num_bands: Number of standard deviation bands
            reset_period: When to reset VWAP
        """
        self.use_bands = use_bands
        if use_bands:
            self.vwap_bands = VWAPBands(num_bands, reset_period)
        else:
            self.vwap = VWAP(reset_period)
            
        self.last_price = None
        self.last_vwap_data = None
        
    def update(self, high: float, low: float, close: float, volume: float,
               timestamp: Optional[datetime] = None) -> dict:
        """
        Update VWAP and generate trading signals
        
        Returns:
            Dictionary with signals and VWAP data
        """
        if self.use_bands:
            vwap_data = self.vwap_bands.update(high, low, close, volume, timestamp)
            vwap_val = vwap_data['vwap']
        else:
            vwap_val = self.vwap.update(high, low, close, volume, timestamp)
            vwap_data = {'vwap': vwap_val}
            
        signals = self._generate_signals(close, vwap_data)
        
        self.last_price = close
        self.last_vwap_data = vwap_data
        
        return {
            'signals': signals,
            'vwap_data': vwap_data,
            'price': close
        }
    
    def _generate_signals(self, price: float, vwap_data: dict) -> List[str]:
        """Generate trading signals based on VWAP"""
        signals = []
        vwap_val = vwap_data['vwap']
        
        # Basic VWAP signals
        if price > vwap_val:
            signals.append('ABOVE_VWAP')
        elif price < vwap_val:
            signals.append('BELOW_VWAP')
            
        # Band-based signals
        if self.use_bands and 'upper_band_1' in vwap_data:
            if price > vwap_data['upper_band_2']:
                signals.append('OVERBOUGHT_EXTREME')
            elif price > vwap_data['upper_band_1']:
                signals.append('OVERBOUGHT')
            elif price < vwap_data['lower_band_2']:
                signals.append('OVERSOLD_EXTREME')
            elif price < vwap_data['lower_band_1']:
                signals.append('OVERSOLD')
                
        # Trend signals (requires previous data)
        if self.last_price and self.last_vwap_data:
            last_vwap = self.last_vwap_data['vwap']
            
            # Price crossing VWAP
            if (self.last_price <= last_vwap and price > vwap_val):
                signals.append('BULLISH_CROSS')
            elif (self.last_price >= last_vwap and price < vwap_val):
                signals.append('BEARISH_CROSS')
                
        return signals


# Utility functions for common VWAP strategies
def vwap_mean_reversion_signal(price: float, vwap: float, 
                              upper_band: float, lower_band: float) -> Optional[str]:
    """
    Mean reversion signal based on VWAP bands
    
    Returns:
        'BUY', 'SELL', or None
    """
    if price <= lower_band:
        return 'BUY'  # Oversold, expect reversion to VWAP
    elif price >= upper_band:
        return 'SELL'  # Overbought, expect reversion to VWAP
    return None


def vwap_trend_following_signal(price: float, vwap: float, 
                               prev_price: float, prev_vwap: float) -> Optional[str]:
    """
    Trend following signal based on VWAP
    
    Returns:
        'BUY', 'SELL', or None
    """
    # Bullish: Price above VWAP and VWAP trending up
    if price > vwap and vwap > prev_vwap:
        return 'BUY'
    # Bearish: Price below VWAP and VWAP trending down  
    elif price < vwap and vwap < prev_vwap:
        return 'SELL'
    return None


# Example usage and testing
if __name__ == "__main__":
    # Generate sample data
    np.random.seed(42)
    n_points = 200
    
    # Create realistic intraday price data
    base_price = 100
    prices = []
    volumes = []
    timestamps = []
    
    current_time = datetime(2024, 1, 15, 9, 30)  # Market open
    
    for i in range(n_points):
        # Simulate intraday price movement
        if i == 0:
            price = base_price
        else:
            price = prices[-1] + np.random.randn() * 0.5
            
        high = price + abs(np.random.randn() * 0.3)
        low = price - abs(np.random.randn() * 0.3) 
        close = price + np.random.randn() * 0.2
        volume = np.random.randint(1000, 5000)
        
        prices.append([high, low, close])
        volumes.append(volume)
        timestamps.append(current_time)
        
        # Increment time by 1 minute
        current_time = current_time.replace(minute=current_time.minute + 1)
        if current_time.minute >= 60:
            current_time = current_time.replace(hour=current_time.hour + 1, minute=0)
    
    # Create DataFrame
    df = pd.DataFrame({
        'timestamp': timestamps,
        'high': [p[0] for p in prices],
        'low': [p[1] for p in prices], 
        'close': [p[2] for p in prices],
        'volume': volumes
    })
    
    print("VWAP Indicator Demo")
    print("=" * 50)
    
    # Demo 1: Basic VWAP calculation
    print("1. Basic VWAP Calculation:")
    vwap_indicator = VWAP(reset_period='daily')
    
    for i in range(10):
        row = df.iloc[i]
        vwap_val = vwap_indicator.update(
            row['high'], row['low'], row['close'], row['volume'], row['timestamp']
        )
        print(f"Time: {row['timestamp'].strftime('%H:%M')}, "
              f"Close: {row['close']:.2f}, VWAP: {vwap_val:.2f}")
    
    print("\n" + "=" * 50 + "\n")
    
    # Demo 2: VWAP with bands
    print("2. VWAP with Standard Deviation Bands:")
    vwap_bands = VWAPBands(num_bands=2, reset_period='daily')
    
    for i in range(20, 30):
        row = df.iloc[i]
        result = vwap_bands.update(
            row['high'], row['low'], row['close'], row['volume'], row['timestamp']
        )
        print(f"Time: {row['timestamp'].strftime('%H:%M')}, "
              f"Close: {row['close']:.2f}, VWAP: {result['vwap']:.2f}, "
              f"Upper: {result['upper_band_1']:.2f}, Lower: {result['lower_band_1']:.2f}")
    
    print("\n" + "=" * 50 + "\n")
    
    # Demo 3: Signal generation
    print("3. VWAP Signal Generation:")
    signal_gen = VWAPSignalGenerator(use_bands=True, num_bands=2)
    
    signals_generated = []
    for i in range(50, 70):
        row = df.iloc[i]
        result = signal_gen.update(
            row['high'], row['low'], row['close'], row['volume'], row['timestamp']
        )
        
        if result['signals']:
            print(f"Time: {row['timestamp'].strftime('%H:%M')}, "
                  f"Price: {row['close']:.2f}, "
                  f"Signals: {', '.join(result['signals'])}")
            signals_generated.extend(result['signals'])
    
    print(f"\nUnique signals generated: {set(signals_generated)}")
    
    print("\n" + "=" * 50 + "\n")
    
    # Demo 4: DataFrame calculation
    print("4. DataFrame VWAP Calculation:")
    df['vwap'] = calculate_vwap_series(df, reset_period='daily', timestamp_col='timestamp')
    
    print("Last 10 rows:")
    print(df[['timestamp', 'close', 'volume', 'vwap']].tail(10).round(2))