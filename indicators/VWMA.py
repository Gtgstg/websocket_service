import pandas as pd
import numpy as np
from typing import Union, Optional
from collections import deque

class VWMA:
    """
    Volume Weighted Moving Average (VWMA) Indicator
    
    VWMA gives more weight to periods with higher volume, making it more responsive
    to price movements that occur on high volume.
    
    Formula: VWMA = Σ(Price × Volume) / Σ(Volume) over n periods
    """
    
    def __init__(self, period: int = 20):
        """
        Initialize VWMA indicator
        
        Args:
            period: Number of periods for the moving average
        """
        self.period = period
        self.prices = deque(maxlen=period)
        self.volumes = deque(maxlen=period)
        self.vwma_values = []
        
    def update(self, price: float, volume: float) -> Optional[float]:
        """
        Update VWMA with new price and volume data
        
        Args:
            price: Current price (typically close price)
            volume: Current volume
            
        Returns:
            Current VWMA value or None if insufficient data
        """
        self.prices.append(price)
        self.volumes.append(volume)
        
        if len(self.prices) < self.period:
            return None
            
        # Calculate VWMA
        price_volume_sum = sum(p * v for p, v in zip(self.prices, self.volumes))
        volume_sum = sum(self.volumes)
        
        if volume_sum == 0:
            return None
            
        vwma = price_volume_sum / volume_sum
        self.vwma_values.append(vwma)
        
        return vwma
    
    def reset(self):
        """Reset the indicator"""
        self.prices.clear()
        self.volumes.clear()
        self.vwma_values.clear()
    
    @property
    def current_value(self) -> Optional[float]:
        """Get the current VWMA value"""
        return self.vwma_values[-1] if self.vwma_values else None


def calculate_vwma_series(prices: Union[list, pd.Series], 
                         volumes: Union[list, pd.Series], 
                         period: int = 20) -> pd.Series:
    """
    Calculate VWMA for entire price/volume series
    
    Args:
        prices: Series of prices
        volumes: Series of volumes
        period: Period for VWMA calculation
        
    Returns:
        Series of VWMA values
    """
    if len(prices) != len(volumes):
        raise ValueError("Prices and volumes must have same length")
    
    prices = pd.Series(prices) if not isinstance(prices, pd.Series) else prices
    volumes = pd.Series(volumes) if not isinstance(volumes, pd.Series) else volumes
    
    vwma_values = []
    
    for i in range(len(prices)):
        if i < period - 1:
            vwma_values.append(np.nan)
        else:
            start_idx = i - period + 1
            price_slice = prices.iloc[start_idx:i+1]
            volume_slice = volumes.iloc[start_idx:i+1]
            
            price_volume_sum = (price_slice * volume_slice).sum()
            volume_sum = volume_slice.sum()
            
            if volume_sum == 0:
                vwma_values.append(np.nan)
            else:
                vwma_values.append(price_volume_sum / volume_sum)
    
    return pd.Series(vwma_values, index=prices.index)


def calculate_vwma_pandas(df: pd.DataFrame, 
                         price_col: str = 'close', 
                         volume_col: str = 'volume', 
                         period: int = 20) -> pd.Series:
    """
    Calculate VWMA using pandas rolling window (more efficient for large datasets)
    
    Args:
        df: DataFrame with price and volume data
        price_col: Name of price column
        volume_col: Name of volume column
        period: Period for VWMA calculation
        
    Returns:
        Series of VWMA values
    """
    price_volume = df[price_col] * df[volume_col]
    
    # Use rolling window for efficient calculation
    numerator = price_volume.rolling(window=period).sum()
    denominator = df[volume_col].rolling(window=period).sum()
    
    # Avoid division by zero
    vwma = numerator / denominator.replace(0, np.nan)
    
    return vwma


class VWMASignalGenerator:
    """
    Signal generator using VWMA for trading decisions
    """
    
    def __init__(self, fast_period: int = 10, slow_period: int = 20):
        """
        Initialize dual VWMA signal generator
        
        Args:
            fast_period: Period for fast VWMA
            slow_period: Period for slow VWMA
        """
        self.fast_vwma = VWMA(fast_period)
        self.slow_vwma = VWMA(slow_period)
        self.signals = []
        
    def update(self, price: float, volume: float) -> Optional[str]:
        """
        Update VWMAs and generate trading signal
        
        Args:
            price: Current price
            volume: Current volume
            
        Returns:
            'BUY', 'SELL', or None
        """
        fast_val = self.fast_vwma.update(price, volume)
        slow_val = self.slow_vwma.update(price, volume)
        
        if fast_val is None or slow_val is None:
            return None
            
        # Generate signals based on VWMA crossover
        if len(self.signals) == 0:
            self.signals.append('HOLD')
            return None
            
        prev_signal = self.signals[-1]
        
        if fast_val > slow_val and prev_signal != 'BUY':
            signal = 'BUY'
        elif fast_val < slow_val and prev_signal != 'SELL':
            signal = 'SELL'
        else:
            signal = 'HOLD'
            
        self.signals.append(signal)
        return signal if signal != 'HOLD' else None


# Example usage and testing
if __name__ == "__main__":
    # Example data
    np.random.seed(42)
    n_points = 100
    
    # Generate sample price and volume data
    prices = 100 + np.cumsum(np.random.randn(n_points) * 0.5)
    volumes = np.random.randint(1000, 10000, n_points)
    
    # Method 1: Using the VWMA class for real-time updates
    print("Real-time VWMA calculation:")
    vwma_indicator = VWMA(period=20)
    
    for i in range(min(25, len(prices))):
        vwma_val = vwma_indicator.update(prices[i], volumes[i])
        if vwma_val is not None:
            print(f"Period {i+1}: Price={prices[i]:.2f}, Volume={volumes[i]}, VWMA={vwma_val:.2f}")
    
    print("\n" + "="*50 + "\n")
    
    # Method 2: Calculate for entire series
    print("Series VWMA calculation:")
    df = pd.DataFrame({
        'close': prices,
        'volume': volumes
    })
    
    df['vwma_20'] = calculate_vwma_pandas(df, period=20)
    df['vwma_10'] = calculate_vwma_pandas(df, period=10)
    
    # Display last 10 rows
    print(df.tail(10)[['close', 'volume', 'vwma_10', 'vwma_20']].round(2))
    
    print("\n" + "="*50 + "\n")
    
    # Method 3: Signal generation
    print("VWMA Signal Generation:")
    signal_gen = VWMASignalGenerator(fast_period=10, slow_period=20)
    
    signals = []
    for i in range(len(prices)):
        signal = signal_gen.update(prices[i], volumes[i])
        if signal:
            print(f"Period {i+1}: {signal} signal generated at price {prices[i]:.2f}")
            signals.append((i+1, signal, prices[i]))
    
    print(f"\nTotal signals generated: {len(signals)}")