import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

@dataclass
class VolumeProfileResult:
    """Container for Volume Profile calculation results"""
    price_levels: np.ndarray
    volume_at_price: np.ndarray
    poc_price: float
    poc_volume: float
    value_area_high: float
    value_area_low: float
    value_area_volume: float
    total_volume: float

class VolumeProfile:
    """
    Volume Profile Indicator for Algorithmic Trading
    
    Calculates volume distribution across price levels and identifies:
    - Point of Control (POC): Price level with highest volume
    - Value Area: Price range containing 70% of total volume
    - Volume at Price (VAP): Volume traded at each price level
    """
    
    def __init__(self, num_bins: int = 50, value_area_percentage: float = 0.70):
        """
        Initialize Volume Profile indicator
        
        Args:
            num_bins: Number of price bins to create
            value_area_percentage: Percentage of volume for Value Area (default 70%)
        """
        self.num_bins = num_bins
        self.value_area_percentage = value_area_percentage
    
    def calculate(self, high: np.ndarray, low: np.ndarray, close: np.ndarray, 
                 volume: np.ndarray) -> VolumeProfileResult:
        """
        Calculate Volume Profile for given OHLCV data
        
        Args:
            high: Array of high prices
            low: Array of low prices  
            close: Array of close prices
            volume: Array of volumes
            
        Returns:
            VolumeProfileResult containing all calculated metrics
        """
        if len(high) != len(low) != len(close) != len(volume):
            raise ValueError("All input arrays must have the same length")
        
        # Calculate price range
        min_price = np.min(low)
        max_price = np.max(high)
        
        # Create price bins
        price_bins = np.linspace(min_price, max_price, self.num_bins + 1)
        price_levels = (price_bins[:-1] + price_bins[1:]) / 2
        
        # Initialize volume at price array
        volume_at_price = np.zeros(self.num_bins)
        
        # Distribute volume across price levels
        for i in range(len(high)):
            h, l, v = high[i], low[i], volume[i]
            
            # Find bins that overlap with this bar's price range
            start_bin = np.searchsorted(price_bins, l, side='right') - 1
            end_bin = np.searchsorted(price_bins, h, side='left')
            
            start_bin = max(0, start_bin)
            end_bin = min(self.num_bins - 1, end_bin)
            
            if start_bin <= end_bin:
                # Calculate volume distribution within the price range
                price_range = h - l
                if price_range > 0:
                    for bin_idx in range(start_bin, end_bin + 1):
                        bin_low = price_bins[bin_idx]
                        bin_high = price_bins[bin_idx + 1]
                        
                        # Calculate overlap between bar range and bin range
                        overlap_low = max(l, bin_low)
                        overlap_high = min(h, bin_high)
                        
                        if overlap_high > overlap_low:
                            overlap_ratio = (overlap_high - overlap_low) / price_range
                            volume_at_price[bin_idx] += v * overlap_ratio
                else:
                    # Single price point
                    volume_at_price[start_bin] += v
        
        # Find Point of Control (POC)
        poc_idx = np.argmax(volume_at_price)
        poc_price = price_levels[poc_idx]
        poc_volume = volume_at_price[poc_idx]
        
        # Calculate Value Area
        total_volume = np.sum(volume_at_price)
        target_volume = total_volume * self.value_area_percentage
        
        value_area_high, value_area_low, value_area_volume = self._calculate_value_area(
            price_levels, volume_at_price, poc_idx, target_volume
        )
        
        return VolumeProfileResult(
            price_levels=price_levels,
            volume_at_price=volume_at_price,
            poc_price=poc_price,
            poc_volume=poc_volume,
            value_area_high=value_area_high,
            value_area_low=value_area_low,
            value_area_volume=value_area_volume,
            total_volume=total_volume
        )
    
    def _calculate_value_area(self, price_levels: np.ndarray, volume_at_price: np.ndarray, 
                            poc_idx: int, target_volume: float) -> Tuple[float, float, float]:
        """Calculate Value Area High and Low"""
        
        value_area_volume = volume_at_price[poc_idx]
        upper_idx = poc_idx
        lower_idx = poc_idx
        
        # Expand from POC until we reach target volume
        while value_area_volume < target_volume:
            # Determine which direction to expand
            upper_volume = volume_at_price[upper_idx + 1] if upper_idx + 1 < len(volume_at_price) else 0
            lower_volume = volume_at_price[lower_idx - 1] if lower_idx - 1 >= 0 else 0
            
            if upper_volume == 0 and lower_volume == 0:
                break
            
            # Expand in direction with higher volume
            if upper_volume >= lower_volume and upper_idx + 1 < len(volume_at_price):
                upper_idx += 1
                value_area_volume += volume_at_price[upper_idx]
            elif lower_idx - 1 >= 0:
                lower_idx -= 1
                value_area_volume += volume_at_price[lower_idx]
            else:
                break
        
        value_area_high = price_levels[upper_idx]
        value_area_low = price_levels[lower_idx]
        
        return value_area_high, value_area_low, value_area_volume
    
    def get_support_resistance_levels(self, result: VolumeProfileResult, 
                                    min_volume_threshold: float = 0.05) -> Dict[str, List[float]]:
        """
        Identify potential support and resistance levels from volume profile
        
        Args:
            result: VolumeProfileResult from calculate()
            min_volume_threshold: Minimum volume threshold as percentage of total volume
            
        Returns:
            Dictionary with 'support' and 'resistance' price levels
        """
        min_volume = result.total_volume * min_volume_threshold
        
        # Find local volume peaks
        significant_levels = []
        for i in range(1, len(result.volume_at_price) - 1):
            if (result.volume_at_price[i] > result.volume_at_price[i-1] and 
                result.volume_at_price[i] > result.volume_at_price[i+1] and
                result.volume_at_price[i] >= min_volume):
                significant_levels.append(result.price_levels[i])
        
        # Sort levels and classify as support/resistance based on current price context
        significant_levels.sort()
        
        return {
            'support': significant_levels,
            'resistance': significant_levels,
            'poc': [result.poc_price],
            'value_area_high': [result.value_area_high],
            'value_area_low': [result.value_area_low]
        }

# Example usage and testing functions
def create_sample_data(n_bars: int = 100) -> pd.DataFrame:
    """Create sample OHLCV data for testing"""
    np.random.seed(42)
    
    # Generate price data with trend and noise
    base_price = 100
    price_data = []
    current_price = base_price
    
    for i in range(n_bars):
        # Add some trend and randomness
        change = np.random.normal(0, 0.5) + 0.02  # Small upward bias
        current_price += change
        
        # Generate OHLC from current price
        volatility = np.random.uniform(0.5, 2.0)
        high = current_price + np.random.uniform(0, volatility)
        low = current_price - np.random.uniform(0, volatility)
        close = current_price + np.random.uniform(-volatility/2, volatility/2)
        
        # Ensure OHLC relationships are valid
        high = max(high, current_price, close)
        low = min(low, current_price, close)
        
        # Generate volume (higher volume near significant price levels)
        volume = np.random.uniform(1000, 5000)
        if i % 10 == 0:  # Simulate higher volume every 10 bars
            volume *= 2
        
        price_data.append({
            'high': high,
            'low': low,
            'close': close,
            'volume': volume
        })
        
        current_price = close
    
    return pd.DataFrame(price_data)

def analyze_volume_profile(df: pd.DataFrame, num_bins: int = 30) -> None:
    """Analyze and display volume profile results"""
    vp = VolumeProfile(num_bins=num_bins)
    
    result = vp.calculate(
        high=df['high'].values,
        low=df['low'].values, 
        close=df['close'].values,
        volume=df['volume'].values
    )
    
    # Get support/resistance levels
    levels = vp.get_support_resistance_levels(result)
    
    print("=== Volume Profile Analysis ===")
    print(f"Point of Control (POC): ${result.poc_price:.2f} (Volume: {result.poc_volume:,.0f})")
    print(f"Value Area High: ${result.value_area_high:.2f}")
    print(f"Value Area Low: ${result.value_area_low:.2f}")
    print(f"Value Area Volume: {result.value_area_volume:,.0f} ({result.value_area_volume/result.total_volume*100:.1f}%)")
    print(f"Total Volume: {result.total_volume:,.0f}")
    print(f"\nKey Support/Resistance Levels: {len(levels['support'])} levels identified")
    
    return result, levels

# Example usage
if __name__ == "__main__":
    # Create sample data
    sample_data = create_sample_data(200)
    
    # Analyze volume profile
    result, levels = analyze_volume_profile(sample_data)
    
    # Example of how to use in trading logic
    current_price = sample_data['close'].iloc[-1]
    print(f"\nCurrent Price: ${current_price:.2f}")
    
    if current_price > result.value_area_high:
        print("Price is above Value Area - potential resistance zone")
    elif current_price < result.value_area_low:
        print("Price is below Value Area - potential support zone")
    else:
        print("Price is within Value Area - neutral zone")
    
    if abs(current_price - result.poc_price) / result.poc_price < 0.01:  # Within 1%
        print("Price is near Point of Control - high activity zone")