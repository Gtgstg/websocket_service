import numpy as np
import pandas as pd
from typing import Union, List, Optional
from collections import deque

class StandardDeviationIndicator:
    """
    Standard Deviation indicator for algorithmic trading.
    
    This indicator calculates the rolling standard deviation of price data,
    which is useful for measuring volatility and identifying potential
    trading opportunities.
    """
    
    def __init__(self, period: int = 20, ddof: int = 1):
        """
        Initialize the Standard Deviation indicator.
        
        Args:
            period (int): Number of periods for calculation (default: 20)
            ddof (int): Delta degrees of freedom (default: 1 for sample std dev)
        """
        self.period = period
        self.ddof = ddof
        self.data_buffer = deque(maxlen=period)
        self.std_values = []
        
    def update(self, value: float) -> Optional[float]:
        """
        Update the indicator with a new price value.
        
        Args:
            value (float): New price value
            
        Returns:
            Optional[float]: Standard deviation value or None if insufficient data
        """
        self.data_buffer.append(value)
        
        if len(self.data_buffer) < self.period:
            return None
        
        std_dev = np.std(list(self.data_buffer), ddof=self.ddof)
        self.std_values.append(std_dev)
        return std_dev
    
    def calculate_series(self, data: Union[List[float], pd.Series, np.ndarray]) -> pd.Series:
        """
        Calculate standard deviation for an entire data series.
        
        Args:
            data: Price data series
            
        Returns:
            pd.Series: Standard deviation values
        """
        if isinstance(data, (list, np.ndarray)):
            data = pd.Series(data)
        
        return data.rolling(window=self.period, min_periods=self.period).std(ddof=self.ddof)
    
    def get_current_value(self) -> Optional[float]:
        """Get the most recent standard deviation value."""
        return self.std_values[-1] if self.std_values else None
    
    def get_history(self, n: int = None) -> List[float]:
        """
        Get historical standard deviation values.
        
        Args:
            n (int): Number of recent values to return (default: all)
            
        Returns:
            List[float]: Historical standard deviation values
        """
        if n is None:
            return self.std_values.copy()
        return self.std_values[-n:] if len(self.std_values) >= n else self.std_values.copy()
    
    def reset(self):
        """Reset the indicator state."""
        self.data_buffer.clear()
        self.std_values.clear()

class AdvancedStdDevIndicator(StandardDeviationIndicator):
    """
    Advanced Standard Deviation indicator with additional features for trading.
    """
    
    def __init__(self, period: int = 20, ddof: int = 1, 
                 upper_threshold: float = 2.0, lower_threshold: float = 0.5):
        """
        Initialize the Advanced Standard Deviation indicator.
        
        Args:
            period (int): Number of periods for calculation
            ddof (int): Delta degrees of freedom
            upper_threshold (float): Upper volatility threshold
            lower_threshold (float): Lower volatility threshold
        """
        super().__init__(period, ddof)
        self.upper_threshold = upper_threshold
        self.lower_threshold = lower_threshold
        self.mean_std = None
        
    def update(self, value: float) -> dict:
        """
        Update indicator and return comprehensive volatility analysis.
        
        Args:
            value (float): New price value
            
        Returns:
            dict: Volatility analysis including signals
        """
        std_dev = super().update(value)
        
        if std_dev is None:
            return {"std_dev": None, "signal": "INSUFFICIENT_DATA"}
        
        # Calculate mean standard deviation for comparison
        if len(self.std_values) >= 10:  # Need some history for mean calculation
            self.mean_std = np.mean(self.std_values[-10:])
        
        # Generate trading signals based on volatility
        signal = self._generate_signal(std_dev)
        
        return {
            "std_dev": std_dev,
            "mean_std": self.mean_std,
            "signal": signal,
            "volatility_ratio": std_dev / self.mean_std if self.mean_std else None,
            "is_high_volatility": std_dev > (self.mean_std * self.upper_threshold) if self.mean_std else False,
            "is_low_volatility": std_dev < (self.mean_std * self.lower_threshold) if self.mean_std else False
        }
    
    def _generate_signal(self, current_std: float) -> str:
        """Generate trading signal based on volatility."""
        if self.mean_std is None:
            return "NEUTRAL"
        
        ratio = current_std / self.mean_std
        
        if ratio > self.upper_threshold:
            return "HIGH_VOLATILITY"  # Consider volatility contraction trades
        elif ratio < self.lower_threshold:
            return "LOW_VOLATILITY"   # Consider breakout trades
        else:
            return "NORMAL_VOLATILITY"

# Example usage and testing
def example_usage():
    """Example of how to use the Standard Deviation indicators."""
    
    # Sample price data (could be from your data feed)
    sample_prices = [100, 101, 99, 102, 98, 103, 97, 104, 96, 105, 
                    95, 106, 94, 107, 93, 108, 92, 109, 91, 110,
                    89, 112, 88, 113, 87, 114, 86, 115, 85, 116]
    
    print("=== Basic Standard Deviation Indicator ===")
    basic_indicator = StandardDeviationIndicator(period=10)
    
    for i, price in enumerate(sample_prices):
        std_dev = basic_indicator.update(price)
        if std_dev is not None:
            print(f"Price: {price:6.2f} | Std Dev: {std_dev:6.4f}")
    
    print(f"\nCurrent Std Dev: {basic_indicator.get_current_value():.4f}")
    print(f"Last 5 values: {[round(x, 4) for x in basic_indicator.get_history(5)]}")
    
    print("\n=== Advanced Standard Deviation Indicator ===")
    advanced_indicator = AdvancedStdDevIndicator(period=10, upper_threshold=1.5, lower_threshold=0.7)
    
    for i, price in enumerate(sample_prices):
        result = advanced_indicator.update(price)
        if result["std_dev"] is not None:
            print(f"Price: {price:6.2f} | Std Dev: {result['std_dev']:6.4f} | Signal: {result['signal']}")
    
    print("\n=== Pandas Series Calculation ===")
    # Calculate for entire series at once
    series_indicator = StandardDeviationIndicator(period=10)
    std_series = series_indicator.calculate_series(sample_prices)
    
    print("Standard Deviation Series:")
    for i, (price, std) in enumerate(zip(sample_prices, std_series)):
        if not pd.isna(std):
            print(f"Index {i:2d}: Price {price:6.2f} | Std Dev: {std:6.4f}")

# Bollinger Bands using Standard Deviation (bonus implementation)
class BollingerBands:
    """
    Bollinger Bands implementation using the Standard Deviation indicator.
    """
    
    def __init__(self, period: int = 20, std_multiplier: float = 2.0):
        self.period = period
        self.std_multiplier = std_multiplier
        self.price_buffer = deque(maxlen=period)
        self.std_indicator = StandardDeviationIndicator(period)
        
    def update(self, price: float) -> dict:
        """
        Update Bollinger Bands with new price.
        
        Returns:
            dict: Contains upper_band, middle_band, lower_band, and position
        """
        self.price_buffer.append(price)
        std_dev = self.std_indicator.update(price)
        
        if std_dev is None or len(self.price_buffer) < self.period:
            return {"upper_band": None, "middle_band": None, "lower_band": None, "position": None}
        
        middle_band = np.mean(list(self.price_buffer))  # Simple Moving Average
        upper_band = middle_band + (std_dev * self.std_multiplier)
        lower_band = middle_band - (std_dev * self.std_multiplier)
        
        # Determine position relative to bands
        if price > upper_band:
            position = "ABOVE_UPPER"
        elif price < lower_band:
            position = "BELOW_LOWER"
        elif price > middle_band:
            position = "UPPER_HALF"
        else:
            position = "LOWER_HALF"
        
        return {
            "upper_band": upper_band,
            "middle_band": middle_band,
            "lower_band": lower_band,
            "position": position,
            "bandwidth": (upper_band - lower_band) / middle_band * 100  # Bandwidth percentage
        }

if __name__ == "__main__":
    example_usage()