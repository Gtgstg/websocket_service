import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

class TrendDirection(Enum):
    UPTREND = "uptrend"
    DOWNTREND = "downtrend"

@dataclass
class FibLevel:
    """Represents a Fibonacci retracement level"""
    ratio: float
    price: float
    label: str

@dataclass
class SwingPoint:
    """Represents a swing high/low point"""
    index: int
    price: float
    timestamp: Optional[str] = None

class FibonacciRetracement:
    """
    Fibonacci Retracement tool for algorithmic trading
    
    Features:
    - Automatic swing high/low detection
    - Multiple Fibonacci levels calculation
    - Support and resistance level identification
    - Trading signal generation
    """
    
    # Standard Fibonacci ratios
    DEFAULT_RATIOS = [0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0]
    EXTENDED_RATIOS = [0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0, 1.272, 1.618, 2.618]
    
    def __init__(self, ratios: List[float] = None, lookback_period: int = 20):
        """
        Initialize Fibonacci Retracement tool
        
        Args:
            ratios: List of Fibonacci ratios to calculate
            lookback_period: Period for swing point detection
        """
        self.ratios = ratios or self.DEFAULT_RATIOS
        self.lookback_period = lookback_period
        self.fib_levels = []
        
    def find_swing_points(self, data: pd.Series, 
                         swing_type: str = "both") -> Dict[str, List[SwingPoint]]:
        """
        Find swing highs and lows in price data
        
        Args:
            data: Price data (typically close prices)
            swing_type: "high", "low", or "both"
            
        Returns:
            Dictionary with swing highs and lows
        """
        highs = []
        lows = []
        
        for i in range(self.lookback_period, len(data) - self.lookback_period):
            # Check for swing high
            if swing_type in ["high", "both"]:
                is_high = True
                for j in range(i - self.lookback_period, i + self.lookback_period + 1):
                    if j != i and data.iloc[j] >= data.iloc[i]:
                        is_high = False
                        break
                if is_high:
                    highs.append(SwingPoint(i, data.iloc[i]))
            
            # Check for swing low
            if swing_type in ["low", "both"]:
                is_low = True
                for j in range(i - self.lookback_period, i + self.lookback_period + 1):
                    if j != i and data.iloc[j] <= data.iloc[i]:
                        is_low = False
                        break
                if is_low:
                    lows.append(SwingPoint(i, data.iloc[i]))
        
        return {"highs": highs, "lows": lows}
    
    def get_latest_swing_points(self, data: pd.Series) -> Tuple[SwingPoint, SwingPoint]:
        """
        Get the most recent significant swing high and low
        
        Args:
            data: Price data
            
        Returns:
            Tuple of (swing_high, swing_low)
        """
        swing_points = self.find_swing_points(data)
        
        if not swing_points["highs"] or not swing_points["lows"]:
            raise ValueError("Not enough swing points found in data")
        
        # Get most recent swing points
        latest_high = max(swing_points["highs"], key=lambda x: x.index)
        latest_low = max(swing_points["lows"], key=lambda x: x.index)
        
        return latest_high, latest_low
    
    def calculate_fibonacci_levels(self, high_point: SwingPoint, 
                                 low_point: SwingPoint) -> List[FibLevel]:
        """
        Calculate Fibonacci retracement levels
        
        Args:
            high_point: Swing high point
            low_point: Swing low point
            
        Returns:
            List of Fibonacci levels
        """
        price_range = high_point.price - low_point.price
        
        # Determine trend direction
        if high_point.index > low_point.index:
            trend = TrendDirection.UPTREND
            base_price = low_point.price
        else:
            trend = TrendDirection.DOWNTREND
            base_price = high_point.price
            price_range = -price_range
        
        fib_levels = []
        for ratio in self.ratios:
            if trend == TrendDirection.UPTREND:
                level_price = high_point.price - (price_range * ratio)
            else:
                level_price = low_point.price + (price_range * ratio)
            
            fib_levels.append(FibLevel(
                ratio=ratio,
                price=level_price,
                label=f"Fib {ratio:.1%}"
            ))
        
        self.fib_levels = fib_levels
        return fib_levels
    
    def get_support_resistance_levels(self, current_price: float) -> Dict[str, List[FibLevel]]:
        """
        Identify support and resistance levels based on current price
        
        Args:
            current_price: Current market price
            
        Returns:
            Dictionary with support and resistance levels
        """
        if not self.fib_levels:
            raise ValueError("No Fibonacci levels calculated")
        
        support_levels = []
        resistance_levels = []
        
        for level in self.fib_levels:
            if level.price < current_price:
                support_levels.append(level)
            else:
                resistance_levels.append(level)
        
        # Sort support levels (highest first)
        support_levels.sort(key=lambda x: x.price, reverse=True)
        # Sort resistance levels (lowest first)
        resistance_levels.sort(key=lambda x: x.price)
        
        return {
            "support": support_levels,
            "resistance": resistance_levels
        }
    
    def generate_trading_signals(self, data: pd.Series, 
                               current_price: float,
                               tolerance: float = 0.001) -> Dict[str, any]:
        """
        Generate trading signals based on Fibonacci levels
        
        Args:
            data: Price data
            current_price: Current market price
            tolerance: Price tolerance for level detection (as percentage)
            
        Returns:
            Dictionary with trading signals and analysis
        """
        if not self.fib_levels:
            # Auto-calculate if not done
            high, low = self.get_latest_swing_points(data)
            self.calculate_fibonacci_levels(high, low)
        
        levels = self.get_support_resistance_levels(current_price)
        
        signals = {
            "timestamp": pd.Timestamp.now(),
            "current_price": current_price,
            "signal": "HOLD",
            "confidence": 0.0,
            "nearest_support": None,
            "nearest_resistance": None,
            "key_levels": [],
            "analysis": ""
        }
        
        # Find nearest levels
        if levels["support"]:
            signals["nearest_support"] = levels["support"][0]
        if levels["resistance"]:
            signals["nearest_resistance"] = levels["resistance"][0]
        
        # Check if price is near a Fibonacci level
        for level in self.fib_levels:
            price_diff = abs(current_price - level.price) / current_price
            if price_diff <= tolerance:
                signals["key_levels"].append(level)
                
                # Generate signals based on key Fibonacci ratios
                if level.ratio in [0.382, 0.5, 0.618]:  # Strong retracement levels
                    if current_price > level.price:
                        signals["signal"] = "BUY"
                        signals["confidence"] = 0.7 + (0.3 * (1 - price_diff / tolerance))
                        signals["analysis"] = f"Price bouncing off {level.label} support level"
                    else:
                        signals["signal"] = "SELL"
                        signals["confidence"] = 0.7 + (0.3 * (1 - price_diff / tolerance))
                        signals["analysis"] = f"Price rejected at {level.label} resistance level"
        
        return signals
    
    def analyze_trend_strength(self, data: pd.Series) -> Dict[str, any]:
        """
        Analyze trend strength using Fibonacci levels
        
        Args:
            data: Price data
            
        Returns:
            Trend analysis
        """
        high, low = self.get_latest_swing_points(data)
        levels = self.calculate_fibonacci_levels(high, low)
        
        current_price = data.iloc[-1]
        
        # Determine trend direction and strength
        if high.index > low.index:
            trend_direction = "UPTREND"
            retracement_level = (high.price - current_price) / (high.price - low.price)
        else:
            trend_direction = "DOWNTREND"
            retracement_level = (current_price - low.price) / (high.price - low.price)
        
        # Classify trend strength
        if retracement_level < 0.382:
            strength = "STRONG"
        elif retracement_level < 0.618:
            strength = "MODERATE"
        else:
            strength = "WEAK"
        
        return {
            "trend_direction": trend_direction,
            "trend_strength": strength,
            "retracement_level": retracement_level,
            "swing_high": high.price,
            "swing_low": low.price,
            "current_price": current_price
        }

# Example usage and testing
def example_usage():
    """Example of how to use the Fibonacci Retracement tool"""
    
    # Create sample price data
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=100, freq='D')
    
    # Simulate price movement with trend
    base_price = 100
    trend = np.linspace(0, 20, 100)  # Upward trend
    noise = np.random.randn(100) * 2
    prices = base_price + trend + noise
    
    price_data = pd.Series(prices, index=dates)
    
    # Initialize Fibonacci tool
    fib_tool = FibonacciRetracement(lookback_period=10)
    
    # Find swing points
    swing_points = fib_tool.find_swing_points(price_data)
    print(f"Found {len(swing_points['highs'])} swing highs and {len(swing_points['lows'])} swing lows")
    
    # Calculate Fibonacci levels
    if swing_points['highs'] and swing_points['lows']:
        latest_high, latest_low = fib_tool.get_latest_swing_points(price_data)
        fib_levels = fib_tool.calculate_fibonacci_levels(latest_high, latest_low)
        
        print("\nFibonacci Retracement Levels:")
        for level in fib_levels:
            print(f"{level.label}: ${level.price:.2f}")
        
        # Generate trading signals
        current_price = price_data.iloc[-1]
        signals = fib_tool.generate_trading_signals(price_data, current_price)
        
        print(f"\nTrading Analysis:")
        print(f"Current Price: ${signals['current_price']:.2f}")
        print(f"Signal: {signals['signal']}")
        print(f"Confidence: {signals['confidence']:.2%}")
        print(f"Analysis: {signals['analysis']}")
        
        # Trend analysis
        trend_analysis = fib_tool.analyze_trend_strength(price_data)
        print(f"\nTrend Analysis:")
        print(f"Direction: {trend_analysis['trend_direction']}")
        print(f"Strength: {trend_analysis['trend_strength']}")
        print(f"Retracement: {trend_analysis['retracement_level']:.1%}")

if __name__ == "__main__":
    example_usage()