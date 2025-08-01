import numpy as np
import pandas as pd
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
from enum import Enum
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.linear_model import LinearRegression

class WedgeType(Enum):
    RISING_WEDGE = "rising_wedge"
    FALLING_WEDGE = "falling_wedge"
    NONE = "none"

@dataclass
class WedgePattern:
    wedge_type: WedgeType
    start_idx: int
    end_idx: int
    upper_trendline: Tuple[float, float]  # (slope, intercept)
    lower_trendline: Tuple[float, float]  # (slope, intercept)
    convergence_point: Tuple[float, float]  # (x, y) where lines meet
    strength: float  # Pattern strength (0-1)
    breakout_direction: Optional[str] = None
    volume_confirmation: bool = False

class WedgeDetector:
    def __init__(self, 
                 min_points: int = 4,
                 min_period: int = 20,
                 max_period: int = 100,
                 min_slope_diff: float = 0.0001,
                 max_slope_diff: float = 0.01,
                 r_squared_threshold: float = 0.7,
                 volume_confirmation: bool = True):
        """
        Initialize Wedge Pattern Detector
        
        Args:
            min_points: Minimum number of touch points for each trendline
            min_period: Minimum number of periods for pattern formation
            max_period: Maximum number of periods to look back
            min_slope_diff: Minimum difference between trendline slopes
            max_slope_diff: Maximum difference between trendline slopes
            r_squared_threshold: Minimum R-squared for trendline validity
            volume_confirmation: Whether to use volume for confirmation
        """
        self.min_points = min_points
        self.min_period = min_period
        self.max_period = max_period
        self.min_slope_diff = min_slope_diff
        self.max_slope_diff = max_slope_diff
        self.r_squared_threshold = r_squared_threshold
        self.volume_confirmation = volume_confirmation
    
    def find_pivots(self, data: pd.DataFrame, window: int = 5) -> Dict[str, List[int]]:
        """Find swing highs and lows (pivot points)"""
        highs = []
        lows = []
        
        high_col = 'High' if 'High' in data.columns else 'high'
        low_col = 'Low' if 'Low' in data.columns else 'low'
        
        for i in range(window, len(data) - window):
            # Check for swing high
            is_high = True
            for j in range(i - window, i + window + 1):
                if j != i and data[high_col].iloc[j] >= data[high_col].iloc[i]:
                    is_high = False
                    break
            if is_high:
                highs.append(i)
            
            # Check for swing low
            is_low = True
            for j in range(i - window, i + window + 1):
                if j != i and data[low_col].iloc[j] <= data[low_col].iloc[i]:
                    is_low = False
                    break
            if is_low:
                lows.append(i)
        
        return {'highs': highs, 'lows': lows}
    
    def fit_trendline(self, x_points: List[int], y_points: List[float]) -> Tuple[float, float, float]:
        """Fit a trendline and return slope, intercept, and R-squared"""
        if len(x_points) < 2:
            return 0, 0, 0
        
        x_array = np.array(x_points).reshape(-1, 1)
        y_array = np.array(y_points)
        
        model = LinearRegression()
        model.fit(x_array, y_array)
        
        y_pred = model.predict(x_array)
        r_squared = stats.pearsonr(y_array, y_pred)[0] ** 2
        
        return model.coef_[0], model.intercept_, r_squared
    
    def find_trendline_touches(self, data: pd.DataFrame, pivots: List[int], 
                              is_upper: bool, tolerance: float = 0.002) -> List[int]:
        """Find points that touch a potential trendline"""
        if len(pivots) < 2:
            return []
        
        price_col = 'High' if is_upper else 'Low'
        if price_col not in data.columns:
            price_col = price_col.lower()
        
        best_touches = []
        best_r_squared = 0
        
        # Try different combinations of pivot points
        for i in range(len(pivots)):
            for j in range(i + 1, len(pivots)):
                start_idx, end_idx = pivots[i], pivots[j]
                if end_idx - start_idx < self.min_period:
                    continue
                
                # Get initial trendline from two points
                x_init = [start_idx, end_idx]
                y_init = [data[price_col].iloc[start_idx], data[price_col].iloc[end_idx]]
                
                slope, intercept, _ = self.fit_trendline(x_init, y_init)
                
                # Find all points that touch this trendline
                touches = []
                for k in range(start_idx, end_idx + 1):
                    expected_price = slope * k + intercept
                    actual_price = data[price_col].iloc[k]
                    
                    if is_upper:
                        # For upper trendline, price should be close to or below the line
                        if abs(actual_price - expected_price) / expected_price <= tolerance:
                            touches.append(k)
                    else:
                        # For lower trendline, price should be close to or above the line
                        if abs(actual_price - expected_price) / expected_price <= tolerance:
                            touches.append(k)
                
                if len(touches) >= self.min_points:
                    # Refit trendline with all touching points
                    x_touches = touches
                    y_touches = [data[price_col].iloc[k] for k in touches]
                    _, _, r_squared = self.fit_trendline(x_touches, y_touches)
                    
                    if r_squared > best_r_squared:
                        best_r_squared = r_squared
                        best_touches = touches
        
        return best_touches if best_r_squared >= self.r_squared_threshold else []
    
    def calculate_convergence_point(self, upper_line: Tuple[float, float], 
                                  lower_line: Tuple[float, float]) -> Tuple[float, float]:
        """Calculate where two trendlines converge"""
        slope1, intercept1 = upper_line
        slope2, intercept2 = lower_line
        
        if abs(slope1 - slope2) < 1e-10:  # Parallel lines
            return None, None
        
        x_converge = (intercept2 - intercept1) / (slope1 - slope2)
        y_converge = slope1 * x_converge + intercept1
        
        return x_converge, y_converge
    
    def detect_wedge_pattern(self, data: pd.DataFrame, 
                           end_idx: Optional[int] = None) -> Optional[WedgePattern]:
        """Detect wedge pattern in the given data"""
        if end_idx is None:
            end_idx = len(data) - 1
        
        start_idx = max(0, end_idx - self.max_period)
        subset_data = data.iloc[start_idx:end_idx + 1].copy()
        subset_data.reset_index(drop=True, inplace=True)
        
        # Find pivot points
        pivots = self.find_pivots(subset_data)
        
        if len(pivots['highs']) < 2 or len(pivots['lows']) < 2:
            return None
        
        # Find upper and lower trendlines
        high_col = 'High' if 'High' in subset_data.columns else 'high'
        low_col = 'Low' if 'Low' in subset_data.columns else 'low'
        
        upper_touches = self.find_trendline_touches(subset_data, pivots['highs'], True)
        lower_touches = self.find_trendline_touches(subset_data, pivots['lows'], False)
        
        if len(upper_touches) < self.min_points or len(lower_touches) < self.min_points:
            return None
        
        # Fit final trendlines
        upper_x = upper_touches
        upper_y = [subset_data[high_col].iloc[i] for i in upper_touches]
        upper_slope, upper_intercept, upper_r2 = self.fit_trendline(upper_x, upper_y)
        
        lower_x = lower_touches
        lower_y = [subset_data[low_col].iloc[i] for i in lower_touches]
        lower_slope, lower_intercept, lower_r2 = self.fit_trendline(lower_x, lower_y)
        
        # Check if slopes are converging
        slope_diff = abs(upper_slope - lower_slope)
        if slope_diff < self.min_slope_diff or slope_diff > self.max_slope_diff:
            return None
        
        # Determine wedge type
        wedge_type = WedgeType.NONE
        if upper_slope < 0 and lower_slope < 0 and upper_slope > lower_slope:
            # Both slopes negative, upper slope less steep = Falling Wedge
            wedge_type = WedgeType.FALLING_WEDGE
        elif upper_slope > 0 and lower_slope > 0 and upper_slope < lower_slope:
            # Both slopes positive, upper slope less steep = Rising Wedge
            wedge_type = WedgeType.RISING_WEDGE
        elif upper_slope > 0 and lower_slope < 0:
            # Diverging lines, not a wedge
            return None
        
        if wedge_type == WedgeType.NONE:
            return None
        
        # Calculate convergence point
        convergence_x, convergence_y = self.calculate_convergence_point(
            (upper_slope, upper_intercept), (lower_slope, lower_intercept)
        )
        
        # Calculate pattern strength
        strength = min(upper_r2, lower_r2) * (len(upper_touches) + len(lower_touches)) / 20
        strength = min(strength, 1.0)
        
        # Volume confirmation (if available and enabled)
        volume_confirmation = False
        if self.volume_confirmation and 'Volume' in subset_data.columns:
            recent_volume = subset_data['Volume'].tail(5).mean()
            earlier_volume = subset_data['Volume'].head(10).mean()
            volume_confirmation = recent_volume < earlier_volume * 0.8  # Decreasing volume
        
        return WedgePattern(
            wedge_type=wedge_type,
            start_idx=start_idx + min(min(upper_touches), min(lower_touches)),
            end_idx=start_idx + max(max(upper_touches), max(lower_touches)),
            upper_trendline=(upper_slope, upper_intercept),
            lower_trendline=(lower_slope, lower_intercept),
            convergence_point=(convergence_x, convergence_y),
            strength=strength,
            volume_confirmation=volume_confirmation
        )
    
    def detect_breakout(self, data: pd.DataFrame, pattern: WedgePattern, 
                       current_idx: int) -> Optional[str]:
        """Detect if price has broken out of the wedge pattern"""
        if current_idx <= pattern.end_idx:
            return None
        
        close_col = 'Close' if 'Close' in data.columns else 'close'
        current_price = data[close_col].iloc[current_idx]
        
        # Calculate expected trendline values at current time
        upper_expected = pattern.upper_trendline[0] * current_idx + pattern.upper_trendline[1]
        lower_expected = pattern.lower_trendline[0] * current_idx + pattern.lower_trendline[1]
        
        # Check for breakout
        if current_price > upper_expected * 1.002:  # 0.2% buffer
            return "upward"
        elif current_price < lower_expected * 0.998:  # 0.2% buffer
            return "downward"
        
        return None
    
    def plot_pattern(self, data: pd.DataFrame, pattern: WedgePattern, 
                    title: str = "Wedge Pattern"):
        """Plot the detected wedge pattern"""
        plt.figure(figsize=(12, 8))
        
        close_col = 'Close' if 'Close' in data.columns else 'close'
        
        # Plot price data
        plt.plot(data.index, data[close_col], 'b-', linewidth=1, alpha=0.7, label='Price')
        
        # Plot trendlines
        x_range = np.arange(pattern.start_idx, len(data))
        upper_line = pattern.upper_trendline[0] * x_range + pattern.upper_trendline[1]
        lower_line = pattern.lower_trendline[0] * x_range + pattern.lower_trendline[1]
        
        plt.plot(x_range, upper_line, 'r--', linewidth=2, label='Upper Trendline')
        plt.plot(x_range, lower_line, 'g--', linewidth=2, label='Lower Trendline')
        
        # Highlight pattern area
        plt.axvspan(pattern.start_idx, pattern.end_idx, alpha=0.2, color='yellow')
        
        # Mark convergence point if within visible range
        if (pattern.convergence_point[0] is not None and 
            pattern.start_idx <= pattern.convergence_point[0] <= len(data) * 1.2):
            plt.plot(pattern.convergence_point[0], pattern.convergence_point[1], 
                    'ro', markersize=8, label='Convergence Point')
        
        plt.title(f"{title} - {pattern.wedge_type.value.replace('_', ' ').title()}")
        plt.xlabel('Time')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

# Example usage and trading signals
class WedgeTradingBot:
    def __init__(self, detector: WedgeDetector):
        self.detector = detector
        self.active_patterns = []
    
    def generate_signals(self, data: pd.DataFrame) -> Dict[str, any]:
        """Generate trading signals based on wedge patterns"""
        current_idx = len(data) - 1
        
        # Detect new pattern
        pattern = self.detector.detect_wedge_pattern(data, current_idx)
        
        signals = {
            'pattern_detected': pattern is not None,
            'pattern_type': pattern.wedge_type.value if pattern else None,
            'signal': None,
            'confidence': 0,
            'stop_loss': None,
            'take_profit': None
        }
        
        if pattern:
            # Check for breakout
            breakout = self.detector.detect_breakout(data, pattern, current_idx)
            
            if breakout:
                close_price = data['Close' if 'Close' in data.columns else 'close'].iloc[-1]
                
                if pattern.wedge_type == WedgeType.FALLING_WEDGE and breakout == "upward":
                    # Bullish breakout from falling wedge
                    signals['signal'] = 'BUY'
                    signals['confidence'] = pattern.strength * 0.8
                    signals['stop_loss'] = close_price * 0.98  # 2% stop loss
                    signals['take_profit'] = close_price * 1.06  # 6% take profit
                    
                elif pattern.wedge_type == WedgeType.RISING_WEDGE and breakout == "downward":
                    # Bearish breakout from rising wedge
                    signals['signal'] = 'SELL'
                    signals['confidence'] = pattern.strength * 0.8
                    signals['stop_loss'] = close_price * 1.02  # 2% stop loss
                    signals['take_profit'] = close_price * 0.94  # 6% take profit
        
        return signals

# Example usage
if __name__ == "__main__":
    # Create sample data (replace with your actual data source)
    np.random.seed(42)
    dates = pd.date_range(start='2024-01-01', periods=200, freq='D')
    
    # Generate sample OHLCV data with wedge pattern
    base_price = 100
    trend = np.linspace(0, 10, 200)  # Slight upward trend
    noise = np.random.normal(0, 2, 200)
    
    # Create wedge-like price action
    wedge_factor = np.linspace(5, 1, 200)  # Decreasing volatility (wedge)
    
    close_prices = base_price + trend + noise * wedge_factor
    high_prices = close_prices + np.abs(np.random.normal(0, 1, 200))
    low_prices = close_prices - np.abs(np.random.normal(0, 1, 200))
    volume = np.random.randint(1000, 10000, 200)
    
    sample_data = pd.DataFrame({
        'Date': dates,
        'Open': close_prices,
        'High': high_prices,
        'Low': low_prices,
        'Close': close_prices,
        'Volume': volume
    })
    
    # Initialize detector and bot
    detector = WedgeDetector(
        min_points=3,
        min_period=15,
        max_period=80,
        r_squared_threshold=0.6
    )
    
    bot = WedgeTradingBot(detector)
    
    # Detect pattern
    pattern = detector.detect_wedge_pattern(sample_data)
    
    if pattern:
        print(f"Pattern detected: {pattern.wedge_type.value}")
        print(f"Pattern strength: {pattern.strength:.2f}")
        print(f"Start index: {pattern.start_idx}, End index: {pattern.end_idx}")
        print(f"Upper trendline: slope={pattern.upper_trendline[0]:.6f}")
        print(f"Lower trendline: slope={pattern.lower_trendline[0]:.6f}")
        
        # Generate trading signals
        signals = bot.generate_signals(sample_data)
        print(f"Trading signals: {signals}")
        
        # Plot the pattern (uncomment to visualize)
        # detector.plot_pattern(sample_data, pattern)
    else:
        print("No wedge pattern detected")