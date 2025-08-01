import numpy as np
import pandas as pd
from scipy import stats
from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass
from enum import Enum

class TriangleType(Enum):
    ASCENDING = "ascending"
    DESCENDING = "descending"
    SYMMETRICAL = "symmetrical"
    NONE = "none"

@dataclass
class TrianglePattern:
    type: TriangleType
    start_idx: int
    end_idx: int
    upper_line: Tuple[float, float]  # (slope, intercept)
    lower_line: Tuple[float, float]  # (slope, intercept)
    breakout_direction: Optional[str] = None
    confidence: float = 0.0
    volume_confirmation: bool = False

class TrianglePatternDetector:
    def __init__(self, min_touches=4, min_length=20, max_length=200, 
                 slope_tolerance=0.001, r_squared_threshold=0.7):
        """
        Initialize Triangle Pattern Detector
        
        Args:
            min_touches: Minimum number of touches on trendlines
            min_length: Minimum number of bars for pattern
            max_length: Maximum number of bars for pattern
            slope_tolerance: Tolerance for horizontal lines
            r_squared_threshold: Minimum R-squared for trendline validity
        """
        self.min_touches = min_touches
        self.min_length = min_length
        self.max_length = max_length
        self.slope_tolerance = slope_tolerance
        self.r_squared_threshold = r_squared_threshold
    
    def find_pivot_points(self, data: pd.DataFrame, window: int = 5) -> Dict:
        """Find pivot highs and lows in the data"""
        highs = []
        lows = []
        
        high_col = 'high' if 'high' in data.columns else 'High'
        low_col = 'low' if 'low' in data.columns else 'Low'
        
        for i in range(window, len(data) - window):
            # Check for pivot high
            if all(data[high_col].iloc[i] >= data[high_col].iloc[i-j] for j in range(1, window+1)) and \
               all(data[high_col].iloc[i] >= data[high_col].iloc[i+j] for j in range(1, window+1)):
                highs.append((i, data[high_col].iloc[i]))
            
            # Check for pivot low
            if all(data[low_col].iloc[i] <= data[low_col].iloc[i-j] for j in range(1, window+1)) and \
               all(data[low_col].iloc[i] <= data[low_col].iloc[i+j] for j in range(1, window+1)):
                lows.append((i, data[low_col].iloc[i]))
        
        return {'highs': highs, 'lows': lows}
    
    def fit_trendline(self, points: List[Tuple[int, float]]) -> Tuple[float, float, float]:
        """Fit a trendline to given points and return slope, intercept, r_squared"""
        if len(points) < 2:
            return 0, 0, 0
        
        x_vals = np.array([p[0] for p in points])
        y_vals = np.array([p[1] for p in points])
        
        slope, intercept, r_value, p_value, std_err = stats.linregress(x_vals, y_vals)
        r_squared = r_value ** 2
        
        return slope, intercept, r_squared
    
    def find_touching_points(self, all_points: List[Tuple[int, float]], 
                           slope: float, intercept: float, tolerance: float = 0.02) -> List[Tuple[int, float]]:
        """Find points that touch or are close to the trendline"""
        touching = []
        
        for idx, price in all_points:
            expected_price = slope * idx + intercept
            price_diff = abs(price - expected_price) / expected_price
            
            if price_diff <= tolerance:
                touching.append((idx, price))
        
        return touching
    
    def detect_triangle_patterns(self, data: pd.DataFrame) -> List[TrianglePattern]:
        """Main method to detect triangle patterns"""
        patterns = []
        pivots = self.find_pivot_points(data)
        
        # Look for patterns in sliding windows
        for start in range(0, len(data) - self.min_length):
            for end in range(start + self.min_length, min(start + self.max_length, len(data))):
                pattern = self._analyze_triangle_in_range(data, pivots, start, end)
                if pattern and pattern.type != TriangleType.NONE:
                    patterns.append(pattern)
        
        # Remove overlapping patterns, keep the best ones
        return self._filter_overlapping_patterns(patterns)
    
    def _analyze_triangle_in_range(self, data: pd.DataFrame, pivots: Dict, 
                                 start: int, end: int) -> Optional[TrianglePattern]:
        """Analyze a specific range for triangle patterns"""
        # Filter pivots within range
        range_highs = [(i, p) for i, p in pivots['highs'] if start <= i <= end]
        range_lows = [(i, p) for i, p in pivots['lows'] if start <= i <= end]
        
        if len(range_highs) < 2 or len(range_lows) < 2:
            return None
        
        # Try different combinations of pivot points for trendlines
        best_pattern = None
        best_score = 0
        
        # Try combinations of high points for upper trendline
        for i in range(len(range_highs)):
            for j in range(i + 1, len(range_highs)):
                upper_points = [range_highs[i], range_highs[j]]
                upper_slope, upper_intercept, upper_r2 = self.fit_trendline(upper_points)
                
                if upper_r2 < self.r_squared_threshold:
                    continue
                
                # Find additional touching points
                upper_touches = self.find_touching_points(range_highs, upper_slope, upper_intercept)
                
                # Try combinations of low points for lower trendline
                for k in range(len(range_lows)):
                    for l in range(k + 1, len(range_lows)):
                        lower_points = [range_lows[k], range_lows[l]]
                        lower_slope, lower_intercept, lower_r2 = self.fit_trendline(lower_points)
                        
                        if lower_r2 < self.r_squared_threshold:
                            continue
                        
                        lower_touches = self.find_touching_points(range_lows, lower_slope, lower_intercept)
                        
                        # Check if we have enough touches
                        if len(upper_touches) + len(lower_touches) < self.min_touches:
                            continue
                        
                        # Determine triangle type
                        triangle_type = self._classify_triangle(upper_slope, lower_slope)
                        
                        if triangle_type == TriangleType.NONE:
                            continue
                        
                        # Calculate pattern quality score
                        score = self._calculate_pattern_score(upper_r2, lower_r2, 
                                                           len(upper_touches), len(lower_touches))
                        
                        if score > best_score:
                            best_score = score
                            best_pattern = TrianglePattern(
                                type=triangle_type,
                                start_idx=start,
                                end_idx=end,
                                upper_line=(upper_slope, upper_intercept),
                                lower_line=(lower_slope, lower_intercept),
                                confidence=score
                            )
        
        return best_pattern
    
    def _classify_triangle(self, upper_slope: float, lower_slope: float) -> TriangleType:
        """Classify triangle type based on trendline slopes"""
        # Check if lines are converging
        if upper_slope >= lower_slope:
            return TriangleType.NONE
        
        # Ascending triangle: horizontal resistance, rising support
        if abs(upper_slope) <= self.slope_tolerance and lower_slope > self.slope_tolerance:
            return TriangleType.ASCENDING
        
        # Descending triangle: falling resistance, horizontal support
        if upper_slope < -self.slope_tolerance and abs(lower_slope) <= self.slope_tolerance:
            return TriangleType.DESCENDING
        
        # Symmetrical triangle: falling resistance, rising support
        if upper_slope < -self.slope_tolerance and lower_slope > self.slope_tolerance:
            return TriangleType.SYMMETRICAL
        
        return TriangleType.NONE
    
    def _calculate_pattern_score(self, upper_r2: float, lower_r2: float, 
                               upper_touches: int, lower_touches: int) -> float:
        """Calculate a quality score for the pattern"""
        r2_score = (upper_r2 + lower_r2) / 2
        touch_score = min((upper_touches + lower_touches) / 8, 1.0)  # Normalize to max 8 touches
        
        return (r2_score * 0.7) + (touch_score * 0.3)
    
    def _filter_overlapping_patterns(self, patterns: List[TrianglePattern]) -> List[TrianglePattern]:
        """Remove overlapping patterns, keeping the best ones"""
        if not patterns:
            return []
        
        # Sort by confidence score
        patterns.sort(key=lambda x: x.confidence, reverse=True)
        
        filtered = []
        for pattern in patterns:
            # Check if this pattern overlaps significantly with any already selected
            overlaps = False
            for existing in filtered:
                overlap_start = max(pattern.start_idx, existing.start_idx)
                overlap_end = min(pattern.end_idx, existing.end_idx)
                overlap_length = max(0, overlap_end - overlap_start)
                
                pattern_length = pattern.end_idx - pattern.start_idx
                existing_length = existing.end_idx - existing.start_idx
                
                # If overlap is more than 50% of either pattern, skip
                if (overlap_length / pattern_length > 0.5 or 
                    overlap_length / existing_length > 0.5):
                    overlaps = True
                    break
            
            if not overlaps:
                filtered.append(pattern)
        
        return filtered
    
    def check_breakout(self, data: pd.DataFrame, pattern: TrianglePattern, 
                      current_idx: int) -> Optional[str]:
        """Check if there's a breakout from the triangle pattern"""
        if current_idx <= pattern.end_idx:
            return None
        
        close_col = 'close' if 'close' in data.columns else 'Close'
        volume_col = 'volume' if 'volume' in data.columns else 'Volume'
        
        current_price = data[close_col].iloc[current_idx]
        
        # Calculate trendline values at current index
        upper_value = pattern.upper_line[0] * current_idx + pattern.upper_line[1]
        lower_value = pattern.lower_line[0] * current_idx + pattern.lower_line[1]
        
        # Check for breakout
        if current_price > upper_value * 1.01:  # 1% buffer for noise
            return "bullish"
        elif current_price < lower_value * 0.99:  # 1% buffer for noise
            return "bearish"
        
        return None
    
    def get_trading_signals(self, data: pd.DataFrame, patterns: List[TrianglePattern]) -> List[Dict]:
        """Generate trading signals based on triangle patterns"""
        signals = []
        
        close_col = 'close' if 'close' in data.columns else 'Close'
        
        for pattern in patterns:
            # Calculate apex (convergence point)
            apex_x = (pattern.lower_line[1] - pattern.upper_line[1]) / (pattern.upper_line[0] - pattern.lower_line[0])
            apex_y = pattern.upper_line[0] * apex_x + pattern.upper_line[1]
            
            # Calculate target based on triangle height
            triangle_start = pattern.start_idx
            start_high = pattern.upper_line[0] * triangle_start + pattern.upper_line[1]
            start_low = pattern.lower_line[0] * triangle_start + pattern.lower_line[1]
            triangle_height = start_high - start_low
            
            signal = {
                'pattern_type': pattern.type.value,
                'start_idx': pattern.start_idx,
                'end_idx': pattern.end_idx,
                'apex_idx': int(apex_x),
                'apex_price': apex_y,
                'confidence': pattern.confidence,
                'triangle_height': triangle_height,
                'bullish_target': None,
                'bearish_target': None,
                'stop_loss_bull': None,
                'stop_loss_bear': None
            }
            
            # Set targets based on triangle type
            current_price = data[close_col].iloc[pattern.end_idx]
            
            if pattern.type == TriangleType.ASCENDING:
                signal['bullish_target'] = current_price + triangle_height
                signal['stop_loss_bull'] = current_price - triangle_height * 0.3
            elif pattern.type == TriangleType.DESCENDING:
                signal['bearish_target'] = current_price - triangle_height
                signal['stop_loss_bear'] = current_price + triangle_height * 0.3
            else:  # SYMMETRICAL
                signal['bullish_target'] = current_price + triangle_height
                signal['bearish_target'] = current_price - triangle_height
                signal['stop_loss_bull'] = current_price - triangle_height * 0.3
                signal['stop_loss_bear'] = current_price + triangle_height * 0.3
            
            signals.append(signal)
        
        return signals

# Example usage
def example_usage():
    """Example of how to use the Triangle Pattern Detector"""
    
    # Create sample data (replace with your actual OHLCV data)
    np.random.seed(42)
    dates = pd.date_range(start='2024-01-01', periods=200, freq='D')
    
    # Generate sample price data with triangle patterns
    price = 100
    prices = []
    highs = []
    lows = []
    volumes = []
    
    for i in range(200):
        # Add some triangle-like behavior
        if 50 <= i <= 100:
            # Ascending triangle pattern
            resistance = 110
            support = 100 + (i - 50) * 0.2
        elif 120 <= i <= 170:
            # Descending triangle pattern
            resistance = 115 - (i - 120) * 0.2
            support = 105
        else:
            resistance = price * 1.05
            support = price * 0.95
        
        daily_range = np.random.uniform(0.5, 2.0)
        high = min(price + daily_range, resistance + np.random.uniform(-0.5, 0.5))
        low = max(price - daily_range, support + np.random.uniform(-0.5, 0.5))
        close = np.random.uniform(low, high)
        
        prices.append(close)
        highs.append(high)
        lows.append(low)
        volumes.append(np.random.randint(1000, 10000))
        price = close
    
    data = pd.DataFrame({
        'Date': dates,
        'Open': prices,
        'High': highs,
        'Low': lows,
        'Close': prices,
        'Volume': volumes
    })
    
    # Initialize detector
    detector = TrianglePatternDetector(
        min_touches=4,
        min_length=20,
        max_length=100,
        r_squared_threshold=0.6
    )
    
    # Detect patterns
    patterns = detector.detect_triangle_patterns(data)
    
    print(f"Found {len(patterns)} triangle patterns:")
    for i, pattern in enumerate(patterns):
        print(f"\nPattern {i+1}:")
        print(f"  Type: {pattern.type.value}")
        print(f"  Range: {pattern.start_idx} to {pattern.end_idx}")
        print(f"  Confidence: {pattern.confidence:.3f}")
        print(f"  Upper line: slope={pattern.upper_line[0]:.4f}, intercept={pattern.upper_line[1]:.2f}")
        print(f"  Lower line: slope={pattern.lower_line[0]:.4f}, intercept={pattern.lower_line[1]:.2f}")
    
    # Generate trading signals
    signals = detector.get_trading_signals(data, patterns)
    
    print(f"\nGenerated {len(signals)} trading signals:")
    for i, signal in enumerate(signals):
        print(f"\nSignal {i+1}:")
        print(f"  Pattern: {signal['pattern_type']}")
        print(f"  Confidence: {signal['confidence']:.3f}")
        if signal['bullish_target']:
            print(f"  Bullish target: {signal['bullish_target']:.2f}")
        if signal['bearish_target']:
            print(f"  Bearish target: {signal['bearish_target']:.2f}")

if __name__ == "__main__":
    example_usage()