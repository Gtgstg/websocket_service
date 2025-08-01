import numpy as np
import pandas as pd
from scipy import stats
from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass
from enum import Enum
import warnings
warnings.filterwarnings('ignore')

class RectangleType(Enum):
    BULLISH = "bullish"  # Rectangle at bottom of trend (support holding)
    BEARISH = "bearish"  # Rectangle at top of trend (resistance holding)
    NEUTRAL = "neutral"  # Rectangle in sideways market
    NONE = "none"

@dataclass
class RectanglePattern:
    type: RectangleType
    start_idx: int
    end_idx: int
    resistance_level: float
    support_level: float
    resistance_touches: List[Tuple[int, float]]
    support_touches: List[Tuple[int, float]]
    height: float
    width: int
    volume_trend: str  # "decreasing", "increasing", "stable"
    breakout_direction: Optional[str] = None
    confidence: float = 0.0
    avg_volume: float = 0.0

class RectanglePatternDetector:
    def __init__(self, min_touches=3, min_width=15, max_width=150, 
                 min_height_ratio=0.02, max_height_ratio=0.15,
                 level_tolerance=0.015, volume_analysis=True):
        """
        Initialize Rectangle Pattern Detector
        
        Args:
            min_touches: Minimum touches for each level (support/resistance)
            min_width: Minimum number of bars for rectangle
            max_width: Maximum number of bars for rectangle
            min_height_ratio: Minimum height as ratio of price (2%)
            max_height_ratio: Maximum height as ratio of price (15%)
            level_tolerance: Price tolerance for level touches (1.5%)
            volume_analysis: Whether to analyze volume patterns
        """
        self.min_touches = min_touches
        self.min_width = min_width
        self.max_width = max_width
        self.min_height_ratio = min_height_ratio
        self.max_height_ratio = max_height_ratio
        self.level_tolerance = level_tolerance
        self.volume_analysis = volume_analysis
    
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
    
    def find_horizontal_levels(self, points: List[Tuple[int, float]], 
                             tolerance: float) -> List[Tuple[float, List[Tuple[int, float]]]]:
        """Find horizontal price levels with multiple touches"""
        if len(points) < 2:
            return []
        
        levels = []
        used_points = set()
        
        for i, (idx1, price1) in enumerate(points):
            if i in used_points:
                continue
                
            level_points = [(idx1, price1)]
            level_price = price1
            
            # Find other points that touch this level
            for j, (idx2, price2) in enumerate(points):
                if i != j and j not in used_points:
                    price_diff = abs(price2 - level_price) / level_price
                    if price_diff <= tolerance:
                        level_points.append((idx2, price2))
                        level_price = np.mean([p[1] for p in level_points])  # Update average
            
            if len(level_points) >= self.min_touches:
                # Sort by index
                level_points.sort(key=lambda x: x[0])
                levels.append((level_price, level_points))
                used_points.update([points.index((idx, price)) for idx, price in level_points 
                                  if (idx, price) in points])
        
        return levels
    
    def analyze_volume_trend(self, data: pd.DataFrame, start_idx: int, end_idx: int) -> Tuple[str, float]:
        """Analyze volume trend within the rectangle"""
        if not self.volume_analysis:
            return "stable", 0.0
        
        volume_col = 'volume' if 'volume' in data.columns else 'Volume'
        if volume_col not in data.columns:
            return "stable", 0.0
        
        volume_data = data[volume_col].iloc[start_idx:end_idx+1]
        avg_volume = volume_data.mean()
        
        # Split into first and second half
        mid_point = len(volume_data) // 2
        first_half = volume_data[:mid_point].mean()
        second_half = volume_data[mid_point:].mean()
        
        change_ratio = (second_half - first_half) / first_half if first_half > 0 else 0
        
        if change_ratio > 0.15:
            return "increasing", avg_volume
        elif change_ratio < -0.15:
            return "decreasing", avg_volume
        else:
            return "stable", avg_volume
    
    def classify_rectangle_context(self, data: pd.DataFrame, start_idx: int, end_idx: int,
                                 support_level: float, resistance_level: float) -> RectangleType:
        """Classify rectangle based on market context"""
        close_col = 'close' if 'close' in data.columns else 'Close'
        
        # Look at trend before rectangle
        pre_window = min(20, start_idx)
        if pre_window < 10:
            return RectangleType.NEUTRAL
        
        pre_start = max(0, start_idx - pre_window)
        pre_prices = data[close_col].iloc[pre_start:start_idx]
        
        if len(pre_prices) < 5:
            return RectangleType.NEUTRAL
        
        # Calculate trend slope before rectangle
        x_vals = np.arange(len(pre_prices))
        slope, _, r_value, _, _ = stats.linregress(x_vals, pre_prices.values)
        
        # Get price level relative to rectangle
        rect_mid = (support_level + resistance_level) / 2
        pre_avg = pre_prices.mean()
        
        # Classification logic
        strong_uptrend = slope > 0 and r_value > 0.7
        strong_downtrend = slope < 0 and r_value < -0.7
        
        if strong_uptrend and pre_avg < rect_mid:
            return RectangleType.BULLISH
        elif strong_downtrend and pre_avg > rect_mid:
            return RectangleType.BEARISH
        else:
            return RectangleType.NEUTRAL
    
    def detect_rectangle_patterns(self, data: pd.DataFrame) -> List[RectanglePattern]:
        """Main method to detect rectangle patterns"""
        patterns = []
        pivots = self.find_pivot_points(data)
        
        close_col = 'close' if 'close' in data.columns else 'Close'
        
        # Find horizontal resistance and support levels
        resistance_levels = self.find_horizontal_levels(pivots['highs'], self.level_tolerance)
        support_levels = self.find_horizontal_levels(pivots['lows'], self.level_tolerance)
        
        # Try to match resistance and support levels to form rectangles
        for resistance_price, resistance_touches in resistance_levels:
            for support_price, support_touches in support_levels:
                
                # Check if levels are appropriately spaced
                height = resistance_price - support_price
                mid_price = (resistance_price + support_price) / 2
                height_ratio = height / mid_price
                
                if height_ratio < self.min_height_ratio or height_ratio > self.max_height_ratio:
                    continue
                
                # Find overlapping time period
                res_start = min([idx for idx, _ in resistance_touches])
                res_end = max([idx for idx, _ in resistance_touches])
                sup_start = min([idx for idx, _ in support_touches])
                sup_end = max([idx for idx, _ in support_touches])
                
                # Rectangle boundaries
                rect_start = max(res_start, sup_start)
                rect_end = min(res_end, sup_end)
                width = rect_end - rect_start
                
                if width < self.min_width or width > self.max_width:
                    continue
                
                # Filter touches within rectangle timeframe
                rect_res_touches = [(idx, price) for idx, price in resistance_touches 
                                  if rect_start <= idx <= rect_end]
                rect_sup_touches = [(idx, price) for idx, price in support_touches 
                                  if rect_start <= idx <= rect_end]
                
                if len(rect_res_touches) < self.min_touches or len(rect_sup_touches) < self.min_touches:
                    continue
                
                # Validate that price stays mostly within rectangle
                rect_data = data.iloc[rect_start:rect_end+1]
                high_col = 'high' if 'high' in data.columns else 'High'
                low_col = 'low' if 'low' in data.columns else 'Low'
                
                # Check containment
                breaks_above = (rect_data[high_col] > resistance_price * (1 + self.level_tolerance)).sum()
                breaks_below = (rect_data[low_col] < support_price * (1 - self.level_tolerance)).sum()
                
                containment_ratio = 1 - (breaks_above + breaks_below) / len(rect_data)
                
                if containment_ratio < 0.8:  # At least 80% containment
                    continue
                
                # Analyze volume trend
                volume_trend, avg_volume = self.analyze_volume_trend(data, rect_start, rect_end)
                
                # Classify rectangle type
                rect_type = self.classify_rectangle_context(data, rect_start, rect_end, 
                                                          support_price, resistance_price)
                
                # Calculate confidence score
                confidence = self._calculate_confidence(
                    rect_res_touches, rect_sup_touches, containment_ratio, 
                    height_ratio, width, volume_trend
                )
                
                pattern = RectanglePattern(
                    type=rect_type,
                    start_idx=rect_start,
                    end_idx=rect_end,
                    resistance_level=resistance_price,
                    support_level=support_price,
                    resistance_touches=rect_res_touches,
                    support_touches=rect_sup_touches,
                    height=height,
                    width=width,
                    volume_trend=volume_trend,
                    confidence=confidence,
                    avg_volume=avg_volume
                )
                
                patterns.append(pattern)
        
        # Remove overlapping patterns
        return self._filter_overlapping_patterns(patterns)
    
    def _calculate_confidence(self, res_touches: List, sup_touches: List, 
                            containment: float, height_ratio: float, 
                            width: int, volume_trend: str) -> float:
        """Calculate confidence score for rectangle pattern"""
        
        # Touch quality (more touches = higher confidence)
        touch_score = min((len(res_touches) + len(sup_touches)) / 8, 1.0)
        
        # Containment quality
        containment_score = containment
        
        # Height ratio quality (prefer moderate heights)
        optimal_height = 0.05  # 5%
        height_score = 1 - abs(height_ratio - optimal_height) / optimal_height
        height_score = max(0, min(1, height_score))
        
        # Width quality (prefer reasonable widths)
        optimal_width = 40
        width_score = 1 - abs(width - optimal_width) / optimal_width
        width_score = max(0.3, min(1, width_score))
        
        # Volume trend bonus
        volume_bonus = 0
        if volume_trend == "decreasing":
            volume_bonus = 0.1  # Decreasing volume in consolidation is bullish
        
        # Weighted combination
        confidence = (touch_score * 0.3 + 
                     containment_score * 0.4 + 
                     height_score * 0.15 + 
                     width_score * 0.15 + 
                     volume_bonus)
        
        return min(1.0, confidence)
    
    def _filter_overlapping_patterns(self, patterns: List[RectanglePattern]) -> List[RectanglePattern]:
        """Remove overlapping patterns, keeping the best ones"""
        if not patterns:
            return []
        
        # Sort by confidence
        patterns.sort(key=lambda x: x.confidence, reverse=True)
        
        filtered = []
        for pattern in patterns:
            overlaps = False
            for existing in filtered:
                overlap_start = max(pattern.start_idx, existing.start_idx)
                overlap_end = min(pattern.end_idx, existing.end_idx)
                overlap_length = max(0, overlap_end - overlap_start)
                
                pattern_length = pattern.end_idx - pattern.start_idx
                existing_length = existing.end_idx - existing.start_idx
                
                # If overlap is more than 60% of either pattern, skip
                if (overlap_length / pattern_length > 0.6 or 
                    overlap_length / existing_length > 0.6):
                    overlaps = True
                    break
            
            if not overlaps:
                filtered.append(pattern)
        
        return filtered
    
    def check_breakout(self, data: pd.DataFrame, pattern: RectanglePattern, 
                      current_idx: int, volume_confirmation: bool = True) -> Optional[str]:
        """Check for breakout from rectangle pattern"""
        if current_idx <= pattern.end_idx:
            return None
        
        close_col = 'close' if 'close' in data.columns else 'Close'
        volume_col = 'volume' if 'volume' in data.columns else 'Volume'
        
        current_price = data[close_col].iloc[current_idx]
        
        # Check for price breakout
        breakout_threshold = 0.01  # 1% beyond level
        
        bullish_breakout = current_price > pattern.resistance_level * (1 + breakout_threshold)
        bearish_breakout = current_price < pattern.support_level * (1 - breakout_threshold)
        
        if not (bullish_breakout or bearish_breakout):
            return None
        
        # Volume confirmation if requested
        if volume_confirmation and self.volume_analysis and volume_col in data.columns:
            recent_volume = data[volume_col].iloc[current_idx-2:current_idx+1].mean()
            avg_volume = pattern.avg_volume
            
            if recent_volume < avg_volume * 1.2:  # Volume should be 20% above average
                return None
        
        return "bullish" if bullish_breakout else "bearish"
    
    def get_trading_signals(self, data: pd.DataFrame, patterns: List[RectanglePattern]) -> List[Dict]:
        """Generate trading signals based on rectangle patterns"""
        signals = []
        
        close_col = 'close' if 'close' in data.columns else 'Close'
        
        for pattern in patterns:
            current_price = data[close_col].iloc[pattern.end_idx]
            
            # Calculate risk-reward ratios
            height = pattern.height
            
            signal = {
                'pattern_type': pattern.type.value,
                'start_idx': pattern.start_idx,
                'end_idx': pattern.end_idx,
                'resistance_level': pattern.resistance_level,
                'support_level': pattern.support_level,
                'height': height,
                'width': pattern.width,
                'confidence': pattern.confidence,
                'volume_trend': pattern.volume_trend,
                'current_price': current_price,
                
                # Trading levels
                'buy_level': pattern.support_level,
                'sell_level': pattern.resistance_level,
                'bullish_target': pattern.resistance_level + height,  # Height projection
                'bearish_target': pattern.support_level - height,    # Height projection
                'bullish_stop': pattern.support_level - height * 0.2,  # Tight stop
                'bearish_stop': pattern.resistance_level + height * 0.2,  # Tight stop
                
                # Risk management
                'risk_reward_bull': None,
                'risk_reward_bear': None
            }
            
            # Calculate risk-reward ratios
            bull_profit = signal['bullish_target'] - pattern.resistance_level
            bull_risk = pattern.resistance_level - signal['bullish_stop']
            if bull_risk > 0:
                signal['risk_reward_bull'] = bull_profit / bull_risk
            
            bear_profit = pattern.support_level - signal['bearish_target']
            bear_risk = signal['bearish_stop'] - pattern.support_level
            if bear_risk > 0:
                signal['risk_reward_bear'] = bear_profit / bear_risk
            
            # Strategy recommendations based on pattern type
            if pattern.type == RectangleType.BULLISH:
                signal['primary_strategy'] = 'buy_support_breakout'
                signal['bias'] = 'bullish'
            elif pattern.type == RectangleType.BEARISH:
                signal['primary_strategy'] = 'sell_resistance_breakout'
                signal['bias'] = 'bearish'
            else:
                signal['primary_strategy'] = 'range_trading'
                signal['bias'] = 'neutral'
            
            signals.append(signal)
        
        return signals
    
    def get_range_trading_signals(self, data: pd.DataFrame, pattern: RectanglePattern, 
                                current_idx: int) -> Optional[Dict]:
        """Generate range trading signals within rectangle"""
        if current_idx < pattern.start_idx or current_idx > pattern.end_idx:
            return None
        
        close_col = 'close' if 'close' in data.columns else 'Close'
        current_price = data[close_col].iloc[current_idx]
        
        # Calculate position within rectangle
        rect_height = pattern.height
        distance_from_support = current_price - pattern.support_level
        position_ratio = distance_from_support / rect_height
        
        signal = None
        
        # Near support - potential buy
        if position_ratio <= 0.2:  # Within 20% of support
            signal = {
                'action': 'buy',
                'entry_price': current_price,
                'target': pattern.resistance_level * 0.95,  # Near resistance
                'stop_loss': pattern.support_level * 0.98,   # Just below support
                'confidence': pattern.confidence * (1 - position_ratio),
                'reason': 'near_support'
            }
        
        # Near resistance - potential sell
        elif position_ratio >= 0.8:  # Within 20% of resistance
            signal = {
                'action': 'sell',
                'entry_price': current_price,
                'target': pattern.support_level * 1.05,     # Near support
                'stop_loss': pattern.resistance_level * 1.02, # Just above resistance
                'confidence': pattern.confidence * position_ratio,
                'reason': 'near_resistance'
            }
        
        return signal

# Example usage and backtesting framework
def example_usage():
    """Example of how to use the Rectangle Pattern Detector"""
    
    # Create sample data with rectangle patterns
    np.random.seed(42)
    dates = pd.date_range(start='2024-01-01', periods=300, freq='D')
    
    # Generate sample data with rectangle patterns
    price = 100
    prices = []
    highs = []
    lows = []
    volumes = []
    
    for i in range(300):
        # Create rectangle patterns at specific intervals
        if 50 <= i <= 100:
            # Rectangle pattern 1
            resistance = 110
            support = 105
        elif 150 <= i <= 200:
            # Rectangle pattern 2
            resistance = 125
            support = 120
        elif 220 <= i <= 270:
            # Rectangle pattern 3
            resistance = 135
            support = 128
        else:
            # Trending or random movement
            resistance = price * 1.03
            support = price * 0.97
        
        # Add noise and stay mostly within bounds
        if support < price < resistance:
            daily_change = np.random.uniform(-1, 1)
        else:
            # Mean reversion toward rectangle
            if price > resistance:
                daily_change = np.random.uniform(-2, 0)
            else:
                daily_change = np.random.uniform(0, 2)
        
        price += daily_change
        high = min(price + abs(np.random.uniform(0, 1)), resistance + np.random.uniform(-0.5, 1))
        low = max(price - abs(np.random.uniform(0, 1)), support + np.random.uniform(-1, 0.5))
        close = np.random.uniform(low, high)
        
        prices.append(close)
        highs.append(high)
        lows.append(low)
        volumes.append(np.random.randint(5000, 15000))
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
    detector = RectanglePatternDetector(
        min_touches=3,
        min_width=20,
        max_width=80,
        min_height_ratio=0.025,
        max_height_ratio=0.12,
        level_tolerance=0.02
    )
    
    # Detect patterns
    patterns = detector.detect_rectangle_patterns(data)
    
    print(f"Found {len(patterns)} rectangle patterns:")
    for i, pattern in enumerate(patterns):
        print(f"\nPattern {i+1}:")
        print(f"  Type: {pattern.type.value}")
        print(f"  Range: {pattern.start_idx} to {pattern.end_idx} (width: {pattern.width})")
        print(f"  Resistance: {pattern.resistance_level:.2f} ({len(pattern.resistance_touches)} touches)")
        print(f"  Support: {pattern.support_level:.2f} ({len(pattern.support_touches)} touches)")
        print(f"  Height: {pattern.height:.2f} ({pattern.height/((pattern.resistance_level + pattern.support_level)/2)*100:.1f}%)")
        print(f"  Volume trend: {pattern.volume_trend}")
        print(f"  Confidence: {pattern.confidence:.3f}")
    
    # Generate trading signals
    signals = detector.get_trading_signals(data, patterns)
    
    print(f"\nGenerated {len(signals)} trading signals:")
    for i, signal in enumerate(signals):
        print(f"\nSignal {i+1}:")
        print(f"  Pattern: {signal['pattern_type']}")
        print(f"  Strategy: {signal['primary_strategy']}")
        print(f"  Bias: {signal['bias']}")
        print(f"  Confidence: {signal['confidence']:.3f}")
        print(f"  Support: {signal['support_level']:.2f}")
        print(f"  Resistance: {signal['resistance_level']:.2f}")
        print(f"  Bullish target: {signal['bullish_target']:.2f} (R/R: {signal['risk_reward_bull']:.2f})")
        print(f"  Bearish target: {signal['bearish_target']:.2f} (R/R: {signal['risk_reward_bear']:.2f})")
        
    # Example of checking for breakouts
    print(f"\nChecking for breakouts after patterns:")
    for i, pattern in enumerate(patterns):
        if pattern.end_idx + 10 < len(data):
            breakout = detector.check_breakout(data, pattern, pattern.end_idx + 10)
            if breakout:
                print(f"  Pattern {i+1}: {breakout} breakout detected!")

if __name__ == "__main__":
    example_usage()