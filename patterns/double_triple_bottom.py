import numpy as np
import pandas as pd
from typing import Tuple, Optional, Dict, List, Union
from dataclasses import dataclass
from scipy.signal import find_peaks, find_peaks_cwt
from scipy.stats import linregress
import warnings
warnings.filterwarnings('ignore')

@dataclass
class PatternSignal:
    """Data class to store pattern detection results"""
    pattern_type: str  # 'double_top', 'triple_top', 'double_bottom', 'triple_bottom'
    entry_price: float
    stop_loss: float
    take_profit: float
    confidence: float  # 0-1 scale
    neckline_level: float
    peak_prices: List[float]  # Prices of the peaks/troughs
    peak_indices: List[int]   # Indices of the peaks/troughs
    neckline_break_index: int
    pattern_height: float

class DoubleTripletopsBottomsDetector:
    """
    Detects Double/Triple Top and Bottom chart patterns for algorithmic trading
    
    Double Top: Two peaks at similar levels with valley between
    Triple Top: Three peaks at similar levels with valleys between
    Double Bottom: Two troughs at similar levels with peak between
    Triple Bottom: Three troughs at similar levels with peaks between
    """
    
    def __init__(self, 
                 peak_similarity_pct: float = 2.0,    # Max % difference between peaks
                 min_valley_depth_pct: float = 3.0,   # Min depth of valley between peaks
                 min_pattern_bars: int = 20,          # Min bars for complete pattern
                 max_pattern_bars: int = 100,         # Max bars for complete pattern
                 volume_confirmation: bool = True,
                 neckline_break_pct: float = 1.0):    # % break of neckline required
        
        self.peak_similarity_pct = peak_similarity_pct / 100
        self.min_valley_depth_pct = min_valley_depth_pct / 100
        self.min_pattern_bars = min_pattern_bars
        self.max_pattern_bars = max_pattern_bars
        self.volume_confirmation = volume_confirmation
        self.neckline_break_pct = neckline_break_pct / 100
    
    def detect_patterns(self, df: pd.DataFrame) -> List[PatternSignal]:
        """
        Main method to detect double/triple top and bottom patterns
        
        Args:
            df: DataFrame with columns ['open', 'high', 'low', 'close', 'volume']
        
        Returns:
            List of PatternSignal objects
        """
        signals = []
        
        # Ensure we have enough data
        if len(df) < self.max_pattern_bars:
            return signals
        
        # Find peaks and troughs
        peaks = self._find_peaks(df['high'].values)
        troughs = self._find_troughs(df['low'].values)
        
        # Look for double/triple top patterns
        top_signals = self._detect_top_patterns(df, peaks)
        signals.extend(top_signals)
        
        # Look for double/triple bottom patterns
        bottom_signals = self._detect_bottom_patterns(df, troughs)
        signals.extend(bottom_signals)
        
        # Remove overlapping patterns (keep highest confidence)
        signals = self._remove_overlapping_patterns(signals)
        
        return signals
    
    def _find_peaks(self, prices: np.ndarray) -> List[int]:
        """Find significant peaks in price data"""
        
        # Use scipy's find_peaks with minimum distance and prominence
        peaks, properties = find_peaks(
            prices, 
            distance=5,  # Minimum 5 bars between peaks
            prominence=np.std(prices) * 0.5  # Minimum prominence
        )
        
        return peaks.tolist()
    
    def _find_troughs(self, prices: np.ndarray) -> List[int]:
        """Find significant troughs in price data"""
        
        # Invert prices to find troughs as peaks
        inverted_prices = -prices
        troughs, properties = find_peaks(
            inverted_prices,
            distance=5,  # Minimum 5 bars between troughs
            prominence=np.std(inverted_prices) * 0.5  # Minimum prominence
        )
        
        return troughs.tolist()
    
    def _detect_top_patterns(self, df: pd.DataFrame, peaks: List[int]) -> List[PatternSignal]:
        """Detect double and triple top patterns"""
        
        signals = []
        
        # Check each combination of peaks for patterns
        for i in range(len(peaks) - 1):
            for j in range(i + 1, len(peaks)):
                
                # Check for double top
                double_top = self._check_double_top(df, peaks[i], peaks[j])
                if double_top:
                    signals.append(double_top)
                
                # Check for triple top (if we have enough peaks)
                for k in range(j + 1, len(peaks)):
                    triple_top = self._check_triple_top(df, peaks[i], peaks[j], peaks[k])
                    if triple_top:
                        signals.append(triple_top)
        
        return signals
    
    def _detect_bottom_patterns(self, df: pd.DataFrame, troughs: List[int]) -> List[PatternSignal]:
        """Detect double and triple bottom patterns"""
        
        signals = []
        
        # Check each combination of troughs for patterns
        for i in range(len(troughs) - 1):
            for j in range(i + 1, len(troughs)):
                
                # Check for double bottom
                double_bottom = self._check_double_bottom(df, troughs[i], troughs[j])
                if double_bottom:
                    signals.append(double_bottom)
                
                # Check for triple bottom (if we have enough troughs)
                for k in range(j + 1, len(troughs)):
                    triple_bottom = self._check_triple_bottom(df, troughs[i], troughs[j], troughs[k])
                    if triple_bottom:
                        signals.append(triple_bottom)
        
        return signals
    
    def _check_double_top(self, df: pd.DataFrame, peak1_idx: int, peak2_idx: int) -> Optional[PatternSignal]:
        """Check if two peaks form a valid double top pattern"""
        
        # Validate timing constraints
        bars_between = peak2_idx - peak1_idx
        if bars_between < self.min_pattern_bars or bars_between > self.max_pattern_bars:
            return None
        
        # Get peak prices
        peak1_price = df.iloc[peak1_idx]['high']
        peak2_price = df.iloc[peak2_idx]['high']
        
        # Check if peaks are similar in height
        if not self._are_peaks_similar(peak1_price, peak2_price):
            return None
        
        # Find the valley between peaks
        valley_data = df.iloc[peak1_idx:peak2_idx+1]
        valley_idx = valley_data['low'].idxmin()
        valley_price = valley_data.loc[valley_idx, 'low']
        valley_idx_relative = valley_idx - df.index[0]  # Convert to relative index
        
        # Check valley depth
        avg_peak_price = (peak1_price + peak2_price) / 2
        valley_depth = (avg_peak_price - valley_price) / avg_peak_price
        
        if valley_depth < self.min_valley_depth_pct:
            return None
        
        # Check for neckline break (price breaking below valley)
        neckline_level = valley_price
        break_idx = self._check_neckline_break(df, peak2_idx, neckline_level, 'bearish')
        
        if break_idx is None:
            return None
        
        # Calculate pattern metrics
        pattern_height = avg_peak_price - valley_price
        confidence = self._calculate_top_confidence(df, peak1_idx, peak2_idx, valley_idx_relative, 'double')
        
        # Calculate trading levels
        entry_price = df.iloc[break_idx]['close']
        stop_loss = max(peak1_price, peak2_price) * 1.02  # 2% above highest peak
        take_profit = entry_price - pattern_height  # Pattern height projection
        
        return PatternSignal(
            pattern_type='double_top',
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            confidence=confidence,
            neckline_level=neckline_level,
            peak_prices=[peak1_price, peak2_price],
            peak_indices=[peak1_idx, peak2_idx],
            neckline_break_index=break_idx,
            pattern_height=pattern_height
        )
    
    def _check_triple_top(self, df: pd.DataFrame, peak1_idx: int, 
                         peak2_idx: int, peak3_idx: int) -> Optional[PatternSignal]:
        """Check if three peaks form a valid triple top pattern"""
        
        # Validate timing constraints
        total_bars = peak3_idx - peak1_idx
        if total_bars < self.min_pattern_bars or total_bars > self.max_pattern_bars:
            return None
        
        # Get peak prices
        peak1_price = df.iloc[peak1_idx]['high']
        peak2_price = df.iloc[peak2_idx]['high']
        peak3_price = df.iloc[peak3_idx]['high']
        
        # Check if all peaks are similar in height
        if not (self._are_peaks_similar(peak1_price, peak2_price) and 
                self._are_peaks_similar(peak2_price, peak3_price) and
                self._are_peaks_similar(peak1_price, peak3_price)):
            return None
        
        # Find valleys between peaks
        valley1_data = df.iloc[peak1_idx:peak2_idx+1]
        valley1_idx = valley1_data['low'].idxmin()
        valley1_price = valley1_data.loc[valley1_idx, 'low']
        valley1_idx_relative = valley1_idx - df.index[0]
        
        valley2_data = df.iloc[peak2_idx:peak3_idx+1]
        valley2_idx = valley2_data['low'].idxmin()
        valley2_price = valley2_data.loc[valley2_idx, 'low']
        valley2_idx_relative = valley2_idx - df.index[0]
        
        # Check valley depths
        avg_peak_price = (peak1_price + peak2_price + peak3_price) / 3
        valley1_depth = (avg_peak_price - valley1_price) / avg_peak_price
        valley2_depth = (avg_peak_price - valley2_price) / avg_peak_price
        
        if valley1_depth < self.min_valley_depth_pct or valley2_depth < self.min_valley_depth_pct:
            return None
        
        # Neckline is the higher of the two valley lows
        neckline_level = max(valley1_price, valley2_price)
        break_idx = self._check_neckline_break(df, peak3_idx, neckline_level, 'bearish')
        
        if break_idx is None:
            return None
        
        # Calculate pattern metrics
        pattern_height = avg_peak_price - neckline_level
        confidence = self._calculate_top_confidence(df, peak1_idx, peak3_idx, 
                                                   min(valley1_idx_relative, valley2_idx_relative), 'triple')
        
        # Calculate trading levels
        entry_price = df.iloc[break_idx]['close']
        stop_loss = max(peak1_price, peak2_price, peak3_price) * 1.02
        take_profit = entry_price - pattern_height
        
        return PatternSignal(
            pattern_type='triple_top',
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            confidence=confidence,
            neckline_level=neckline_level,
            peak_prices=[peak1_price, peak2_price, peak3_price],
            peak_indices=[peak1_idx, peak2_idx, peak3_idx],
            neckline_break_index=break_idx,
            pattern_height=pattern_height
        )
    
    def _check_double_bottom(self, df: pd.DataFrame, trough1_idx: int, trough2_idx: int) -> Optional[PatternSignal]:
        """Check if two troughs form a valid double bottom pattern"""
        
        # Validate timing constraints
        bars_between = trough2_idx - trough1_idx
        if bars_between < self.min_pattern_bars or bars_between > self.max_pattern_bars:
            return None
        
        # Get trough prices
        trough1_price = df.iloc[trough1_idx]['low']
        trough2_price = df.iloc[trough2_idx]['low']
        
        # Check if troughs are similar in depth
        if not self._are_peaks_similar(trough1_price, trough2_price):
            return None
        
        # Find the peak between troughs
        peak_data = df.iloc[trough1_idx:trough2_idx+1]
        peak_idx = peak_data['high'].idxmax()
        peak_price = peak_data.loc[peak_idx, 'high']
        peak_idx_relative = peak_idx - df.index[0]
        
        # Check peak height
        avg_trough_price = (trough1_price + trough2_price) / 2
        peak_height = (peak_price - avg_trough_price) / avg_trough_price
        
        if peak_height < self.min_valley_depth_pct:
            return None
        
        # Check for neckline break (price breaking above peak)
        neckline_level = peak_price
        break_idx = self._check_neckline_break(df, trough2_idx, neckline_level, 'bullish')
        
        if break_idx is None:
            return None
        
        # Calculate pattern metrics
        pattern_height = peak_price - avg_trough_price
        confidence = self._calculate_bottom_confidence(df, trough1_idx, trough2_idx, peak_idx_relative, 'double')
        
        # Calculate trading levels
        entry_price = df.iloc[break_idx]['close']
        stop_loss = min(trough1_price, trough2_price) * 0.98  # 2% below lowest trough
        take_profit = entry_price + pattern_height  # Pattern height projection
        
        return PatternSignal(
            pattern_type='double_bottom',
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            confidence=confidence,
            neckline_level=neckline_level,
            peak_prices=[trough1_price, trough2_price],
            peak_indices=[trough1_idx, trough2_idx],
            neckline_break_index=break_idx,
            pattern_height=pattern_height
        )
    
    def _check_triple_bottom(self, df: pd.DataFrame, trough1_idx: int, 
                            trough2_idx: int, trough3_idx: int) -> Optional[PatternSignal]:
        """Check if three troughs form a valid triple bottom pattern"""
        
        # Validate timing constraints
        total_bars = trough3_idx - trough1_idx
        if total_bars < self.min_pattern_bars or total_bars > self.max_pattern_bars:
            return None
        
        # Get trough prices
        trough1_price = df.iloc[trough1_idx]['low']
        trough2_price = df.iloc[trough2_idx]['low']
        trough3_price = df.iloc[trough3_idx]['low']
        
        # Check if all troughs are similar in depth
        if not (self._are_peaks_similar(trough1_price, trough2_price) and 
                self._are_peaks_similar(trough2_price, trough3_price) and
                self._are_peaks_similar(trough1_price, trough3_price)):
            return None
        
        # Find peaks between troughs
        peak1_data = df.iloc[trough1_idx:trough2_idx+1]
        peak1_idx = peak1_data['high'].idxmax()
        peak1_price = peak1_data.loc[peak1_idx, 'high']
        peak1_idx_relative = peak1_idx - df.index[0]
        
        peak2_data = df.iloc[trough2_idx:trough3_idx+1]
        peak2_idx = peak2_data['high'].idxmax()
        peak2_price = peak2_data.loc[peak2_idx, 'high']
        peak2_idx_relative = peak2_idx - df.index[0]
        
        # Check peak heights
        avg_trough_price = (trough1_price + trough2_price + trough3_price) / 3
        peak1_height = (peak1_price - avg_trough_price) / avg_trough_price
        peak2_height = (peak2_price - avg_trough_price) / avg_trough_price
        
        if peak1_height < self.min_valley_depth_pct or peak2_height < self.min_valley_depth_pct:
            return None
        
        # Neckline is the lower of the two peak highs
        neckline_level = min(peak1_price, peak2_price)
        break_idx = self._check_neckline_break(df, trough3_idx, neckline_level, 'bullish')
        
        if break_idx is None:
            return None
        
        # Calculate pattern metrics
        pattern_height = neckline_level - avg_trough_price
        confidence = self._calculate_bottom_confidence(df, trough1_idx, trough3_idx,
                                                      max(peak1_idx_relative, peak2_idx_relative), 'triple')
        
        # Calculate trading levels
        entry_price = df.iloc[break_idx]['close']
        stop_loss = min(trough1_price, trough2_price, trough3_price) * 0.98
        take_profit = entry_price + pattern_height
        
        return PatternSignal(
            pattern_type='triple_bottom',
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            confidence=confidence,
            neckline_level=neckline_level,
            peak_prices=[trough1_price, trough2_price, trough3_price],
            peak_indices=[trough1_idx, trough2_idx, trough3_idx],
            neckline_break_index=break_idx,
            pattern_height=pattern_height
        )
    
    def _are_peaks_similar(self, price1: float, price2: float) -> bool:
        """Check if two prices are similar within tolerance"""
        
        avg_price = (price1 + price2) / 2
        diff_pct = abs(price1 - price2) / avg_price
        
        return diff_pct <= self.peak_similarity_pct
    
    def _check_neckline_break(self, df: pd.DataFrame, pattern_end_idx: int, 
                             neckline_level: float, direction: str) -> Optional[int]:
        """Check for neckline break after pattern completion"""
        
        # Look for break in next 10 bars
        end_search = min(pattern_end_idx + 10, len(df))
        
        for i in range(pattern_end_idx + 1, end_search):
            if direction == 'bearish':
                # Looking for break below neckline
                if df.iloc[i]['close'] < neckline_level * (1 - self.neckline_break_pct):
                    # Confirm with volume if required
                    if self._confirm_break_volume(df, i, pattern_end_idx):
                        return i
            else:
                # Looking for break above neckline
                if df.iloc[i]['close'] > neckline_level * (1 + self.neckline_break_pct):
                    # Confirm with volume if required
                    if self._confirm_break_volume(df, i, pattern_end_idx):
                        return i
        
        return None
    
    def _confirm_break_volume(self, df: pd.DataFrame, break_idx: int, pattern_end_idx: int) -> bool:
        """Confirm neckline break with volume analysis"""
        
        if not self.volume_confirmation or 'volume' not in df.columns:
            return True
        
        # Calculate average volume during pattern formation
        pattern_volume = df.iloc[max(0, pattern_end_idx-20):pattern_end_idx]['volume'].mean()
        break_volume = df.iloc[break_idx]['volume']
        
        # Break volume should be at least 1.5x average
        return break_volume >= pattern_volume * 1.5
    
    def _calculate_top_confidence(self, df: pd.DataFrame, start_idx: int, 
                                 end_idx: int, valley_idx: int, pattern_type: str) -> float:
        """Calculate confidence score for top patterns"""
        
        confidence = 0.5  # Base confidence
        
        # Factor 1: Pattern symmetry
        left_bars = valley_idx - start_idx
        right_bars = end_idx - valley_idx
        symmetry = 1 - abs(left_bars - right_bars) / max(left_bars, right_bars)
        confidence += symmetry * 0.2
        
        # Factor 2: Volume pattern (declining volume typical in tops)
        if 'volume' in df.columns:
            volume_pattern = self._analyze_top_volume_pattern(df, start_idx, end_idx)
            confidence += volume_pattern * 0.15
        
        # Factor 3: Pattern type bonus (triple patterns are more reliable)
        if pattern_type == 'triple':
            confidence += 0.1
        
        # Factor 4: Market context
        trend_context = self._check_trend_context(df, start_idx, 'bearish')
        confidence += trend_context * 0.15
        
        return min(1.0, max(0.1, confidence))
    
    def _calculate_bottom_confidence(self, df: pd.DataFrame, start_idx: int, 
                                    end_idx: int, peak_idx: int, pattern_type: str) -> float:
        """Calculate confidence score for bottom patterns"""
        
        confidence = 0.5  # Base confidence
        
        # Factor 1: Pattern symmetry
        left_bars = peak_idx - start_idx
        right_bars = end_idx - peak_idx
        symmetry = 1 - abs(left_bars - right_bars) / max(left_bars, right_bars)
        confidence += symmetry * 0.2
        
        # Factor 2: Volume pattern (increasing volume typical in bottoms)
        if 'volume' in df.columns:
            volume_pattern = self._analyze_bottom_volume_pattern(df, start_idx, end_idx)
            confidence += volume_pattern * 0.15
        
        # Factor 3: Pattern type bonus
        if pattern_type == 'triple':
            confidence += 0.1
        
        # Factor 4: Market context
        trend_context = self._check_trend_context(df, start_idx, 'bullish')
        confidence += trend_context * 0.15
        
        return min(1.0, max(0.1, confidence))
    
    def _analyze_top_volume_pattern(self, df: pd.DataFrame, start_idx: int, end_idx: int) -> float:
        """Analyze volume pattern for top formations (should decline)"""
        
        volumes = df.iloc[start_idx:end_idx+1]['volume'].values
        
        # Volume should generally decline in top patterns
        x = np.arange(len(volumes))
        try:
            slope, _, r_value, _, _ = linregress(x, volumes)
            # Negative slope is good for tops
            if slope < 0 and abs(r_value) > 0.3:
                return 1.0
            else:
                return 0.0
        except:
            return 0.5
    
    def _analyze_bottom_volume_pattern(self, df: pd.DataFrame, start_idx: int, end_idx: int) -> float:
        """Analyze volume pattern for bottom formations (should increase)"""
        
        volumes = df.iloc[start_idx:end_idx+1]['volume'].values
        
        # Volume should generally increase in bottom patterns
        x = np.arange(len(volumes))
        try:
            slope, _, r_value, _, _ = linregress(x, volumes)
            # Positive slope is good for bottoms
            if slope > 0 and abs(r_value) > 0.3:
                return 1.0
            else:
                return 0.0
        except:
            return 0.5
    
    def _check_trend_context(self, df: pd.DataFrame, pattern_start: int, expected_reversal: str) -> float:
        """Check if pattern appears in appropriate trend context"""
        
        # Look at trend before pattern
        lookback_start = max(0, pattern_start - 50)
        trend_data = df.iloc[lookback_start:pattern_start]
        
        if len(trend_data) < 10:
            return 0.5
        
        # Calculate trend direction
        start_price = trend_data.iloc[0]['close']
        end_price = trend_data.iloc[-1]['close']
        
        if expected_reversal == 'bearish':
            # Top patterns should appear after uptrend
            if end_price > start_price:
                return 1.0
            else:
                return 0.0
        else:
            # Bottom patterns should appear after downtrend
            if end_price < start_price:
                return 1.0
            else:
                return 0.0
    
    def _remove_overlapping_patterns(self, signals: List[PatternSignal]) -> List[PatternSignal]:
        """Remove overlapping patterns, keeping the highest confidence ones"""
        
        if not signals:
            return signals
        
        # Sort by confidence descending
        signals.sort(key=lambda x: x.confidence, reverse=True)
        
        filtered_signals = []
        used_ranges = []
        
        for signal in signals:
            # Check if this signal overlaps with any already selected
            signal_range = (min(signal.peak_indices), max(signal.peak_indices))
            
            overlaps = False
            for used_range in used_ranges:
                if (signal_range[0] <= used_range[1] and signal_range[1] >= used_range[0]):
                    overlaps = True
                    break
            
            if not overlaps:
                filtered_signals.append(signal)
                used_ranges.append(signal_range)
        
        return filtered_signals

# Example usage and testing
def example_usage():
    """Example of how to use the DoubleTripletopsBottomsDetector"""
    
    # Create sample data with some patterns
    dates = pd.date_range('2023-01-01', periods=300, freq='D')
    np.random.seed(42)
    
    # Generate sample price data with embedded patterns
    base_trend = np.linspace(100, 120, 300)
    noise = np.random.randn(300) * 2
    
    # Add some double top pattern around index 150-200
    prices = base_trend + noise
    prices[150:160] += 5  # First peak
    prices[160:170] -= 3  # Valley
    prices[170:180] += 4  # Second peak
    prices[180:190] -= 8  # Breakdown
    
    df = pd.DataFrame({
        'date': dates,
        'open': prices + np.random.randn(300) * 0.5,
        'high': prices + np.abs(np.random.randn(300) * 1.5),
        'low': prices - np.abs(np.random.randn(300) * 1.5),
        'close': prices,
        'volume': np.random.randint(1000, 10000, 300)
    })
    
    # Initialize detector
    detector = DoubleTripletopsBottomsDetector(
        peak_similarity_pct=3.0,
        min_valley_depth_pct=2.0,
        min_pattern_bars=15,
        max_pattern_bars=80,
        volume_confirmation=True
    )
    
    # Detect patterns
    signals = detector.detect_patterns(df)
    
    # Print results
    print(f"Found {len(signals)} patterns:")
    for i, signal in enumerate(signals):
        print(f"\nPattern {i+1}:")
        print(f"Type: {signal.pattern_type}")
        print(f"Peak Prices: {[f'${p:.2f}' for p in signal.peak_prices]}")
        print(f"Neckline: ${signal.neckline_level:.2f}")
        print(f"Entry: ${signal.entry_price:.2f}")
        print(f"Stop Loss: ${signal.stop_loss:.2f}")
        print(f"Take Profit: ${signal.take_profit:.2f}")
        print(f"Confidence: {signal.confidence:.2f}")
        print(f"Risk/Reward: {abs(signal.take_profit - signal.entry_price) / abs(signal.entry_price - signal.stop_loss):.2f}")

if __name__ == "__main__":
    example_usage()