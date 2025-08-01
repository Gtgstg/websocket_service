import numpy as np
import pandas as pd
from typing import Tuple, Optional, Dict, List
from dataclasses import dataclass
from scipy.stats import linregress
import warnings
warnings.filterwarnings('ignore')

@dataclass
class PatternSignal:
    """Data class to store pattern detection results"""
    pattern_type: str  # 'bull_flag', 'bear_flag', 'bull_pennant', 'bear_pennant'
    entry_price: float
    stop_loss: float
    take_profit: float
    confidence: float  # 0-1 scale
    flagpole_height: float
    consolidation_start: int
    consolidation_end: int
    breakout_index: int

class FlagsPennantsDetector:
    """
    Detects Flag and Pennant chart patterns for algorithmic trading
    
    Flags: Brief consolidation after strong move, parallel trend lines
    Pennants: Brief consolidation after strong move, converging trend lines
    """
    
    def __init__(self, 
                 min_flagpole_pct: float = 5.0,  # Minimum flagpole move %
                 max_consolidation_bars: int = 20,  # Max consolidation period
                 min_consolidation_bars: int = 5,   # Min consolidation period
                 volume_confirmation: bool = True,
                 breakout_volume_multiplier: float = 1.5):
        
        self.min_flagpole_pct = min_flagpole_pct / 100
        self.max_consolidation_bars = max_consolidation_bars
        self.min_consolidation_bars = min_consolidation_bars
        self.volume_confirmation = volume_confirmation
        self.breakout_volume_multiplier = breakout_volume_multiplier
    
    def detect_patterns(self, df: pd.DataFrame) -> List[PatternSignal]:
        """
        Main method to detect flag and pennant patterns
        
        Args:
            df: DataFrame with columns ['open', 'high', 'low', 'close', 'volume']
        
        Returns:
            List of PatternSignal objects
        """
        signals = []
        
        # Ensure we have enough data
        if len(df) < self.max_consolidation_bars + 10:
            return signals
        
        # Look for patterns starting from sufficient history
        for i in range(50, len(df) - 5):
            
            # Check for bullish patterns
            bull_signal = self._detect_bullish_pattern(df, i)
            if bull_signal:
                signals.append(bull_signal)
            
            # Check for bearish patterns  
            bear_signal = self._detect_bearish_pattern(df, i)
            if bear_signal:
                signals.append(bear_signal)
                
        return signals
    
    def _detect_bullish_pattern(self, df: pd.DataFrame, current_idx: int) -> Optional[PatternSignal]:
        """Detect bullish flag or pennant patterns"""
        
        # Step 1: Find the flagpole (strong upward move)
        flagpole_data = self._find_bullish_flagpole(df, current_idx)
        if not flagpole_data:
            return None
            
        flagpole_start, flagpole_end, flagpole_height = flagpole_data
        
        # Step 2: Find consolidation period after flagpole
        consolidation_data = self._find_consolidation(df, flagpole_end, current_idx)
        if not consolidation_data:
            return None
            
        cons_start, cons_end, cons_highs, cons_lows = consolidation_data
        
        # Step 3: Determine if it's a flag or pennant
        pattern_type = self._classify_pattern(cons_highs, cons_lows, 'bullish')
        if not pattern_type:
            return None
        
        # Step 4: Check for breakout
        breakout_idx = self._check_bullish_breakout(df, cons_end, max(cons_highs))
        if breakout_idx is None:
            return None
        
        # Step 5: Calculate confidence score
        confidence = self._calculate_confidence(df, flagpole_start, cons_end, 
                                              flagpole_height, pattern_type, 'bullish')
        
        # Step 6: Calculate trading levels
        entry_price = df.iloc[breakout_idx]['close']
        stop_loss = min(cons_lows) * 0.98  # 2% below consolidation low
        take_profit = entry_price + flagpole_height  # Flagpole height projection
        
        return PatternSignal(
            pattern_type=pattern_type,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            confidence=confidence,
            flagpole_height=flagpole_height,
            consolidation_start=cons_start,
            consolidation_end=cons_end,
            breakout_index=breakout_idx
        )
    
    def _detect_bearish_pattern(self, df: pd.DataFrame, current_idx: int) -> Optional[PatternSignal]:
        """Detect bearish flag or pennant patterns"""
        
        # Step 1: Find the flagpole (strong downward move)
        flagpole_data = self._find_bearish_flagpole(df, current_idx)
        if not flagpole_data:
            return None
            
        flagpole_start, flagpole_end, flagpole_height = flagpole_data
        
        # Step 2: Find consolidation period after flagpole
        consolidation_data = self._find_consolidation(df, flagpole_end, current_idx)
        if not consolidation_data:
            return None
            
        cons_start, cons_end, cons_highs, cons_lows = consolidation_data
        
        # Step 3: Determine if it's a flag or pennant
        pattern_type = self._classify_pattern(cons_highs, cons_lows, 'bearish')
        if not pattern_type:
            return None
        
        # Step 4: Check for breakout
        breakout_idx = self._check_bearish_breakout(df, cons_end, min(cons_lows))
        if breakout_idx is None:
            return None
        
        # Step 5: Calculate confidence score
        confidence = self._calculate_confidence(df, flagpole_start, cons_end, 
                                              flagpole_height, pattern_type, 'bearish')
        
        # Step 6: Calculate trading levels
        entry_price = df.iloc[breakout_idx]['close']
        stop_loss = max(cons_highs) * 1.02  # 2% above consolidation high
        take_profit = entry_price - flagpole_height  # Flagpole height projection
        
        return PatternSignal(
            pattern_type=pattern_type,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            confidence=confidence,
            flagpole_height=flagpole_height,
            consolidation_start=cons_start,
            consolidation_end=cons_end,
            breakout_index=breakout_idx
        )
    
    def _find_bullish_flagpole(self, df: pd.DataFrame, end_idx: int) -> Optional[Tuple[int, int, float]]:
        """Find a strong bullish move (flagpole)"""
        
        # Look back for the start of the strong move
        for lookback in range(5, 30):
            if end_idx - lookback < 0:
                continue
                
            start_idx = end_idx - lookback
            start_price = df.iloc[start_idx]['low']
            end_price = df.iloc[end_idx]['high']
            
            # Calculate percentage move
            pct_move = (end_price - start_price) / start_price
            
            # Check if move is strong enough
            if pct_move >= self.min_flagpole_pct:
                # Verify it's mostly upward movement
                if self._is_strong_directional_move(df, start_idx, end_idx, 'up'):
                    return start_idx, end_idx, end_price - start_price
        
        return None
    
    def _find_bearish_flagpole(self, df: pd.DataFrame, end_idx: int) -> Optional[Tuple[int, int, float]]:
        """Find a strong bearish move (flagpole)"""
        
        # Look back for the start of the strong move
        for lookback in range(5, 30):
            if end_idx - lookback < 0:
                continue
                
            start_idx = end_idx - lookback
            start_price = df.iloc[start_idx]['high']
            end_price = df.iloc[end_idx]['low']
            
            # Calculate percentage move
            pct_move = abs((end_price - start_price) / start_price)
            
            # Check if move is strong enough
            if pct_move >= self.min_flagpole_pct:
                # Verify it's mostly downward movement
                if self._is_strong_directional_move(df, start_idx, end_idx, 'down'):
                    return start_idx, end_idx, start_price - end_price
        
        return None
    
    def _is_strong_directional_move(self, df: pd.DataFrame, start_idx: int, 
                                   end_idx: int, direction: str) -> bool:
        """Verify the move is consistently in one direction"""
        
        prices = df.iloc[start_idx:end_idx+1]['close'].values
        
        if direction == 'up':
            # Count bars that closed higher than previous
            up_bars = sum(1 for i in range(1, len(prices)) if prices[i] > prices[i-1])
            return up_bars / (len(prices) - 1) >= 0.6  # 60% of bars should be up
        else:
            # Count bars that closed lower than previous  
            down_bars = sum(1 for i in range(1, len(prices)) if prices[i] < prices[i-1])
            return down_bars / (len(prices) - 1) >= 0.6  # 60% of bars should be down
    
    def _find_consolidation(self, df: pd.DataFrame, start_idx: int, 
                          max_idx: int) -> Optional[Tuple[int, int, List[float], List[float]]]:
        """Find consolidation period after flagpole"""
        
        for end_idx in range(start_idx + self.min_consolidation_bars, 
                           min(start_idx + self.max_consolidation_bars, max_idx)):
            
            # Get consolidation period data
            cons_data = df.iloc[start_idx:end_idx+1]
            highs = cons_data['high'].tolist()
            lows = cons_data['low'].tolist()
            
            # Check if it's a valid consolidation (trading range)
            if self._is_valid_consolidation(highs, lows):
                return start_idx, end_idx, highs, lows
        
        return None
    
    def _is_valid_consolidation(self, highs: List[float], lows: List[float]) -> bool:
        """Check if price action represents valid consolidation"""
        
        high_range = max(highs) - min(highs)
        low_range = max(lows) - min(lows)
        overall_range = max(highs) - min(lows)
        
        # Consolidation should have relatively small range
        avg_price = (max(highs) + min(lows)) / 2
        range_pct = overall_range / avg_price
        
        # Range should be less than 8% of price
        return range_pct < 0.08
    
    def _classify_pattern(self, highs: List[float], lows: List[float], 
                         direction: str) -> Optional[str]:
        """Classify as flag or pennant based on trend line slopes"""
        
        if len(highs) < 4 or len(lows) < 4:
            return None
        
        # Create x values (time indices)
        x = list(range(len(highs)))
        
        # Calculate trend lines
        try:
            high_slope, _, high_r, _, _ = linregress(x, highs)
            low_slope, _, low_r, _, _ = linregress(x, lows)
        except:
            return None
        
        # Check if trend lines are valid (reasonable correlation)
        if abs(high_r) < 0.5 or abs(low_r) < 0.5:
            return None
        
        # Classify based on slope relationship
        slope_diff = abs(high_slope - low_slope)
        
        if slope_diff < 0.001:  # Parallel lines = Flag
            if direction == 'bullish':
                return 'bull_flag'
            else:
                return 'bear_flag'
        else:  # Converging lines = Pennant
            # Check if lines are actually converging
            if (high_slope < 0 and low_slope > 0) or (high_slope > 0 and low_slope < 0):
                if direction == 'bullish':
                    return 'bull_pennant' 
                else:
                    return 'bear_pennant'
        
        return None
    
    def _check_bullish_breakout(self, df: pd.DataFrame, cons_end: int, 
                               resistance_level: float) -> Optional[int]:
        """Check for bullish breakout above resistance"""
        
        # Look for breakout in next few bars
        for i in range(cons_end + 1, min(cons_end + 5, len(df))):
            if df.iloc[i]['close'] > resistance_level * 1.01:  # 1% above resistance
                # Confirm with volume if required
                if self.volume_confirmation:
                    if self._confirm_breakout_volume(df, i, cons_end):
                        return i
                else:
                    return i
        
        return None
    
    def _check_bearish_breakout(self, df: pd.DataFrame, cons_end: int, 
                               support_level: float) -> Optional[int]:
        """Check for bearish breakout below support"""
        
        # Look for breakout in next few bars
        for i in range(cons_end + 1, min(cons_end + 5, len(df))):
            if df.iloc[i]['close'] < support_level * 0.99:  # 1% below support
                # Confirm with volume if required
                if self.volume_confirmation:
                    if self._confirm_breakout_volume(df, i, cons_end):
                        return i
                else:
                    return i
        
        return None
    
    def _confirm_breakout_volume(self, df: pd.DataFrame, breakout_idx: int, 
                                cons_end: int) -> bool:
        """Confirm breakout with volume analysis"""
        
        if 'volume' not in df.columns:
            return True  # Skip volume check if no volume data
        
        # Calculate average volume during consolidation
        cons_volume = df.iloc[max(0, cons_end-10):cons_end]['volume'].mean()
        breakout_volume = df.iloc[breakout_idx]['volume']
        
        # Breakout volume should be higher than average
        return breakout_volume >= cons_volume * self.breakout_volume_multiplier
    
    def _calculate_confidence(self, df: pd.DataFrame, flagpole_start: int, 
                            cons_end: int, flagpole_height: float, 
                            pattern_type: str, direction: str) -> float:
        """Calculate confidence score for the pattern (0-1)"""
        
        confidence = 0.5  # Base confidence
        
        # Factor 1: Flagpole strength (stronger flagpole = higher confidence)
        flagpole_bars = cons_end - flagpole_start
        if flagpole_bars <= 10:
            confidence += 0.1
        
        # Factor 2: Pattern clarity (cleaner consolidation = higher confidence)
        cons_data = df.iloc[flagpole_start:cons_end+1]
        volatility = cons_data['close'].std() / cons_data['close'].mean()
        if volatility < 0.02:  # Low volatility consolidation
            confidence += 0.15
        
        # Factor 3: Volume pattern (if available)
        if 'volume' in df.columns:
            volume_trend = self._analyze_volume_pattern(df, flagpole_start, cons_end)
            confidence += volume_trend * 0.1
        
        # Factor 4: Market context (trend alignment)
        trend_alignment = self._check_trend_alignment(df, flagpole_start, direction)
        confidence += trend_alignment * 0.15
        
        return min(1.0, max(0.1, confidence))
    
    def _analyze_volume_pattern(self, df: pd.DataFrame, start_idx: int, end_idx: int) -> float:
        """Analyze volume pattern during consolidation"""
        
        volumes = df.iloc[start_idx:end_idx+1]['volume'].values
        
        # Volume should generally decrease during consolidation
        first_half = volumes[:len(volumes)//2].mean()
        second_half = volumes[len(volumes)//2:].mean()
        
        if second_half < first_half:
            return 1.0  # Good volume pattern
        else:
            return 0.0  # Poor volume pattern
    
    def _check_trend_alignment(self, df: pd.DataFrame, start_idx: int, direction: str) -> float:
        """Check if pattern aligns with broader trend"""
        
        # Look at longer term trend before the pattern
        lookback_start = max(0, start_idx - 50)
        trend_data = df.iloc[lookback_start:start_idx]
        
        if len(trend_data) < 10:
            return 0.5  # Neutral if insufficient data
        
        # Calculate trend direction
        start_price = trend_data.iloc[0]['close']
        end_price = trend_data.iloc[-1]['close']
        
        trend_direction = 'up' if end_price > start_price else 'down'
        
        # Check alignment
        if (direction == 'bullish' and trend_direction == 'up') or \
           (direction == 'bearish' and trend_direction == 'down'):
            return 1.0  # Good alignment
        else:
            return 0.0  # Poor alignment

# Example usage and testing
def example_usage():
    """Example of how to use the FlagsPennantsDetector"""
    
    # Create sample data (replace with your actual data)
    dates = pd.date_range('2023-01-01', periods=200, freq='D')
    np.random.seed(42)
    
    # Generate sample price data with some patterns
    prices = 100 + np.cumsum(np.random.randn(200) * 0.5)
    
    df = pd.DataFrame({
        'date': dates,
        'open': prices + np.random.randn(200) * 0.1,
        'high': prices + np.abs(np.random.randn(200) * 0.3),
        'low': prices - np.abs(np.random.randn(200) * 0.3),
        'close': prices,
        'volume': np.random.randint(1000, 10000, 200)
    })
    
    # Initialize detector
    detector = FlagsPennantsDetector(
        min_flagpole_pct=3.0,
        max_consolidation_bars=15,
        min_consolidation_bars=5,
        volume_confirmation=True
    )
    
    # Detect patterns
    signals = detector.detect_patterns(df)
    
    # Print results
    print(f"Found {len(signals)} patterns:")
    for i, signal in enumerate(signals):
        print(f"\nPattern {i+1}:")
        print(f"Type: {signal.pattern_type}")
        print(f"Entry: ${signal.entry_price:.2f}")
        print(f"Stop Loss: ${signal.stop_loss:.2f}")
        print(f"Take Profit: ${signal.take_profit:.2f}")
        print(f"Confidence: {signal.confidence:.2f}")
        print(f"Risk/Reward: {abs(signal.take_profit - signal.entry_price) / abs(signal.entry_price - signal.stop_loss):.2f}")

if __name__ == "__main__":
    example_usage()