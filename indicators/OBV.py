import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from enum import Enum

class OBVSignal(Enum):
    """OBV Trading Signals"""
    BULLISH_DIVERGENCE = "bullish_divergence"
    BEARISH_DIVERGENCE = "bearish_divergence"
    BULLISH_BREAKOUT = "bullish_breakout"
    BEARISH_BREAKDOWN = "bearish_breakdown"
    TREND_CONFIRMATION = "trend_confirmation"
    NEUTRAL = "neutral"

@dataclass
class OBVResult:
    """Container for OBV calculation results"""
    obv: np.ndarray
    obv_ma: Optional[np.ndarray] = None
    obv_slope: Optional[np.ndarray] = None
    price_slope: Optional[np.ndarray] = None
    signals: Optional[List[Dict]] = None
    support_levels: Optional[List[float]] = None
    resistance_levels: Optional[List[float]] = None

class OnBalanceVolume:
    """
    On-Balance Volume (OBV) Indicator for Algorithmic Trading
    
    OBV measures buying and selling pressure as a cumulative indicator
    that adds volume on up days and subtracts volume on down days.
    
    Features:
    - Basic OBV calculation
    - Moving average smoothing
    - Divergence detection
    - Breakout/breakdown signals
    - Support/resistance level identification
    - Trend confirmation signals
    """
    
    def __init__(self, ma_period: int = 20, slope_period: int = 14, 
                 divergence_lookback: int = 20, min_divergence_bars: int = 5):
        """
        Initialize OBV indicator
        
        Args:
            ma_period: Period for OBV moving average
            slope_period: Period for slope calculation
            divergence_lookback: Lookback period for divergence detection
            min_divergence_bars: Minimum bars for valid divergence
        """
        self.ma_period = ma_period
        self.slope_period = slope_period
        self.divergence_lookback = divergence_lookback
        self.min_divergence_bars = min_divergence_bars
    
    def calculate(self, close: np.ndarray, volume: np.ndarray, 
                 calculate_signals: bool = True) -> OBVResult:
        """
        Calculate On-Balance Volume
        
        Args:
            close: Array of closing prices
            volume: Array of volumes
            calculate_signals: Whether to calculate trading signals
            
        Returns:
            OBVResult containing OBV values and analysis
        """
        if len(close) != len(volume):
            raise ValueError("Close and volume arrays must have the same length")
        
        if len(close) < 2:
            raise ValueError("Need at least 2 data points to calculate OBV")
        
        # Calculate basic OBV
        obv = self._calculate_obv(close, volume)
        
        # Calculate moving average if requested
        obv_ma = None
        if self.ma_period > 0 and len(obv) >= self.ma_period:
            obv_ma = self._moving_average(obv, self.ma_period)
        
        # Calculate slopes for trend analysis
        obv_slope = None
        price_slope = None
        if self.slope_period > 0 and len(obv) >= self.slope_period:
            obv_slope = self._calculate_slope(obv, self.slope_period)
            price_slope = self._calculate_slope(close, self.slope_period)
        
        # Generate trading signals
        signals = None
        support_levels = None
        resistance_levels = None
        
        if calculate_signals and len(obv) >= self.divergence_lookback:
            signals = self._detect_signals(close, obv, obv_slope, price_slope)
            support_levels, resistance_levels = self._find_support_resistance(obv)
        
        return OBVResult(
            obv=obv,
            obv_ma=obv_ma,
            obv_slope=obv_slope,
            price_slope=price_slope,
            signals=signals,
            support_levels=support_levels,
            resistance_levels=resistance_levels
        )
    
    def _calculate_obv(self, close: np.ndarray, volume: np.ndarray) -> np.ndarray:
        """Calculate basic OBV values"""
        obv = np.zeros(len(close))
        obv[0] = volume[0]  # Initialize first value
        
        for i in range(1, len(close)):
            if close[i] > close[i-1]:
                # Up day: add volume
                obv[i] = obv[i-1] + volume[i]
            elif close[i] < close[i-1]:
                # Down day: subtract volume
                obv[i] = obv[i-1] - volume[i]
            else:
                # Unchanged: keep same OBV
                obv[i] = obv[i-1]
        
        return obv
    
    def _moving_average(self, data: np.ndarray, period: int) -> np.ndarray:
        """Calculate simple moving average"""
        ma = np.full(len(data), np.nan)
        for i in range(period - 1, len(data)):
            ma[i] = np.mean(data[i - period + 1:i + 1])
        return ma
    
    def _calculate_slope(self, data: np.ndarray, period: int) -> np.ndarray:
        """Calculate slope over specified period"""
        slopes = np.full(len(data), np.nan)
        
        for i in range(period - 1, len(data)):
            y = data[i - period + 1:i + 1]
            x = np.arange(period)
            
            # Linear regression slope
            n = len(x)
            slope = (n * np.sum(x * y) - np.sum(x) * np.sum(y)) / (n * np.sum(x**2) - np.sum(x)**2)
            slopes[i] = slope
        
        return slopes
    
    def _detect_signals(self, close: np.ndarray, obv: np.ndarray, 
                       obv_slope: Optional[np.ndarray], 
                       price_slope: Optional[np.ndarray]) -> List[Dict]:
        """Detect OBV trading signals"""
        signals = []
        
        for i in range(self.divergence_lookback, len(close)):
            signal_info = {
                'index': i,
                'price': close[i],
                'obv': obv[i],
                'signal': OBVSignal.NEUTRAL,
                'strength': 0.0,
                'description': ""
            }
            
            # Check for divergences
            divergence_signal = self._check_divergence(close, obv, i)
            if divergence_signal:
                signal_info.update(divergence_signal)
                signals.append(signal_info.copy())
                continue
            
            # Check for breakouts/breakdowns
            breakout_signal = self._check_breakout(obv, i)
            if breakout_signal:
                signal_info.update(breakout_signal)
                signals.append(signal_info.copy())
                continue
            
            # Check for trend confirmation
            if obv_slope is not None and price_slope is not None:
                trend_signal = self._check_trend_confirmation(
                    obv_slope[i], price_slope[i]
                )
                if trend_signal:
                    signal_info.update(trend_signal)
                    signals.append(signal_info.copy())
        
        return signals
    
    def _check_divergence(self, close: np.ndarray, obv: np.ndarray, 
                         current_idx: int) -> Optional[Dict]:
        """Check for bullish/bearish divergences"""
        lookback_start = current_idx - self.divergence_lookback
        
        # Find recent high/low in price and OBV
        price_segment = close[lookback_start:current_idx + 1]
        obv_segment = obv[lookback_start:current_idx + 1]
        
        # Find peaks and troughs
        price_high_idx = np.argmax(price_segment)
        price_low_idx = np.argmin(price_segment)
        obv_high_idx = np.argmax(obv_segment)
        obv_low_idx = np.argmin(obv_segment)
        
        # Bullish divergence: price making lower lows, OBV making higher lows
        if (price_low_idx > len(price_segment) // 2 and 
            obv_low_idx < len(obv_segment) // 2):
            
            recent_price_low = price_segment[price_low_idx]
            earlier_price_low = np.min(price_segment[:len(price_segment)//2])
            recent_obv_low = obv_segment[obv_low_idx]
            earlier_obv_low = np.min(obv_segment[:len(obv_segment)//2])
            
            if recent_price_low < earlier_price_low and recent_obv_low > earlier_obv_low:
                return {
                    'signal': OBVSignal.BULLISH_DIVERGENCE,
                    'strength': 0.8,
                    'description': "Bullish divergence: Price lower low, OBV higher low"
                }
        
        # Bearish divergence: price making higher highs, OBV making lower highs
        if (price_high_idx > len(price_segment) // 2 and 
            obv_high_idx < len(obv_segment) // 2):
            
            recent_price_high = price_segment[price_high_idx]
            earlier_price_high = np.max(price_segment[:len(price_segment)//2])
            recent_obv_high = obv_segment[obv_high_idx]
            earlier_obv_high = np.max(obv_segment[:len(obv_segment)//2])
            
            if recent_price_high > earlier_price_high and recent_obv_high < earlier_obv_high:
                return {
                    'signal': OBVSignal.BEARISH_DIVERGENCE,
                    'strength': 0.8,
                    'description': "Bearish divergence: Price higher high, OBV lower high"
                }
        
        return None
    
    def _check_breakout(self, obv: np.ndarray, current_idx: int) -> Optional[Dict]:
        """Check for OBV breakouts/breakdowns"""
        if current_idx < 20:  # Need enough history
            return None
        
        # Calculate recent resistance/support levels
        lookback_period = min(20, current_idx)
        recent_obv = obv[current_idx - lookback_period:current_idx]
        current_obv = obv[current_idx]
        
        resistance_level = np.max(recent_obv)
        support_level = np.min(recent_obv)
        
        # Check for breakout above resistance
        if current_obv > resistance_level * 1.02:  # 2% buffer
            return {
                'signal': OBVSignal.BULLISH_BREAKOUT,
                'strength': 0.7,
                'description': f"OBV breakout above resistance at {resistance_level:.0f}"
            }
        
        # Check for breakdown below support
        if current_obv < support_level * 0.98:  # 2% buffer
            return {
                'signal': OBVSignal.BEARISH_BREAKDOWN,
                'strength': 0.7,
                'description': f"OBV breakdown below support at {support_level:.0f}"
            }
        
        return None
    
    def _check_trend_confirmation(self, obv_slope: float, 
                                 price_slope: float) -> Optional[Dict]:
        """Check if OBV confirms price trend"""
        # Both slopes should be in same direction for confirmation
        if obv_slope > 0 and price_slope > 0:
            strength = min(abs(obv_slope), abs(price_slope)) / max(abs(obv_slope), abs(price_slope))
            return {
                'signal': OBVSignal.TREND_CONFIRMATION,
                'strength': strength * 0.6,  # Medium strength signal
                'description': "OBV confirms uptrend"
            }
        elif obv_slope < 0 and price_slope < 0:
            strength = min(abs(obv_slope), abs(price_slope)) / max(abs(obv_slope), abs(price_slope))
            return {
                'signal': OBVSignal.TREND_CONFIRMATION,
                'strength': strength * 0.6,
                'description': "OBV confirms downtrend"
            }
        
        return None
    
    def _find_support_resistance(self, obv: np.ndarray) -> Tuple[List[float], List[float]]:
        """Find support and resistance levels in OBV"""
        # Use recent data for current S/R levels
        recent_period = min(50, len(obv))
        recent_obv = obv[-recent_period:]
        
        # Find local peaks and troughs
        support_levels = []
        resistance_levels = []
        
        for i in range(2, len(recent_obv) - 2):
            # Local minimum (support)
            if (recent_obv[i] < recent_obv[i-1] and recent_obv[i] < recent_obv[i-2] and
                recent_obv[i] < recent_obv[i+1] and recent_obv[i] < recent_obv[i+2]):
                support_levels.append(recent_obv[i])
            
            # Local maximum (resistance)
            if (recent_obv[i] > recent_obv[i-1] and recent_obv[i] > recent_obv[i-2] and
                recent_obv[i] > recent_obv[i+1] and recent_obv[i] > recent_obv[i+2]):
                resistance_levels.append(recent_obv[i])
        
        # Remove duplicates and sort
        support_levels = sorted(list(set(support_levels)))
        resistance_levels = sorted(list(set(resistance_levels)), reverse=True)
        
        return support_levels, resistance_levels
    
    def get_current_signal(self, result: OBVResult) -> Dict:
        """Get the most recent trading signal"""
        if not result.signals or len(result.signals) == 0:
            return {
                'signal': OBVSignal.NEUTRAL,
                'strength': 0.0,
                'description': "No signals detected"
            }
        
        # Return the most recent signal
        latest_signal = result.signals[-1]
        return {
            'signal': latest_signal['signal'],
            'strength': latest_signal['strength'],
            'description': latest_signal['description']
        }
    
    def analyze_obv_trend(self, result: OBVResult, periods: int = 10) -> Dict:
        """Analyze OBV trend over specified periods"""
        if len(result.obv) < periods:
            return {'trend': 'insufficient_data'}
        
        recent_obv = result.obv[-periods:]
        obv_change = (recent_obv[-1] - recent_obv[0]) / abs(recent_obv[0]) * 100
        
        # Determine trend strength
        if abs(obv_change) > 10:
            strength = "strong"
        elif abs(obv_change) > 5:
            strength = "moderate"
        else:
            strength = "weak"
        
        trend_direction = "bullish" if obv_change > 0 else "bearish"
        
        return {
            'trend': trend_direction,
            'strength': strength,
            'change_percent': obv_change,
            'current_obv': result.obv[-1],
            'periods_analyzed': periods
        }

# Utility functions for integration with trading systems
def create_sample_data(n_bars: int = 200) -> pd.DataFrame:
    """Create sample OHLCV data for testing"""
    np.random.seed(42)
    
    data = []
    base_price = 100
    base_volume = 1000000
    
    for i in range(n_bars):
        # Generate price with trend and noise
        trend = 0.02 if i > n_bars // 2 else -0.01  # Trend change halfway
        price_change = np.random.normal(trend, 1.0)
        base_price += price_change
        
        # Generate volume with correlation to price moves
        volume_multiplier = 1 + abs(price_change) * 0.5  # Higher volume on big moves
        volume = int(base_volume * volume_multiplier * np.random.uniform(0.8, 1.2))
        
        data.append({
            'close': base_price,
            'volume': volume,
            'timestamp': i
        })
    
    return pd.DataFrame(data)

def run_obv_analysis(df: pd.DataFrame) -> None:
    """Run complete OBV analysis on dataset"""
    obv_indicator = OnBalanceVolume(ma_period=20, slope_period=14)
    
    result = obv_indicator.calculate(
        close=df['close'].values,
        volume=df['volume'].values
    )
    
    # Get current signal
    current_signal = obv_indicator.get_current_signal(result)
    
    # Analyze trend
    trend_analysis = obv_indicator.analyze_obv_trend(result)
    
    print("=== On-Balance Volume Analysis ===")
    print(f"Current OBV: {result.obv[-1]:,.0f}")
    print(f"Current Signal: {current_signal['signal'].value}")
    print(f"Signal Strength: {current_signal['strength']:.2f}")
    print(f"Description: {current_signal['description']}")
    print(f"\nTrend Analysis:")
    print(f"Direction: {trend_analysis['trend']}")
    print(f"Strength: {trend_analysis['strength']}")
    print(f"Change: {trend_analysis['change_percent']:.2f}%")
    
    if result.support_levels:
        print(f"\nSupport Levels: {[f'{level:,.0f}' for level in result.support_levels[:3]]}")
    if result.resistance_levels:
        print(f"Resistance Levels: {[f'{level:,.0f}' for level in result.resistance_levels[:3]]}")
    
    if result.signals:
        print(f"\nTotal Signals Generated: {len(result.signals)}")
        recent_signals = result.signals[-3:] if len(result.signals) >= 3 else result.signals
        print("Recent Signals:")
        for signal in recent_signals:
            print(f"  {signal['signal'].value}: {signal['description']} (Strength: {signal['strength']:.2f})")

# Example usage
if __name__ == "__main__":
    # Create sample data
    sample_data = create_sample_data(300)
    
    # Run analysis
    run_obv_analysis(sample_data)
    
    # Example of using in trading bot
    obv = OnBalanceVolume()
    result = obv.calculate(
        close=sample_data['close'].values,
        volume=sample_data['volume'].values
    )
    
    # Trading logic example
    current_signal = obv.get_current_signal(result)
    if current_signal['signal'] == OBVSignal.BULLISH_DIVERGENCE and current_signal['strength'] > 0.7:
        print("\n>>> STRONG BUY SIGNAL DETECTED <<<")
    elif current_signal['signal'] == OBVSignal.BEARISH_DIVERGENCE and current_signal['strength'] > 0.7:
        print("\n>>> STRONG SELL SIGNAL DETECTED <<<")