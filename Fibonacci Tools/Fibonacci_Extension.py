import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from enum import Enum

class ExtensionType(Enum):
    EXTERNAL = "external"  # Standard extensions (1.272, 1.618, etc.)
    INTERNAL = "internal"  # Internal projections (0.618, 1.0, etc.)

class TrendDirection(Enum):
    BULLISH = "bullish"
    BEARISH = "bearish"

@dataclass
class FibExtensionLevel:
    """Represents a Fibonacci extension level"""
    ratio: float
    price: float
    label: str
    extension_type: ExtensionType

@dataclass
class SwingPoint:
    """Represents a swing high/low point"""
    index: int
    price: float
    timestamp: Optional[str] = None
    point_type: Optional[str] = None  # 'high', 'low'

@dataclass
class ExtensionSetup:
    """Three-point Fibonacci extension setup"""
    point_a: SwingPoint  # Start of move
    point_b: SwingPoint  # End of move (retracement start)
    point_c: SwingPoint  # End of retracement (extension start)
    trend_direction: TrendDirection

class FibonacciExtension:
    """
    Fibonacci Extension tool for algorithmic trading
    
    Features:
    - Multiple extension calculation methods (ABC, AB=CD patterns)
    - Price target identification
    - Confluence level detection
    - Risk/reward ratio calculations
    - Dynamic stop-loss and take-profit levels
    """
    
    # Standard Fibonacci extension ratios
    STANDARD_RATIOS = [0.618, 1.0, 1.272, 1.414, 1.618, 2.0, 2.618, 3.618, 4.236]
    INTERNAL_RATIOS = [0.382, 0.5, 0.618, 0.786, 1.0]
    
    def __init__(self, 
                 extension_ratios: List[float] = None,
                 internal_ratios: List[float] = None,
                 lookback_period: int = 20,
                 min_swing_percentage: float = 0.02):
        """
        Initialize Fibonacci Extension tool
        
        Args:
            extension_ratios: List of external extension ratios
            internal_ratios: List of internal projection ratios
            lookback_period: Period for swing point detection
            min_swing_percentage: Minimum percentage move to qualify as swing
        """
        self.extension_ratios = extension_ratios or self.STANDARD_RATIOS
        self.internal_ratios = internal_ratios or self.INTERNAL_RATIOS
        self.lookback_period = lookback_period
        self.min_swing_percentage = min_swing_percentage
        self.extension_levels = []
        self.current_setup = None
        
    def find_swing_points(self, data: pd.Series, 
                         min_move: float = None) -> Dict[str, List[SwingPoint]]:
        """
        Find significant swing highs and lows
        
        Args:
            data: Price data
            min_move: Minimum price move to qualify as swing
            
        Returns:
            Dictionary with swing highs and lows
        """
        if min_move is None:
            min_move = data.mean() * self.min_swing_percentage
            
        highs = []
        lows = []
        
        for i in range(self.lookback_period, len(data) - self.lookback_period):
            # Check for swing high
            is_high = True
            peak_value = data.iloc[i]
            
            # Verify it's a local maximum
            for j in range(i - self.lookback_period, i + self.lookback_period + 1):
                if j != i and data.iloc[j] >= peak_value:
                    is_high = False
                    break
            
            # Check minimum move requirement
            if is_high:
                left_low = data.iloc[i - self.lookback_period:i].min()
                right_low = data.iloc[i:i + self.lookback_period].min()
                min_low = min(left_low, right_low)
                
                if peak_value - min_low >= min_move:
                    highs.append(SwingPoint(i, peak_value, point_type='high'))
            
            # Check for swing low
            is_low = True
            trough_value = data.iloc[i]
            
            # Verify it's a local minimum
            for j in range(i - self.lookback_period, i + self.lookback_period + 1):
                if j != i and data.iloc[j] <= trough_value:
                    is_low = False
                    break
            
            # Check minimum move requirement
            if is_low:
                left_high = data.iloc[i - self.lookback_period:i].max()
                right_high = data.iloc[i:i + self.lookback_period].max()
                max_high = max(left_high, right_high)
                
                if max_high - trough_value >= min_move:
                    lows.append(SwingPoint(i, trough_value, point_type='low'))
        
        return {"highs": highs, "lows": lows}
    
    def identify_abc_pattern(self, data: pd.Series) -> Optional[ExtensionSetup]:
        """
        Identify ABC correction pattern for extension calculation
        
        Args:
            data: Price data
            
        Returns:
            ExtensionSetup if valid pattern found
        """
        swing_points = self.find_swing_points(data)
        
        if len(swing_points["highs"]) < 2 or len(swing_points["lows"]) < 2:
            return None
        
        # Get recent swing points
        all_points = []
        all_points.extend([(p, 'high') for p in swing_points["highs"]])
        all_points.extend([(p, 'low') for p in swing_points["lows"]])
        
        # Sort by index (chronological order)
        all_points.sort(key=lambda x: x[0].index)
        
        if len(all_points) < 3:
            return None
        
        # Take the last 3 significant points
        recent_points = all_points[-3:]
        
        point_a = recent_points[0][0]
        point_b = recent_points[1][0]
        point_c = recent_points[2][0]
        
        # Determine trend direction
        if (recent_points[0][1] == 'low' and 
            recent_points[1][1] == 'high' and 
            recent_points[2][1] == 'low'):
            # Bullish ABC: Low -> High -> Low
            if point_c.price > point_a.price:  # Higher low
                trend_direction = TrendDirection.BULLISH
            else:
                return None
        elif (recent_points[0][1] == 'high' and 
              recent_points[1][1] == 'low' and 
              recent_points[2][1] == 'high'):
            # Bearish ABC: High -> Low -> High
            if point_c.price < point_a.price:  # Lower high
                trend_direction = TrendDirection.BEARISH
            else:
                return None
        else:
            return None
        
        setup = ExtensionSetup(point_a, point_b, point_c, trend_direction)
        self.current_setup = setup
        return setup
    
    def calculate_abc_extensions(self, setup: ExtensionSetup) -> List[FibExtensionLevel]:
        """
        Calculate Fibonacci extensions using ABC pattern
        
        Args:
            setup: ExtensionSetup with A, B, C points
            
        Returns:
            List of extension levels
        """
        # Calculate AB move
        ab_move = abs(setup.point_b.price - setup.point_a.price)
        
        extension_levels = []
        
        # External extensions (beyond C)
        for ratio in self.extension_ratios:
            if setup.trend_direction == TrendDirection.BULLISH:
                target_price = setup.point_c.price + (ab_move * ratio)
            else:
                target_price = setup.point_c.price - (ab_move * ratio)
            
            extension_levels.append(FibExtensionLevel(
                ratio=ratio,
                price=target_price,
                label=f"Ext {ratio:.3f}",
                extension_type=ExtensionType.EXTERNAL
            ))
        
        # Internal projections (between B and theoretical 100% extension)
        for ratio in self.internal_ratios:
            if setup.trend_direction == TrendDirection.BULLISH:
                target_price = setup.point_c.price + (ab_move * ratio)
            else:
                target_price = setup.point_c.price - (ab_move * ratio)
            
            extension_levels.append(FibExtensionLevel(
                ratio=ratio,
                price=target_price,
                label=f"Int {ratio:.3f}",
                extension_type=ExtensionType.INTERNAL
            ))
        
        self.extension_levels = sorted(extension_levels, 
                                     key=lambda x: x.price, 
                                     reverse=(setup.trend_direction == TrendDirection.BEARISH))
        return self.extension_levels
    
    def calculate_abcd_extensions(self, setup: ExtensionSetup) -> List[FibExtensionLevel]:
        """
        Calculate AB=CD pattern extensions
        
        Args:
            setup: ExtensionSetup with A, B, C points
            
        Returns:
            List of ABCD extension levels
        """
        ab_move = abs(setup.point_b.price - setup.point_a.price)
        bc_move = abs(setup.point_c.price - setup.point_b.price)
        
        # Calculate potential D points where CD = AB * ratio
        abcd_levels = []
        
        ratios = [0.618, 0.786, 1.0, 1.272, 1.618]  # Common ABCD ratios
        
        for ratio in ratios:
            cd_target = ab_move * ratio
            
            if setup.trend_direction == TrendDirection.BULLISH:
                d_price = setup.point_c.price + cd_target
            else:
                d_price = setup.point_c.price - cd_target
            
            abcd_levels.append(FibExtensionLevel(
                ratio=ratio,
                price=d_price,
                label=f"ABCD {ratio:.3f}",
                extension_type=ExtensionType.EXTERNAL
            ))
        
        return abcd_levels
    
    def find_confluence_levels(self, tolerance: float = 0.005) -> List[Dict]:
        """
        Find confluence zones where multiple Fibonacci levels cluster
        
        Args:
            tolerance: Price tolerance for confluence (as percentage)
            
        Returns:
            List of confluence zones
        """
        if not self.extension_levels:
            return []
        
        confluence_zones = []
        used_levels = set()
        
        for i, level1 in enumerate(self.extension_levels):
            if i in used_levels:
                continue
                
            confluence_group = [level1]
            used_levels.add(i)
            
            for j, level2 in enumerate(self.extension_levels[i+1:], i+1):
                if j in used_levels:
                    continue
                    
                price_diff = abs(level1.price - level2.price) / level1.price
                if price_diff <= tolerance:
                    confluence_group.append(level2)
                    used_levels.add(j)
            
            if len(confluence_group) > 1:
                avg_price = np.mean([level.price for level in confluence_group])
                confluence_zones.append({
                    'price': avg_price,
                    'levels': confluence_group,
                    'strength': len(confluence_group),
                    'range': (min(level.price for level in confluence_group),
                             max(level.price for level in confluence_group))
                })
        
        return sorted(confluence_zones, key=lambda x: x['strength'], reverse=True)
    
    def generate_trading_targets(self, current_price: float,
                               risk_percentage: float = 0.02) -> Dict[str, any]:
        """
        Generate trading targets and risk management levels
        
        Args:
            current_price: Current market price
            risk_percentage: Risk percentage for position sizing
            
        Returns:
            Dictionary with trading targets and risk levels
        """
        if not self.current_setup or not self.extension_levels:
            return {"error": "No valid setup or extension levels calculated"}
        
        # Determine entry direction
        is_bullish = self.current_setup.trend_direction == TrendDirection.BULLISH
        
        # Find relevant targets based on current price
        if is_bullish:
            targets = [level for level in self.extension_levels 
                      if level.price > current_price]
            stop_loss = self.current_setup.point_c.price * (1 - risk_percentage)
        else:
            targets = [level for level in self.extension_levels 
                      if level.price < current_price]
            stop_loss = self.current_setup.point_c.price * (1 + risk_percentage)
        
        if not targets:
            return {"error": "No valid targets found"}
        
        # Sort targets by proximity
        targets.sort(key=lambda x: abs(x.price - current_price))
        
        # Calculate risk/reward ratios
        risk_amount = abs(current_price - stop_loss)
        
        target_analysis = []
        for target in targets[:5]:  # Top 5 targets
            reward_amount = abs(target.price - current_price)
            rr_ratio = reward_amount / risk_amount if risk_amount > 0 else 0
            
            target_analysis.append({
                'level': target,
                'distance_percent': abs(target.price - current_price) / current_price,
                'risk_reward_ratio': rr_ratio
            })
        
        # Find confluence zones
        confluence_zones = self.find_confluence_levels()
        
        return {
            'setup_type': 'ABC_Extension',
            'trend_direction': self.current_setup.trend_direction.value,
            'entry_price': current_price,
            'stop_loss': stop_loss,
            'risk_amount': risk_amount,
            'targets': target_analysis,
            'confluence_zones': confluence_zones,
            'setup_points': {
                'A': {'price': self.current_setup.point_a.price, 'index': self.current_setup.point_a.index},
                'B': {'price': self.current_setup.point_b.price, 'index': self.current_setup.point_b.index},
                'C': {'price': self.current_setup.point_c.price, 'index': self.current_setup.point_c.index}
            }
        }
    
    def generate_trading_signals(self, data: pd.Series, 
                               current_price: float,
                               min_rr_ratio: float = 2.0) -> Dict[str, any]:
        """
        Generate trading signals based on Fibonacci extensions
        
        Args:
            data: Price data
            current_price: Current market price
            min_rr_ratio: Minimum risk/reward ratio for signals
            
        Returns:
            Trading signal analysis
        """
        # Identify pattern
        setup = self.identify_abc_pattern(data)
        if not setup:
            return {
                'signal': 'NO_SETUP',
                'confidence': 0.0,
                'analysis': 'No valid ABC pattern found'
            }
        
        # Calculate extensions
        self.calculate_abc_extensions(setup)
        
        # Generate targets
        targets = self.generate_trading_targets(current_price)
        
        if 'error' in targets:
            return {
                'signal': 'NO_SIGNAL',
                'confidence': 0.0,
                'analysis': targets['error']
            }
        
        # Determine signal strength
        signal_strength = 0.0
        signal_type = 'HOLD'
        analysis_parts = []
        
        # Check if we have good risk/reward targets
        good_targets = [t for t in targets['targets'] if t['risk_reward_ratio'] >= min_rr_ratio]
        
        if good_targets:
            signal_strength += 0.4
            best_rr = max(t['risk_reward_ratio'] for t in good_targets)
            analysis_parts.append(f"Best R/R ratio: {best_rr:.2f}")
        
        # Check for confluence zones
        if targets['confluence_zones']:
            signal_strength += 0.3
            strongest_zone = max(targets['confluence_zones'], key=lambda x: x['strength'])
            analysis_parts.append(f"Confluence zone with {strongest_zone['strength']} levels")
        
        # Check trend strength (based on ABC pattern quality)
        ab_move = abs(setup.point_b.price - setup.point_a.price)
        bc_move = abs(setup.point_c.price - setup.point_b.price)
        retracement_ratio = bc_move / ab_move
        
        if 0.382 <= retracement_ratio <= 0.618:  # Ideal retracement
            signal_strength += 0.3
            analysis_parts.append(f"Ideal retracement: {retracement_ratio:.1%}")
        
        # Determine signal direction
        if signal_strength >= 0.6 and good_targets:
            if setup.trend_direction == TrendDirection.BULLISH:
                signal_type = 'BUY'
            else:
                signal_type = 'SELL'
        
        return {
            'signal': signal_type,
            'confidence': signal_strength,
            'analysis': '; '.join(analysis_parts) if analysis_parts else 'Weak setup',
            'setup': setup,
            'targets': targets,
            'recommended_targets': good_targets[:3]  # Top 3 targets with good R/R
        }
    
    def calculate_position_size(self, account_balance: float,
                              risk_percentage: float,
                              entry_price: float,
                              stop_loss: float) -> Dict[str, float]:
        """
        Calculate position size based on risk management
        
        Args:
            account_balance: Account balance
            risk_percentage: Risk percentage (e.g., 0.02 for 2%)
            entry_price: Entry price
            stop_loss: Stop loss price
            
        Returns:
            Position sizing information
        """
        risk_amount = account_balance * risk_percentage
        price_risk = abs(entry_price - stop_loss)
        
        if price_risk == 0:
            return {"error": "Invalid price risk"}
        
        position_size = risk_amount / price_risk
        position_value = position_size * entry_price
        
        return {
            'position_size': position_size,
            'position_value': position_value,
            'risk_amount': risk_amount,
            'risk_per_unit': price_risk,
            'risk_percentage_actual': (risk_amount / account_balance) * 100
        }

# Example usage and testing
def example_usage():
    """Example of how to use the Fibonacci Extension tool"""
    
    # Create sample price data with ABC pattern
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=150, freq='H')
    
    # Simulate ABC correction pattern
    prices = []
    base_price = 100
    
    # A to B (upward move)
    for i in range(50):
        prices.append(base_price + i * 0.3 + np.random.randn() * 0.5)
    
    # B to C (retracement)
    peak_price = prices[-1]
    for i in range(30):
        retracement = peak_price - i * 0.2 + np.random.randn() * 0.3
        prices.append(retracement)
    
    # C onwards (potential extension)
    c_price = prices[-1]
    for i in range(70):
        prices.append(c_price + i * 0.1 + np.random.randn() * 0.4)
    
    price_data = pd.Series(prices, index=dates)
    
    # Initialize Fibonacci Extension tool
    fib_ext = FibonacciExtension(lookback_period=8)
    
    # Identify ABC pattern
    setup = fib_ext.identify_abc_pattern(price_data)
    
    if setup:
        print(f"ABC Pattern Identified:")
        print(f"Trend Direction: {setup.trend_direction.value}")
        print(f"Point A: ${setup.point_a.price:.2f} at index {setup.point_a.index}")
        print(f"Point B: ${setup.point_b.price:.2f} at index {setup.point_b.index}")
        print(f"Point C: ${setup.point_c.price:.2f} at index {setup.point_c.index}")
        
        # Calculate extensions
        extensions = fib_ext.calculate_abc_extensions(setup)
        
        print(f"\nFibonacci Extension Levels:")
        for ext in extensions[:8]:  # Show first 8 levels
            print(f"{ext.label}: ${ext.price:.2f} ({ext.extension_type.value})")
        
        # Generate trading analysis
        current_price = price_data.iloc[-1]
        signals = fib_ext.generate_trading_signals(price_data, current_price)
        
        print(f"\nTrading Analysis:")
        print(f"Current Price: ${current_price:.2f}")
        print(f"Signal: {signals['signal']}")
        print(f"Confidence: {signals['confidence']:.1%}")
        print(f"Analysis: {signals['analysis']}")
        
        if 'recommended_targets' in signals and signals['recommended_targets']:
            print(f"\nRecommended Targets:")
            for i, target in enumerate(signals['recommended_targets'], 1):
                print(f"Target {i}: ${target['level'].price:.2f} "
                      f"(R/R: {target['risk_reward_ratio']:.2f})")
        
        # Position sizing example
        account_balance = 10000
        if signals['signal'] in ['BUY', 'SELL']:
            targets_info = signals['targets']
            if 'stop_loss' in targets_info:
                position_info = fib_ext.calculate_position_size(
                    account_balance, 0.02, current_price, targets_info['stop_loss']
                )
                print(f"\nPosition Sizing (2% risk):")
                print(f"Position Size: {position_info['position_size']:.2f} units")
                print(f"Position Value: ${position_info['position_value']:.2f}")
                print(f"Risk Amount: ${position_info['risk_amount']:.2f}")
    
    else:
        print("No valid ABC pattern found in the data")

if __name__ == "__main__":
    example_usage()