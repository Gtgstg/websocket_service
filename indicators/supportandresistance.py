import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
from scipy.signal import argrelextrema
import warnings
warnings.filterwarnings('ignore')

class SupportResistanceIndicator:
    """
    Advanced Support & Resistance Indicator for Algorithmic Trading
    
    Combines multiple methods:
    1. Pivot Points (Classical, Fibonacci, Camarilla)
    2. Swing Highs/Lows
    3. Volume Profile levels
    4. Psychological levels
    5. Dynamic S/R zones
    6. Breakout detection
    """
    
    def __init__(self, 
                 swing_window: int = 20,
                 min_touches: int = 2,
                 proximity_threshold: float = 0.002,  # 0.2% for level clustering
                 volume_profile_bins: int = 50,
                 zone_width: float = 0.001):  # 0.1% for zone width
        
        self.swing_window = swing_window
        self.min_touches = min_touches
        self.proximity_threshold = proximity_threshold
        self.volume_profile_bins = volume_profile_bins
        self.zone_width = zone_width
    
    def calculate_pivot_points(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """Calculate various pivot point levels"""
        high, low, close = data['high'], data['low'], data['close']
        
        # Previous day's HLC for pivot calculation
        prev_high = high.shift(1)
        prev_low = low.shift(1)
        prev_close = close.shift(1)
        
        # Classical Pivot Points
        pivot = (prev_high + prev_low + prev_close) / 3
        
        # Support levels
        s1 = (2 * pivot) - prev_high
        s2 = pivot - (prev_high - prev_low)
        s3 = prev_low - 2 * (prev_high - pivot)
        
        # Resistance levels
        r1 = (2 * pivot) - prev_low
        r2 = pivot + (prev_high - prev_low)
        r3 = prev_high + 2 * (pivot - prev_low)
        
        # Fibonacci Pivot Points
        fib_pivot = pivot
        fib_s1 = pivot - 0.382 * (prev_high - prev_low)
        fib_s2 = pivot - 0.618 * (prev_high - prev_low)
        fib_s3 = pivot - 1.000 * (prev_high - prev_low)
        fib_r1 = pivot + 0.382 * (prev_high - prev_low)
        fib_r2 = pivot + 0.618 * (prev_high - prev_low)
        fib_r3 = pivot + 1.000 * (prev_high - prev_low)
        
        # Camarilla Pivot Points
        cam_s1 = prev_close - 1.1 * (prev_high - prev_low) / 12
        cam_s2 = prev_close - 1.1 * (prev_high - prev_low) / 6
        cam_s3 = prev_close - 1.1 * (prev_high - prev_low) / 4
        cam_s4 = prev_close - 1.1 * (prev_high - prev_low) / 2
        cam_r1 = prev_close + 1.1 * (prev_high - prev_low) / 12
        cam_r2 = prev_close + 1.1 * (prev_high - prev_low) / 6
        cam_r3 = prev_close + 1.1 * (prev_high - prev_low) / 4
        cam_r4 = prev_close + 1.1 * (prev_high - prev_low) / 2
        
        return {
            # Classical
            'pivot': pivot, 's1': s1, 's2': s2, 's3': s3, 'r1': r1, 'r2': r2, 'r3': r3,
            # Fibonacci
            'fib_pivot': fib_pivot, 'fib_s1': fib_s1, 'fib_s2': fib_s2, 'fib_s3': fib_s3,
            'fib_r1': fib_r1, 'fib_r2': fib_r2, 'fib_r3': fib_r3,
            # Camarilla
            'cam_s1': cam_s1, 'cam_s2': cam_s2, 'cam_s3': cam_s3, 'cam_s4': cam_s4,
            'cam_r1': cam_r1, 'cam_r2': cam_r2, 'cam_r3': cam_r3, 'cam_r4': cam_r4
        }
    
    def find_swing_points(self, data: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        """Find swing highs and lows using local extrema"""
        high, low = data['high'].values, data['low'].values
        
        # Find local maxima and minima
        high_peaks = argrelextrema(high, np.greater, order=self.swing_window//2)[0]
        low_peaks = argrelextrema(low, np.less, order=self.swing_window//2)[0]
        
        # Create series with swing points
        swing_highs = pd.Series(index=data.index, dtype=float)
        swing_lows = pd.Series(index=data.index, dtype=float)
        
        for peak in high_peaks:
            if peak < len(data):
                swing_highs.iloc[peak] = high[peak]
        
        for peak in low_peaks:
            if peak < len(data):
                swing_lows.iloc[peak] = low[peak]
        
        return swing_highs, swing_lows
    
    def calculate_volume_profile(self, data: pd.DataFrame) -> Dict[str, float]:
        """Calculate Volume Profile to find high volume levels"""
        if 'volume' not in data.columns:
            return {'poc': np.nan, 'vah': np.nan, 'val': np.nan}
        
        price_range = data['high'].max() - data['low'].min()
        bin_size = price_range / self.volume_profile_bins
        
        # Create price bins
        min_price = data['low'].min()
        bins = [min_price + i * bin_size for i in range(self.volume_profile_bins + 1)]
        
        # Calculate volume for each price level
        volume_at_price = {}
        
        for i in range(len(data)):
            row = data.iloc[i]
            # Distribute volume across the price range for this candle
            price_levels = np.linspace(row['low'], row['high'], 10)
            volume_per_level = row['volume'] / len(price_levels)
            
            for price in price_levels:
                bin_idx = min(int((price - min_price) / bin_size), self.volume_profile_bins - 1)
                bin_price = bins[bin_idx]
                if bin_price not in volume_at_price:
                    volume_at_price[bin_price] = 0
                volume_at_price[bin_price] += volume_per_level
        
        if not volume_at_price:
            return {'poc': np.nan, 'vah': np.nan, 'val': np.nan}
        
        # Find Point of Control (highest volume)
        poc = max(volume_at_price.keys(), key=lambda x: volume_at_price[x])
        
        # Find Value Area (70% of volume)
        total_volume = sum(volume_at_price.values())
        target_volume = total_volume * 0.7
        
        sorted_levels = sorted(volume_at_price.items(), key=lambda x: x[1], reverse=True)
        cumulative_volume = 0
        value_area_levels = []
        
        for price, volume in sorted_levels:
            cumulative_volume += volume
            value_area_levels.append(price)
            if cumulative_volume >= target_volume:
                break
        
        vah = max(value_area_levels) if value_area_levels else poc
        val = min(value_area_levels) if value_area_levels else poc
        
        return {'poc': poc, 'vah': vah, 'val': val}
    
    def find_psychological_levels(self, data: pd.DataFrame) -> List[float]:
        """Find psychological round number levels"""
        price_range = data['close'].max() - data['close'].min()
        current_price = data['close'].iloc[-1]
        
        # Determine appropriate round number intervals
        if current_price < 1:
            intervals = [0.01, 0.05, 0.1]
        elif current_price < 10:
            intervals = [0.1, 0.5, 1.0]
        elif current_price < 100:
            intervals = [1, 5, 10]
        elif current_price < 1000:
            intervals = [10, 25, 50, 100]
        else:
            intervals = [100, 250, 500, 1000]
        
        psychological_levels = []
        
        for interval in intervals:
            # Find levels within reasonable range
            min_level = int((current_price - price_range) / interval) * interval
            max_level = int((current_price + price_range) / interval + 1) * interval
            
            level = min_level
            while level <= max_level:
                if abs(level - current_price) <= price_range:
                    psychological_levels.append(level)
                level += interval
        
        return sorted(list(set(psychological_levels)))
    
    def cluster_levels(self, levels: List[float]) -> List[Dict]:
        """Cluster nearby levels and calculate their strength"""
        if not levels:
            return []
        
        levels = sorted([l for l in levels if not np.isnan(l)])
        clusters = []
        current_cluster = [levels[0]]
        
        for i in range(1, len(levels)):
            # Check if level is close to current cluster
            cluster_center = np.mean(current_cluster)
            if abs(levels[i] - cluster_center) / cluster_center <= self.proximity_threshold:
                current_cluster.append(levels[i])
            else:
                # Finalize current cluster
                if len(current_cluster) >= self.min_touches:
                    clusters.append({
                        'level': np.mean(current_cluster),
                        'strength': len(current_cluster),
                        'touches': len(current_cluster)
                    })
                current_cluster = [levels[i]]
        
        # Don't forget the last cluster
        if len(current_cluster) >= self.min_touches:
            clusters.append({
                'level': np.mean(current_cluster),
                'strength': len(current_cluster),
                'touches': len(current_cluster)
            })
        
        return sorted(clusters, key=lambda x: x['strength'], reverse=True)
    
    def calculate_dynamic_sr(self, data: pd.DataFrame, window: int = 50) -> Dict[str, pd.Series]:
        """Calculate dynamic support and resistance using rolling windows"""
        
        # Rolling support and resistance
        rolling_support = data['low'].rolling(window=window).min()
        rolling_resistance = data['high'].rolling(window=window).max()
        
        # Exponential moving average based levels
        ema_20 = data['close'].ewm(span=20).mean()
        ema_50 = data['close'].ewm(span=50).mean()
        
        # Bollinger Bands as dynamic S/R
        bb_middle = data['close'].rolling(window=20).mean()
        bb_std = data['close'].rolling(window=20).std()
        bb_upper = bb_middle + (bb_std * 2)
        bb_lower = bb_middle - (bb_std * 2)
        
        return {
            'dynamic_support': rolling_support,
            'dynamic_resistance': rolling_resistance,
            'ema_20': ema_20,
            'ema_50': ema_50,
            'bb_upper': bb_upper,
            'bb_lower': bb_lower,
            'bb_middle': bb_middle
        }
    
    def detect_breakouts(self, data: pd.DataFrame, sr_levels: List[Dict]) -> Dict[str, pd.Series]:
        """Detect breakouts and breakdowns from S/R levels"""
        
        close = data['close']
        volume = data.get('volume', pd.Series(index=data.index, data=1))
        
        breakout_signals = pd.Series(index=data.index, data=0)  # 1 = breakout, -1 = breakdown
        breakout_strength = pd.Series(index=data.index, data=0.0)
        
        for i in range(1, len(data)):
            current_price = close.iloc[i]
            prev_price = close.iloc[i-1]
            current_volume = volume.iloc[i]
            avg_volume = volume.rolling(window=20).mean().iloc[i]
            
            for level_info in sr_levels:
                level = level_info['level']
                strength = level_info['strength']
                
                # Check for breakout (price moves above resistance)
                if prev_price <= level and current_price > level:
                    volume_confirmation = current_volume > avg_volume * 1.2  # 20% above average
                    if volume_confirmation:
                        breakout_signals.iloc[i] = 1
                        breakout_strength.iloc[i] = strength
                        break
                
                # Check for breakdown (price moves below support)
                elif prev_price >= level and current_price < level:
                    volume_confirmation = current_volume > avg_volume * 1.2
                    if volume_confirmation:
                        breakout_signals.iloc[i] = -1
                        breakout_strength.iloc[i] = strength
                        break
        
        return {
            'breakout_signals': breakout_signals,
            'breakout_strength': breakout_strength
        }
    
    def calculate_sr_zones(self, levels: List[Dict]) -> List[Dict]:
        """Convert point levels to zones for more realistic trading"""
        zones = []
        
        for level_info in levels:
            level = level_info['level']
            strength = level_info['strength']
            
            zone_half_width = level * self.zone_width
            
            zones.append({
                'upper': level + zone_half_width,
                'lower': level - zone_half_width,
                'center': level,
                'strength': strength,
                'type': 'support' if level < level else 'resistance'  # Will be determined by context
            })
        
        return zones
    
    def get_current_sr_context(self, data: pd.DataFrame, sr_levels: List[Dict]) -> Dict:
        """Determine current price context relative to S/R levels"""
        current_price = data['close'].iloc[-1]
        
        # Find nearest support and resistance
        supports = [l for l in sr_levels if l['level'] < current_price]
        resistances = [l for l in sr_levels if l['level'] > current_price]
        
        nearest_support = max(supports, key=lambda x: x['level']) if supports else None
        nearest_resistance = min(resistances, key=lambda x: x['level']) if resistances else None
        
        # Calculate distances
        support_distance = (current_price - nearest_support['level']) / current_price if nearest_support else float('inf')
        resistance_distance = (nearest_resistance['level'] - current_price) / current_price if nearest_resistance else float('inf')
        
        return {
            'current_price': current_price,
            'nearest_support': nearest_support,
            'nearest_resistance': nearest_resistance,
            'support_distance_pct': support_distance * 100,
            'resistance_distance_pct': resistance_distance * 100,
            'in_no_trade_zone': min(support_distance, resistance_distance) < 0.005  # Within 0.5%
        }
    
    def generate_trading_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate comprehensive S/R based trading signals"""
        
        # Calculate all components
        pivot_points = self.calculate_pivot_points(data)
        swing_highs, swing_lows = self.find_swing_points(data)
        volume_profile = self.calculate_volume_profile(data)
        psychological_levels = self.find_psychological_levels(data)
        dynamic_sr = self.calculate_dynamic_sr(data)
        
        # Collect all levels
        all_levels = []
        
        # Add pivot points
        for key, series in pivot_points.items():
            if not series.isna().all():
                all_levels.extend(series.dropna().tolist())
        
        # Add swing points
        all_levels.extend(swing_highs.dropna().tolist())
        all_levels.extend(swing_lows.dropna().tolist())
        
        # Add volume profile levels
        for key, value in volume_profile.items():
            if not np.isnan(value):
                all_levels.append(value)
        
        # Add psychological levels
        all_levels.extend(psychological_levels)
        
        # Cluster levels
        sr_levels = self.cluster_levels(all_levels)
        
        # Detect breakouts
        breakout_data = self.detect_breakouts(data, sr_levels)
        
        # Create comprehensive signals DataFrame
        signals_df = pd.DataFrame(index=data.index)
        
        # Add price data
        signals_df['open'] = data['open']
        signals_df['high'] = data['high']
        signals_df['low'] = data['low']
        signals_df['close'] = data['close']
        if 'volume' in data.columns:
            signals_df['volume'] = data['volume']
        
        # Add pivot points
        for key, series in pivot_points.items():
            signals_df[key] = series
        
        # Add dynamic S/R
        for key, series in dynamic_sr.items():
            signals_df[key] = series
        
        # Add breakout signals
        signals_df['breakout_signal'] = breakout_data['breakout_signals']
        signals_df['breakout_strength'] = breakout_data['breakout_strength']
        
        # Add current context for last row
        if sr_levels:
            context = self.get_current_sr_context(data, sr_levels)
            signals_df.loc[signals_df.index[-1], 'nearest_support'] = context['nearest_support']['level'] if context['nearest_support'] else np.nan
            signals_df.loc[signals_df.index[-1], 'nearest_resistance'] = context['nearest_resistance']['level'] if context['nearest_resistance'] else np.nan
            signals_df.loc[signals_df.index[-1], 'support_distance_pct'] = context['support_distance_pct']
            signals_df.loc[signals_df.index[-1], 'resistance_distance_pct'] = context['resistance_distance_pct']
            signals_df.loc[signals_df.index[-1], 'in_no_trade_zone'] = context['in_no_trade_zone']
        
        # Generate trading signals
        signals_df['buy_signal'] = (
            (signals_df['breakout_signal'] == 1) & 
            (signals_df['breakout_strength'] >= 2)
        )
        
        signals_df['sell_signal'] = (
            (signals_df['breakout_signal'] == -1) & 
            (signals_df['breakout_strength'] >= 2)
        )
        
        # Support/Resistance bounce signals
        signals_df['support_bounce'] = (
            (signals_df['low'] <= signals_df['nearest_support'] * 1.002) & 
            (signals_df['close'] > signals_df['nearest_support'])
        ).fillna(False)
        
        signals_df['resistance_rejection'] = (
            (signals_df['high'] >= signals_df['nearest_resistance'] * 0.998) & 
            (signals_df['close'] < signals_df['nearest_resistance'])
        ).fillna(False)
        
        return signals_df, sr_levels


# Trading Bot Implementation
class SRTradingBot:
    """Support/Resistance based trading bot"""
    
    def __init__(self, initial_capital: float = 10000):
        self.sr_indicator = SupportResistanceIndicator()
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.position = 0
        self.entry_price = 0
        self.trades = []
        
    def calculate_position_size(self, price: float, stop_loss: float, risk_per_trade: float = 0.02) -> float:
        """Calculate position size based on risk management"""
        risk_amount = self.capital * risk_per_trade
        price_risk = abs(price - stop_loss)
        if price_risk == 0:
            return 0
        position_size = risk_amount / price_risk
        max_position = self.capital * 0.1 / price  # Max 10% of capital in one trade
        return min(position_size, max_position)
    
    def backtest(self, data: pd.DataFrame) -> Dict:
        """Backtest the S/R strategy"""
        
        signals_df, sr_levels = self.sr_indicator.generate_trading_signals(data)
        
        for i in range(50, len(signals_df)):  # Start after warm-up period
            row = signals_df.iloc[i]
            
            if self.position == 0:  # No position
                # Long entry on breakout or support bounce
                if row['buy_signal'] or row['support_bounce']:
                    stop_loss = row['nearest_support'] if not pd.isna(row['nearest_support']) else row['close'] * 0.98
                    take_profit = row['nearest_resistance'] if not pd.isna(row['nearest_resistance']) else row['close'] * 1.04
                    
                    self.enter_long(row['close'], stop_loss, take_profit, i)
                
                # Short entry on breakdown or resistance rejection
                elif row['sell_signal'] or row['resistance_rejection']:
                    stop_loss = row['nearest_resistance'] if not pd.isna(row['nearest_resistance']) else row['close'] * 1.02
                    take_profit = row['nearest_support'] if not pd.isna(row['nearest_support']) else row['close'] * 0.96
                    
                    self.enter_short(row['close'], stop_loss, take_profit, i)
            
            else:  # Have position - check exits
                if self.position > 0:  # Long position
                    if (row['close'] <= self.stop_loss or 
                        row['close'] >= self.take_profit or
                        row['sell_signal']):
                        self.exit_position(row['close'], i)
                
                elif self.position < 0:  # Short position
                    if (row['close'] >= self.stop_loss or 
                        row['close'] <= self.take_profit or
                        row['buy_signal']):
                        self.exit_position(row['close'], i)
        
        return self.get_performance_stats()
    
    def enter_long(self, price: float, stop_loss: float, take_profit: float, index: int):
        position_size = self.calculate_position_size(price, stop_loss)
        if position_size > 0:
            self.position = position_size
            self.entry_price = price
            self.stop_loss = stop_loss
            self.take_profit = take_profit
    
    def enter_short(self, price: float, stop_loss: float, take_profit: float, index: int):
        position_size = self.calculate_position_size(price, stop_loss)
        if position_size > 0:
            self.position = -position_size
            self.entry_price = price
            self.stop_loss = stop_loss
            self.take_profit = take_profit
    
    def exit_position(self, price: float, index: int):
        if self.position != 0:
            pnl = (price - self.entry_price) * self.position
            pnl_pct = (pnl / (self.entry_price * abs(self.position))) * 100
            
            self.trades.append({
                'entry_price': self.entry_price,
                'exit_price': price,
                'position_size': self.position,
                'pnl': pnl,
                'pnl_pct': pnl_pct,
                'position_type': 'long' if self.position > 0 else 'short'
            })
            
            self.capital += pnl
            self.position = 0
    
    def get_performance_stats(self) -> Dict:
        if not self.trades:
            return {'total_return': 0, 'win_rate': 0, 'total_trades': 0, 'sharpe_ratio': 0}
        
        df_trades = pd.DataFrame(self.trades)
        total_return = ((self.capital - self.initial_capital) / self.initial_capital) * 100
        win_rate = (df_trades['pnl_pct'] > 0).mean() * 100
        
        avg_return = df_trades['pnl_pct'].mean()
        return_std = df_trades['pnl_pct'].std()
        sharpe_ratio = avg_return / return_std if return_std != 0 else 0
        
        return {
            'total_return': total_return,
            'win_rate': win_rate,
            'total_trades': len(self.trades),
            'avg_return_per_trade': avg_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': self.calculate_max_drawdown(),
            'final_capital': self.capital
        }
    
    def calculate_max_drawdown(self) -> float:
        if not self.trades:
            return 0
        
        cumulative_returns = []
        running_capital = self.initial_capital
        
        for trade in self.trades:
            running_capital += trade['pnl']
            cumulative_returns.append(running_capital)
        
        peak = self.initial_capital
        max_dd = 0
        
        for capital in cumulative_returns:
            if capital > peak:
                peak = capital
            drawdown = (peak - capital) / peak * 100
            max_dd = max(max_dd, drawdown)
        
        return max_dd


# Example usage
if __name__ == "__main__":
    """
    # Sample usage with your data
    data = pd.DataFrame({
        'open': [...],
        'high': [...],
        'low': [...],
        'close': [...],
        'volume': [...]  # Optional but recommended
    })
    
    # Initialize and run
    bot = SRTradingBot(initial_capital=10000)
    results = bot.backtest(data)
    
    print(f"Total Return: {results['total_return']:.2f}%")
    print(f"Win Rate: {results['win_rate']:.2f}%")
    print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
    print(f"Max Drawdown: {results['max_drawdown']:.2f}%")
    
    # Get current S/R levels
    indicator = SupportResistanceIndicator()
    signals_df, sr_levels = indicator.generate_trading_signals(data)
    
    print("\\nCurrent S/R Levels:")
    for level in sr_levels[:10]:  # Top 10 levels
        print(f"Level: {level['level']:.2f}, Strength: {level['strength']}")
    """
    pass