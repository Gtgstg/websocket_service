"""
Fibonacci Levels Indicator Module
Provides classes and functions for calculating Fibonacci retracement and extension levels.
"""
import pandas as pd
import numpy as np

class FibonacciCalculator:
    def __init__(self):
        """
        Initialize Fibonacci calculator with standard ratios
        
        Standard Fibonacci ratios:
        Retracements: 0.236, 0.382, 0.5, 0.618, 0.786
        Extensions: 1.272, 1.618, 2.618, 3.618, 4.236
        """
        # Common Fibonacci retracement levels
        self.retracement_levels = {
            '23.6%': 0.236,
            '38.2%': 0.382,
            '50.0%': 0.500,  # Not Fibonacci but widely used
            '61.8%': 0.618,
            '78.6%': 0.786
        }
        
        # Common Fibonacci extension levels
        self.extension_levels = {
            '127.2%': 1.272,
            '161.8%': 1.618,
            '261.8%': 2.618,
            '361.8%': 3.618,
            '423.6%': 4.236
        }
    
    def calculate_retracements(self, high, low, trend='uptrend'):
        """
        Calculate Fibonacci retracement levels
        
        Parameters:
        high (float): Highest price point
        low (float): Lowest price point
        trend (str): 'uptrend' or 'downtrend'
        
        Returns:
        dict: Fibonacci retracement levels
        """
        levels = {}
        price_range = high - low
        
        for label, ratio in self.retracement_levels.items():
            if trend == 'uptrend':
                # In uptrend, retrace from high to low
                levels[label] = high - (price_range * ratio)
            else:
                # In downtrend, retrace from low to high
                levels[label] = low + (price_range * ratio)
        
        # Add the high and low as reference points
        levels['0%'] = high if trend == 'uptrend' else low
        levels['100%'] = low if trend == 'uptrend' else high
        
        return dict(sorted(levels.items(), key=lambda x: float(x[1]), reverse=(trend=='uptrend')))
    
    def calculate_extensions(self, high, low, trend='uptrend'):
        """
        Calculate Fibonacci extension levels
        
        Parameters:
        high (float): Highest price point
        low (float): Lowest price point
        trend (str): 'uptrend' or 'downtrend'
        
        Returns:
        dict: Fibonacci extension levels
        """
        levels = {}
        price_range = high - low
        
        for label, ratio in self.extension_levels.items():
            if trend == 'uptrend':
                # Project up from the high in an uptrend
                levels[label] = high + (price_range * ratio)
            else:
                # Project down from the low in a downtrend
                levels[label] = low - (price_range * ratio)
        
        # Add the high/low as reference points
        levels['0%'] = high if trend == 'uptrend' else low
        levels['100%'] = high + price_range if trend == 'uptrend' else low - price_range
        
        return dict(sorted(levels.items(), key=lambda x: float(x[1]), reverse=(trend=='uptrend')))
    
    def calculate_all_levels(self, swing_high, swing_low, trend='uptrend'):
        """
        Calculate both retracement and extension levels
        
        Parameters:
        swing_high (float): Swing high price
        swing_low (float): Swing low price
        trend (str): 'uptrend' or 'downtrend'
        
        Returns:
        dict: All Fibonacci levels
        """
        return {
            'retracements': self.calculate_retracements(swing_high, swing_low, trend),
            'extensions': self.calculate_extensions(swing_high, swing_low, trend)
        }
    
    def find_nearest_level(self, price, levels):
        """
        Find the nearest Fibonacci level to current price
        
        Parameters:
        price (float): Current price
        levels (dict): Dictionary of Fibonacci levels
        
        Returns:
        tuple: (nearest_level_label, nearest_level_price, distance)
        """
        nearest_level = min(levels.items(), key=lambda x: abs(float(x[1]) - price))
        distance = abs(float(nearest_level[1]) - price)
        return nearest_level[0], nearest_level[1], distance
    
    def validate_level_breakout(self, price, prev_price, level, tolerance=0.001):
        """
        Validate if price has broken through a Fibonacci level
        
        Parameters:
        price (float): Current price
        prev_price (float): Previous price
        level (float): Fibonacci level to check
        tolerance (float): Percentage tolerance for breakout confirmation
        
        Returns:
        dict: Breakout information
        """
        level_value = float(level)
        tolerance_range = level_value * tolerance
        
        # Check if price crossed the level
        crossed_up = prev_price < level_value <= price
        crossed_down = prev_price > level_value >= price
        
        return {
            'breakout': crossed_up or crossed_down,
            'direction': 'up' if crossed_up else 'down' if crossed_down else None,
            'level_price': level_value,
            'distance': abs(price - level_value)
        }

    def detect_swings(self, df, window=20):
        """
        Detect swing highs and lows in price data
        
        Parameters:
        df: DataFrame with OHLC data
        window (int): Lookback window for swing detection
        
        Returns:
        dict: Detected swing points
        """
        swings = {
            'highs': [],
            'lows': []
        }
        
        for i in range(window, len(df) - window):
            # Check for swing high
            if df['high'].iloc[i] == df['high'].iloc[i-window:i+window+1].max():
                swings['highs'].append({
                    'price': df['high'].iloc[i],
                    'index': df.index[i]
                })
            
            # Check for swing low
            if df['low'].iloc[i] == df['low'].iloc[i-window:i+window+1].min():
                swings['lows'].append({
                    'price': df['low'].iloc[i],
                    'index': df.index[i]
                })
        
        return swings
