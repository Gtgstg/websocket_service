import numpy as np
import pandas as pd
from scipy.signal import argrelextrema

def find_local_extrema(prices, order=5):
    """Identify local maxima and minima for pattern detection."""
    local_max = argrelextrema(prices.values, np.greater_equal, order=order)[0]
    local_min = argrelextrema(prices.values, np.less_equal, order=order)[0]
    return local_max, local_min

def detect_cup_and_handle(prices, order=5, depth_tol=0.25, handle_max_len=0.3):
    """
    Detects Cup and Handle pattern in price series.
    - order: pivot detection window
    - depth_tol: max allowed cup depth as fraction of peak price (e.g. 0.25 = 25%)
    - handle_max_len: max length of handle as fraction of cup length (e.g. 0.3 = 30%)
    Returns list of tuples: (cup_start, cup_bottom, cup_end, handle_end)
    """
    local_max, local_min = find_local_extrema(prices, order=order)
    patterns = []
    for i in range(len(local_max)-1):
        left_peak = local_max[i]
        for j in range(i+1, len(local_max)):
            right_peak = local_max[j]
            if right_peak <= left_peak:
                continue
            # Find local minima between peaks
            mids = [m for m in local_min if left_peak < m < right_peak]
            if not mids:
                continue
            cup_bottom = mids[0]
            # Cup depth (avoid deep or shallow)
            cup_depth = (prices[left_peak] - prices[cup_bottom]) / prices[left_peak]
            if cup_depth > depth_tol or cup_depth < 0.05:
                continue
            # Rounded bottom: lowest price at cup_bottom
            window = prices[max(left_peak, cup_bottom-3):min(right_peak+1, cup_bottom+4)]
            if prices[cup_bottom] != window.min():
                continue
            # Handle: short, shallow pullback after right_peak
            handle_start = right_peak + 1
            handle_end = handle_start
            max_handle_len = int((right_peak - left_peak) * handle_max_len)
            handle_depth_max = cup_depth / 3
            while handle_end < len(prices)-1 and handle_end - handle_start < max_handle_len:
                if prices[handle_end] < prices[right_peak] * (1 - handle_depth_max):
                    break
                handle_end += 1
            if handle_end == handle_start:
                continue
            handle_prices = prices[handle_start:handle_end]
            # Avoid sharp handle drop
            if len(handle_prices) == 0 or handle_prices.min() < prices[right_peak] * (1 - handle_depth_max):
                continue
            patterns.append((left_peak, cup_bottom, right_peak, handle_end))
    return patterns

# Usage Example:
# df = pd.read_csv('prices.csv')
# prices = df['Close']
# patterns = detect_cup_and_handle(prices)
# print('Cup and Handle Patterns:', patterns)
# # To trigger trades, check for breakout above right_peak after handle_end

