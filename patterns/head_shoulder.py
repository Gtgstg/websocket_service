import numpy as np
import pandas as pd

def find_local_extrema(prices, order=3):
    """Identify local maxima and minima in the price series using a rolling window."""
    from scipy.signal import argrelextrema

    local_max = argrelextrema(prices.values, np.greater, order=order)[0]
    local_min = argrelextrema(prices.values, np.less, order=order)[0]
    return local_max, local_min

def detect_head_and_shoulders(prices, order=3):
    local_max, local_min = find_local_extrema(prices, order=order)
    pivots = sorted(list(local_max) + list(local_min))
    patterns = []

    for i in range(4, len(pivots)):
        a, b, c, d, e = pivots[i-4:i+1]
        pts = prices[[a, b, c, d, e]].values
        
        # Head and Shoulders conditions (bearish)
        # Left Shoulder < Head > Right Shoulder; Head is highest
        if pts[0] < pts[2] and pts[4] < pts[2] and pts[2] > pts[1] and pts[2] > pts[3]:
            # Shoulders roughly the same height, symmetry, etc.
            if abs(pts[0] - pts[4]) < 0.03*pts[2]:
                patterns.append((a, b, c, d, e))
        
        # Inverse Head and Shoulders (bullish)
        # Left Shoulder > Head < Right Shoulder; Head is lowest
        if pts[0] > pts[2] and pts[4] > pts[2] and pts[2] < pts[1] and pts[2] < pts[3]:
            if abs(pts[0] - pts[4]) < 0.03*pts[2]:
                patterns.append((a, b, c, d, e))
    return patterns

# Example usage with OHLCV price data (Closing Price)
df = pd.read_csv('prices.csv', index_col=0)
patterns = detect_head_and_shoulders(df['Close'])

# Trading signal generation
for pattern in patterns:
    l_shld, l_arm, head, r_arm, r_shld = pattern
    # Confirm breakdown below "neckline" for entry
    neckline = (df['Close'][l_arm] + df['Close'][r_arm]) / 2
    # If Close crosses below neckline after r_shld, short entry signal
    if df['Close'][r_shld+1] < neckline:
        print(f'Bearish H&S detected at {df.index[r_shld]}, signal SHORT')
