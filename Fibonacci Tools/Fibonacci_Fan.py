import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def read_data(file_path):
    """Read OHLC data from CSV into DataFrame."""
    df = pd.read_csv(file_path, parse_dates=['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    return df

def find_swing_lows_highs(df, window=5):
    """Find indices of local swing lows and highs using window."""
    from scipy.signal import argrelextrema

    lows = argrelextrema(df['Low'].values, np.less_equal, order=window)[0]
    highs = argrelextrema(df['High'].values, np.greater_equal, order=window)[0]
    return lows, highs

def fibonacci_fan(df, swing_low_idx, swing_high_idx):
    x0, y0 = swing_low_idx, df['Low'].iloc[swing_low_idx]
    x1, y1 = swing_high_idx, df['High'].iloc[swing_high_idx]
    dx, dy = x1 - x0, y1 - y0

    fib_levels = [0.382, 0.5, 0.618]
    fan_lines = []

    for level in fib_levels:
        target_y = y0 + level * dy
        slope = (target_y - y0) / dx
        fan_lines.append({'level': level, 'slope': slope, 'origin_idx': x0, 'origin_price': y0})

    # Plotting
    plt.figure(figsize=(14, 7))
    plt.plot(df['Date'], df['Close'], label='Close Price', color='black')
    for fan in fan_lines:
        x_vals = np.arange(x0, len(df))
        y_vals = fan['slope'] * (x_vals - x0) + y0
        plt.plot(df['Date'].iloc[x_vals], y_vals, label=f'Fan {fan["level"]*100:.1f}%')
    plt.scatter([df['Date'].iloc[x0], df['Date'].iloc[x1]], [y0, y1], color='red', label='Swing Low/High', zorder=5)
    plt.title('Fibonacci Fan')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.show()
    return fan_lines

def check_price_touch(df, fan_lines):
    signals = []
    for idx in range(fan_lines[0]['origin_idx'], len(df)):
        price = df['Close'].iloc[idx]
        for fan in fan_lines:
            fan_price = fan['slope'] * (idx - fan['origin_idx']) + fan['origin_price']
            if np.isclose(price, fan_price, rtol=0.005):  # 0.5% tolerance
                signals.append({'date': df['Date'].iloc[idx], 'level': fan['level'], 'price': price, 'type': 'touch'})
    return signals

# === Usage Example ===
# df = read_data('your_historical_data.csv')
# swing_lows, swing_highs = find_swing_lows_highs(df)
# swing_low_idx = swing_lows[-1]
# swing_high_idx = swing_highs[-1]
# fan_lines = fibonacci_fan(df, swing_low_idx, swing_high_idx)
# signals = check_price_touch(df, fan_lines)
# print(signals)
