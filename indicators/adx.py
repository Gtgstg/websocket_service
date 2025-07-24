import numpy as np
import pandas as pd

def calculate_adx(df, period=14):
    """
    Calculate the Average Directional Index (ADX) along with +DI and -DI
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with 'high', 'low', and 'close' columns
    period : int, default 14
        The period for calculating the ADX
        
    Returns:
    --------
    pandas.DataFrame
        Original DataFrame with 'adx', 'plus_di', and 'minus_di' columns added
    """
    # Make a copy of the dataframe to avoid modifying the original
    df = df.copy()
    
    # Calculate True Range (TR)
    df['high_low'] = df['high'] - df['low']
    df['high_close'] = np.abs(df['high'] - df['close'].shift(1))
    df['low_close'] = np.abs(df['low'] - df['close'].shift(1))
    df['tr'] = df[['high_low', 'high_close', 'low_close']].max(axis=1)
    
    # Calculate Directional Movement (+DM and -DM)
    df['up_move'] = df['high'].diff()
    df['down_move'] = df['low'].diff(-1).abs()  # Multiply by -1 to make positive
    
    # Calculate +DM and -DM
    df['plus_dm'] = np.where(
        (df['up_move'] > df['down_move']) & (df['up_move'] > 0),
        df['up_move'],
        0
    )
    df['minus_dm'] = np.where(
        (df['down_move'] > df['up_move']) & (df['down_move'] > 0),
        df['down_move'],
        0
    )
    
    # Calculate smoothed averages
    # Smoothed TR, +DM and -DM for the first period
    df['smoothed_tr'] = df['tr'].rolling(window=period).sum()
    df['smoothed_plus_dm'] = df['plus_dm'].rolling(window=period).sum()
    df['smoothed_minus_dm'] = df['minus_dm'].rolling(window=period).sum()
    
    # Calculate subsequent values with Wilder's smoothing
    for i in range(period + 1, len(df)):
        df.loc[df.index[i], 'smoothed_tr'] = (
            df.loc[df.index[i-1], 'smoothed_tr'] - 
            (df.loc[df.index[i-1], 'smoothed_tr'] / period) + 
            df.loc[df.index[i], 'tr']
        )
        
        df.loc[df.index[i], 'smoothed_plus_dm'] = (
            df.loc[df.index[i-1], 'smoothed_plus_dm'] - 
            (df.loc[df.index[i-1], 'smoothed_plus_dm'] / period) + 
            df.loc[df.index[i], 'plus_dm']
        )
        
        df.loc[df.index[i], 'smoothed_minus_dm'] = (
            df.loc[df.index[i-1], 'smoothed_minus_dm'] - 
            (df.loc[df.index[i-1], 'smoothed_minus_dm'] / period) + 
            df.loc[df.index[i], 'minus_dm']
        )
    
    # Calculate +DI and -DI
    df['plus_di'] = 100 * (df['smoothed_plus_dm'] / df['smoothed_tr'])
    df['minus_di'] = 100 * (df['smoothed_minus_dm'] / df['smoothed_tr'])
    
    # Calculate DX (Directional Index)
    df['dx'] = 100 * np.abs(df['plus_di'] - df['minus_di']) / (df['plus_di'] + df['minus_di'])
    
    # Calculate ADX (smoothed DX)
    df['adx'] = df['dx'].rolling(window=period).mean()
    
    # Clean up intermediate columns
    columns_to_drop = [
        'high_low', 'high_close', 'low_close', 'tr', 
        'up_move', 'down_move', 'plus_dm', 'minus_dm',
        'smoothed_tr', 'smoothed_plus_dm', 'smoothed_minus_dm', 'dx'
    ]
    df = df.drop(columns=columns_to_drop)
    
    return df


def detect_adx_divergence(df, window=10):
    """
    Detect ADX divergence patterns
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with 'close' and 'adx' columns
    window : int, default 10
        Lookback window for detecting local highs/lows
        
    Returns:
    --------
    pandas.DataFrame
        Original DataFrame with divergence columns added:
        'bullish_divergence' and 'bearish_divergence'
    """
    df = df.copy()
    
    # Initialize divergence columns
    df['bullish_divergence'] = False
    df['bearish_divergence'] = False
    
    # Need at least 2*window+1 data points to detect divergence
    if len(df) < 2*window+1:
        return df
    
    # Function to find local extrema (peaks and troughs)
    def is_local_max(series, i, window):
        if i - window < 0 or i + window >= len(series):
            return False
        return all(series.iloc[i] > series.iloc[i-j] for j in range(1, window+1)) and \
               all(series.iloc[i] > series.iloc[i+j] for j in range(1, window+1))
               
    def is_local_min(series, i, window):
        if i - window < 0 or i + window >= len(series):
            return False
        return all(series.iloc[i] < series.iloc[i-j] for j in range(1, window+1)) and \
               all(series.iloc[i] < series.iloc[i+j] for j in range(1, window+1))
    
    # Detect divergences
    for i in range(window, len(df) - window):
        # Bearish Divergence: Price makes higher high but ADX makes lower high
        if is_local_max(df['close'], i, window):
            # Look for a previous local high
            for j in range(i-window, i):
                if is_local_max(df['close'], j, window//2):
                    # If price makes higher high
                    if df['close'].iloc[i] > df['close'].iloc[j]:
                        # But ADX makes lower high or equal high
                        if df['adx'].iloc[i] < df['adx'].iloc[j]:
                            df.loc[df.index[i], 'bearish_divergence'] = True
                            break
        
        # Bullish Divergence: Price makes lower low but ADX makes higher low
        if is_local_min(df['close'], i, window):
            # Look for a previous local low
            for j in range(i-window, i):
                if is_local_min(df['close'], j, window//2):
                    # If price makes lower low
                    if df['close'].iloc[i] < df['close'].iloc[j]:
                        # But ADX makes higher low or equal low
                        if df['adx'].iloc[i] > df['adx'].iloc[j]:
                            df.loc[df.index[i], 'bullish_divergence'] = True
                            break
    
    return df


def adx_trading_signals(df, adx_threshold=25, use_divergence=True, window=10):
    """
    Generate trading signals based on ADX indicator and divergence
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with 'adx', 'plus_di', and 'minus_di' columns
    adx_threshold : int, default 25
        The threshold value for ADX to consider a strong trend
    use_divergence : bool, default True
        Whether to include divergence signals
    window : int, default 10
        Lookback window for detecting divergence
        
    Returns:
    --------
    pandas.DataFrame
        Original DataFrame with 'signal' column added:
          1 for buy signal
         -1 for sell signal
          0 for no signal/hold
    """
    df = df.copy()
    
    # Initialize signals
    df['signal'] = 0
    
    # Detect divergences if enabled
    if use_divergence:
        df = detect_adx_divergence(df, window=window)
    
    # Generate signals based on ADX and DI crossovers
    for i in range(1, len(df)):
        # Strong trend (ADX > threshold)
        if df['adx'].iloc[i] > adx_threshold:
            # Bullish signal: +DI crosses above -DI
            if (df['plus_di'].iloc[i] > df['minus_di'].iloc[i] and 
                df['plus_di'].iloc[i-1] <= df['minus_di'].iloc[i-1]):
                df.loc[df.index[i], 'signal'] = 1
                
            # Bearish signal: -DI crosses above +DI
            elif (df['minus_di'].iloc[i] > df['plus_di'].iloc[i] and 
                  df['minus_di'].iloc[i-1] <= df['plus_di'].iloc[i-1]):
                df.loc[df.index[i], 'signal'] = -1
        
        # Add divergence signals if enabled
        if use_divergence:
            # Bullish divergence detected
            if df['bullish_divergence'].iloc[i] and df['signal'].iloc[i] == 0:
                df.loc[df.index[i], 'signal'] = 1
                
            # Bearish divergence detected
            elif df['bearish_divergence'].iloc[i] and df['signal'].iloc[i] == 0:
                df.loc[df.index[i], 'signal'] = -1
    
    return df


# Example usage
if __name__ == "__main__":
    # Create a sample dataframe or load your data
    # Example with random data
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from datetime import datetime, timedelta
    
    # Generate sample date range
    start_date = datetime(2023, 1, 1)
    dates = [start_date + timedelta(days=i) for i in range(200)]
    
    # Generate sample OHLC data with more price movement for better divergence examples
    np.random.seed(42)
    
    # Create a more realistic price series with trends and reversals
    close = np.zeros(200)
    close[0] = 1000
    
    # Generate a price series with some trends and reversals
    for i in range(1, 200):
        if i < 50:  # Uptrend
            close[i] = close[i-1] + np.random.normal(0.5, 1)
        elif i < 80:  # Downtrend
            close[i] = close[i-1] - np.random.normal(0.6, 1.2)
        elif i < 130:  # Uptrend
            close[i] = close[i-1] + np.random.normal(0.7, 1.1)
        elif i < 160:  # Sideways
            close[i] = close[i-1] + np.random.normal(0, 1.5)
        else:  # Downtrend
            close[i] = close[i-1] - np.random.normal(0.5, 1)
    
    # Generate high, low, and open prices based on close
    high = close + np.random.uniform(0, 5, 200)
    low = close - np.random.uniform(0, 5, 200)
    open_price = close + np.random.normal(0, 2, 200)
    
    # Create dataframe
    df = pd.DataFrame({
        'date': dates,
        'open': open_price,
        'high': high,
        'low': low,
        'close': close
    })
    df.set_index('date', inplace=True)
    
    # Calculate ADX
    df = calculate_adx(df, period=14)
    
    # Generate trading signals with divergence
    df = adx_trading_signals(df, adx_threshold=25, use_divergence=True, window=10)
    
    # Plot the results
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    
    # Plot price
    ax1.plot(df.index, df['close'], label='Close Price')
    
    # Add buy signals
    buy_signals = df[df['signal'] == 1]
    ax1.scatter(buy_signals.index, buy_signals['close'], color='green', marker='^', s=100, label='Buy Signal')
    
    # Add sell signals
    sell_signals = df[df['signal'] == -1]
    ax1.scatter(sell_signals.index, sell_signals['close'], color='red', marker='v', s=100, label='Sell Signal')
    
    # Highlight bullish divergence points
    bullish_div = df[df['bullish_divergence']]
    ax1.scatter(bullish_div.index, bullish_div['close'] - 15, color='blue', marker='*', s=150, label='Bullish Divergence')
    
    # Highlight bearish divergence points
    bearish_div = df[df['bearish_divergence']]
    ax1.scatter(bearish_div.index, bearish_div['close'] + 15, color='purple', marker='*', s=150, label='Bearish Divergence')
    
    ax1.set_ylabel('Price')
    ax1.legend()
    ax1.grid(True)
    ax1.set_title('Price with ADX Signals Including Divergence')
    
    # Plot ADX, +DI, -DI
    ax2.plot(df.index, df['adx'], label='ADX', color='black')
    ax2.plot(df.index, df['plus_di'], label='+DI', color='green')
    ax2.plot(df.index, df['minus_di'], label='-DI', color='red')
    ax2.axhline(y=25, color='blue', linestyle='--', label='Threshold (25)')
    
    # Highlight the same divergence points on ADX
    ax2.scatter(bullish_div.index, bullish_div['adx'], color='blue', marker='*', s=150)
    ax2.scatter(bearish_div.index, bearish_div['adx'], color='purple', marker='*', s=150)
    
    ax2.set_ylabel('ADX/DI Values')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # Print trading signals
    print("Trading Signals:")
    signal_df = df[df['signal'] != 0][['close', 'adx', 'plus_di', 'minus_di', 'signal', 'bullish_divergence', 'bearish_divergence']]
    print(signal_df.head(10))
    
    # Print divergence information
    print("\nBullish Divergences:")
    print(df[df['bullish_divergence']].head())
    
    print("\nBearish Divergences:")
    print(df[df['bearish_divergence']].head())