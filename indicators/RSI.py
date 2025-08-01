import numpy as np
import pandas as pd

def calculate_rsi(df, period=14, method='wilders'):
    """
    Calculate the Relative Strength Index (RSI)
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with 'close' column
    period : int, default 14
        The period for calculating the RSI
    method : str, default 'wilders'
        Method for smoothing: 'wilders' (Wilder's smoothing) or 'ema' (exponential moving average)
        
    Returns:
    --------
    pandas.DataFrame
        Original DataFrame with 'rsi' column added
    """
    # Make a copy of the dataframe to avoid modifying the original
    df = df.copy()
    
    # Calculate price changes
    df['price_change'] = df['close'].diff()
    
    # Separate gains and losses
    df['gain'] = np.where(df['price_change'] > 0, df['price_change'], 0)
    df['loss'] = np.where(df['price_change'] < 0, -df['price_change'], 0)
    
    if method == 'wilders':
        # Wilder's smoothing method (original RSI calculation)
        # Calculate initial averages using simple moving average
        df['avg_gain'] = df['gain'].rolling(window=period).mean()
        df['avg_loss'] = df['loss'].rolling(window=period).mean()
        
        # Apply Wilder's smoothing for subsequent values
        for i in range(period, len(df)):
            df.loc[df.index[i], 'avg_gain'] = (
                (df.loc[df.index[i-1], 'avg_gain'] * (period - 1) + df.loc[df.index[i], 'gain']) / period
            )
            df.loc[df.index[i], 'avg_loss'] = (
                (df.loc[df.index[i-1], 'avg_loss'] * (period - 1) + df.loc[df.index[i], 'loss']) / period
            )
    
    elif method == 'ema':
        # Exponential moving average method
        alpha = 2.0 / (period + 1)
        df['avg_gain'] = df['gain'].ewm(alpha=alpha, adjust=False).mean()
        df['avg_loss'] = df['loss'].ewm(alpha=alpha, adjust=False).mean()
    
    else:
        raise ValueError("Method must be 'wilders' or 'ema'")
    
    # Calculate Relative Strength (RS) and RSI
    df['rs'] = df['avg_gain'] / df['avg_loss']
    df['rsi'] = 100 - (100 / (1 + df['rs']))
    
    # Handle division by zero (when avg_loss is 0)
    df['rsi'] = np.where(df['avg_loss'] == 0, 100, df['rsi'])
    df['rsi'] = np.where(df['avg_gain'] == 0, 0, df['rsi'])
    
    # Clean up intermediate columns
    columns_to_drop = ['price_change', 'gain', 'loss', 'avg_gain', 'avg_loss', 'rs']
    df = df.drop(columns=columns_to_drop)
    
    return df


def detect_rsi_divergence(df, window=10, rsi_threshold_high=70, rsi_threshold_low=30):
    """
    Detect RSI divergence patterns
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with 'close' and 'rsi' columns
    window : int, default 10
        Lookback window for detecting local highs/lows
    rsi_threshold_high : float, default 70
        RSI level above which to look for bearish divergences
    rsi_threshold_low : float, default 30
        RSI level below which to look for bullish divergences
        
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
        return all(series.iloc[i] >= series.iloc[i-j] for j in range(1, window+1)) and \
               all(series.iloc[i] >= series.iloc[i+j] for j in range(1, window+1))
               
    def is_local_min(series, i, window):
        if i - window < 0 or i + window >= len(series):
            return False
        return all(series.iloc[i] <= series.iloc[i-j] for j in range(1, window+1)) and \
               all(series.iloc[i] <= series.iloc[i+j] for j in range(1, window+1))
    
    # Detect divergences
    for i in range(window, len(df) - window):
        # Bearish Divergence: Price makes higher high but RSI makes lower high (in overbought zone)
        if (is_local_max(df['close'], i, window) and 
            df['rsi'].iloc[i] > rsi_threshold_high):
            
            # Look for a previous local high in overbought zone
            for j in range(max(0, i-50), i):  # Look back up to 50 periods
                if (is_local_max(df['close'], j, window//2) and 
                    df['rsi'].iloc[j] > rsi_threshold_high):
                    
                    # If price makes higher high but RSI makes lower high
                    if (df['close'].iloc[i] > df['close'].iloc[j] and 
                        df['rsi'].iloc[i] < df['rsi'].iloc[j]):
                        df.loc[df.index[i], 'bearish_divergence'] = True
                        break
        
        # Bullish Divergence: Price makes lower low but RSI makes higher low (in oversold zone)
        if (is_local_min(df['close'], i, window) and 
            df['rsi'].iloc[i] < rsi_threshold_low):
            
            # Look for a previous local low in oversold zone
            for j in range(max(0, i-50), i):  # Look back up to 50 periods
                if (is_local_min(df['close'], j, window//2) and 
                    df['rsi'].iloc[j] < rsi_threshold_low):
                    
                    # If price makes lower low but RSI makes higher low
                    if (df['close'].iloc[i] < df['close'].iloc[j] and 
                        df['rsi'].iloc[i] > df['rsi'].iloc[j]):
                        df.loc[df.index[i], 'bullish_divergence'] = True
                        break
    
    return df


def rsi_trading_signals(df, overbought=70, oversold=30, use_divergence=True, window=10):
    """
    Generate trading signals based on RSI indicator
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with 'rsi' column
    overbought : float, default 70
        RSI level considered overbought (sell signal threshold)
    oversold : float, default 30
        RSI level considered oversold (buy signal threshold)
    use_divergence : bool, default True
        Whether to include divergence signals
    window : int, default 10
        Lookback window for detecting divergence
        
    Returns:
    --------
    pandas.DataFrame
        Original DataFrame with signal columns added:
        'signal': 1 for buy, -1 for sell, 0 for hold
        'rsi_overbought': True when RSI > overbought threshold
        'rsi_oversold': True when RSI < oversold threshold
    """
    df = df.copy()
    
    # Initialize signal columns
    df['signal'] = 0
    df['rsi_overbought'] = df['rsi'] > overbought
    df['rsi_oversold'] = df['rsi'] < oversold
    
    # Detect divergences if enabled
    if use_divergence:
        df = detect_rsi_divergence(df, window=window, 
                                 rsi_threshold_high=overbought, 
                                 rsi_threshold_low=oversold)
    
    # Generate signals
    for i in range(1, len(df)):
        # Standard RSI signals
        # Buy signal: RSI crosses above oversold level
        if (df['rsi'].iloc[i] > oversold and 
            df['rsi'].iloc[i-1] <= oversold):
            df.loc[df.index[i], 'signal'] = 1
            
        # Sell signal: RSI crosses below overbought level
        elif (df['rsi'].iloc[i] < overbought and 
              df['rsi'].iloc[i-1] >= overbought):
            df.loc[df.index[i], 'signal'] = -1
        
        # Add divergence signals if enabled
        if use_divergence:
            # Bullish divergence signal (overrides standard signal if stronger)
            if df['bullish_divergence'].iloc[i]:
                df.loc[df.index[i], 'signal'] = 1
                
            # Bearish divergence signal (overrides standard signal if stronger)
            elif df['bearish_divergence'].iloc[i]:
                df.loc[df.index[i], 'signal'] = -1
    
    return df


def rsi_multi_timeframe(df, periods=[14, 21, 28]):
    """
    Calculate RSI for multiple timeframes
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with 'close' column
    periods : list, default [14, 21, 28]
        List of periods for RSI calculation
        
    Returns:
    --------
    pandas.DataFrame
        Original DataFrame with RSI columns for each period
    """
    df = df.copy()
    
    for period in periods:
        temp_df = calculate_rsi(df[['close']], period=period)
        df[f'rsi_{period}'] = temp_df['rsi']
    
    return df


def rsi_stochastic_rsi(df, period=14, stoch_period=14):
    """
    Calculate Stochastic RSI - RSI of RSI
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with 'close' column
    period : int, default 14
        Period for RSI calculation
    stoch_period : int, default 14
        Period for Stochastic calculation on RSI
        
    Returns:
    --------
    pandas.DataFrame
        Original DataFrame with 'stoch_rsi', '%K', and '%D' columns added
    """
    df = df.copy()
    
    # First calculate RSI
    df = calculate_rsi(df, period=period)
    
    # Calculate Stochastic RSI
    df['rsi_low'] = df['rsi'].rolling(window=stoch_period).min()
    df['rsi_high'] = df['rsi'].rolling(window=stoch_period).max()
    
    # Calculate %K (fast stochastic)
    df['stoch_rsi'] = (df['rsi'] - df['rsi_low']) / (df['rsi_high'] - df['rsi_low']) * 100
    df['%K'] = df['stoch_rsi']
    
    # Calculate %D (slow stochastic - 3-period SMA of %K)
    df['%D'] = df['%K'].rolling(window=3).mean()
    
    # Handle division by zero
    df['stoch_rsi'] = df['stoch_rsi'].fillna(50)
    df['%K'] = df['%K'].fillna(50)
    df['%D'] = df['%D'].fillna(50)
    
    # Clean up intermediate columns
    df = df.drop(columns=['rsi_low', 'rsi_high'])
    
    return df


class RSI:
    """
    Relative Strength Index (RSI) implementation.
    RSI measures the magnitude of recent price changes to evaluate
    overbought or oversold conditions.
    """
    
    def __init__(self, period: int = 14):
        """
        Initialize RSI indicator.
        
        Args:
            period (int): The period over which to calculate RSI (default: 14)
        """
        self.period = period
    
    def calculate(self, prices: pd.Series) -> pd.DataFrame:
        """
        Calculate RSI for the given price series.
        
        Args:
            prices (pd.Series): Price data series
            
        Returns:
            pd.DataFrame: DataFrame with RSI values and signals
        """
        # Calculate price changes
        delta = prices.diff()
        
        # Separate gains and losses
        gains = delta.copy()
        losses = delta.copy()
        
        gains[gains < 0] = 0
        losses[losses > 0] = 0
        losses = abs(losses)
        
        # Calculate average gains and losses
        avg_gains = gains.rolling(window=self.period, min_periods=1).mean()
        avg_losses = losses.rolling(window=self.period, min_periods=1).mean()
        
        # Calculate RS and RSI
        rs = avg_gains / avg_losses
        rsi = 100 - (100 / (1 + rs))
        
        # Generate signals
        signals = pd.DataFrame(index=prices.index)
        signals['rsi'] = rsi
        signals['signal'] = 0
        
        # Oversold (RSI < 30) -> Buy Signal
        signals.loc[signals['rsi'] < 30, 'signal'] = 1
        
        # Overbought (RSI > 70) -> Sell Signal
        signals.loc[signals['rsi'] > 70, 'signal'] = -1
        
        return signals
    
    def get_signal(self, rsi_value: float) -> int:
        """
        Get trading signal based on RSI value.
        
        Args:
            rsi_value (float): Current RSI value
            
        Returns:
            int: 1 for buy, -1 for sell, 0 for hold
        """
        if rsi_value < 30:  # Oversold
            return 1
        elif rsi_value > 70:  # Overbought
            return -1
        return 0  # Hold


# Example usage
if __name__ == "__main__":
    # Create a sample dataframe or load your data
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from datetime import datetime, timedelta
    
    # Generate sample date range
    start_date = datetime(2023, 1, 1)
    dates = [start_date + timedelta(days=i) for i in range(200)]
    
    # Generate sample price data with various market conditions
    np.random.seed(42)
    
    # Create a price series with trends, reversals, and volatility
    close = np.zeros(200)
    close[0] = 100
    
    # Generate a realistic price series
    for i in range(1, 200):
        if i < 40:  # Strong uptrend
            close[i] = close[i-1] + np.random.normal(0.8, 1.2)
        elif i < 60:  # Correction/pullback
            close[i] = close[i-1] - np.random.normal(0.5, 1.5)
        elif i < 100:  # Consolidation/sideways
            close[i] = close[i-1] + np.random.normal(0, 1.8)
        elif i < 140:  # Another uptrend
            close[i] = close[i-1] + np.random.normal(0.6, 1.1)
        elif i < 170:  # Sharp decline
            close[i] = close[i-1] - np.random.normal(1.2, 1.5)
        else:  # Recovery
            close[i] = close[i-1] + np.random.normal(0.4, 1.0)
    
    # Generate high, low, and open prices based on close
    high = close + np.random.uniform(0.5, 2.5, 200)
    low = close - np.random.uniform(0.5, 2.5, 200)
    open_price = close + np.random.normal(0, 1, 200)
    volume = np.random.randint(10000, 100000, 200)
    
    # Create dataframe
    df = pd.DataFrame({
        'date': dates,
        'open': open_price,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume
    })
    df.set_index('date', inplace=True)
    
    # Calculate RSI
    df = calculate_rsi(df, period=14)
    
    # Generate trading signals with divergence
    df = rsi_trading_signals(df, overbought=70, oversold=30, use_divergence=True, window=8)
    
    # Calculate multi-timeframe RSI
    df = rsi_multi_timeframe(df, periods=[14, 21])
    
    # Calculate Stochastic RSI
    df = rsi_stochastic_rsi(df, period=14, stoch_period=14)
    
    # Plot the results
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
    
    # Plot price with signals
    ax1.plot(df.index, df['close'], label='Close Price', color='black', linewidth=1)
    
    # Add buy signals
    buy_signals = df[df['signal'] == 1]
    ax1.scatter(buy_signals.index, buy_signals['close'], color='green', marker='^', s=100, label='Buy Signal', zorder=5)
    
    # Add sell signals
    sell_signals = df[df['signal'] == -1]
    ax1.scatter(sell_signals.index, sell_signals['close'], color='red', marker='v', s=100, label='Sell Signal', zorder=5)
    
    # Highlight bullish divergence points
    if 'bullish_divergence' in df.columns:
        bullish_div = df[df['bullish_divergence']]
        ax1.scatter(bullish_div.index, bullish_div['close'] - 5, color='blue', marker='*', s=150, label='Bullish Divergence', zorder=5)
    
    # Highlight bearish divergence points
    if 'bearish_divergence' in df.columns:
        bearish_div = df[df['bearish_divergence']]
        ax1.scatter(bearish_div.index, bearish_div['close'] + 5, color='purple', marker='*', s=150, label='Bearish Divergence', zorder=5)
    
    ax1.set_ylabel('Price')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_title('Price with RSI Signals and Divergences')
    
    # Plot RSI with overbought/oversold levels
    ax2.plot(df.index, df['rsi'], label='RSI (14)', color='blue', linewidth=1.5)
    ax2.plot(df.index, df['rsi_21'], label='RSI (21)', color='orange', linewidth=1, alpha=0.7)
    ax2.axhline(y=70, color='red', linestyle='--', alpha=0.7, label='Overbought (70)')
    ax2.axhline(y=30, color='green', linestyle='--', alpha=0.7, label='Oversold (30)')
    ax2.axhline(y=50, color='gray', linestyle='-', alpha=0.5)
    
    # Fill overbought and oversold areas
    ax2.fill_between(df.index, 70, 100, alpha=0.1, color='red')
    ax2.fill_between(df.index, 0, 30, alpha=0.1, color='green')
    
    # Highlight the same divergence points on RSI
    if 'bullish_divergence' in df.columns:
        ax2.scatter(bullish_div.index, bullish_div['rsi'], color='blue', marker='*', s=150, zorder=5)
    if 'bearish_divergence' in df.columns:
        ax2.scatter(bearish_div.index, bearish_div['rsi'], color='purple', marker='*', s=150, zorder=5)
    
    ax2.set_ylabel('RSI')
    ax2.set_ylim(0, 100)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot Stochastic RSI
    ax3.plot(df.index, df['%K'], label='Stochastic RSI %K', color='blue', linewidth=1)
    ax3.plot(df.index, df['%D'], label='Stochastic RSI %D', color='red', linewidth=1)
    ax3.axhline(y=80, color='red', linestyle='--', alpha=0.7, label='Overbought (80)')
    ax3.axhline(y=20, color='green', linestyle='--', alpha=0.7, label='Oversold (20)')
    ax3.axhline(y=50, color='gray', linestyle='-', alpha=0.5)
    
    # Fill overbought and oversold areas
    ax3.fill_between(df.index, 80, 100, alpha=0.1, color='red')
    ax3.fill_between(df.index, 0, 20, alpha=0.1, color='green')
    
    ax3.set_ylabel('Stochastic RSI')
    ax3.set_ylim(0, 100)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_xlabel('Date')
    
    plt.tight_layout()
    plt.show()
    
    # Print trading signals summary
    print("RSI Trading Signals Summary:")
    print(f"Total signals: {(df['signal'] != 0).sum()}")
    print(f"Buy signals: {(df['signal'] == 1).sum()}")
    print(f"Sell signals: {(df['signal'] == -1).sum()}")
    
    if 'bullish_divergence' in df.columns:
        print(f"Bullish divergences: {df['bullish_divergence'].sum()}")
    if 'bearish_divergence' in df.columns:
        print(f"Bearish divergences: {df['bearish_divergence'].sum()}")
    
    # Print recent signals
    print("\nRecent Trading Signals:")
    recent_signals = df[df['signal'] != 0][['close', 'rsi', 'signal']].tail(10)
    if not recent_signals.empty:
        for idx, row in recent_signals.iterrows():
            signal_type = "BUY" if row['signal'] == 1 else "SELL"
            print(f"{idx.strftime('%Y-%m-%d')}: {signal_type} - Price: {row['close']:.2f}, RSI: {row['rsi']:.2f}")
    
    # Print current RSI levels
    print(f"\nCurrent RSI: {df['rsi'].iloc[-1]:.2f}")
    print(f"Current Stochastic RSI %K: {df['%K'].iloc[-1]:.2f}")
    print(f"Current Stochastic RSI %D: {df['%D'].iloc[-1]:.2f}")