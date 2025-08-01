import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional

class StochasticOscillator:
    """
    Stochastic Oscillator implementation for algorithmic trading.
    
    The Stochastic Oscillator is a momentum indicator that compares a security's 
    closing price to its price range over a specific period.
    
    %K = ((C - L14) / (H14 - L14)) * 100
    %D = 3-period SMA of %K
    
    Where:
    C = Current closing price
    L14 = Lowest low over past 14 periods
    H14 = Highest high over past 14 periods
    """
    
    def __init__(self, k_period: int = 14, d_period: int = 3, smooth_k: int = 1):
        """
        Initialize Stochastic Oscillator parameters.
        
        Args:
            k_period: Period for %K calculation (default: 14)
            d_period: Period for %D smoothing (default: 3)
            smooth_k: Smoothing period for %K (default: 1, no smoothing)
        """
        self.k_period = k_period
        self.d_period = d_period
        self.smooth_k = smooth_k
    
    def calculate(self, high: pd.Series, low: pd.Series, close: pd.Series) -> Tuple[pd.Series, pd.Series]:
        """
        Calculate Stochastic Oscillator %K and %D values.
        
        Args:
            high: High prices series
            low: Low prices series
            close: Close prices series
            
        Returns:
            Tuple of (%K, %D) pandas Series
        """
        # Calculate rolling highs and lows
        rolling_high = high.rolling(window=self.k_period).max()
        rolling_low = low.rolling(window=self.k_period).min()
        
        # Calculate raw %K
        k_raw = ((close - rolling_low) / (rolling_high - rolling_low)) * 100
        
        # Smooth %K if specified
        if self.smooth_k > 1:
            k_percent = k_raw.rolling(window=self.smooth_k).mean()
        else:
            k_percent = k_raw
        
        # Calculate %D (smoothed %K)
        d_percent = k_percent.rolling(window=self.d_period).mean()
        
        return k_percent, d_percent
    
    def generate_signals(self, k_percent: pd.Series, d_percent: pd.Series, 
                        overbought: float = 80, oversold: float = 20) -> pd.DataFrame:
        """
        Generate trading signals based on Stochastic Oscillator.
        
        Args:
            k_percent: %K values
            d_percent: %D values
            overbought: Overbought threshold (default: 80)
            oversold: Oversold threshold (default: 20)
            
        Returns:
            DataFrame with trading signals
        """
        signals = pd.DataFrame(index=k_percent.index)
        signals['%K'] = k_percent
        signals['%D'] = d_percent
        
        # Basic overbought/oversold signals
        signals['overbought'] = (k_percent > overbought) & (d_percent > overbought)
        signals['oversold'] = (k_percent < oversold) & (d_percent < oversold)
        
        # Crossover signals
        signals['k_above_d'] = k_percent > d_percent
        signals['bullish_crossover'] = (k_percent > d_percent) & (k_percent.shift(1) <= d_percent.shift(1))
        signals['bearish_crossover'] = (k_percent < d_percent) & (k_percent.shift(1) >= d_percent.shift(1))
        
        # Combined signals
        signals['buy_signal'] = (
            (signals['bullish_crossover'] & (k_percent < oversold)) |
            (signals['oversold'] & signals['bullish_crossover'])
        )
        
        signals['sell_signal'] = (
            (signals['bearish_crossover'] & (k_percent > overbought)) |
            (signals['overbought'] & signals['bearish_crossover'])
        )
        
        # Signal strength (0-3 scale)
        signals['signal_strength'] = 0
        signals.loc[signals['oversold'] | signals['overbought'], 'signal_strength'] = 1
        signals.loc[signals['bullish_crossover'] | signals['bearish_crossover'], 'signal_strength'] = 2
        signals.loc[signals['buy_signal'] | signals['sell_signal'], 'signal_strength'] = 3
        
        return signals
    
    def plot(self, data: pd.DataFrame, signals: pd.DataFrame, 
             title: str = "Stochastic Oscillator", figsize: Tuple[int, int] = (15, 10)):
        """
        Plot price data with Stochastic Oscillator and signals.
        
        Args:
            data: DataFrame with OHLC data
            signals: DataFrame with stochastic signals
            title: Plot title
            figsize: Figure size tuple
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True)
        
        # Price plot
        ax1.plot(data.index, data['close'], label='Close Price', linewidth=1.5)
        
        # Mark buy/sell signals on price chart
        buy_signals = signals[signals['buy_signal']]
        sell_signals = signals[signals['sell_signal']]
        
        if not buy_signals.empty:
            ax1.scatter(buy_signals.index, data.loc[buy_signals.index, 'close'], 
                       color='green', marker='^', s=100, label='Buy Signal', zorder=5)
        
        if not sell_signals.empty:
            ax1.scatter(sell_signals.index, data.loc[sell_signals.index, 'close'], 
                       color='red', marker='v', s=100, label='Sell Signal', zorder=5)
        
        ax1.set_title(f'{title} - Price Action')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Stochastic plot
        ax2.plot(signals.index, signals['%K'], label='%K', linewidth=1.5, color='blue')
        ax2.plot(signals.index, signals['%D'], label='%D', linewidth=1.5, color='red')
        
        # Add threshold lines
        ax2.axhline(y=80, color='gray', linestyle='--', alpha=0.7, label='Overbought (80)')
        ax2.axhline(y=20, color='gray', linestyle='--', alpha=0.7, label='Oversold (20)')
        ax2.axhline(y=50, color='black', linestyle='-', alpha=0.3)
        
        # Highlight overbought/oversold regions
        ax2.fill_between(signals.index, 80, 100, alpha=0.1, color='red')
        ax2.fill_between(signals.index, 0, 20, alpha=0.1, color='green')
        
        ax2.set_title('Stochastic Oscillator')
        ax2.set_ylabel('Stochastic %')
        ax2.set_ylim(0, 100)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

# Example usage and backtesting framework
class StochasticBacktester:
    """Simple backtesting framework for Stochastic Oscillator strategies."""
    
    def __init__(self, initial_capital: float = 10000):
        self.initial_capital = initial_capital
    
    def backtest(self, data: pd.DataFrame, signals: pd.DataFrame, 
                commission: float = 0.001) -> pd.DataFrame:
        """
        Perform simple backtest of stochastic signals.
        
        Args:
            data: OHLC price data
            signals: Stochastic signals
            commission: Commission rate per trade
            
        Returns:
            DataFrame with backtest results
        """
        results = pd.DataFrame(index=data.index)
        results['price'] = data['close']
        results['buy_signal'] = signals['buy_signal']
        results['sell_signal'] = signals['sell_signal']
        
        position = 0
        cash = self.initial_capital
        portfolio_value = []
        trades = []
        
        for i, (date, row) in enumerate(results.iterrows()):
            price = row['price']
            
            if row['buy_signal'] and position <= 0:
                # Buy signal
                shares_to_buy = cash // price
                if shares_to_buy > 0:
                    cost = shares_to_buy * price * (1 + commission)
                    if cost <= cash:
                        position += shares_to_buy
                        cash -= cost
                        trades.append({'date': date, 'action': 'BUY', 'price': price, 'shares': shares_to_buy})
            
            elif row['sell_signal'] and position > 0:
                # Sell signal
                proceeds = position * price * (1 - commission)
                cash += proceeds
                trades.append({'date': date, 'action': 'SELL', 'price': price, 'shares': position})
                position = 0
            
            # Calculate portfolio value
            total_value = cash + (position * price)
            portfolio_value.append(total_value)
        
        results['portfolio_value'] = portfolio_value
        results['returns'] = results['portfolio_value'].pct_change()
        results['cumulative_returns'] = (results['portfolio_value'] / self.initial_capital - 1) * 100
        
        return results, trades

# Example implementation
def example_usage():
    """Example of how to use the Stochastic Oscillator."""
    
    # Generate sample data (replace with your actual data source)
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=252, freq='D')
    
    # Simulate price data with trend and volatility
    price = 100
    prices = []
    for i in range(len(dates)):
        price += np.random.normal(0.1, 2)  # Random walk with slight upward bias
        prices.append(max(price, 10))  # Prevent negative prices
    
    # Create OHLC data
    closes = np.array(prices)
    highs = closes * (1 + np.abs(np.random.normal(0, 0.02, len(closes))))
    lows = closes * (1 - np.abs(np.random.normal(0, 0.02, len(closes))))
    
    data = pd.DataFrame({
        'high': highs,
        'low': lows,
        'close': closes
    }, index=dates)
    
    # Initialize and calculate Stochastic Oscillator
    stoch = StochasticOscillator(k_period=14, d_period=3)
    k_percent, d_percent = stoch.calculate(data['high'], data['low'], data['close'])
    
    # Generate trading signals
    signals = stoch.generate_signals(k_percent, d_percent)
    
    # Backtest the strategy
    backtester = StochasticBacktester(initial_capital=10000)
    results, trades = backtester.backtest(data, signals)
    
    # Display results
    print("Stochastic Oscillator Trading Strategy Results")
    print("=" * 50)
    print(f"Initial Capital: ${backtester.initial_capital:,.2f}")
    print(f"Final Portfolio Value: ${results['portfolio_value'].iloc[-1]:,.2f}")
    print(f"Total Return: {results['cumulative_returns'].iloc[-1]:.2f}%")
    print(f"Number of Trades: {len(trades)}")
    
    if len(trades) > 0:
        print(f"Average Trade: {results['cumulative_returns'].iloc[-1] / (len(trades)/2):.2f}%")
    
    # Plot results
    stoch.plot(data, signals)
    
    return data, signals, results, trades

if __name__ == "__main__":
    # Run example
    data, signals, results, trades = example_usage()