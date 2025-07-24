"""
Bollinger Bands Indicator Module
Provides classes and functions for calculating Bollinger Bands and related signals.
"""
import pandas as pd
import numpy as np
from typing import Tuple, Dict, Optional
import matplotlib.pyplot as plt

class BollingerBands:
    """
    Bollinger Bands technical indicator for algorithmic trading.
    
    Bollinger Bands consist of:
    - Middle Band: Simple Moving Average (SMA)
    - Upper Band: SMA + (standard deviation * multiplier)
    - Lower Band: SMA - (standard deviation * multiplier)
    """
    
    def __init__(self, period: int = 20, std_dev: float = 2.0):
        """
        Initialize Bollinger Bands indicator.
        
        Args:
            period (int): Period for moving average calculation (default: 20)
            std_dev (float): Standard deviation multiplier (default: 2.0)
        """
        self.period = period
        self.std_dev = std_dev
    
    def calculate(self, prices: pd.Series) -> pd.DataFrame:
        """
        Calculate Bollinger Bands for given price series.
        
        Args:
            prices (pd.Series): Price data (typically closing prices)
            
        Returns:
            pd.DataFrame: DataFrame with columns ['middle', 'upper', 'lower', 'bandwidth', 'percent_b']
        """
        if len(prices) < self.period:
            raise ValueError(f"Not enough data points. Need at least {self.period} points.")
        
        # Calculate Simple Moving Average (Middle Band)
        middle = prices.rolling(window=self.period).mean()
        
        # Calculate Standard Deviation
        rolling_std = prices.rolling(window=self.period).std()
        
        # Calculate Upper and Lower Bands
        upper = middle + (rolling_std * self.std_dev)
        lower = middle - (rolling_std * self.std_dev)
        
        # Calculate additional metrics
        bandwidth = ((upper - lower) / middle) * 100  # Bandwidth as percentage
        percent_b = (prices - lower) / (upper - lower)  # %B indicator
        
        result = pd.DataFrame({
            'middle': middle,
            'upper': upper,
            'lower': lower,
            'bandwidth': bandwidth,
            'percent_b': percent_b,
            'price': prices
        })
        
        return result
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals based on Bollinger Bands.
        
        Args:
            data (pd.DataFrame): DataFrame with Bollinger Bands data
            
        Returns:
            pd.DataFrame: DataFrame with additional signal columns
        """
        signals = data.copy()
        
        # Initialize signal columns
        signals['signal'] = 0  # 0: Hold, 1: Buy, -1: Sell
        signals['position'] = 0  # Current position
        signals['squeeze'] = False  # Bollinger Band Squeeze
        
        # Generate signals based on price touching bands
        signals.loc[signals['price'] <= signals['lower'], 'signal'] = 1  # Buy signal
        signals.loc[signals['price'] >= signals['upper'], 'signal'] = -1  # Sell signal
        
        # Alternative: %B based signals
        signals.loc[signals['percent_b'] <= 0, 'signal_percent_b'] = 1  # Oversold
        signals.loc[signals['percent_b'] >= 1, 'signal_percent_b'] = -1  # Overbought
        
        # Bollinger Band Squeeze detection (low volatility)
        # Squeeze occurs when bandwidth is below a threshold (e.g., 10th percentile)
        bandwidth_threshold = signals['bandwidth'].quantile(0.1)
        signals['squeeze'] = signals['bandwidth'] < bandwidth_threshold
        
        # Calculate position (simple strategy: follow signals)
        signals['position'] = signals['signal'].fillna(0)
        
        return signals
    
    def backtest_simple(self, data: pd.DataFrame, initial_capital: float = 10000) -> Dict:
        """
        Simple backtest of Bollinger Bands strategy.
        
        Args:
            data (pd.DataFrame): DataFrame with signals
            initial_capital (float): Starting capital
            
        Returns:
            Dict: Backtest results
        """
        portfolio = initial_capital
        position = 0
        trades = []
        portfolio_values = [initial_capital]
        
        for i in range(1, len(data)):
            current_price = data.iloc[i]['price']
            signal = data.iloc[i]['signal']
            
            if signal == 1 and position == 0:  # Buy
                position = portfolio / current_price
                portfolio = 0
                trades.append(('BUY', current_price, data.index[i]))
                
            elif signal == -1 and position > 0:  # Sell
                portfolio = position * current_price
                position = 0
                trades.append(('SELL', current_price, data.index[i]))
            
            # Calculate current portfolio value
            current_value = portfolio + (position * current_price)
            portfolio_values.append(current_value)
        
        # Final portfolio value
        final_price = data.iloc[-1]['price']
        final_value = portfolio + (position * final_price)
        
        # Calculate returns
        total_return = (final_value - initial_capital) / initial_capital * 100
        
        return {
            'initial_capital': initial_capital,
            'final_value': final_value,
            'total_return': total_return,
            'num_trades': len(trades),
            'trades': trades,
            'portfolio_values': portfolio_values
        }

def create_sample_data(days: int = 252) -> pd.Series:
    """Create sample price data for testing."""
    np.random.seed(42)
    dates = pd.date_range(start='2023-01-01', periods=days, freq='D')
    
    # Generate realistic price movement
    returns = np.random.normal(0.001, 0.02, days)  # Daily returns
    prices = [100]  # Starting price
    
    for ret in returns[1:]:
        prices.append(prices[-1] * (1 + ret))
    
    return pd.Series(prices, index=dates)

# Example usage and demonstration
if __name__ == "__main__":
    # Create sample data
    price_data = create_sample_data(252)
    
    # Initialize Bollinger Bands
    bb = BollingerBands(period=20, std_dev=2.0)
    
    # Calculate Bollinger Bands
    bb_data = bb.calculate(price_data)
    
    # Generate trading signals
    signals = bb.generate_signals(bb_data)
    
    # Run backtest
    backtest_results = bb.backtest_simple(signals)
    
    # Print results
    print("=== Bollinger Bands Backtest Results ===")
    print(f"Initial Capital: ${backtest_results['initial_capital']:,.2f}")
    print(f"Final Value: ${backtest_results['final_value']:,.2f}")
    print(f"Total Return: {backtest_results['total_return']:.2f}%")
    print(f"Number of Trades: {backtest_results['num_trades']}")
    
    # Display recent signals
    print("\n=== Recent Signals ===")
    recent_signals = signals.tail(10)[['price', 'upper', 'middle', 'lower', 'signal', 'percent_b']]
    print(recent_signals.round(4))
    
    # Plot results
    plt.figure(figsize=(15, 10))
    
    # Price and Bollinger Bands
    plt.subplot(2, 1, 1)
    plt.plot(signals.index, signals['price'], label='Price', linewidth=2)
    plt.plot(signals.index, signals['upper'], label='Upper Band', alpha=0.7)
    plt.plot(signals.index, signals['middle'], label='Middle Band (SMA)', alpha=0.7)
    plt.plot(signals.index, signals['lower'], label='Lower Band', alpha=0.7)
    plt.fill_between(signals.index, signals['upper'], signals['lower'], alpha=0.1)
    
    # Mark buy/sell signals
    buy_signals = signals[signals['signal'] == 1]
    sell_signals = signals[signals['signal'] == -1]
    
    plt.scatter(buy_signals.index, buy_signals['price'], color='green', marker='^', s=100, label='Buy Signal')
    plt.scatter(sell_signals.index, sell_signals['price'], color='red', marker='v', s=100, label='Sell Signal')
    
    plt.title('Bollinger Bands Trading Strategy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # %B Indicator
    plt.subplot(2, 1, 2)
    plt.plot(signals.index, signals['percent_b'], label='%B', color='purple')
    plt.axhline(y=1, color='red', linestyle='--', alpha=0.7, label='Overbought (1.0)')
    plt.axhline(y=0, color='green', linestyle='--', alpha=0.7, label='Oversold (0.0)')
    plt.axhline(y=0.5, color='gray', linestyle='-', alpha=0.5, label='Middle (0.5)')
    plt.fill_between(signals.index, 0, 1, alpha=0.1, color='gray')
    
    plt.title('%B Indicator (Price Position within Bands)')
    plt.ylabel('%B')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# Integration example for live trading bot
class BollingerBandsTradingBot:
    """
    Example integration class for a live trading bot.
    """
    
    def __init__(self, symbol: str, period: int = 20, std_dev: float = 2.0):
        self.symbol = symbol
        self.bb = BollingerBands(period, std_dev)
        self.position = 0
        self.price_history = []
    
    def update_price(self, new_price: float, timestamp: Optional[str] = None):
        """Update with new price data and check for signals."""
        self.price_history.append(new_price)
        
        # Keep only necessary history
        if len(self.price_history) > self.bb.period * 2:
            self.price_history = self.price_history[-self.bb.period * 2:]
        
        # Need minimum data points
        if len(self.price_history) < self.bb.period:
            return None
        
        # Calculate Bollinger Bands
        prices = pd.Series(self.price_history)
        bb_data = self.bb.calculate(prices)
        signals = self.bb.generate_signals(bb_data)
        
        # Get latest signal
        latest_signal = signals.iloc[-1]
        
        return {
            'price': new_price,
            'upper_band': latest_signal['upper'],
            'middle_band': latest_signal['middle'],
            'lower_band': latest_signal['lower'],
            'signal': latest_signal['signal'],
            'percent_b': latest_signal['percent_b'],
            'bandwidth': latest_signal['bandwidth'],
            'squeeze': latest_signal['squeeze']
        }
    
    def should_buy(self, signal_data: Dict) -> bool:
        """Determine if should buy based on signal."""
        return signal_data['signal'] == 1 and self.position == 0
    
    def should_sell(self, signal_data: Dict) -> bool:
        """Determine if should sell based on signal."""
        return signal_data['signal'] == -1 and self.position > 0