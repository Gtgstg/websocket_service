import pandas as pd
import numpy as np
from typing import Union, Tuple

class DeMarkerIndicator:
    """
    DeMarker Indicator implementation for algorithmic trading.
    
    The DeMarker oscillates between 0 and 1:
    - Values above 0.7 typically indicate overbought conditions
    - Values below 0.3 typically indicate oversold conditions
    """
    
    def __init__(self, period: int = 14):
        """
        Initialize DeMarker indicator.
        
        Args:
            period (int): Period for DeMarker calculation (default: 14)
        """
        self.period = period
    
    def calculate(self, high: Union[pd.Series, list, np.array], 
                  low: Union[pd.Series, list, np.array]) -> pd.Series:
        """
        Calculate DeMarker indicator values.
        
        Args:
            high: High prices series
            low: Low prices series
            
        Returns:
            pd.Series: DeMarker values
        """
        # Convert to pandas Series if needed
        if not isinstance(high, pd.Series):
            high = pd.Series(high)
        if not isinstance(low, pd.Series):
            low = pd.Series(low)
        
        # Calculate DeMax and DeMin
        demax = self._calculate_demax(high)
        demin = self._calculate_demin(low)
        
        # Calculate moving averages
        demax_ma = demax.rolling(window=self.period).mean()
        demin_ma = demin.rolling(window=self.period).mean()
        
        # Calculate DeMarker
        demarker = demax_ma / (demax_ma + demin_ma)
        
        return demarker
    
    def _calculate_demax(self, high: pd.Series) -> pd.Series:
        """Calculate DeMax values."""
        demax = pd.Series(index=high.index, dtype=float)
        
        for i in range(1, len(high)):
            if high.iloc[i] > high.iloc[i-1]:
                demax.iloc[i] = high.iloc[i] - high.iloc[i-1]
            else:
                demax.iloc[i] = 0
        
        demax.iloc[0] = 0  # First value is always 0
        return demax
    
    def _calculate_demin(self, low: pd.Series) -> pd.Series:
        """Calculate DeMin values."""
        demin = pd.Series(index=low.index, dtype=float)
        
        for i in range(1, len(low)):
            if low.iloc[i] < low.iloc[i-1]:
                demin.iloc[i] = low.iloc[i-1] - low.iloc[i]
            else:
                demin.iloc[i] = 0
        
        demin.iloc[0] = 0  # First value is always 0
        return demin
    
    def get_signals(self, demarker_values: pd.Series, 
                   overbought: float = 0.7, oversold: float = 0.3) -> pd.DataFrame:
        """
        Generate trading signals based on DeMarker values.
        
        Args:
            demarker_values: DeMarker indicator values
            overbought: Overbought threshold (default: 0.7)
            oversold: Oversold threshold (default: 0.3)
            
        Returns:
            pd.DataFrame: DataFrame with signals
        """
        signals = pd.DataFrame(index=demarker_values.index)
        signals['demarker'] = demarker_values
        signals['signal'] = 0
        signals['position'] = 0
        
        # Generate signals
        signals.loc[demarker_values <= oversold, 'signal'] = 1  # Buy signal
        signals.loc[demarker_values >= overbought, 'signal'] = -1  # Sell signal
        
        # Generate positions (1 for long, -1 for short, 0 for neutral)
        current_position = 0
        for i in range(len(signals)):
            if signals['signal'].iloc[i] == 1 and current_position <= 0:
                current_position = 1  # Enter long
            elif signals['signal'].iloc[i] == -1 and current_position >= 0:
                current_position = -1  # Enter short
            
            signals['position'].iloc[i] = current_position
        
        return signals


# Alternative vectorized implementation for better performance
class DeMarkerVectorized:
    """
    Vectorized DeMarker implementation for better performance on large datasets.
    """
    
    def __init__(self, period: int = 14):
        self.period = period
    
    def calculate(self, high: Union[pd.Series, np.array], 
                  low: Union[pd.Series, np.array]) -> np.array:
        """
        Vectorized DeMarker calculation.
        
        Args:
            high: High prices
            low: Low prices
            
        Returns:
            np.array: DeMarker values
        """
        # Convert to numpy arrays
        high = np.array(high)
        low = np.array(low)
        
        # Calculate DeMax
        high_diff = np.diff(high)
        demax = np.where(high_diff > 0, high_diff, 0)
        demax = np.concatenate([[0], demax])  # Add 0 for first value
        
        # Calculate DeMin
        low_diff = np.diff(low)
        demin = np.where(low_diff < 0, -low_diff, 0)
        demin = np.concatenate([[0], demin])  # Add 0 for first value
        
        # Calculate moving averages using pandas for convenience
        demax_series = pd.Series(demax)
        demin_series = pd.Series(demin)
        
        demax_ma = demax_series.rolling(window=self.period).mean().values
        demin_ma = demin_series.rolling(window=self.period).mean().values
        
        # Calculate DeMarker
        with np.errstate(divide='ignore', invalid='ignore'):
            demarker = demax_ma / (demax_ma + demin_ma)
            demarker = np.where(np.isnan(demarker), 0.5, demarker)  # Handle NaN values
        
        return demarker


# Example usage and backtesting framework
class DeMarkerStrategy:
    """
    Complete trading strategy using DeMarker indicator.
    """
    
    def __init__(self, period: int = 14, overbought: float = 0.7, 
                 oversold: float = 0.3):
        self.indicator = DeMarkerIndicator(period)
        self.overbought = overbought
        self.oversold = oversold
    
    def backtest(self, data: pd.DataFrame, initial_capital: float = 10000) -> dict:
        """
        Simple backtest of DeMarker strategy.
        
        Args:
            data: DataFrame with 'high', 'low', 'close' columns
            initial_capital: Starting capital
            
        Returns:
            dict: Backtest results
        """
        # Calculate DeMarker
        demarker = self.indicator.calculate(data['high'], data['low'])
        
        # Generate signals
        signals = self.indicator.get_signals(demarker, self.overbought, self.oversold)
        
        # Calculate returns
        data['returns'] = data['close'].pct_change()
        signals['strategy_returns'] = signals['position'].shift(1) * data['returns']
        
        # Calculate cumulative returns
        signals['cumulative_returns'] = (1 + signals['strategy_returns']).cumprod()
        signals['cumulative_market'] = (1 + data['returns']).cumprod()
        
        # Calculate performance metrics
        total_return = signals['cumulative_returns'].iloc[-1] - 1
        market_return = signals['cumulative_market'].iloc[-1] - 1
        
        # Calculate Sharpe ratio (assuming 252 trading days)
        strategy_std = signals['strategy_returns'].std() * np.sqrt(252)
        strategy_mean = signals['strategy_returns'].mean() * 252
        sharpe_ratio = strategy_mean / strategy_std if strategy_std != 0 else 0
        
        # Calculate maximum drawdown
        cumulative = signals['cumulative_returns']
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()
        
        return {
            'total_return': total_return,
            'market_return': market_return,
            'excess_return': total_return - market_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'signals': signals,
            'demarker_values': demarker
        }


# Example usage
if __name__ == "__main__":
    # Create sample data
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=1000, freq='D')
    
    # Generate random walk price data
    price = 100
    prices = []
    for _ in range(1000):
        price += np.random.normal(0, 1)
        prices.append(price)
    
    # Create OHLC data
    data = pd.DataFrame({
        'date': dates,
        'close': prices,
        'high': np.array(prices) + np.random.uniform(0, 2, 1000),
        'low': np.array(prices) - np.random.uniform(0, 2, 1000)
    })
    
    # Initialize strategy
    strategy = DeMarkerStrategy(period=14, overbought=0.7, oversold=0.3)
    
    # Run backtest
    results = strategy.backtest(data)
    
    print("DeMarker Strategy Results:")
    print(f"Total Return: {results['total_return']:.2%}")
    print(f"Market Return: {results['market_return']:.2%}")
    print(f"Excess Return: {results['excess_return']:.2%}")
    print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
    print(f"Max Drawdown: {results['max_drawdown']:.2%}")
    
    # Show recent signals
    print("\nRecent DeMarker values and signals:")
    recent_data = results['signals'].tail(10)[['demarker', 'signal', 'position']]
    print(recent_data.round(3))