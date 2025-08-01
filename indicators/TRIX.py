import pandas as pd
import numpy as np
from typing import Union, Tuple, Optional

class TRIXIndicator:
    """
    TRIX (Triple Exponential Average) Indicator implementation for algorithmic trading.
    
    TRIX is a momentum oscillator that:
    - Shows the percentage rate of change of a triple exponentially smoothed moving average
    - Oscillates around zero line
    - Positive values indicate upward momentum
    - Negative values indicate downward momentum
    - Zero line crossovers generate buy/sell signals
    """
    
    def __init__(self, period: int = 14, signal_period: int = 9):
        """
        Initialize TRIX indicator.
        
        Args:
            period (int): Period for triple EMA calculation (default: 14)
            signal_period (int): Period for signal line EMA (default: 9)
        """
        self.period = period
        self.signal_period = signal_period
    
    def calculate(self, prices: Union[pd.Series, list, np.array]) -> pd.DataFrame:
        """
        Calculate TRIX indicator values.
        
        Args:
            prices: Price series (typically closing prices)
            
        Returns:
            pd.DataFrame: DataFrame with TRIX, Signal line, and Histogram
        """
        # Convert to pandas Series if needed
        if not isinstance(prices, pd.Series):
            prices = pd.Series(prices)
        
        # Calculate triple exponential moving average
        ema1 = self._calculate_ema(prices, self.period)
        ema2 = self._calculate_ema(ema1, self.period)
        ema3 = self._calculate_ema(ema2, self.period)
        
        # Calculate TRIX as percentage rate of change
        trix = ((ema3 - ema3.shift(1)) / ema3.shift(1)) * 10000  # Multiply by 10000 for better scaling
        
        # Calculate signal line (EMA of TRIX)
        signal_line = self._calculate_ema(trix, self.signal_period)
        
        # Calculate histogram (TRIX - Signal)
        histogram = trix - signal_line
        
        # Create result DataFrame
        result = pd.DataFrame({
            'trix': trix,
            'signal': signal_line,
            'histogram': histogram,
            'ema1': ema1,
            'ema2': ema2,
            'ema3': ema3
        })
        
        return result
    
    def _calculate_ema(self, prices: pd.Series, period: int) -> pd.Series:
        """
        Calculate Exponential Moving Average.
        
        Args:
            prices: Price series
            period: EMA period
            
        Returns:
            pd.Series: EMA values
        """
        return prices.ewm(span=period, adjust=False).mean()
    
    def get_signals(self, trix_data: pd.DataFrame, 
                   use_signal_line: bool = True) -> pd.DataFrame:
        """
        Generate trading signals based on TRIX values.
        
        Args:
            trix_data: DataFrame from calculate() method
            use_signal_line: If True, use TRIX-Signal crossovers; if False, use zero-line crossovers
            
        Returns:
            pd.DataFrame: DataFrame with trading signals
        """
        signals = trix_data.copy()
        signals['signal'] = 0
        signals['position'] = 0
        
        if use_signal_line:
            # TRIX crossing above signal line = Buy
            # TRIX crossing below signal line = Sell
            trix_above_signal = signals['trix'] > signals['signal']
            trix_above_signal_prev = signals['trix'].shift(1) > signals['signal'].shift(1)
            
            # Buy signal: TRIX crosses above signal line
            buy_condition = trix_above_signal & ~trix_above_signal_prev
            # Sell signal: TRIX crosses below signal line
            sell_condition = ~trix_above_signal & trix_above_signal_prev
            
        else:
            # Zero line crossovers
            trix_above_zero = signals['trix'] > 0
            trix_above_zero_prev = signals['trix'].shift(1) > 0
            
            # Buy signal: TRIX crosses above zero
            buy_condition = trix_above_zero & ~trix_above_zero_prev
            # Sell signal: TRIX crosses below zero
            sell_condition = ~trix_above_zero & trix_above_zero_prev
        
        signals.loc[buy_condition, 'signal'] = 1
        signals.loc[sell_condition, 'signal'] = -1
        
        # Generate positions
        current_position = 0
        for i in range(len(signals)):
            if signals['signal'].iloc[i] == 1:
                current_position = 1
            elif signals['signal'].iloc[i] == -1:
                current_position = -1
            
            signals['position'].iloc[i] = current_position
        
        return signals
    
    def get_divergence_signals(self, prices: pd.Series, trix_data: pd.DataFrame, 
                              lookback_period: int = 20) -> pd.DataFrame:
        """
        Detect bullish and bearish divergences between price and TRIX.
        
        Args:
            prices: Price series
            trix_data: TRIX data from calculate()
            lookback_period: Period to look back for divergence detection
            
        Returns:
            pd.DataFrame: DataFrame with divergence signals
        """
        signals = trix_data.copy()
        signals['price'] = prices
        signals['bullish_divergence'] = False
        signals['bearish_divergence'] = False
        
        for i in range(lookback_period, len(signals)):
            # Get recent data
            recent_prices = signals['price'].iloc[i-lookback_period:i+1]
            recent_trix = signals['trix'].iloc[i-lookback_period:i+1]
            
            # Find local highs and lows
            price_high_idx = recent_prices.idxmax()
            price_low_idx = recent_prices.idxmin()
            trix_high_idx = recent_trix.idxmax()
            trix_low_idx = recent_trix.idxmin()
            
            # Bearish divergence: Price makes higher high, TRIX makes lower high
            if (price_high_idx == signals.index[i] and 
                recent_prices.iloc[-1] > recent_prices.iloc[0] and
                recent_trix.iloc[-1] < recent_trix.iloc[0]):
                signals.loc[signals.index[i], 'bearish_divergence'] = True
            
            # Bullish divergence: Price makes lower low, TRIX makes higher low
            if (price_low_idx == signals.index[i] and 
                recent_prices.iloc[-1] < recent_prices.iloc[0] and
                recent_trix.iloc[-1] > recent_trix.iloc[0]):
                signals.loc[signals.index[i], 'bullish_divergence'] = True
        
        return signals


class TRIXStrategy:
    """
    Complete trading strategy using TRIX indicator.
    """
    
    def __init__(self, period: int = 14, signal_period: int = 9, 
                 use_signal_line: bool = True, use_divergence: bool = False):
        """
        Initialize TRIX strategy.
        
        Args:
            period: TRIX calculation period
            signal_period: Signal line period
            use_signal_line: Use TRIX-Signal crossovers instead of zero-line
            use_divergence: Include divergence analysis
        """
        self.indicator = TRIXIndicator(period, signal_period)
        self.use_signal_line = use_signal_line
        self.use_divergence = use_divergence
    
    def generate_signals(self, prices: pd.Series) -> pd.DataFrame:
        """
        Generate complete trading signals.
        
        Args:
            prices: Price series
            
        Returns:
            pd.DataFrame: Complete signal DataFrame
        """
        # Calculate TRIX
        trix_data = self.indicator.calculate(prices)
        
        # Get basic signals
        signals = self.indicator.get_signals(trix_data, self.use_signal_line)
        
        # Add divergence signals if requested
        if self.use_divergence:
            div_signals = self.indicator.get_divergence_signals(prices, trix_data)
            signals['bullish_divergence'] = div_signals['bullish_divergence']
            signals['bearish_divergence'] = div_signals['bearish_divergence']
            
            # Modify signals based on divergence
            # Strengthen buy signals with bullish divergence
            bullish_div_buy = (signals['signal'] == 1) & signals['bullish_divergence']
            signals.loc[bullish_div_buy, 'signal'] = 2  # Strong buy
            
            # Strengthen sell signals with bearish divergence
            bearish_div_sell = (signals['signal'] == -1) & signals['bearish_divergence']
            signals.loc[bearish_div_sell, 'signal'] = -2  # Strong sell
        
        return signals
    
    def backtest(self, data: pd.DataFrame, initial_capital: float = 10000,
                 transaction_cost: float = 0.001) -> dict:
        """
        Backtest TRIX strategy.
        
        Args:
            data: DataFrame with 'close' column
            initial_capital: Starting capital
            transaction_cost: Transaction cost per trade (0.1% default)
            
        Returns:
            dict: Backtest results
        """
        # Generate signals
        signals = self.generate_signals(data['close'])
        
        # Calculate returns
        data['returns'] = data['close'].pct_change()
        
        # Apply transaction costs
        position_changes = signals['position'].diff().abs()
        transaction_costs = position_changes * transaction_cost
        
        # Calculate strategy returns
        signals['strategy_returns'] = (signals['position'].shift(1) * data['returns']) - transaction_costs
        
        # Calculate cumulative returns
        signals['cumulative_returns'] = (1 + signals['strategy_returns']).cumprod()
        signals['cumulative_market'] = (1 + data['returns']).cumprod()
        
        # Performance metrics
        total_return = signals['cumulative_returns'].iloc[-1] - 1
        market_return = signals['cumulative_market'].iloc[-1] - 1
        
        # Sharpe ratio (annualized)
        strategy_std = signals['strategy_returns'].std() * np.sqrt(252)
        strategy_mean = signals['strategy_returns'].mean() * 252
        sharpe_ratio = strategy_mean / strategy_std if strategy_std != 0 else 0
        
        # Maximum drawdown
        cumulative = signals['cumulative_returns']
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Win rate
        winning_trades = signals['strategy_returns'][signals['strategy_returns'] > 0]
        total_trades = len(signals[signals['strategy_returns'] != 0])
        win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0
        
        # Number of trades
        num_trades = (signals['position'].diff() != 0).sum()
        
        return {
            'total_return': total_return,
            'market_return': market_return,
            'excess_return': total_return - market_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'num_trades': num_trades,
            'signals': signals
        }


# Alternative implementation with different smoothing methods
class TRIXAdvanced:
    """
    Advanced TRIX implementation with different smoothing options.
    """
    
    def __init__(self, period: int = 14, smoothing_method: str = 'ema'):
        """
        Initialize advanced TRIX.
        
        Args:
            period: Smoothing period
            smoothing_method: 'ema', 'sma', or 'hull'
        """
        self.period = period
        self.smoothing_method = smoothing_method
    
    def calculate(self, prices: pd.Series) -> pd.Series:
        """Calculate TRIX with different smoothing methods."""
        if self.smoothing_method == 'ema':
            ma1 = prices.ewm(span=self.period).mean()
            ma2 = ma1.ewm(span=self.period).mean()
            ma3 = ma2.ewm(span=self.period).mean()
        elif self.smoothing_method == 'sma':
            ma1 = prices.rolling(self.period).mean()
            ma2 = ma1.rolling(self.period).mean()
            ma3 = ma2.rolling(self.period).mean()
        elif self.smoothing_method == 'hull':
            ma1 = self._hull_ma(prices, self.period)
            ma2 = self._hull_ma(ma1, self.period)
            ma3 = self._hull_ma(ma2, self.period)
        else:
            raise ValueError("Unsupported smoothing method")
        
        # Calculate TRIX
        trix = ((ma3 - ma3.shift(1)) / ma3.shift(1)) * 10000
        return trix
    
    def _hull_ma(self, prices: pd.Series, period: int) -> pd.Series:
        """Calculate Hull Moving Average."""
        half_period = period // 2
        sqrt_period = int(np.sqrt(period))
        
        wma_half = prices.rolling(half_period).apply(
            lambda x: np.average(x, weights=np.arange(1, len(x) + 1)), raw=True
        )
        wma_full = prices.rolling(period).apply(
            lambda x: np.average(x, weights=np.arange(1, len(x) + 1)), raw=True
        )
        
        hull_values = 2 * wma_half - wma_full
        hull_ma = hull_values.rolling(sqrt_period).apply(
            lambda x: np.average(x, weights=np.arange(1, len(x) + 1)), raw=True
        )
        
        return hull_ma


# Example usage and testing
if __name__ == "__main__":
    # Create sample data
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=500, freq='D')
    
    # Generate trending price data with noise
    trend = np.linspace(100, 150, 500)
    noise = np.cumsum(np.random.normal(0, 1, 500))
    prices = trend + noise
    
    data = pd.DataFrame({
        'date': dates,
        'close': prices
    })
    data.set_index('date', inplace=True)
    
    # Test basic TRIX strategy
    print("=== TRIX Strategy Results ===")
    
    # Test with signal line crossovers
    strategy1 = TRIXStrategy(period=14, signal_period=9, use_signal_line=True)
    results1 = strategy1.backtest(data)
    
    print("Signal Line Crossover Strategy:")
    print(f"Total Return: {results1['total_return']:.2%}")
    print(f"Market Return: {results1['market_return']:.2%}")
    print(f"Excess Return: {results1['excess_return']:.2%}")
    print(f"Sharpe Ratio: {results1['sharpe_ratio']:.2f}")
    print(f"Max Drawdown: {results1['max_drawdown']:.2%}")
    print(f"Win Rate: {results1['win_rate']:.2%}")
    print(f"Number of Trades: {results1['num_trades']}")
    
    print("\n" + "="*50)
    
    # Test with zero line crossovers
    strategy2 = TRIXStrategy(period=14, signal_period=9, use_signal_line=False)
    results2 = strategy2.backtest(data)
    
    print("Zero Line Crossover Strategy:")
    print(f"Total Return: {results2['total_return']:.2%}")
    print(f"Market Return: {results2['market_return']:.2%}")
    print(f"Excess Return: {results2['excess_return']:.2%}")
    print(f"Sharpe Ratio: {results2['sharpe_ratio']:.2f}")
    print(f"Max Drawdown: {results2['max_drawdown']:.2%}")
    print(f"Win Rate: {results2['win_rate']:.2%}")
    print(f"Number of Trades: {results2['num_trades']}")
    
    # Show recent signals
    print("\n=== Recent TRIX Values and Signals ===")
    recent_signals = results1['signals'].tail(10)[['trix', 'signal', 'histogram', 'position']]
    print(recent_signals.round(4))
    
    # Test advanced TRIX with Hull MA
    print("\n=== Advanced TRIX with Hull MA ===")
    trix_hull = TRIXAdvanced(period=14, smoothing_method='hull')
    hull_trix = trix_hull.calculate(data['close'])
    print("Hull MA TRIX (last 5 values):")
    print(hull_trix.tail().round(4))