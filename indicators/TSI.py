import pandas as pd
import numpy as np
from typing import Tuple, Optional

class TSIIndicator:
    """
    True Strength Index (TSI) Indicator
    
    TSI is a momentum oscillator that uses double-smoothed price changes
    to filter out price noise and identify trend direction and momentum.
    
    Formula:
    TSI = 100 * (Double Smoothed PC / Double Smoothed |PC|)
    
    Where PC = Price Change (current close - previous close)
    """
    
    def __init__(self, long_period: int = 25, short_period: int = 13, signal_period: int = 7):
        """
        Initialize TSI parameters
        
        Args:
            long_period: First smoothing period (default: 25)
            short_period: Second smoothing period (default: 13)  
            signal_period: Signal line smoothing period (default: 7)
        """
        self.long_period = long_period
        self.short_period = short_period
        self.signal_period = signal_period
        
    def _ema(self, data: pd.Series, period: int) -> pd.Series:
        """Calculate Exponential Moving Average"""
        return data.ewm(span=period, adjust=False).mean()
    
    def calculate(self, prices: pd.Series) -> Tuple[pd.Series, pd.Series]:
        """
        Calculate TSI and Signal line
        
        Args:
            prices: Series of closing prices
            
        Returns:
            Tuple of (TSI values, Signal line values)
        """
        # Calculate price changes
        price_changes = prices.diff()
        
        # Calculate absolute price changes
        abs_price_changes = price_changes.abs()
        
        # First smoothing (long period)
        smoothed_pc = self._ema(price_changes, self.long_period)
        smoothed_abs_pc = self._ema(abs_price_changes, self.long_period)
        
        # Second smoothing (short period)
        double_smoothed_pc = self._ema(smoothed_pc, self.short_period)
        double_smoothed_abs_pc = self._ema(smoothed_abs_pc, self.short_period)
        
        # Calculate TSI
        tsi = 100 * (double_smoothed_pc / double_smoothed_abs_pc)
        
        # Calculate signal line
        signal = self._ema(tsi, self.signal_period)
        
        return tsi, signal
    
    def generate_signals(self, prices: pd.Series) -> pd.DataFrame:
        """
        Generate trading signals based on TSI
        
        Args:
            prices: Series of closing prices
            
        Returns:
            DataFrame with TSI, Signal, and trading signals
        """
        tsi, signal = self.calculate(prices)
        
        df = pd.DataFrame({
            'price': prices,
            'tsi': tsi,
            'signal': signal
        })
        
        # Generate signals
        df['tsi_cross_above'] = (df['tsi'] > df['signal']) & (df['tsi'].shift(1) <= df['signal'].shift(1))
        df['tsi_cross_below'] = (df['tsi'] < df['signal']) & (df['tsi'].shift(1) >= df['signal'].shift(1))
        
        # Zero line crosses
        df['tsi_above_zero'] = df['tsi'] > 0
        df['tsi_below_zero'] = df['tsi'] < 0
        df['zero_cross_up'] = (df['tsi'] > 0) & (df['tsi'].shift(1) <= 0)
        df['zero_cross_down'] = (df['tsi'] < 0) & (df['tsi'].shift(1) >= 0)
        
        # Momentum signals
        df['bullish_momentum'] = (df['tsi'] > df['signal']) & (df['tsi'] > 0)
        df['bearish_momentum'] = (df['tsi'] < df['signal']) & (df['tsi'] < 0)
        
        # Overbought/Oversold levels (typical thresholds)
        df['overbought'] = df['tsi'] > 25
        df['oversold'] = df['tsi'] < -25
        
        return df

# Example usage and backtesting framework
class TSITradingBot:
    """
    Simple trading bot using TSI signals
    """
    
    def __init__(self, long_period: int = 25, short_period: int = 13, signal_period: int = 7):
        self.tsi = TSIIndicator(long_period, short_period, signal_period)
        self.position = 0  # 0: no position, 1: long, -1: short
        self.trades = []
        
    def backtest(self, prices: pd.Series, strategy: str = 'crossover') -> dict:
        """
        Backtest TSI strategy
        
        Args:
            prices: Historical price data
            strategy: 'crossover', 'zero_cross', or 'momentum'
            
        Returns:
            Dictionary with backtest results
        """
        signals_df = self.tsi.generate_signals(prices)
        
        trades = []
        position = 0
        entry_price = 0
        
        for i in range(1, len(signals_df)):
            current_price = signals_df['price'].iloc[i]
            
            if strategy == 'crossover':
                # TSI crosses above signal line - Buy
                if signals_df['tsi_cross_above'].iloc[i] and position != 1:
                    if position == -1:  # Close short
                        pnl = entry_price - current_price
                        trades.append({'type': 'close_short', 'price': current_price, 'pnl': pnl})
                    # Open long
                    position = 1
                    entry_price = current_price
                    trades.append({'type': 'buy', 'price': current_price, 'pnl': 0})
                
                # TSI crosses below signal line - Sell
                elif signals_df['tsi_cross_below'].iloc[i] and position != -1:
                    if position == 1:  # Close long
                        pnl = current_price - entry_price
                        trades.append({'type': 'close_long', 'price': current_price, 'pnl': pnl})
                    # Open short
                    position = -1
                    entry_price = current_price
                    trades.append({'type': 'sell', 'price': current_price, 'pnl': 0})
            
            elif strategy == 'zero_cross':
                # TSI crosses above zero - Buy
                if signals_df['zero_cross_up'].iloc[i] and position != 1:
                    if position == -1:
                        pnl = entry_price - current_price
                        trades.append({'type': 'close_short', 'price': current_price, 'pnl': pnl})
                    position = 1
                    entry_price = current_price
                    trades.append({'type': 'buy', 'price': current_price, 'pnl': 0})
                    
                # TSI crosses below zero - Sell
                elif signals_df['zero_cross_down'].iloc[i] and position != -1:
                    if position == 1:
                        pnl = current_price - entry_price
                        trades.append({'type': 'close_long', 'price': current_price, 'pnl': pnl})
                    position = -1
                    entry_price = current_price
                    trades.append({'type': 'sell', 'price': current_price, 'pnl': 0})
        
        # Calculate performance metrics
        trade_pnls = [t['pnl'] for t in trades if t['pnl'] != 0]
        total_return = sum(trade_pnls)
        win_rate = len([p for p in trade_pnls if p > 0]) / len(trade_pnls) if trade_pnls else 0
        
        return {
            'total_return': total_return,
            'num_trades': len(trade_pnls),
            'win_rate': win_rate,
            'trades': trades,
            'signals_df': signals_df
        }

# Example usage
def example_usage():
    """Example of how to use the TSI indicator"""
    
    # Generate sample price data (you would use real market data)
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=500, freq='D')
    prices = pd.Series(100 + np.cumsum(np.random.randn(500) * 0.5), index=dates)
    
    # Initialize TSI indicator
    tsi_indicator = TSIIndicator(long_period=25, short_period=13, signal_period=7)
    
    # Calculate TSI values
    tsi_values, signal_values = tsi_indicator.calculate(prices)
    
    # Generate trading signals
    signals_df = tsi_indicator.generate_signals(prices)
    
    # Print recent signals
    print("Recent TSI signals:")
    print(signals_df[['price', 'tsi', 'signal', 'tsi_cross_above', 'tsi_cross_below']].tail(10))
    
    # Run backtest
    bot = TSITradingBot()
    results = bot.backtest(prices, strategy='crossover')
    
    print(f"\nBacktest Results:")
    print(f"Total Return: {results['total_return']:.2f}")
    print(f"Number of Trades: {results['num_trades']}")
    print(f"Win Rate: {results['win_rate']:.2%}")
    
    return signals_df, results

if __name__ == "__main__":
    example_usage()