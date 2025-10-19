import pandas as pd
import numpy as np
from typing import Tuple, Optional

class VortexIndicator:
    """
    Vortex Indicator (VI)
    
    The Vortex Indicator identifies the start of a new trend or the continuation 
    of an existing trend by measuring the relationship between closing prices 
    and the trading range (high-low).
    
    VI+ = Sum of |High[i] - Low[i-1]| over n periods / Sum of True Range over n periods
    VI- = Sum of |Low[i] - High[i-1]| over n periods / Sum of True Range over n periods
    
    Created by Etienne Botes and Douglas Siepman
    """
    
    def __init__(self, period: int = 14):
        """
        Initialize Vortex Indicator parameters
        
        Args:
            period: Period for calculation (default: 14)
        """
        self.period = period
        
    def _true_range(self, high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
        """
        Calculate True Range
        TR = max(H-L, |H-C_prev|, |L-C_prev|)
        """
        hl = high - low
        hc = np.abs(high - close.shift(1))
        lc = np.abs(low - close.shift(1))
        
        tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
        return tr
    
    def calculate(self, high: pd.Series, low: pd.Series, close: pd.Series) -> Tuple[pd.Series, pd.Series]:
        """
        Calculate Vortex Indicator values
        
        Args:
            high: Series of high prices
            low: Series of low prices
            close: Series of closing prices
            
        Returns:
            Tuple of (VI+, VI-) values
        """
        # Calculate vortex movements
        vm_plus = np.abs(high - low.shift(1))  # |High[i] - Low[i-1]|
        vm_minus = np.abs(low - high.shift(1))  # |Low[i] - High[i-1]|
        
        # Calculate true range
        tr = self._true_range(high, low, close)
        
        # Sum over the period
        sum_vm_plus = vm_plus.rolling(window=self.period).sum()
        sum_vm_minus = vm_minus.rolling(window=self.period).sum()
        sum_tr = tr.rolling(window=self.period).sum()
        
        # Calculate VI+ and VI-
        vi_plus = sum_vm_plus / sum_tr
        vi_minus = sum_vm_minus / sum_tr
        
        return vi_plus, vi_minus
    
    def generate_signals(self, high: pd.Series, low: pd.Series, close: pd.Series) -> pd.DataFrame:
        """
        Generate trading signals based on Vortex Indicator
        
        Args:
            high: Series of high prices
            low: Series of low prices
            close: Series of closing prices
            
        Returns:
            DataFrame with VI values and trading signals
        """
        vi_plus, vi_minus = self.calculate(high, low, close)
        
        df = pd.DataFrame({
            'high': high,
            'low': low,
            'close': close,
            'vi_plus': vi_plus,
            'vi_minus': vi_minus
        })
        
        # Calculate VI difference and ratio
        df['vi_diff'] = df['vi_plus'] - df['vi_minus']
        df['vi_ratio'] = df['vi_plus'] / df['vi_minus']
        
        # Primary signals - crossovers
        df['vi_bullish_cross'] = (df['vi_plus'] > df['vi_minus']) & (df['vi_plus'].shift(1) <= df['vi_minus'].shift(1))
        df['vi_bearish_cross'] = (df['vi_plus'] < df['vi_minus']) & (df['vi_plus'].shift(1) >= df['vi_minus'].shift(1))
        
        # Trend strength signals
        df['strong_uptrend'] = (df['vi_plus'] > df['vi_minus']) & (df['vi_plus'] > 1.0)
        df['strong_downtrend'] = (df['vi_minus'] > df['vi_plus']) & (df['vi_minus'] > 1.0)
        
        # Momentum signals
        df['bullish_momentum'] = (df['vi_plus'] > df['vi_minus']) & (df['vi_diff'] > df['vi_diff'].shift(1))
        df['bearish_momentum'] = (df['vi_minus'] > df['vi_plus']) & (df['vi_diff'] < df['vi_diff'].shift(1))
        
        # Consolidation/sideways market
        df['consolidation'] = (np.abs(df['vi_plus'] - df['vi_minus']) < 0.02) & (df['vi_plus'] < 1.0) & (df['vi_minus'] < 1.0)
        
        # Extreme readings (potential reversal zones)
        df['vi_plus_extreme'] = df['vi_plus'] > 1.3
        df['vi_minus_extreme'] = df['vi_minus'] > 1.3
        
        return df

class VortexTradingBot:
    """
    Advanced trading bot using Vortex Indicator signals
    """
    
    def __init__(self, period: int = 14, risk_per_trade: float = 0.02):
        """
        Initialize the trading bot
        
        Args:
            period: VI calculation period
            risk_per_trade: Risk percentage per trade (default: 2%)
        """
        self.vi = VortexIndicator(period)
        self.risk_per_trade = risk_per_trade
        self.position = 0  # 0: no position, 1: long, -1: short
        self.trades = []
        
    def calculate_position_size(self, account_balance: float, entry_price: float, stop_loss: float) -> float:
        """
        Calculate position size based on risk management
        
        Args:
            account_balance: Current account balance
            entry_price: Entry price for the trade
            stop_loss: Stop loss price
            
        Returns:
            Position size in units
        """
        risk_amount = account_balance * self.risk_per_trade
        price_risk = abs(entry_price - stop_loss)
        
        if price_risk == 0:
            return 0
            
        position_size = risk_amount / price_risk
        return position_size
    
    def backtest(self, high: pd.Series, low: pd.Series, close: pd.Series, 
                 strategy: str = 'crossover', account_balance: float = 10000) -> dict:
        """
        Backtest Vortex Indicator strategy
        
        Args:
            high: High prices
            low: Low prices  
            close: Close prices
            strategy: Trading strategy ('crossover', 'trend_strength', 'momentum')
            account_balance: Starting account balance
            
        Returns:
            Dictionary with backtest results
        """
        signals_df = self.vi.generate_signals(high, low, close)
        
        trades = []
        position = 0
        entry_price = 0
        stop_loss = 0
        current_balance = account_balance
        
        for i in range(1, len(signals_df)):
            current_high = signals_df['high'].iloc[i]
            current_low = signals_df['low'].iloc[i]
            current_close = signals_df['close'].iloc[i]
            
            # Calculate ATR for stop loss (using 2-period ATR approximation)
            atr = (current_high - current_low + abs(current_high - signals_df['close'].iloc[i-1]) + 
                   abs(current_low - signals_df['close'].iloc[i-1])) / 3
            
            if strategy == 'crossover':
                # VI+ crosses above VI- - Buy signal
                if signals_df['vi_bullish_cross'].iloc[i] and position != 1:
                    if position == -1:  # Close short position
                        pnl = (entry_price - current_close) * abs(position)
                        current_balance += pnl
                        trades.append({
                            'type': 'close_short', 
                            'price': current_close, 
                            'pnl': pnl,
                            'balance': current_balance
                        })
                    
                    # Open long position
                    entry_price = current_close
                    stop_loss = current_close - (2 * atr)  # 2 ATR stop loss
                    position_size = self.calculate_position_size(current_balance, entry_price, stop_loss)
                    position = position_size
                    
                    trades.append({
                        'type': 'buy', 
                        'price': entry_price, 
                        'stop_loss': stop_loss,
                        'position_size': position_size,
                        'pnl': 0
                    })
                
                # VI+ crosses below VI- - Sell signal
                elif signals_df['vi_bearish_cross'].iloc[i] and position != -1:
                    if position > 0:  # Close long position
                        pnl = (current_close - entry_price) * position
                        current_balance += pnl
                        trades.append({
                            'type': 'close_long', 
                            'price': current_close, 
                            'pnl': pnl,
                            'balance': current_balance
                        })
                    
                    # Open short position
                    entry_price = current_close
                    stop_loss = current_close + (2 * atr)  # 2 ATR stop loss
                    position_size = self.calculate_position_size(current_balance, entry_price, stop_loss)
                    position = -position_size
                    
                    trades.append({
                        'type': 'sell', 
                        'price': entry_price, 
                        'stop_loss': stop_loss,
                        'position_size': position_size,
                        'pnl': 0
                    })
            
            elif strategy == 'trend_strength':
                # Strong uptrend - Buy
                if signals_df['strong_uptrend'].iloc[i] and not signals_df['strong_uptrend'].iloc[i-1] and position <= 0:
                    if position < 0:  # Close short
                        pnl = (entry_price - current_close) * abs(position)
                        current_balance += pnl
                        trades.append({
                            'type': 'close_short', 
                            'price': current_close, 
                            'pnl': pnl,
                            'balance': current_balance
                        })
                    
                    # Open long
                    entry_price = current_close
                    stop_loss = current_close - (2 * atr)
                    position_size = self.calculate_position_size(current_balance, entry_price, stop_loss)
                    position = position_size
                    
                    trades.append({
                        'type': 'buy', 
                        'price': entry_price, 
                        'stop_loss': stop_loss,
                        'position_size': position_size,
                        'pnl': 0
                    })
                
                # Strong downtrend - Sell
                elif signals_df['strong_downtrend'].iloc[i] and not signals_df['strong_downtrend'].iloc[i-1] and position >= 0:
                    if position > 0:  # Close long
                        pnl = (current_close - entry_price) * position
                        current_balance += pnl
                        trades.append({
                            'type': 'close_long', 
                            'price': current_close, 
                            'pnl': pnl,
                            'balance': current_balance
                        })
                    
                    # Open short
                    entry_price = current_close
                    stop_loss = current_close + (2 * atr)
                    position_size = self.calculate_position_size(current_balance, entry_price, stop_loss)
                    position = -position_size
                    
                    trades.append({
                        'type': 'sell', 
                        'price': entry_price, 
                        'stop_loss': stop_loss,
                        'position_size': position_size,
                        'pnl': 0
                    })
            
            # Check stop loss
            if position > 0 and current_low <= stop_loss:  # Long position stopped out
                pnl = (stop_loss - entry_price) * position
                current_balance += pnl
                trades.append({
                    'type': 'stop_loss_long', 
                    'price': stop_loss, 
                    'pnl': pnl,
                    'balance': current_balance
                })
                position = 0
                
            elif position < 0 and current_high >= stop_loss:  # Short position stopped out
                pnl = (entry_price - stop_loss) * abs(position)
                current_balance += pnl
                trades.append({
                    'type': 'stop_loss_short', 
                    'price': stop_loss, 
                    'pnl': pnl,
                    'balance': current_balance
                })
                position = 0
        
        # Calculate performance metrics
        trade_pnls = [t['pnl'] for t in trades if 'pnl' in t and t['pnl'] != 0]
        total_return = current_balance - account_balance
        total_return_pct = (total_return / account_balance) * 100
        
        winning_trades = [p for p in trade_pnls if p > 0]
        losing_trades = [p for p in trade_pnls if p < 0]
        
        win_rate = len(winning_trades) / len(trade_pnls) if trade_pnls else 0
        avg_win = np.mean(winning_trades) if winning_trades else 0
        avg_loss = np.mean(losing_trades) if losing_trades else 0
        profit_factor = abs(sum(winning_trades) / sum(losing_trades)) if losing_trades else float('inf')
        
        return {
            'total_return': total_return,
            'total_return_pct': total_return_pct,
            'final_balance': current_balance,
            'num_trades': len(trade_pnls),
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'trades': trades,
            'signals_df': signals_df
        }

# Real-time trading class
class VortexLiveTrader:
    """
    Live trading implementation using Vortex Indicator
    """
    
    def __init__(self, period: int = 14):
        self.vi = VortexIndicator(period)
        self.data_buffer = pd.DataFrame()
        self.current_position = 0
        
    def update_data(self, timestamp, high: float, low: float, close: float):
        """
        Update data buffer with new price data
        
        Args:
            timestamp: Current timestamp
            high: High price
            low: Low price  
            close: Close price
        """
        new_row = pd.DataFrame({
            'timestamp': [timestamp],
            'high': [high],
            'low': [low], 
            'close': [close]
        })
        
        self.data_buffer = pd.concat([self.data_buffer, new_row], ignore_index=True)
        
        # Keep only necessary data (period + 1 for calculations)
        if len(self.data_buffer) > self.vi.period + 10:
            self.data_buffer = self.data_buffer.tail(self.vi.period + 10).reset_index(drop=True)
    
    def get_current_signal(self) -> dict:
        """
        Get current trading signal based on latest data
        
        Returns:
            Dictionary with current VI values and signals
        """
        if len(self.data_buffer) < self.vi.period + 1:
            return {'signal': 'insufficient_data'}
        
        signals_df = self.vi.generate_signals(
            self.data_buffer['high'], 
            self.data_buffer['low'], 
            self.data_buffer['close']
        )
        
        latest = signals_df.iloc[-1]
        
        return {
            'vi_plus': latest['vi_plus'],
            'vi_minus': latest['vi_minus'],
            'vi_diff': latest['vi_diff'],
            'bullish_cross': latest['vi_bullish_cross'],
            'bearish_cross': latest['vi_bearish_cross'],
            'strong_uptrend': latest['strong_uptrend'],
            'strong_downtrend': latest['strong_downtrend'],
            'consolidation': latest['consolidation']
        }

# Example usage
def example_usage():
    """Example of how to use the Vortex Indicator"""
    
    # Generate sample OHLC data
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=500, freq='D')
    
    # Create realistic OHLC data
    close_prices = 100 + np.cumsum(np.random.randn(500) * 0.8)
    high_prices = close_prices + np.random.uniform(0.5, 3, 500)
    low_prices = close_prices - np.random.uniform(0.5, 3, 500)
    
    # Ensure High >= Close >= Low
    high_prices = np.maximum(high_prices, close_prices)
    low_prices = np.minimum(low_prices, close_prices)
    
    high = pd.Series(high_prices, index=dates)
    low = pd.Series(low_prices, index=dates)
    close = pd.Series(close_prices, index=dates)
    
    # Initialize Vortex Indicator
    vi_indicator = VortexIndicator(period=14)
    
    # Calculate VI values
    vi_plus, vi_minus = vi_indicator.calculate(high, low, close)
    
    # Generate trading signals
    signals_df = vi_indicator.generate_signals(high, low, close)
    
    print("Recent Vortex Indicator signals:")
    print(signals_df[['close', 'vi_plus', 'vi_minus', 'vi_bullish_cross', 'vi_bearish_cross', 'strong_uptrend']].tail(10))
    
    # Run backtest
    bot = VortexTradingBot(period=14)
    results = bot.backtest(high, low, close, strategy='crossover')
    
    print(f"\nBacktest Results:")
    print(f"Total Return: ${results['total_return']:.2f} ({results['total_return_pct']:.2f}%)")
    print(f"Final Balance: ${results['final_balance']:.2f}")
    print(f"Number of Trades: {results['num_trades']}")
    print(f"Win Rate: {results['win_rate']:.2%}")
    print(f"Average Win: ${results['avg_win']:.2f}")
    print(f"Average Loss: ${results['avg_loss']:.2f}")
    print(f"Profit Factor: {results['profit_factor']:.2f}")
    
    return signals_df, results

if __name__ == "__main__":
    example_usage()