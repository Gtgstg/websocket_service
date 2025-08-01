"""
Keltner Channels Indicator Module
Provides classes and functions for calculating Keltner Channels and related signals.
"""
import pandas as pd
import numpy as np
from typing import Tuple, Dict, List, Optional
import matplotlib.pyplot as plt
from dataclasses import dataclass
from enum import Enum

class SignalType(Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"

@dataclass
class TradingSignal:
    timestamp: pd.Timestamp
    signal: SignalType
    price: float
    reason: str
    confidence: float = 0.0

class KeltnerChannels:
    """
    Keltner Channels Technical Indicator
    
    Keltner Channels consist of:
    - Middle Line: Exponential Moving Average (EMA) of price
    - Upper Channel: Middle Line + (multiplier * Average True Range)
    - Lower Channel: Middle Line - (multiplier * Average True Range)
    """
    
    def __init__(self, period: int = 20, multiplier: float = 2.0, ema_period: int = 20):
        self.period = period
        self.multiplier = multiplier
        self.ema_period = ema_period
    
    def calculate_atr(self, high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
        """Calculate Average True Range"""
        tr1 = high - low
        tr2 = np.abs(high - close.shift(1))
        tr3 = np.abs(low - close.shift(1))
        
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return true_range.rolling(window=self.period).mean()
    
    def calculate_ema(self, series: pd.Series, period: int) -> pd.Series:
        """Calculate Exponential Moving Average"""
        return series.ewm(span=period, adjust=False).mean()
    
    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Keltner Channels
        
        Args:
            df: DataFrame with columns ['high', 'low', 'close']
        
        Returns:
            DataFrame with Keltner Channel values
        """
        result = df.copy()
        
        # Calculate components
        result['atr'] = self.calculate_atr(df['high'], df['low'], df['close'])
        result['middle_line'] = self.calculate_ema(df['close'], self.ema_period)
        
        # Calculate channels
        result['upper_channel'] = result['middle_line'] + (self.multiplier * result['atr'])
        result['lower_channel'] = result['middle_line'] - (self.multiplier * result['atr'])
        
        # Calculate channel width and position
        result['channel_width'] = result['upper_channel'] - result['lower_channel']
        result['price_position'] = (df['close'] - result['lower_channel']) / result['channel_width']
        
        return result

class KeltnerTradingStrategy:
    """
    Keltner Channels Trading Strategy
    
    Trading Rules:
    1. BUY when price breaks above upper channel (breakout)
    2. SELL when price breaks below lower channel (breakdown)
    3. Mean reversion: BUY near lower channel, SELL near upper channel
    4. Trend following: Follow breakouts with momentum confirmation
    """
    
    def __init__(self, 
                 keltner_period: int = 20,
                 keltner_multiplier: float = 2.0,
                 ema_period: int = 20,
                 strategy_type: str = "breakout",  # "breakout" or "mean_reversion"
                 volume_confirmation: bool = True):
        
        self.keltner = KeltnerChannels(keltner_period, keltner_multiplier, ema_period)
        self.strategy_type = strategy_type
        self.volume_confirmation = volume_confirmation
        self.position = 0  # 0: no position, 1: long, -1: short
        self.signals = []
        
    def detect_breakout(self, df: pd.DataFrame, idx: int) -> Optional[TradingSignal]:
        """Detect breakout signals"""
        current = df.iloc[idx]
        prev = df.iloc[idx - 1] if idx > 0 else None
        
        if prev is None:
            return None
            
        # Upper channel breakout (BUY signal)
        if (current['close'] > current['upper_channel'] and 
            prev['close'] <= prev['upper_channel']):
            
            confidence = min(1.0, (current['close'] - current['upper_channel']) / current['atr'] * 0.5)
            
            return TradingSignal(
                timestamp=current.name,
                signal=SignalType.BUY,
                price=current['close'],
                reason="Upper channel breakout",
                confidence=confidence
            )
        
        # Lower channel breakdown (SELL signal)
        elif (current['close'] < current['lower_channel'] and 
              prev['close'] >= prev['lower_channel']):
            
            confidence = min(1.0, (current['lower_channel'] - current['close']) / current['atr'] * 0.5)
            
            return TradingSignal(
                timestamp=current.name,
                signal=SignalType.SELL,
                price=current['close'],
                reason="Lower channel breakdown",
                confidence=confidence
            )
        
        return None
    
    def detect_mean_reversion(self, df: pd.DataFrame, idx: int) -> Optional[TradingSignal]:
        """Detect mean reversion signals"""
        current = df.iloc[idx]
        
        # BUY near lower channel
        if current['price_position'] < 0.2:  # Within 20% of lower channel
            return TradingSignal(
                timestamp=current.name,
                signal=SignalType.BUY,
                price=current['close'],
                reason="Mean reversion - near lower channel",
                confidence=1.0 - current['price_position']
            )
        
        # SELL near upper channel
        elif current['price_position'] > 0.8:  # Within 20% of upper channel
            return TradingSignal(
                timestamp=current.name,
                signal=SignalType.SELL,
                price=current['close'],
                reason="Mean reversion - near upper channel",
                confidence=current['price_position']
            )
        
        return None
    
    def add_volume_confirmation(self, df: pd.DataFrame, signal: TradingSignal, idx: int) -> TradingSignal:
        """Add volume confirmation to signals"""
        if not self.volume_confirmation or 'volume' not in df.columns:
            return signal
        
        current = df.iloc[idx]
        avg_volume = df['volume'].rolling(window=20).mean().iloc[idx]
        
        if pd.isna(avg_volume):
            return signal
        
        volume_ratio = current['volume'] / avg_volume
        
        # Adjust confidence based on volume
        if volume_ratio > 1.5:  # High volume confirmation
            signal.confidence = min(1.0, signal.confidence * 1.2)
            signal.reason += " (high volume)"
        elif volume_ratio < 0.7:  # Low volume warning
            signal.confidence *= 0.8
            signal.reason += " (low volume)"
        
        return signal
    
    def generate_signals(self, df: pd.DataFrame) -> List[TradingSignal]:
        """Generate trading signals based on Keltner Channels"""
        
        # Calculate Keltner Channels
        df_with_keltner = self.keltner.calculate(df)
        signals = []
        
        for i in range(1, len(df_with_keltner)):
            signal = None
            
            if self.strategy_type == "breakout":
                signal = self.detect_breakout(df_with_keltner, i)
            elif self.strategy_type == "mean_reversion":
                signal = self.detect_mean_reversion(df_with_keltner, i)
            
            if signal:
                signal = self.add_volume_confirmation(df_with_keltner, signal, i)
                signals.append(signal)
        
        return signals
    
    def backtest(self, df: pd.DataFrame, initial_capital: float = 10000) -> Dict:
        """Simple backtest of the strategy"""
        
        df_with_keltner = self.keltner.calculate(df)
        signals = self.generate_signals(df)
        
        capital = initial_capital
        position = 0
        position_price = 0
        trades = []
        equity_curve = []
        
        signal_dict = {signal.timestamp: signal for signal in signals}
        
        for idx, row in df_with_keltner.iterrows():
            
            if idx in signal_dict:
                signal = signal_dict[idx]
                
                if signal.signal == SignalType.BUY and position <= 0:
                    if position < 0:  # Close short position
                        pnl = (position_price - row['close']) * abs(position)
                        capital += pnl
                        trades.append({
                            'entry_time': position_price,
                            'exit_time': idx,
                            'type': 'short',
                            'pnl': pnl
                        })
                    
                    # Open long position
                    position = capital / row['close']
                    position_price = row['close']
                    capital = 0
                
                elif signal.signal == SignalType.SELL and position >= 0:
                    if position > 0:  # Close long position
                        pnl = (row['close'] - position_price) * position
                        capital += pnl + (position * position_price)
                        trades.append({
                            'entry_time': position_price,
                            'exit_time': idx,
                            'type': 'long',
                            'pnl': pnl
                        })
                    
                    # Open short position (if allowed)
                    position = -capital / row['close']
                    position_price = row['close']
                    capital = 0
            
            # Calculate current equity
            if position > 0:  # Long position
                current_equity = position * row['close']
            elif position < 0:  # Short position
                current_equity = abs(position) * (2 * position_price - row['close'])
            else:  # No position
                current_equity = capital
            
            equity_curve.append(current_equity)
        
        # Calculate performance metrics
        equity_curve = np.array(equity_curve)
        total_return = (equity_curve[-1] - initial_capital) / initial_capital
        
        returns = np.diff(equity_curve) / equity_curve[:-1]
        sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0
        
        max_drawdown = np.max(np.maximum.accumulate(equity_curve) - equity_curve) / np.max(equity_curve)
        
        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'num_trades': len(trades),
            'equity_curve': equity_curve,
            'trades': trades,
            'signals': signals
        }

class KeltnerTradingBot:
    """Complete Keltner Channels Trading Bot"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.strategy = KeltnerTradingStrategy(**config.get('strategy', {}))
        self.is_running = False
        
    def load_data(self, data_source: str) -> pd.DataFrame:
        """Load market data (implement based on your data source)"""
        # Placeholder - implement your data loading logic
        # This could connect to APIs like Alpha Vantage, Yahoo Finance, etc.
        pass
    
    def execute_trade(self, signal: TradingSignal) -> bool:
        """Execute trade based on signal (implement based on your broker API)"""
        # Placeholder - implement your trade execution logic
        # This would connect to your broker's API
        print(f"Executing {signal.signal.value} at {signal.price} - {signal.reason}")
        return True
    
    def run_strategy(self, df: pd.DataFrame) -> None:
        """Run the trading strategy"""
        
        print("Running Keltner Channels Trading Bot...")
        print(f"Strategy Type: {self.strategy.strategy_type}")
        print(f"Keltner Period: {self.strategy.keltner.period}")
        print(f"Multiplier: {self.strategy.keltner.multiplier}")
        
        # Generate signals
        signals = self.strategy.generate_signals(df)
        
        print(f"\nGenerated {len(signals)} signals:")
        for signal in signals[-10:]:  # Show last 10 signals
            print(f"{signal.timestamp}: {signal.signal.value} at {signal.price:.2f} - {signal.reason} (confidence: {signal.confidence:.2f})")
        
        # Run backtest
        backtest_results = self.strategy.backtest(df)
        
        print(f"\nBacktest Results:")
        print(f"Total Return: {backtest_results['total_return']:.2%}")
        print(f"Sharpe Ratio: {backtest_results['sharpe_ratio']:.2f}")
        print(f"Max Drawdown: {backtest_results['max_drawdown']:.2%}")
        print(f"Number of Trades: {backtest_results['num_trades']}")
        
        return backtest_results
    
    def plot_analysis(self, df: pd.DataFrame, backtest_results: Dict) -> None:
        """Plot Keltner Channels and trading signals"""
        
        df_with_keltner = self.strategy.keltner.calculate(df)
        signals = backtest_results['signals']
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
        
        # Plot price and Keltner Channels
        ax1.plot(df_with_keltner.index, df_with_keltner['close'], label='Price', linewidth=1)
        ax1.plot(df_with_keltner.index, df_with_keltner['upper_channel'], label='Upper Channel', alpha=0.7)
        ax1.plot(df_with_keltner.index, df_with_keltner['middle_line'], label='Middle Line', alpha=0.7)
        ax1.plot(df_with_keltner.index, df_with_keltner['lower_channel'], label='Lower Channel', alpha=0.7)
        
        # Fill between channels
        ax1.fill_between(df_with_keltner.index, 
                        df_with_keltner['upper_channel'], 
                        df_with_keltner['lower_channel'], 
                        alpha=0.1, color='gray')
        
        # Plot signals
        buy_signals = [s for s in signals if s.signal == SignalType.BUY]
        sell_signals = [s for s in signals if s.signal == SignalType.SELL]
        
        if buy_signals:
            buy_times = [s.timestamp for s in buy_signals]
            buy_prices = [s.price for s in buy_signals]
            ax1.scatter(buy_times, buy_prices, color='green', marker='^', s=100, label='Buy Signal')
        
        if sell_signals:
            sell_times = [s.timestamp for s in sell_signals]
            sell_prices = [s.price for s in sell_signals]
            ax1.scatter(sell_times, sell_prices, color='red', marker='v', s=100, label='Sell Signal')
        
        ax1.set_title('Keltner Channels Trading Strategy')
        ax1.set_ylabel('Price')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot equity curve
        ax2.plot(df_with_keltner.index, backtest_results['equity_curve'], label='Equity Curve')
        ax2.set_title('Strategy Performance')
        ax2.set_ylabel('Portfolio Value')
        ax2.set_xlabel('Time')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

# Example usage and testing
def create_sample_data(n_days: int = 252) -> pd.DataFrame:
    """Create sample OHLCV data for testing"""
    
    np.random.seed(42)
    dates = pd.date_range(start='2023-01-01', periods=n_days, freq='D')
    
    # Generate realistic price data with trend and volatility
    base_price = 100
    returns = np.random.normal(0.0005, 0.02, n_days)  # Daily returns with slight positive drift
    prices = [base_price]
    
    for r in returns[1:]:
        prices.append(prices[-1] * (1 + r))
    
    prices = np.array(prices)
    
    # Create OHLC data
    high = prices * (1 + np.abs(np.random.normal(0, 0.01, n_days)))
    low = prices * (1 - np.abs(np.random.normal(0, 0.01, n_days)))
    open_price = np.roll(prices, 1)
    open_price[0] = base_price
    volume = np.random.lognormal(10, 0.5, n_days)
    
    df = pd.DataFrame({
        'open': open_price,
        'high': high,
        'low': low,
        'close': prices,
        'volume': volume
    }, index=dates)
    
    return df

if __name__ == "__main__":
    # Configuration
    config = {
        'strategy': {
            'keltner_period': 20,
            'keltner_multiplier': 2.0,
            'ema_period': 20,
            'strategy_type': 'breakout',  # or 'mean_reversion'
            'volume_confirmation': True
        }
    }
    
    # Create sample data
    sample_data = create_sample_data(252)
    
    # Initialize and run bot
    bot = KeltnerTradingBot(config)
    results = bot.run_strategy(sample_data)
    
    # Plot results
    bot.plot_analysis(sample_data, results)