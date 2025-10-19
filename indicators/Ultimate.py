import pandas as pd
import numpy as np
from typing import Tuple, Dict
import talib

class UltimateIndicator:
    """
    Ultimate Trading Indicator combining multiple technical analysis components:
    - Trend Analysis (EMA, MACD)
    - Momentum (RSI, Stochastic)
    - Volatility (Bollinger Bands, ATR)
    - Volume Analysis (OBV, Volume SMA)
    - Support/Resistance levels
    """
    
    def __init__(self, 
                 fast_ema=12, slow_ema=26, signal_ema=9,
                 rsi_period=14, stoch_k=14, stoch_d=3,
                 bb_period=20, bb_std=2,
                 atr_period=14, volume_period=20):
        
        self.fast_ema = fast_ema
        self.slow_ema = slow_ema
        self.signal_ema = signal_ema
        self.rsi_period = rsi_period
        self.stoch_k = stoch_k
        self.stoch_d = stoch_d
        self.bb_period = bb_period
        self.bb_std = bb_std
        self.atr_period = atr_period
        self.volume_period = volume_period
    
    def calculate_trend_score(self, data: pd.DataFrame) -> pd.Series:
        """Calculate trend strength score (-100 to +100)"""
        close = data['close']
        
        # EMA trend
        ema_fast = talib.EMA(close, timeperiod=self.fast_ema)
        ema_slow = talib.EMA(close, timeperiod=self.slow_ema)
        ema_trend = np.where(ema_fast > ema_slow, 1, -1)
        
        # MACD
        macd, macd_signal, macd_hist = talib.MACD(close, 
                                                  fastperiod=self.fast_ema,
                                                  slowperiod=self.slow_ema,
                                                  signalperiod=self.signal_ema)
        macd_trend = np.where(macd > macd_signal, 1, -1)
        
        # Price vs EMAs
        price_vs_fast = np.where(close > ema_fast, 1, -1)
        price_vs_slow = np.where(close > ema_slow, 1, -1)
        
        # Combine trend signals
        trend_score = (ema_trend + macd_trend + price_vs_fast + price_vs_slow) * 25
        
        return pd.Series(trend_score, index=data.index)
    
    def calculate_momentum_score(self, data: pd.DataFrame) -> pd.Series:
        """Calculate momentum score (-100 to +100)"""
        high, low, close = data['high'], data['low'], data['close']
        
        # RSI
        rsi = talib.RSI(close, timeperiod=self.rsi_period)
        rsi_score = ((rsi - 50) / 50) * 100  # Normalize to -100/+100
        
        # Stochastic
        stoch_k, stoch_d = talib.STOCH(high, low, close,
                                       fastk_period=self.stoch_k,
                                       slowk_period=self.stoch_d,
                                       slowd_period=self.stoch_d)
        stoch_score = ((stoch_k - 50) / 50) * 100
        
        # Combine momentum signals
        momentum_score = (rsi_score + stoch_score) / 2
        
        return pd.Series(momentum_score, index=data.index)
    
    def calculate_volatility_score(self, data: pd.DataFrame) -> pd.Series:
        """Calculate volatility-based signals (-100 to +100)"""
        high, low, close = data['high'], data['low'], data['close']
        
        # Bollinger Bands
        bb_upper, bb_middle, bb_lower = talib.BBANDS(close,
                                                     timeperiod=self.bb_period,
                                                     nbdevup=self.bb_std,
                                                     nbdevdn=self.bb_std)
        
        # BB position score
        bb_position = (close - bb_lower) / (bb_upper - bb_lower)
        bb_score = ((bb_position - 0.5) / 0.5) * 100
        
        # ATR-based volatility regime
        atr = talib.ATR(high, low, close, timeperiod=self.atr_period)
        atr_sma = talib.SMA(atr, timeperiod=self.atr_period)
        volatility_regime = np.where(atr > atr_sma, -20, 20)  # High vol = bearish
        
        volatility_score = bb_score + volatility_regime
        volatility_score = np.clip(volatility_score, -100, 100)
        
        return pd.Series(volatility_score, index=data.index)
    
    def calculate_volume_score(self, data: pd.DataFrame) -> pd.Series:
        """Calculate volume-based score (-100 to +100)"""
        close, volume = data['close'], data['volume']
        
        # On-Balance Volume
        obv = talib.OBV(close, volume)
        obv_sma = talib.SMA(obv, timeperiod=self.volume_period)
        obv_trend = np.where(obv > obv_sma, 1, -1)
        
        # Volume relative to average
        volume_sma = talib.SMA(volume, timeperiod=self.volume_period)
        volume_ratio = volume / volume_sma
        volume_strength = np.clip((volume_ratio - 1) * 50, -50, 50)
        
        # Combine volume signals
        volume_score = (obv_trend * 50) + volume_strength
        volume_score = np.clip(volume_score, -100, 100)
        
        return pd.Series(volume_score, index=data.index)
    
    def find_support_resistance(self, data: pd.DataFrame, window: int = 20) -> Tuple[pd.Series, pd.Series]:
        """Find dynamic support and resistance levels"""
        high, low = data['high'], data['low']
        
        # Rolling max/min for support/resistance
        resistance = high.rolling(window=window).max()
        support = low.rolling(window=window).min()
        
        return support, resistance
    
    def calculate_ultimate_score(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """Calculate the ultimate trading indicator score"""
        
        # Calculate individual components
        trend_score = self.calculate_trend_score(data)
        momentum_score = self.calculate_momentum_score(data)
        volatility_score = self.calculate_volatility_score(data)
        volume_score = self.calculate_volume_score(data)
        
        # Weighted combination (adjust weights based on your strategy)
        weights = {
            'trend': 0.35,
            'momentum': 0.25,
            'volatility': 0.20,
            'volume': 0.20
        }
        
        ultimate_score = (
            trend_score * weights['trend'] +
            momentum_score * weights['momentum'] +
            volatility_score * weights['volatility'] +
            volume_score * weights['volume']
        )
        
        # Generate signals
        buy_signal = ultimate_score > 60
        sell_signal = ultimate_score < -60
        hold_signal = (ultimate_score >= -60) & (ultimate_score <= 60)
        
        # Support/Resistance levels
        support, resistance = self.find_support_resistance(data)
        
        return {
            'ultimate_score': ultimate_score,
            'trend_score': trend_score,
            'momentum_score': momentum_score,
            'volatility_score': volatility_score,
            'volume_score': volume_score,
            'buy_signal': buy_signal,
            'sell_signal': sell_signal,
            'hold_signal': hold_signal,
            'support': support,
            'resistance': resistance
        }
    
    def get_trading_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Get comprehensive trading signals with risk management"""
        
        results = self.calculate_ultimate_score(data)
        close = data['close']
        
        # Create signals DataFrame
        signals_df = pd.DataFrame(index=data.index)
        
        for key, value in results.items():
            signals_df[key] = value
        
        # Add price data
        signals_df['close'] = close
        
        # Risk management signals
        atr = talib.ATR(data['high'], data['low'], data['close'], timeperiod=self.atr_period)
        signals_df['stop_loss_long'] = close - (2 * atr)
        signals_df['stop_loss_short'] = close + (2 * atr)
        signals_df['take_profit_long'] = close + (3 * atr)
        signals_df['take_profit_short'] = close - (3 * atr)
        
        # Signal strength
        signals_df['signal_strength'] = abs(signals_df['ultimate_score']) / 100
        
        # Market regime
        signals_df['market_regime'] = np.where(
            signals_df['ultimate_score'] > 20, 'Bullish',
            np.where(signals_df['ultimate_score'] < -20, 'Bearish', 'Neutral')
        )
        
        return signals_df


# Example usage and backtesting framework
class TradingBot:
    """Simple trading bot using the Ultimate Indicator"""
    
    def __init__(self, initial_capital: float = 10000):
        self.indicator = UltimateIndicator()
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.position = 0  # 0 = no position, 1 = long, -1 = short
        self.entry_price = 0
        self.trades = []
    
    def backtest(self, data: pd.DataFrame) -> Dict:
        """Simple backtest of the strategy"""
        
        signals = self.indicator.get_trading_signals(data)
        
        for i in range(1, len(signals)):
            current_price = signals['close'].iloc[i]
            current_signal = signals['ultimate_score'].iloc[i]
            
            # Entry logic
            if self.position == 0:
                if signals['buy_signal'].iloc[i] and current_signal > 70:
                    self.enter_long(current_price, i)
                elif signals['sell_signal'].iloc[i] and current_signal < -70:
                    self.enter_short(current_price, i)
            
            # Exit logic
            elif self.position == 1:  # Long position
                if (signals['sell_signal'].iloc[i] or 
                    current_price <= signals['stop_loss_long'].iloc[i] or
                    current_price >= signals['take_profit_long'].iloc[i]):
                    self.exit_position(current_price, i, 'long')
            
            elif self.position == -1:  # Short position
                if (signals['buy_signal'].iloc[i] or
                    current_price >= signals['stop_loss_short'].iloc[i] or
                    current_price <= signals['take_profit_short'].iloc[i]):
                    self.exit_position(current_price, i, 'short')
        
        return self.get_performance_stats()
    
    def enter_long(self, price: float, index: int):
        self.position = 1
        self.entry_price = price
        
    def enter_short(self, price: float, index: int):
        self.position = -1
        self.entry_price = price
        
    def exit_position(self, price: float, index: int, position_type: str):
        if position_type == 'long':
            pnl = ((price - self.entry_price) / self.entry_price) * 100
        else:  # short
            pnl = ((self.entry_price - price) / self.entry_price) * 100
        
        self.trades.append({
            'entry_price': self.entry_price,
            'exit_price': price,
            'pnl_percent': pnl,
            'position_type': position_type
        })
        
        self.capital *= (1 + pnl/100)
        self.position = 0
        self.entry_price = 0
    
    def get_performance_stats(self) -> Dict:
        if not self.trades:
            return {'total_return': 0, 'win_rate': 0, 'total_trades': 0}
        
        df_trades = pd.DataFrame(self.trades)
        total_return = ((self.capital - self.initial_capital) / self.initial_capital) * 100
        win_rate = (df_trades['pnl_percent'] > 0).mean() * 100
        avg_win = df_trades[df_trades['pnl_percent'] > 0]['pnl_percent'].mean()
        avg_loss = df_trades[df_trades['pnl_percent'] < 0]['pnl_percent'].mean()
        
        return {
            'total_return': total_return,
            'win_rate': win_rate,
            'total_trades': len(self.trades),
            'avg_win': avg_win if not pd.isna(avg_win) else 0,
            'avg_loss': avg_loss if not pd.isna(avg_loss) else 0,
            'final_capital': self.capital
        }


# Example usage with sample data
if __name__ == "__main__":
    # You would replace this with real market data
    # Sample data structure (OHLCV format)
    """
    data = pd.DataFrame({
        'open': [...],
        'high': [...],
        'low': [...],
        'close': [...],
        'volume': [...]
    })
    
    # Initialize and run
    bot = TradingBot(initial_capital=10000)
    results = bot.backtest(data)
    print(f"Total Return: {results['total_return']:.2f}%")
    print(f"Win Rate: {results['win_rate']:.2f}%")
    print(f"Total Trades: {results['total_trades']}")
    
    # Get detailed signals
    indicator = UltimateIndicator()
    signals = indicator.get_trading_signals(data)
    print(signals.tail())
    """
    pass