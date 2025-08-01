import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional, Union
import warnings

class CommodityChannelIndex:
    """
    Commodity Channel Index (CCI) implementation for algorithmic trading.
    
    CCI measures the current price level relative to an average price level over a given period.
    It's designed to identify cyclical trends and detect overbought/oversold conditions.
    
    Formula:
    Typical Price (TP) = (High + Low + Close) / 3
    Simple Moving Average (SMA) = Sum of TP over n periods / n
    Mean Deviation (MD) = Sum of |TP - SMA| over n periods / n
    CCI = (TP - SMA) / (0.015 * MD)
    
    The constant 0.015 ensures approximately 70-80% of CCI values fall between -100 and +100.
    """
    
    def __init__(self, period: int = 20, constant: float = 0.015):
        """
        Initialize CCI parameters.
        
        Args:
            period: Period for CCI calculation (default: 20)
            constant: Scaling constant (default: 0.015, Lambert's constant)
        """
        if period <= 0:
            raise ValueError("Period must be greater than 0")
        if constant <= 0:
            raise ValueError("Constant must be greater than 0")
            
        self.period = period
        self.constant = constant
    
    def calculate_typical_price(self, high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
        """
        Calculate Typical Price (TP).
        
        Args:
            high: High prices series
            low: Low prices series
            close: Close prices series
            
        Returns:
            Typical Price series
        """
        return (high + low + close) / 3
    
    def calculate_mean_deviation(self, typical_price: pd.Series, sma: pd.Series) -> pd.Series:
        """
        Calculate Mean Deviation.
        
        Args:
            typical_price: Typical Price series
            sma: Simple Moving Average of Typical Price
            
        Returns:
            Mean Deviation series
        """
        absolute_deviations = np.abs(typical_price - sma)
        return absolute_deviations.rolling(window=self.period).mean()
    
    def calculate(self, high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
        """
        Calculate Commodity Channel Index.
        
        Args:
            high: High prices series
            low: Low prices series
            close: Close prices series
            
        Returns:
            CCI values series
        """
        # Calculate Typical Price
        typical_price = self.calculate_typical_price(high, low, close)
        
        # Calculate Simple Moving Average of Typical Price
        sma_tp = typical_price.rolling(window=self.period).mean()
        
        # Calculate Mean Deviation
        mean_deviation = self.calculate_mean_deviation(typical_price, sma_tp)
        
        # Calculate CCI
        # Avoid division by zero
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            cci = (typical_price - sma_tp) / (self.constant * mean_deviation)
        
        # Replace inf and -inf with NaN
        cci = cci.replace([np.inf, -np.inf], np.nan)
        
        return cci
    
    def generate_signals(self, cci: pd.Series, overbought: float = 100, oversold: float = -100,
                        extreme_overbought: float = 200, extreme_oversold: float = -200) -> pd.DataFrame:
        """
        Generate trading signals based on CCI values.
        
        Args:
            cci: CCI values series
            overbought: Overbought threshold (default: 100)
            oversold: Oversold threshold (default: -100)
            extreme_overbought: Extreme overbought threshold (default: 200)
            extreme_oversold: Extreme oversold threshold (default: -200)
            
        Returns:
            DataFrame with trading signals
        """
        signals = pd.DataFrame(index=cci.index)
        signals['CCI'] = cci
        
        # Basic overbought/oversold signals
        signals['overbought'] = cci > overbought
        signals['oversold'] = cci < oversold
        signals['extreme_overbought'] = cci > extreme_overbought
        signals['extreme_oversold'] = cci < extreme_oversold
        
        # Zero line crossovers
        signals['above_zero'] = cci > 0
        signals['bullish_zero_cross'] = (cci > 0) & (cci.shift(1) <= 0)
        signals['bearish_zero_cross'] = (cci < 0) & (cci.shift(1) >= 0)
        
        # Threshold crossovers
        signals['bullish_oversold_exit'] = (cci > oversold) & (cci.shift(1) <= oversold)
        signals['bearish_overbought_exit'] = (cci < overbought) & (cci.shift(1) >= overbought)
        
        # Extreme level reversals
        signals['extreme_buy_signal'] = (cci > extreme_oversold) & (cci.shift(1) <= extreme_oversold)
        signals['extreme_sell_signal'] = (cci < extreme_overbought) & (cci.shift(1) >= extreme_overbought)
        
        # Momentum signals
        signals['cci_rising'] = cci > cci.shift(1)
        signals['cci_falling'] = cci < cci.shift(1)
        
        # Divergence detection (simplified)
        signals['potential_bullish_divergence'] = (
            (cci < oversold) & 
            (cci > cci.shift(1)) & 
            (cci.shift(1) < cci.shift(2))
        )
        
        signals['potential_bearish_divergence'] = (
            (cci > overbought) & 
            (cci < cci.shift(1)) & 
            (cci.shift(1) > cci.shift(2))
        )
        
        # Combined buy/sell signals
        signals['buy_signal'] = (
            signals['bullish_zero_cross'] |
            signals['bullish_oversold_exit'] |
            signals['extreme_buy_signal'] |
            signals['potential_bullish_divergence']
        )
        
        signals['sell_signal'] = (
            signals['bearish_zero_cross'] |
            signals['bearish_overbought_exit'] |
            signals['extreme_sell_signal'] |
            signals['potential_bearish_divergence']
        )
        
        # Signal strength (0-4 scale)
        signals['signal_strength'] = 0
        
        # Basic signals (strength 1)
        signals.loc[signals['overbought'] | signals['oversold'], 'signal_strength'] = 1
        
        # Zero crossovers (strength 2)
        signals.loc[signals['bullish_zero_cross'] | signals['bearish_zero_cross'], 'signal_strength'] = 2
        
        # Threshold exits (strength 3)
        signals.loc[signals['bullish_oversold_exit'] | signals['bearish_overbought_exit'], 'signal_strength'] = 3
        
        # Extreme signals (strength 4)
        signals.loc[signals['extreme_buy_signal'] | signals['extreme_sell_signal'], 'signal_strength'] = 4
        
        return signals
    
    def detect_divergence(self, cci: pd.Series, price: pd.Series, 
                         lookback: int = 5, min_bars: int = 3) -> pd.DataFrame:
        """
        Advanced divergence detection between CCI and price.
        
        Args:
            cci: CCI values series
            price: Price series (typically close prices)
            lookback: Lookback period for peak/trough detection
            min_bars: Minimum bars between peaks/troughs
            
        Returns:
            DataFrame with divergence signals
        """
        divergence = pd.DataFrame(index=cci.index)
        divergence['CCI'] = cci
        divergence['Price'] = price
        
        # Find local peaks and troughs
        cci_peaks = self._find_peaks(cci, lookback, min_bars)
        cci_troughs = self._find_troughs(cci, lookback, min_bars)
        price_peaks = self._find_peaks(price, lookback, min_bars)
        price_troughs = self._find_troughs(price, lookback, min_bars)
        
        divergence['cci_peaks'] = cci_peaks
        divergence['cci_troughs'] = cci_troughs
        divergence['price_peaks'] = price_peaks
        divergence['price_troughs'] = price_troughs
        
        # Detect bullish divergence (price makes lower lows, CCI makes higher lows)
        divergence['bullish_divergence'] = False
        divergence['bearish_divergence'] = False
        
        # This is a simplified implementation - in practice, you'd want more sophisticated logic
        for i in range(len(divergence)):
            if i < lookback * 2:
                continue
                
            # Look for recent troughs for bullish divergence
            recent_cci_troughs = cci_troughs[max(0, i-lookback*2):i+1]
            recent_price_troughs = price_troughs[max(0, i-lookback*2):i+1]
            
            if recent_cci_troughs.sum() >= 2 and recent_price_troughs.sum() >= 2:
                # Simplified divergence detection
                cci_trough_indices = recent_cci_troughs[recent_cci_troughs].index
                price_trough_indices = recent_price_troughs[recent_price_troughs].index
                
                if len(cci_trough_indices) >= 2 and len(price_trough_indices) >= 2:
                    last_cci_trough = cci_trough_indices[-1]
                    prev_cci_trough = cci_trough_indices[-2]
                    last_price_trough = price_trough_indices[-1]
                    prev_price_trough = price_trough_indices[-2]
                    
                    # Bullish divergence: price lower low, CCI higher low
                    if (price.loc[last_price_trough] < price.loc[prev_price_trough] and
                        cci.loc[last_cci_trough] > cci.loc[prev_cci_trough]):
                        divergence.loc[divergence.index[i], 'bullish_divergence'] = True
        
        return divergence
    
    def _find_peaks(self, series: pd.Series, lookback: int, min_bars: int) -> pd.Series:
        """Find local peaks in a series."""
        peaks = pd.Series(False, index=series.index)
        
        for i in range(lookback, len(series) - lookback):
            window = series.iloc[i-lookback:i+lookback+1]
            if series.iloc[i] == window.max() and series.iloc[i] > series.iloc[i-min_bars] and series.iloc[i] > series.iloc[i+min_bars]:
                peaks.iloc[i] = True
        
        return peaks
    
    def _find_troughs(self, series: pd.Series, lookback: int, min_bars: int) -> pd.Series:
        """Find local troughs in a series."""
        troughs = pd.Series(False, index=series.index)
        
        for i in range(lookback, len(series) - lookback):
            window = series.iloc[i-lookback:i+lookback+1]
            if series.iloc[i] == window.min() and series.iloc[i] < series.iloc[i-min_bars] and series.iloc[i] < series.iloc[i+min_bars]:
                troughs.iloc[i] = True
        
        return troughs
    
    def plot(self, data: pd.DataFrame, cci: pd.Series, signals: pd.DataFrame,
             title: str = "Commodity Channel Index (CCI)", figsize: Tuple[int, int] = (15, 10)):
        """
        Plot price data with CCI indicator and signals.
        
        Args:
            data: DataFrame with OHLC data
            cci: CCI values series
            signals: DataFrame with CCI signals
            title: Plot title
            figsize: Figure size tuple
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True, height_ratios=[2, 1])
        
        # Price plot
        ax1.plot(data.index, data['close'], label='Close Price', linewidth=1.5, color='black')
        
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
        ax1.set_ylabel('Price')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # CCI plot
        ax2.plot(cci.index, cci, label='CCI', linewidth=1.5, color='blue')
        
        # Add threshold lines
        ax2.axhline(y=100, color='red', linestyle='--', alpha=0.7, label='Overbought (+100)')
        ax2.axhline(y=-100, color='green', linestyle='--', alpha=0.7, label='Oversold (-100)')
        ax2.axhline(y=200, color='darkred', linestyle=':', alpha=0.7, label='Extreme Overbought (+200)')
        ax2.axhline(y=-200, color='darkgreen', linestyle=':', alpha=0.7, label='Extreme Oversold (-200)')
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        
        # Highlight overbought/oversold regions
        ax2.fill_between(cci.index, 100, 200, alpha=0.1, color='red', label='Overbought Zone')
        ax2.fill_between(cci.index, -200, -100, alpha=0.1, color='green', label='Oversold Zone')
        ax2.fill_between(cci.index, 200, cci.max(), alpha=0.2, color='darkred', label='Extreme Overbought')
        ax2.fill_between(cci.index, cci.min(), -200, alpha=0.2, color='darkgreen', label='Extreme Oversold')
        
        # Mark zero line crossovers
        zero_crosses = signals[(signals['bullish_zero_cross']) | (signals['bearish_zero_cross'])]
        if not zero_crosses.empty:
            ax2.scatter(zero_crosses.index, cci.loc[zero_crosses.index], 
                       color='orange', marker='o', s=60, label='Zero Line Cross', zorder=5)
        
        ax2.set_title('Commodity Channel Index (CCI)')
        ax2.set_ylabel('CCI Value')
        ax2.set_xlabel('Date')
        ax2.legend(loc='upper right', fontsize=8)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()


class CCIBacktester:
    """Backtesting framework for CCI-based strategies."""
    
    def __init__(self, initial_capital: float = 10000):
        self.initial_capital = initial_capital
    
    def backtest(self, data: pd.DataFrame, signals: pd.DataFrame, 
                commission: float = 0.001, stop_loss: Optional[float] = None,
                take_profit: Optional[float] = None) -> Tuple[pd.DataFrame, list]:
        """
        Perform backtest of CCI strategy.
        
        Args:
            data: OHLC price data
            signals: CCI signals DataFrame
            commission: Commission rate per trade
            stop_loss: Stop loss percentage (e.g., 0.05 for 5%)
            take_profit: Take profit percentage (e.g., 0.10 for 10%)
            
        Returns:
            Tuple of (results DataFrame, trades list)
        """
        results = pd.DataFrame(index=data.index)
        results['price'] = data['close']
        results['buy_signal'] = signals['buy_signal']
        results['sell_signal'] = signals['sell_signal']
        results['cci'] = signals['CCI']
        
        position = 0
        cash = self.initial_capital
        entry_price = 0
        portfolio_value = []
        trades = []
        
        for i, (date, row) in enumerate(results.iterrows()):
            price = row['price']
            
            # Check stop loss and take profit if in position
            if position > 0 and entry_price > 0:
                if stop_loss and price <= entry_price * (1 - stop_loss):
                    # Stop loss triggered
                    proceeds = position * price * (1 - commission)
                    cash += proceeds
                    trades.append({
                        'date': date, 'action': 'SELL', 'price': price, 
                        'shares': position, 'reason': 'Stop Loss'
                    })
                    position = 0
                    entry_price = 0
                elif take_profit and price >= entry_price * (1 + take_profit):
                    # Take profit triggered
                    proceeds = position * price * (1 - commission)
                    cash += proceeds
                    trades.append({
                        'date': date, 'action': 'SELL', 'price': price, 
                        'shares': position, 'reason': 'Take Profit'
                    })
                    position = 0
                    entry_price = 0
            
            # Regular signal processing
            if row['buy_signal'] and position <= 0:
                # Buy signal
                shares_to_buy = cash // price
                if shares_to_buy > 0:
                    cost = shares_to_buy * price * (1 + commission)
                    if cost <= cash:
                        position += shares_to_buy
                        cash -= cost
                        entry_price = price
                        trades.append({
                            'date': date, 'action': 'BUY', 'price': price, 
                            'shares': shares_to_buy, 'reason': 'CCI Signal'
                        })
            
            elif row['sell_signal'] and position > 0:
                # Sell signal
                proceeds = position * price * (1 - commission)
                cash += proceeds
                trades.append({
                    'date': date, 'action': 'SELL', 'price': price, 
                    'shares': position, 'reason': 'CCI Signal'
                })
                position = 0
                entry_price = 0
            
            # Calculate portfolio value
            total_value = cash + (position * price)
            portfolio_value.append(total_value)
        
        results['portfolio_value'] = portfolio_value
        results['returns'] = results['portfolio_value'].pct_change()
        results['cumulative_returns'] = (results['portfolio_value'] / self.initial_capital - 1) * 100
        
        return results, trades
    
    def calculate_metrics(self, results: pd.DataFrame, trades: list) -> dict:
        """Calculate performance metrics."""
        if len(results) == 0 or results['portfolio_value'].isna().all():
            return {}
        
        total_return = results['cumulative_returns'].iloc[-1]
        returns = results['returns'].dropna()
        
        metrics = {
            'Total Return (%)': total_return,
            'Annualized Return (%)': total_return * (252 / len(results)),
            'Volatility (%)': returns.std() * np.sqrt(252) * 100,
            'Sharpe Ratio': (returns.mean() * 252) / (returns.std() * np.sqrt(252)) if returns.std() > 0 else 0,
            'Max Drawdown (%)': ((results['portfolio_value'] / results['portfolio_value'].expanding().max()) - 1).min() * 100,
            'Total Trades': len(trades),
            'Win Rate (%)': len([t for t in trades if t['action'] == 'SELL' and 'profit' in str(t)]) / max(len([t for t in trades if t['action'] == 'SELL']), 1) * 100
        }
        
        return metrics


# Example usage
def example_usage():
    """Example implementation of CCI indicator."""
    
    # Generate sample data
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=252, freq='D')
    
    # Create realistic price data with trends and cycles
    price_base = 100
    trend = np.linspace(0, 20, 252)  # Upward trend
    cycle = 5 * np.sin(np.linspace(0, 4*np.pi, 252))  # Cyclical component
    noise = np.random.normal(0, 2, 252)  # Random noise
    
    closes = price_base + trend + cycle + np.cumsum(noise * 0.1)
    highs = closes + np.abs(np.random.normal(0, 1, 252))
    lows = closes - np.abs(np.random.normal(0, 1, 252))
    
    data = pd.DataFrame({
        'high': highs,
        'low': lows,
        'close': closes
    }, index=dates)
    
    # Initialize CCI
    cci_indicator = CommodityChannelIndex(period=20)
    
    # Calculate CCI
    cci_values = cci_indicator.calculate(data['high'], data['low'], data['close'])
    
    # Generate signals
    signals = cci_indicator.generate_signals(cci_values)
    
    # Backtest strategy
    backtester = CCIBacktester(initial_capital=10000)
    results, trades = backtester.backtest(data, signals, commission=0.001)
    
    # Calculate metrics
    metrics = backtester.calculate_metrics(results, trades)
    
    # Display results
    print("CCI Trading Strategy Results")
    print("=" * 40)
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{key}: {value:.2f}")
        else:
            print(f"{key}: {value}")
    
    # Plot results
    cci_indicator.plot(data, cci_values, signals)
    
    return data, cci_values, signals, results, trades, metrics


if __name__ == "__main__":
    # Run example
    data, cci, signals, results, trades, metrics = example_usage()