import numpy as np
import pandas as pd
from typing import Union, Optional
import warnings

class HistoricalVolatility:
    """
    Historical Volatility Indicator for Algorithmic Trading
    
    Supports multiple calculation methods:
    - Standard deviation of returns
    - Exponentially weighted moving average (EWMA)
    - GARCH-like volatility estimation
    - Parkinson volatility (high-low based)
    - Garman-Klass volatility
    """
    
    def __init__(self, window: int = 20, annualize: bool = True, trading_days: int = 252):
        """
        Initialize Historical Volatility calculator
        
        Args:
            window: Period for volatility calculation
            annualize: Whether to annualize the volatility
            trading_days: Number of trading days per year for annualization
        """
        self.window = window
        self.annualize = annualize
        self.trading_days = trading_days
    
    def standard_volatility(self, prices: Union[pd.Series, np.ndarray]) -> Union[pd.Series, np.ndarray]:
        """
        Calculate standard historical volatility using close prices
        
        Args:
            prices: Series or array of closing prices
            
        Returns:
            Historical volatility values
        """
        if isinstance(prices, pd.Series):
            returns = prices.pct_change().dropna()
            volatility = returns.rolling(window=self.window).std()
        else:
            prices = np.array(prices)
            returns = np.diff(prices) / prices[:-1]
            volatility = pd.Series(returns).rolling(window=self.window).std().values
        
        if self.annualize:
            volatility = volatility * np.sqrt(self.trading_days)
            
        return volatility
    
    def ewma_volatility(self, prices: Union[pd.Series, np.ndarray], alpha: float = 0.94) -> Union[pd.Series, np.ndarray]:
        """
        Calculate EWMA (Exponentially Weighted Moving Average) volatility
        
        Args:
            prices: Series or array of closing prices
            alpha: Decay factor (0 < alpha < 1, higher = more weight on recent observations)
            
        Returns:
            EWMA volatility values
        """
        if isinstance(prices, pd.Series):
            returns = prices.pct_change().dropna()
        else:
            prices = np.array(prices)
            returns = pd.Series(np.diff(prices) / prices[:-1])
        
        # Calculate EWMA variance
        ewma_var = returns.ewm(alpha=1-alpha, adjust=False).var()
        volatility = np.sqrt(ewma_var)
        
        if self.annualize:
            volatility = volatility * np.sqrt(self.trading_days)
            
        return volatility
    
    def parkinson_volatility(self, high: Union[pd.Series, np.ndarray], 
                           low: Union[pd.Series, np.ndarray]) -> Union[pd.Series, np.ndarray]:
        """
        Calculate Parkinson volatility using high and low prices
        More efficient than close-to-close volatility
        
        Args:
            high: Series or array of high prices
            low: Series or array of low prices
            
        Returns:
            Parkinson volatility values
        """
        if isinstance(high, pd.Series):
            hl_ratio = np.log(high / low)
            parkinson_var = (hl_ratio ** 2) / (4 * np.log(2))
            volatility = np.sqrt(parkinson_var.rolling(window=self.window).mean())
        else:
            high, low = np.array(high), np.array(low)
            hl_ratio = np.log(high / low)
            parkinson_var = (hl_ratio ** 2) / (4 * np.log(2))
            volatility = np.sqrt(pd.Series(parkinson_var).rolling(window=self.window).mean())
        
        if self.annualize:
            volatility = volatility * np.sqrt(self.trading_days)
            
        return volatility
    
    def garman_klass_volatility(self, high: Union[pd.Series, np.ndarray],
                              low: Union[pd.Series, np.ndarray],
                              close: Union[pd.Series, np.ndarray],
                              open_price: Optional[Union[pd.Series, np.ndarray]] = None) -> Union[pd.Series, np.ndarray]:
        """
        Calculate Garman-Klass volatility estimator
        More accurate than standard methods when using OHLC data
        
        Args:
            high: High prices
            low: Low prices  
            close: Close prices
            open_price: Open prices (optional, uses previous close if not provided)
            
        Returns:
            Garman-Klass volatility values
        """
        if isinstance(high, pd.Series):
            if open_price is None:
                open_price = close.shift(1)
            
            # Garman-Klass formula components
            term1 = np.log(high / close) * np.log(high / open_price)
            term2 = np.log(low / close) * np.log(low / open_price)
            
            gk_var = term1 + term2
            volatility = np.sqrt(gk_var.rolling(window=self.window).mean())
        else:
            high, low, close = np.array(high), np.array(low), np.array(close)
            if open_price is None:
                open_price = np.roll(close, 1)
                open_price[0] = close[0]  # Handle first value
            else:
                open_price = np.array(open_price)
            
            term1 = np.log(high / close) * np.log(high / open_price)
            term2 = np.log(low / close) * np.log(low / open_price)
            
            gk_var = term1 + term2
            volatility = np.sqrt(pd.Series(gk_var).rolling(window=self.window).mean())
        
        if self.annualize:
            volatility = volatility * np.sqrt(self.trading_days)
            
        return volatility
    
    def realized_volatility(self, prices: Union[pd.Series, np.ndarray], 
                          intraday_periods: int = 1) -> Union[pd.Series, np.ndarray]:
        """
        Calculate realized volatility (sum of squared returns)
        
        Args:
            prices: Price series
            intraday_periods: Number of intraday periods (for high-frequency data)
            
        Returns:
            Realized volatility values
        """
        if isinstance(prices, pd.Series):
            returns = prices.pct_change().dropna()
        else:
            prices = np.array(prices)
            returns = pd.Series(np.diff(prices) / prices[:-1])
        
        # Sum of squared returns over the window
        squared_returns = returns ** 2
        realized_var = squared_returns.rolling(window=self.window).sum()
        volatility = np.sqrt(realized_var)
        
        if self.annualize:
            # Adjust for intraday periods and annualize
            volatility = volatility * np.sqrt(self.trading_days * intraday_periods)
            
        return volatility
    
    def adaptive_volatility(self, prices: Union[pd.Series, np.ndarray], 
                          fast_period: int = 10, slow_period: int = 30) -> Union[pd.Series, np.ndarray]:
        """
        Calculate adaptive volatility that adjusts based on market conditions
        
        Args:
            prices: Price series
            fast_period: Fast volatility calculation period
            slow_period: Slow volatility calculation period
            
        Returns:
            Adaptive volatility values
        """
        # Calculate fast and slow volatilities
        fast_vol = HistoricalVolatility(window=fast_period, annualize=self.annualize, 
                                      trading_days=self.trading_days).standard_volatility(prices)
        slow_vol = HistoricalVolatility(window=slow_period, annualize=self.annualize,
                                      trading_days=self.trading_days).standard_volatility(prices)
        
        if isinstance(prices, pd.Series):
            returns = prices.pct_change().dropna()
            # Use efficiency ratio to blend fast and slow volatilities
            direction = returns.rolling(window=self.window).sum().abs()
            volatility_sum = returns.abs().rolling(window=self.window).sum()
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                efficiency_ratio = direction / volatility_sum
            
            # Blend volatilities based on efficiency ratio
            adaptive_vol = efficiency_ratio * fast_vol + (1 - efficiency_ratio) * slow_vol
        else:
            # Convert to pandas for calculation then back
            prices_series = pd.Series(prices)
            adaptive_vol = self.adaptive_volatility(prices_series, fast_period, slow_period).values
            
        return adaptive_vol
    
    def volatility_percentile(self, prices: Union[pd.Series, np.ndarray], 
                            lookback_period: int = 252) -> Union[pd.Series, np.ndarray]:
        """
        Calculate current volatility percentile rank over a lookback period
        
        Args:
            prices: Price series
            lookback_period: Period for percentile calculation
            
        Returns:
            Volatility percentile rank (0-100)
        """
        current_vol = self.standard_volatility(prices)
        
        if isinstance(current_vol, pd.Series):
            percentile = current_vol.rolling(window=lookback_period).rank(pct=True) * 100
        else:
            percentile = pd.Series(current_vol).rolling(window=lookback_period).rank(pct=True).values * 100
            
        return percentile

# Example usage and testing functions
def example_usage():
    """Example of how to use the HistoricalVolatility class"""
    
    # Generate sample data
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=500, freq='D')
    
    # Simulate price data with varying volatility
    returns = np.random.normal(0.001, 0.02, 500)
    prices = 100 * np.exp(np.cumsum(returns))
    
    # Add some volatility clustering
    high = prices * (1 + np.abs(np.random.normal(0, 0.01, 500)))
    low = prices * (1 - np.abs(np.random.normal(0, 0.01, 500)))
    
    # Create DataFrame
    df = pd.DataFrame({
        'close': prices,
        'high': high,
        'low': low
    }, index=dates)
    
    # Initialize volatility calculator
    hv = HistoricalVolatility(window=20, annualize=True)
    
    # Calculate different types of volatility
    df['standard_vol'] = hv.standard_volatility(df['close'])
    df['ewma_vol'] = hv.ewma_volatility(df['close'])
    df['parkinson_vol'] = hv.parkinson_volatility(df['high'], df['low'])
    df['gk_vol'] = hv.garman_klass_volatility(df['high'], df['low'], df['close'])
    df['vol_percentile'] = hv.volatility_percentile(df['close'])
    
    print("Sample volatility calculations:")
    print(df[['close', 'standard_vol', 'ewma_vol', 'parkinson_vol', 'gk_vol']].tail(10))
    
    return df

if __name__ == "__main__":
    example_data = example_usage()