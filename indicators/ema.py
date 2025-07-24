"""
Exponential Moving Average (EMA) Indicator Module
Provides classes and functions for calculating the EMA and related signals.
"""
import pandas as pd
import numpy as np

class EMA:
    """
    Exponential Moving Average (EMA) implementation.
    EMA gives more weight to recent prices, making it more responsive to new information.
    """
    
    def __init__(self, period: int = 20):
        """
        Initialize EMA indicator.
        
        Args:
            period (int): Period for EMA calculation (default: 20)
        """
        self.period = period
    
    def calculate(self, prices: pd.Series) -> pd.DataFrame:
        """
        Calculate EMA for the given price series.
        
        Args:
            prices (pd.Series): Price data series
            
        Returns:
            pd.DataFrame: DataFrame with EMA values and signals
        """
        # Calculate EMA
        ema = prices.ewm(span=self.period, adjust=False).mean()
        
        # Create results DataFrame
        results = pd.DataFrame(index=prices.index)
        results['ema'] = ema
        results['signal'] = 0
        
        # Generate signals based on price crossing EMA
        results.loc[prices > ema, 'signal'] = 1  # Price above EMA: Bullish
        results.loc[prices < ema, 'signal'] = -1  # Price below EMA: Bearish
        
        # Add price for reference
        results['price'] = prices
        
        return results
    
    def get_signal(self, price: float, ema_value: float) -> int:
        """
        Get trading signal based on price and EMA value.
        
        Args:
            price (float): Current price
            ema_value (float): Current EMA value
            
        Returns:
            int: 1 for buy (price > EMA), -1 for sell (price < EMA), 0 for hold
        """
        if price > ema_value:
            return 1
        elif price < ema_value:
            return -1
        return 0
