import os
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


@dataclass
class MarketAnalysis:
    """Results from historical market data analysis."""
    total_records: int
    date_range: Tuple[str, str]
    price_range: Tuple[float, float]
    volume_stats: Dict[str, float]
    volatility_metrics: Dict[str, float]
    trend_analysis: Dict[str, float]
    pattern_frequency: Dict[str, int]
    recommendations: List[str]


class HistoricalAnalyzer:
    """Analyzes historical market data to provide context for real-time detection."""
    
    def __init__(self, csv_path: str):
        self.csv_path = csv_path
        self.df = None
        self.analysis = None
        
    def load_data(self) -> pd.DataFrame:
        """Load and preprocess the historical data."""
        print(f"Loading data from {self.csv_path}...")
        
        # Load CSV with proper date parsing
        self.df = pd.read_csv(self.csv_path)
        
        # Convert date columns with flexible parsing
        self.df['Date'] = pd.to_datetime(self.df['Date'], format='mixed', errors='coerce')
        self.df['Final Date format'] = pd.to_datetime(self.df['Final Date format'], format='%Y%m%d', errors='coerce')
        
        # Create datetime index with better error handling
        try:
            self.df['DateTime'] = pd.to_datetime(self.df['Date'].astype(str) + ' ' + self.df['Time'], errors='coerce')
        except:
            # Fallback: use Final Date format if available
            if 'Final Date format' in self.df.columns:
                self.df['DateTime'] = pd.to_datetime(self.df['Final Date format'].astype(str) + ' ' + self.df['Time'], errors='coerce')
            else:
                # Last resort: use index as datetime
                self.df['DateTime'] = pd.to_datetime(self.df.index, errors='coerce')
        
        # Remove rows with invalid datetime
        self.df = self.df.dropna(subset=['DateTime'])
        self.df = self.df.set_index('DateTime').sort_index()
        
        # Clean column names
        self.df.columns = [col.strip() for col in self.df.columns]
        
        # Ensure numeric columns are properly typed
        numeric_cols = ['Open 5', 'High 5', 'Low 5', 'Close 5', 'Volume 5']
        for col in numeric_cols:
            if col in self.df.columns:
                self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
        
        print(f"Loaded {len(self.df)} records from {self.df.index.min()} to {self.df.index.max()}")
        return self.df
    
    def analyze_market_structure(self) -> Dict:
        """Analyze basic market structure and statistics."""
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        # Basic statistics
        close_prices = self.df['Close 5'].dropna()
        volume = self.df['Volume 5'].dropna()
        
        analysis = {
            'total_records': len(self.df),
            'date_range': (self.df.index.min().strftime('%Y-%m-%d'), 
                          self.df.index.max().strftime('%Y-%m-%d')),
            'price_range': (close_prices.min(), close_prices.max()),
            'volume_stats': {
                'mean': volume.mean(),
                'median': volume.median(),
                'std': volume.std(),
                'max': volume.max()
            },
            'price_stats': {
                'mean': close_prices.mean(),
                'median': close_prices.median(),
                'std': close_prices.std()
            }
        }
        
        return analysis
    
    def calculate_volatility_metrics(self) -> Dict:
        """Calculate various volatility measures."""
        close_prices = self.df['Close 5'].dropna()
        
        # Daily returns
        returns = close_prices.pct_change().dropna()
        
        # Volatility metrics
        volatility = {
            'daily_volatility': returns.std(),
            'annualized_volatility': returns.std() * np.sqrt(252),  # Assuming 252 trading days
            'max_drawdown': self._calculate_max_drawdown(close_prices),
            'sharpe_ratio': returns.mean() / returns.std() if returns.std() > 0 else 0,
            'var_95': np.percentile(returns, 5),  # 95% VaR
            'var_99': np.percentile(returns, 1),  # 99% VaR
        }
        
        return volatility
    
    def _calculate_max_drawdown(self, prices: pd.Series) -> float:
        """Calculate maximum drawdown."""
        peak = prices.expanding().max()
        drawdown = (prices - peak) / peak
        return drawdown.min()
    
    def analyze_trends(self) -> Dict:
        """Analyze market trends and patterns."""
        close_prices = self.df['Close 5'].dropna()
        
        # Moving averages
        ma_20 = close_prices.rolling(window=20).mean()
        ma_50 = close_prices.rolling(window=50).mean()
        ma_200 = close_prices.rolling(window=200).mean()
        
        # Trend analysis
        current_price = close_prices.iloc[-1]
        trend_analysis = {
            'current_price': current_price,
            'ma_20': ma_20.iloc[-1] if not pd.isna(ma_20.iloc[-1]) else None,
            'ma_50': ma_50.iloc[-1] if not pd.isna(ma_50.iloc[-1]) else None,
            'ma_200': ma_200.iloc[-1] if not pd.isna(ma_200.iloc[-1]) else None,
            'trend_strength': self._calculate_trend_strength(close_prices),
            'momentum': self._calculate_momentum(close_prices),
        }
        
        # Determine overall trend
        if trend_analysis['ma_20'] and trend_analysis['ma_50'] and trend_analysis['ma_200']:
            if current_price > trend_analysis['ma_20'] > trend_analysis['ma_50'] > trend_analysis['ma_200']:
                trend_analysis['overall_trend'] = 'Strong Uptrend'
            elif current_price < trend_analysis['ma_20'] < trend_analysis['ma_50'] < trend_analysis['ma_200']:
                trend_analysis['overall_trend'] = 'Strong Downtrend'
            elif current_price > trend_analysis['ma_20'] > trend_analysis['ma_50']:
                trend_analysis['overall_trend'] = 'Uptrend'
            elif current_price < trend_analysis['ma_20'] < trend_analysis['ma_50']:
                trend_analysis['overall_trend'] = 'Downtrend'
            else:
                trend_analysis['overall_trend'] = 'Sideways'
        else:
            trend_analysis['overall_trend'] = 'Insufficient Data'
        
        return trend_analysis
    
    def _calculate_trend_strength(self, prices: pd.Series) -> float:
        """Calculate trend strength using linear regression slope."""
        from scipy import stats
        x = np.arange(len(prices))
        slope, _, r_value, _, _ = stats.linregress(x, prices)
        return r_value ** 2  # R-squared
    
    def _calculate_momentum(self, prices: pd.Series) -> float:
        """Calculate price momentum."""
        if len(prices) < 20:
            return 0
        return (prices.iloc[-1] - prices.iloc[-20]) / prices.iloc[-20]
    
    def detect_support_resistance(self) -> Dict:
        """Detect key support and resistance levels."""
        close_prices = self.df['Close 5'].dropna()
        high_prices = self.df['High 5'].dropna()
        low_prices = self.df['Low 5'].dropna()
        
        # Find local maxima and minima
        from scipy.signal import argrelextrema
        
        # Resistance levels (local maxima)
        resistance_indices = argrelextrema(high_prices.values, np.greater, order=5)[0]
        resistance_levels = high_prices.iloc[resistance_indices].values
        
        # Support levels (local minima)
        support_indices = argrelextrema(low_prices.values, np.less, order=5)[0]
        support_levels = low_prices.iloc[support_indices].values
        
        return {
            'resistance_levels': resistance_levels[-10:].tolist(),  # Last 10 resistance levels
            'support_levels': support_levels[-10:].tolist(),  # Last 10 support levels
            'current_resistance': resistance_levels[-1] if len(resistance_levels) > 0 else None,
            'current_support': support_levels[-1] if len(support_levels) > 0 else None,
        }
    
    def generate_recommendations(self, analysis: Dict) -> List[str]:
        """Generate trading recommendations based on analysis."""
        recommendations = []
        
        # Volatility recommendations
        if analysis['volatility_metrics']['annualized_volatility'] > 0.3:
            recommendations.append("High volatility detected - consider wider stop losses")
        elif analysis['volatility_metrics']['annualized_volatility'] < 0.1:
            recommendations.append("Low volatility - market may be consolidating")
        
        # Trend recommendations
        trend = analysis['trend_analysis']['overall_trend']
        if 'Uptrend' in trend:
            recommendations.append("Bullish trend detected - look for long opportunities")
        elif 'Downtrend' in trend:
            recommendations.append("Bearish trend detected - look for short opportunities")
        else:
            recommendations.append("Sideways market - range trading opportunities")
        
        # Support/Resistance recommendations
        if analysis.get('support_resistance'):
            current_price = analysis['trend_analysis']['current_price']
            resistance = analysis['support_resistance'].get('current_resistance')
            support = analysis['support_resistance'].get('current_support')
            
            if resistance and current_price > resistance * 0.98:
                recommendations.append("Near resistance level - watch for reversal signals")
            if support and current_price < support * 1.02:
                recommendations.append("Near support level - watch for bounce signals")
        
        return recommendations
    
    def run_full_analysis(self) -> MarketAnalysis:
        """Run complete historical analysis."""
        print("Starting historical market analysis...")
        
        # Load data
        self.load_data()
        
        # Run all analyses
        basic_stats = self.analyze_market_structure()
        volatility = self.calculate_volatility_metrics()
        trends = self.analyze_trends()
        support_resistance = self.detect_support_resistance()
        
        # Combine all analysis
        combined_analysis = {
            **basic_stats,
            'volatility_metrics': volatility,
            'trend_analysis': trends,
            'support_resistance': support_resistance,
        }
        
        # Generate recommendations
        recommendations = self.generate_recommendations(combined_analysis)
        
        # Create analysis object
        self.analysis = MarketAnalysis(
            total_records=combined_analysis['total_records'],
            date_range=combined_analysis['date_range'],
            price_range=combined_analysis['price_range'],
            volume_stats=combined_analysis['volume_stats'],
            volatility_metrics=combined_analysis['volatility_metrics'],
            trend_analysis=combined_analysis['trend_analysis'],
            pattern_frequency={},  # Will be populated by real-time detection
            recommendations=recommendations
        )
        
        return self.analysis
    
    def print_analysis_summary(self):
        """Print a formatted summary of the analysis."""
        if self.analysis is None:
            print("No analysis available. Run run_full_analysis() first.")
            return
        
        print("\n" + "="*60)
        print("HISTORICAL MARKET ANALYSIS SUMMARY")
        print("="*60)
        
        print(f"\n📊 DATA OVERVIEW:")
        print(f"   • Total Records: {self.analysis.total_records:,}")
        print(f"   • Date Range: {self.analysis.date_range[0]} to {self.analysis.date_range[1]}")
        print(f"   • Price Range: ₹{self.analysis.price_range[0]:,.2f} - ₹{self.analysis.price_range[1]:,.2f}")
        
        print(f"\n📈 VOLATILITY METRICS:")
        vol = self.analysis.volatility_metrics
        print(f"   • Daily Volatility: {vol['daily_volatility']:.4f}")
        print(f"   • Annualized Volatility: {vol['annualized_volatility']:.2%}")
        print(f"   • Max Drawdown: {vol['max_drawdown']:.2%}")
        print(f"   • Sharpe Ratio: {vol['sharpe_ratio']:.3f}")
        
        print(f"\n📊 TREND ANALYSIS:")
        trend = self.analysis.trend_analysis
        print(f"   • Current Price: ₹{trend['current_price']:,.2f}")
        print(f"   • Overall Trend: {trend['overall_trend']}")
        print(f"   • Trend Strength: {trend['trend_strength']:.3f}")
        print(f"   • Momentum: {trend['momentum']:.3f}")
        
        if trend.get('ma_20'):
            print(f"   • MA 20: ₹{trend['ma_20']:,.2f}")
            print(f"   • MA 50: ₹{trend['ma_50']:,.2f}")
            print(f"   • MA 200: ₹{trend['ma_200']:,.2f}")
        
        print(f"\n💡 RECOMMENDATIONS:")
        for i, rec in enumerate(self.analysis.recommendations, 1):
            print(f"   {i}. {rec}")
        
        print("\n" + "="*60)


def analyze_historical_data(csv_path: str) -> MarketAnalysis:
    """Convenience function to analyze historical data."""
    analyzer = HistoricalAnalyzer(csv_path)
    return analyzer.run_full_analysis()
