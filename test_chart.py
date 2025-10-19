#!/usr/bin/env python3
"""
Test script for the trading chart system.
This script tests the chart data provider and creates sample charts.
"""

import sys
import os

# Add the parent directory to the path so we can import indicators
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chart_server import ChartDataProvider
from charting import TradingChart, create_sample_chart_with_indicators

def test_chart_data_provider():
    """Test the chart data provider."""
    print("Testing Chart Data Provider...")
    
    try:
        # Initialize data provider
        provider = ChartDataProvider()
        
        # Test data loading
        print(f"Data loaded: {provider.data is not None}")
        if provider.data is not None:
            print(f"Data shape: {provider.data.shape}")
            print(f"Columns: {provider.data.columns.tolist()}")
            print(f"Date range: {provider.data.index.min()} to {provider.data.index.max()}")
        
        # Test indicator calculation
        provider.calculate_indicators()
        print(f"Indicators calculated: {len(provider.indicators)}")
        print(f"Available indicators: {list(provider.indicators.keys())}")
        
        # Test chart data generation
        chart_data = provider.get_chart_data(period=50)
        print(f"Chart data generated: {len(chart_data['candlesticks'])} candlesticks")
        print(f"Indicators in chart data: {list(chart_data['indicators'].keys())}")
        
        # Test latest values
        latest = provider.get_latest_values()
        print(f"Latest values: {len(latest)} indicators")
        
        return True
        
    except Exception as e:
        print(f"Error testing chart data provider: {e}")
        return False

def test_charting_module():
    """Test the charting module."""
    print("\nTesting Charting Module...")
    
    try:
        # Test with sample data
        csv_path = "BankNiftyFutures.csv"
        if os.path.exists(csv_path):
            print(f"Found CSV file: {csv_path}")
            chart = create_sample_chart_with_indicators(csv_path, "test_charts")
            print("Sample charts created successfully!")
            return True
        else:
            print(f"CSV file not found: {csv_path}")
            print("Creating sample data for testing...")
            
            # Create a simple test chart
            import pandas as pd
            import numpy as np
            
            dates = pd.date_range(start='2023-01-01', end='2024-01-01', freq='D')
            np.random.seed(42)
            
            data = pd.DataFrame({
                'Open': np.random.uniform(45000, 46000, len(dates)),
                'High': np.random.uniform(45100, 46100, len(dates)),
                'Low': np.random.uniform(44900, 45900, len(dates)),
                'Close': np.random.uniform(45000, 46000, len(dates)),
                'Volume': np.random.randint(100000, 1000000, len(dates))
            }, index=dates)
            
            # Create chart instance
            chart = TradingChart(data)
            
            # Add indicators
            chart.add_indicator('SMA 20', data['Close'].rolling(window=20).mean(), 'orange', overlay=True)
            chart.add_indicator('SMA 50', data['Close'].rolling(window=50).mean(), 'blue', overlay=True)
            
            # Save charts
            os.makedirs("test_charts", exist_ok=True)
            chart.save_plotly_html("test_charts/test_chart.html", "Test Trading Chart")
            chart.save_mplfinance_png("test_charts/test_chart.png", "Test Trading Chart")
            
            print("Test charts created successfully!")
            return True
            
    except Exception as e:
        print(f"Error testing charting module: {e}")
        return False

def main():
    """Main test function."""
    print("=" * 50)
    print("TRADING CHART SYSTEM TEST")
    print("=" * 50)
    
    # Test chart data provider
    provider_success = test_chart_data_provider()
    
    # Test charting module
    charting_success = test_charting_module()
    
    print("\n" + "=" * 50)
    print("TEST RESULTS")
    print("=" * 50)
    print(f"Chart Data Provider: {'✅ PASS' if provider_success else '❌ FAIL'}")
    print(f"Charting Module: {'✅ PASS' if charting_success else '❌ FAIL'}")
    
    if provider_success and charting_success:
        print("\n🎉 All tests passed! The charting system is working correctly.")
        print("\nTo start the chart server, run:")
        print("python chart_server.py")
        print("\nThen open http://localhost:8000 in your browser.")
    else:
        print("\n⚠️  Some tests failed. Please check the error messages above.")
    
    print("=" * 50)

if __name__ == "__main__":
    main()