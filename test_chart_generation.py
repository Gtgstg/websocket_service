#!/usr/bin/env python3
"""
Test script to generate sample charts and test pattern detection.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pattern_detection.backtesting import PatternBacktester

def test_chart_generation():
    """Test chart generation and pattern detection."""
    print("🧪 Testing chart generation and pattern detection...")
    
    # Load data
    df = pd.read_csv("BankNiftyFutures.csv")
    df['Date'] = pd.to_datetime(df['Date'], format='mixed', errors='coerce')
    df['DateTime'] = pd.to_datetime(df['Date'].astype(str) + ' ' + df['Time'], errors='coerce')
    df = df.dropna(subset=['DateTime']).set_index('DateTime').sort_index()
    
    # Clean numeric columns
    numeric_cols = ['Open 5', 'High 5', 'Low 5', 'Close 5', 'Volume 5']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    print(f"Loaded {len(df)} records")
    
    # Create a test chart
    test_dir = os.path.join(os.path.expanduser("~"), "test_charts")
    os.makedirs(test_dir, exist_ok=True)
    
    # Take a sample period
    sample_data = df.iloc[1000:1100]  # 100 data points
    chart_path = os.path.join(test_dir, "test_chart.png")
    
    # Create candlestick chart
    fig, ax = plt.subplots(figsize=(12, 8), dpi=100)
    
    # Plot candlesticks
    for i, (idx, row) in enumerate(sample_data.iterrows()):
        open_price = row['Open 5']
        high_price = row['High 5']
        low_price = row['Low 5']
        close_price = row['Close 5']
        
        # Determine color
        color = 'green' if close_price >= open_price else 'red'
        
        # Draw the wick
        ax.plot([i, i], [low_price, high_price], color='black', linewidth=1)
        
        # Draw the body
        body_height = abs(close_price - open_price)
        body_bottom = min(open_price, close_price)
        
        from matplotlib.patches import Rectangle
        rect = Rectangle((i-0.3, body_bottom), 0.6, body_height, 
                       facecolor=color, edgecolor='black', linewidth=0.5, alpha=0.8)
        ax.add_patch(rect)
    
    # Formatting
    ax.set_xlabel('Time Period')
    ax.set_ylabel('Price')
    ax.set_title('Test BankNifty Chart')
    ax.grid(True, alpha=0.3)
    
    # Set x-axis labels
    step = max(1, len(sample_data) // 10)
    ax.set_xticks(range(0, len(sample_data), step))
    ax.set_xticklabels([sample_data.index[i].strftime('%m-%d %H:%M') 
                       for i in range(0, len(sample_data), step)], rotation=45)
    
    # Save chart
    plt.tight_layout()
    plt.savefig(chart_path, dpi=100, bbox_inches='tight')
    plt.close()
    
    print(f"Chart saved to: {chart_path}")
    
    # Test pattern detection
    backtester = PatternBacktester("BankNiftyFutures.csv")
    detections = backtester.detect_patterns_in_chart(chart_path)
    
    print(f"Detections: {detections}")
    
    # Test ground truth
    ground_truth = backtester.identify_ground_truth_patterns(sample_data, 0, len(sample_data))
    print(f"Ground truth patterns: {ground_truth}")
    
    return chart_path, detections, ground_truth

if __name__ == "__main__":
    test_chart_generation()

