#!/usr/bin/env python3
"""
Debug script to test pattern detection with different confidence thresholds.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pattern_detection.backtesting import PatternBacktester

def debug_pattern_detection():
    """Debug pattern detection with different settings."""
    print("🔍 Debugging pattern detection...")
    
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
    
    # Create test chart
    test_dir = os.path.join(os.path.expanduser("~"), "debug_charts")
    os.makedirs(test_dir, exist_ok=True)
    
    # Test multiple periods
    for i in range(3):
        start_idx = 1000 + i * 200
        end_idx = start_idx + 100
        sample_data = df.iloc[start_idx:end_idx]
        
        chart_path = os.path.join(test_dir, f"debug_chart_{i}.png")
        
        # Create candlestick chart
        fig, ax = plt.subplots(figsize=(12, 8), dpi=100)
        
        # Plot candlesticks
        for j, (idx, row) in enumerate(sample_data.iterrows()):
            open_price = row['Open 5']
            high_price = row['High 5']
            low_price = row['Low 5']
            close_price = row['Close 5']
            
            # Determine color
            color = 'green' if close_price >= open_price else 'red'
            
            # Draw the wick
            ax.plot([j, j], [low_price, high_price], color='black', linewidth=1)
            
            # Draw the body
            body_height = abs(close_price - open_price)
            body_bottom = min(open_price, close_price)
            
            from matplotlib.patches import Rectangle
            rect = Rectangle((j-0.3, body_bottom), 0.6, body_height, 
                           facecolor=color, edgecolor='black', linewidth=0.5, alpha=0.8)
            ax.add_patch(rect)
        
        # Formatting
        ax.set_xlabel('Time Period')
        ax.set_ylabel('Price')
        ax.set_title(f'Debug Chart {i} - Period {start_idx} to {end_idx}')
        ax.grid(True, alpha=0.3)
        
        # Save chart
        plt.tight_layout()
        plt.savefig(chart_path, dpi=100, bbox_inches='tight')
        plt.close()
        
        print(f"Chart {i} saved to: {chart_path}")
        
        # Test pattern detection with different confidence thresholds
        backtester = PatternBacktester("BankNiftyFutures.csv")
        
        for conf_threshold in [0.1, 0.2, 0.3, 0.4, 0.5]:
            detections = backtester.detect_patterns_in_chart(chart_path)
            filtered_detections = [d for d in detections if d['confidence'] >= conf_threshold]
            
            print(f"  Confidence {conf_threshold}: {len(filtered_detections)} detections")
            for det in filtered_detections:
                print(f"    - {det['label']}: {det['confidence']:.3f}")
        
        # Test ground truth
        ground_truth = backtester.identify_ground_truth_patterns(sample_data, 0, len(sample_data))
        print(f"  Ground truth: {ground_truth}")
        print()

if __name__ == "__main__":
    debug_pattern_detection()
