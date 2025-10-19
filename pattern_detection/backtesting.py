import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
import cv2
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import time
from datetime import datetime, timedelta
import warnings
import glob
warnings.filterwarnings('ignore')

from .realtime import RealTimePatternDetector, RuntimeConfig, CaptureRegion
from .historical_analysis import HistoricalAnalyzer


@dataclass
class BacktestResult:
    """Results from backtesting pattern detection."""
    total_periods: int
    detected_patterns: int
    true_positives: int
    false_positives: int
    false_negatives: int
    precision: float
    recall: float
    f1_score: float
    accuracy: float
    pattern_breakdown: Dict[str, Dict[str, int]]
    performance_by_period: List[Dict]
    recommendations: List[str]


class PatternBacktester:
    """Backtests pattern detection model on historical data."""
    
    def __init__(self, csv_path: str, model_path: str = "foduucom/stockmarket-pattern-detection-yolov8"):
        self.csv_path = csv_path
        self.model_path = model_path
        self.df = None
        self.model = None
        self.results = None
        
    def load_historical_data(self) -> pd.DataFrame:
        """Load and preprocess historical data for backtesting."""
        print("Loading historical data for backtesting...")
        
        # Load CSV with flexible date parsing
        self.df = pd.read_csv(self.csv_path)
        
        # Convert date columns
        self.df['Date'] = pd.to_datetime(self.df['Date'], format='mixed', errors='coerce')
        self.df['Final Date format'] = pd.to_datetime(self.df['Final Date format'], format='%Y%m%d', errors='coerce')
        
        # Create datetime index
        try:
            self.df['DateTime'] = pd.to_datetime(self.df['Date'].astype(str) + ' ' + self.df['Time'], errors='coerce')
        except:
            if 'Final Date format' in self.df.columns:
                self.df['DateTime'] = pd.to_datetime(self.df['Final Date format'].astype(str) + ' ' + self.df['Time'], errors='coerce')
            else:
                self.df['DateTime'] = pd.to_datetime(self.df.index, errors='coerce')
        
        # Remove invalid dates
        self.df = self.df.dropna(subset=['DateTime'])
        self.df = self.df.set_index('DateTime').sort_index()
        
        # Clean numeric columns
        numeric_cols = ['Open 5', 'High 5', 'Low 5', 'Close 5', 'Volume 5']
        for col in numeric_cols:
            if col in self.df.columns:
                self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
        
        print(f"Loaded {len(self.df)} records for backtesting")
        return self.df
    
    def create_candlestick_chart(self, data: pd.DataFrame, start_idx: int, end_idx: int, 
                                save_path: str, width: int = 800, height: int = 600) -> str:
        """Create a candlestick chart from historical data and save as image."""
        period_data = data.iloc[start_idx:end_idx].copy()
        
        if len(period_data) < 10:  # Need minimum data for meaningful patterns
            return None
            
        # Create figure
        fig, ax = plt.subplots(figsize=(width/100, height/100), dpi=100)
        
        # Plot candlesticks
        for i, (idx, row) in enumerate(period_data.iterrows()):
            open_price = row['Open 5']
            high_price = row['High 5']
            low_price = row['Low 5']
            close_price = row['Close 5']
            
            # Determine color
            color = 'green' if close_price >= open_price else 'red'
            
            # Draw the wick
            ax.plot([i, i], [low_price, high_price], color='black', linewidth=0.8)
            
            # Draw the body
            body_height = abs(close_price - open_price)
            body_bottom = min(open_price, close_price)
            
            rect = Rectangle((i-0.3, body_bottom), 0.6, body_height, 
                           facecolor=color, edgecolor='black', linewidth=0.5, alpha=0.8)
            ax.add_patch(rect)
        
        # Formatting
        ax.set_xlabel('Time Period')
        ax.set_ylabel('Price')
        ax.set_title(f'BankNifty Chart - Period {start_idx} to {end_idx}')
        ax.grid(True, alpha=0.3)
        
        # Set x-axis labels
        step = max(1, len(period_data) // 10)
        ax.set_xticks(range(0, len(period_data), step))
        ax.set_xticklabels([period_data.index[i].strftime('%m-%d %H:%M') 
                           for i in range(0, len(period_data), step)], rotation=45)
        
        # Save chart
        plt.tight_layout()
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        plt.close()
        
        return save_path
    
    def detect_patterns_in_chart(self, chart_path: str) -> List[Dict]:
        """Run pattern detection on a saved chart image."""
        if not os.path.exists(chart_path):
            return []
        
        # Load model if not already loaded
        if self.model is None:
            from ultralytics import YOLO
            from huggingface_hub import snapshot_download
            import glob
            
            # Resolve model weights path (same logic as realtime.py)
            weights_path = self._resolve_model_weights(self.model_path)
            self.model = YOLO(weights_path)
        
        # Run detection
        results = self.model(chart_path, save=False, verbose=False)
        
        detections = []
        if results and hasattr(results[0], 'boxes') and results[0].boxes is not None:
            boxes = results[0].boxes
            if len(boxes) > 0:
                classes = boxes.cls.tolist()
                confidences = boxes.conf.tolist()
                xyxy = boxes.xyxy.cpu().numpy()
                
                for i, cls_idx in enumerate(classes):
                    label = self._get_class_name(int(cls_idx))
                    confidence = float(confidences[i])
                    bbox = xyxy[i].tolist()
                    
                    detections.append({
                        'label': label,
                        'confidence': confidence,
                        'bbox': bbox
                    })
        
        return detections
    
    def _resolve_model_weights(self, model_repo_or_path: str) -> str:
        """Resolve model weights path (same logic as realtime.py)."""
        # If user provided a direct path to a .pt file
        if os.path.isfile(model_repo_or_path) and model_repo_or_path.endswith(".pt"):
            return model_repo_or_path

        # If it's a local directory, pick a .pt inside
        if os.path.isdir(model_repo_or_path):
            candidates = sorted(glob.glob(os.path.join(model_repo_or_path, "**", "*.pt"), recursive=True))
            if not candidates:
                raise FileNotFoundError(f"No .pt weights found under directory: {model_repo_or_path}")
            # Prefer a file named model.pt if present
            preferred = [p for p in candidates if os.path.basename(p).lower() in {"model.pt", "best.pt"}]
            return preferred[0] if preferred else candidates[0]

        # Otherwise assume it's a Hugging Face repo id, download and locate .pt
        from huggingface_hub import snapshot_download
        local_repo_dir = snapshot_download(repo_id=model_repo_or_path)
        candidates = sorted(glob.glob(os.path.join(local_repo_dir, "**", "*.pt"), recursive=True))
        if not candidates:
            raise FileNotFoundError(f"No .pt weights found in Hugging Face repo: {model_repo_or_path}")
        preferred = [p for p in candidates if os.path.basename(p).lower() in {"model.pt", "best.pt"}]
        return preferred[0] if preferred else candidates[0]
    
    def _get_class_name(self, class_idx: int) -> str:
        """Get class name from index."""
        class_names = [
            'Head and shoulders bottom',
            'Head and shoulders top', 
            'M_Head',
            'StockLine',
            'Triangle',
            'W_Bottom'
        ]
        return class_names[class_idx] if class_idx < len(class_names) else f"Class_{class_idx}"
    
    def identify_ground_truth_patterns(self, data: pd.DataFrame, start_idx: int, end_idx: int) -> List[str]:
        """Identify actual patterns in the data using technical analysis."""
        period_data = data.iloc[start_idx:end_idx].copy()
        
        if len(period_data) < 20:
            return []
        
        patterns = []
        
        # Simple pattern detection using price action
        prices = period_data['Close 5'].values
        highs = period_data['High 5'].values
        lows = period_data['Low 5'].values
        
        # Head and Shoulders detection
        if self._detect_head_shoulders(prices, highs, lows):
            patterns.append("Head and shoulders top")
        
        # Double bottom detection
        if self._detect_double_bottom(prices, lows):
            patterns.append("W_Bottom")
        
        # Triangle detection
        if self._detect_triangle(prices, highs, lows):
            patterns.append("Triangle")
        
        # Stock line detection (trend line)
        if self._detect_stock_line(prices):
            patterns.append("StockLine")
        
        return patterns
    
    def _detect_head_shoulders(self, prices: pd.Series, highs: pd.Series, lows: pd.Series) -> bool:
        """Simple head and shoulders detection."""
        if len(prices) < 15:
            return False
        
        # Find local maxima
        from scipy.signal import argrelextrema
        peaks = argrelextrema(highs, np.greater, order=3)[0]
        
        if len(peaks) < 3:
            return False
        
        # Check for H&S pattern (three peaks with middle highest)
        recent_peaks = peaks[-3:] if len(peaks) >= 3 else peaks
        if len(recent_peaks) == 3:
            left_shoulder = highs[recent_peaks[0]]
            head = highs[recent_peaks[1]]
            right_shoulder = highs[recent_peaks[2]]
            
            # Head should be higher than shoulders
            if head > left_shoulder and head > right_shoulder:
                # Shoulders should be roughly equal
                shoulder_diff = abs(left_shoulder - right_shoulder) / left_shoulder
                if shoulder_diff < 0.05:  # 5% tolerance
                    return True
        
        return False
    
    def _detect_double_bottom(self, prices: pd.Series, lows: pd.Series) -> bool:
        """Detect double bottom pattern."""
        if len(prices) < 10:
            return False
        
        # Find local minima
        from scipy.signal import argrelextrema
        valleys = argrelextrema(lows, np.less, order=3)[0]
        
        if len(valleys) < 2:
            return False
        
        # Check for double bottom
        recent_valleys = valleys[-2:] if len(valleys) >= 2 else valleys
        if len(recent_valleys) == 2:
            bottom1 = lows[recent_valleys[0]]
            bottom2 = lows[recent_valleys[1]]
            
            # Bottoms should be roughly equal
            bottom_diff = abs(bottom1 - bottom2) / bottom1
            if bottom_diff < 0.03:  # 3% tolerance
                return True
        
        return False
    
    def _detect_triangle(self, prices: pd.Series, highs: pd.Series, lows: pd.Series) -> bool:
        """Detect triangle pattern."""
        if len(prices) < 10:
            return False
        
        # Check for converging trend lines
        recent_highs = highs[-10:]
        recent_lows = lows[-10:]
        
        # Calculate trend line slopes
        x = np.arange(len(recent_highs))
        high_slope = np.polyfit(x, recent_highs, 1)[0]
        low_slope = np.polyfit(x, recent_lows, 1)[0]
        
        # Triangle: converging lines (opposite slopes)
        if high_slope < 0 and low_slope > 0:
            return True
        
        return False
    
    def _detect_stock_line(self, prices: pd.Series) -> bool:
        """Detect strong trend line."""
        if len(prices) < 10:
            return False
        
        # Calculate trend strength
        x = np.arange(len(prices))
        poly_coeffs = np.polyfit(x, prices, 1)
        slope = poly_coeffs[0]
        # Calculate R-squared manually
        y_pred = np.polyval(poly_coeffs, x)
        ss_res = np.sum((prices - y_pred) ** 2)
        ss_tot = np.sum((prices - np.mean(prices)) ** 2)
        r_value = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        # Strong trend if R² > 0.7
        return r_value ** 2 > 0.7
    
    def run_backtest(self, test_periods: int = 50, period_length: int = 100, 
                    overlap: int = 20, confidence_threshold: float = 0.25) -> BacktestResult:
        """Run comprehensive backtest on historical data."""
        print(f"Starting backtest with {test_periods} periods...")
        
        # Load data
        self.load_historical_data()
        
        # Create temporary directory for charts
        temp_dir = os.path.join(os.path.expanduser("~"), "backtest_charts")
        os.makedirs(temp_dir, exist_ok=True)
        
        # Initialize counters
        total_detections = 0
        true_positives = 0
        false_positives = 0
        false_negatives = 0
        pattern_breakdown = {}
        performance_by_period = []
        
        # Test on different periods
        for i in range(0, min(test_periods, len(self.df) - period_length), overlap):
            print(f"Testing period {i//overlap + 1}/{test_periods//overlap + 1}")
            
            # Create chart for this period
            chart_path = os.path.join(temp_dir, f"chart_{i}.png")
            self.create_candlestick_chart(self.df, i, i + period_length, chart_path)
            
            if not os.path.exists(chart_path):
                continue
            
            # Get ground truth patterns
            ground_truth = self.identify_ground_truth_patterns(self.df, i, i + period_length)
            
            # Run model detection
            detections = self.detect_patterns_in_chart(chart_path)
            detected_labels = [d['label'] for d in detections if d['confidence'] >= confidence_threshold]
            
            # Calculate metrics for this period
            period_tp = len(set(detected_labels) & set(ground_truth))
            period_fp = len(set(detected_labels) - set(ground_truth))
            period_fn = len(set(ground_truth) - set(detected_labels))
            
            true_positives += period_tp
            false_positives += period_fp
            false_negatives += period_fn
            total_detections += len(detected_labels)
            
            # Track pattern breakdown
            for pattern in detected_labels:
                if pattern not in pattern_breakdown:
                    pattern_breakdown[pattern] = {'detected': 0, 'correct': 0, 'incorrect': 0}
                pattern_breakdown[pattern]['detected'] += 1
                if pattern in ground_truth:
                    pattern_breakdown[pattern]['correct'] += 1
                else:
                    pattern_breakdown[pattern]['incorrect'] += 1
            
            # Store period performance
            performance_by_period.append({
                'period': i,
                'ground_truth': ground_truth,
                'detected': detected_labels,
                'tp': period_tp,
                'fp': period_fp,
                'fn': period_fn,
                'precision': period_tp / (period_tp + period_fp) if (period_tp + period_fp) > 0 else 0,
                'recall': period_tp / (period_tp + period_fn) if (period_tp + period_fn) > 0 else 0
            })
        
        # Calculate overall metrics
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        accuracy = true_positives / (true_positives + false_positives + false_negatives) if (true_positives + false_positives + false_negatives) > 0 else 0
        
        # Generate recommendations
        recommendations = self._generate_backtest_recommendations(precision, recall, f1_score, pattern_breakdown)
        
        # Create result object
        self.results = BacktestResult(
            total_periods=test_periods,
            detected_patterns=total_detections,
            true_positives=true_positives,
            false_positives=false_positives,
            false_negatives=false_negatives,
            precision=precision,
            recall=recall,
            f1_score=f1_score,
            accuracy=accuracy,
            pattern_breakdown=pattern_breakdown,
            performance_by_period=performance_by_period,
            recommendations=recommendations
        )
        
        # Cleanup
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)
        
        return self.results
    
    def _generate_backtest_recommendations(self, precision: float, recall: float, 
                                         f1_score: float, pattern_breakdown: Dict) -> List[str]:
        """Generate recommendations based on backtest results."""
        recommendations = []
        
        if precision < 0.5:
            recommendations.append("Low precision detected - consider increasing confidence threshold")
        if recall < 0.3:
            recommendations.append("Low recall detected - model may be missing patterns")
        if f1_score < 0.4:
            recommendations.append("Overall model performance needs improvement")
        
        # Pattern-specific recommendations
        for pattern, stats in pattern_breakdown.items():
            if stats['detected'] > 0:
                accuracy = stats['correct'] / stats['detected']
                if accuracy < 0.3:
                    recommendations.append(f"Poor accuracy for {pattern} - consider retraining")
                elif accuracy > 0.8:
                    recommendations.append(f"Excellent accuracy for {pattern}")
        
        if not recommendations:
            recommendations.append("Model performance is satisfactory")
        
        return recommendations
    
    def print_backtest_summary(self):
        """Print formatted backtest results."""
        if self.results is None:
            print("No backtest results available. Run run_backtest() first.")
            return
        
        print("\n" + "="*60)
        print("PATTERN DETECTION BACKTEST RESULTS")
        print("="*60)
        
        print(f"\n📊 OVERALL PERFORMANCE:")
        print(f"   • Total Periods Tested: {self.results.total_periods}")
        print(f"   • Patterns Detected: {self.results.detected_patterns}")
        print(f"   • True Positives: {self.results.true_positives}")
        print(f"   • False Positives: {self.results.false_positives}")
        print(f"   • False Negatives: {self.results.false_negatives}")
        
        print(f"\n📈 ACCURACY METRICS:")
        print(f"   • Precision: {self.results.precision:.3f}")
        print(f"   • Recall: {self.results.recall:.3f}")
        print(f"   • F1-Score: {self.results.f1_score:.3f}")
        print(f"   • Overall Accuracy: {self.results.accuracy:.3f}")
        
        print(f"\n🎯 PATTERN BREAKDOWN:")
        for pattern, stats in self.results.pattern_breakdown.items():
            if stats['detected'] > 0:
                accuracy = stats['correct'] / stats['detected']
                print(f"   • {pattern}: {stats['detected']} detected, {accuracy:.1%} accuracy")
        
        print(f"\n💡 RECOMMENDATIONS:")
        for i, rec in enumerate(self.results.recommendations, 1):
            print(f"   {i}. {rec}")
        
        print("\n" + "="*60)
    
    def save_detailed_report(self, output_path: str = None):
        """Save detailed backtest report to Excel."""
        if self.results is None:
            print("No backtest results to save.")
            return
        
        if output_path is None:
            output_path = os.path.join(os.path.expanduser("~"), "backtest_report.xlsx")
        
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            # Summary sheet
            summary_data = {
                'Metric': ['Total Periods', 'Patterns Detected', 'True Positives', 
                          'False Positives', 'False Negatives', 'Precision', 'Recall', 
                          'F1-Score', 'Accuracy'],
                'Value': [self.results.total_periods, self.results.detected_patterns,
                         self.results.true_positives, self.results.false_positives,
                         self.results.false_negatives, self.results.precision,
                         self.results.recall, self.results.f1_score, self.results.accuracy]
            }
            pd.DataFrame(summary_data).to_excel(writer, sheet_name='Summary', index=False)
            
            # Pattern breakdown sheet
            pattern_data = []
            for pattern, stats in self.results.pattern_breakdown.items():
                pattern_data.append({
                    'Pattern': pattern,
                    'Detected': stats['detected'],
                    'Correct': stats['correct'],
                    'Incorrect': stats['incorrect'],
                    'Accuracy': stats['correct'] / stats['detected'] if stats['detected'] > 0 else 0
                })
            pd.DataFrame(pattern_data).to_excel(writer, sheet_name='Pattern Breakdown', index=False)
            
            # Period performance sheet
            period_data = []
            for period in self.results.performance_by_period:
                period_data.append({
                    'Period': period['period'],
                    'Ground Truth': ', '.join(period['ground_truth']),
                    'Detected': ', '.join(period['detected']),
                    'True Positives': period['tp'],
                    'False Positives': period['fp'],
                    'False Negatives': period['fn'],
                    'Precision': period['precision'],
                    'Recall': period['recall']
                })
            pd.DataFrame(period_data).to_excel(writer, sheet_name='Period Performance', index=False)
        
        print(f"Detailed backtest report saved to: {output_path}")


def run_backtest(csv_path: str, model_path: str = "foduucom/stockmarket-pattern-detection-yolov8",
                test_periods: int = 50, period_length: int = 100, 
                overlap: int = 20, confidence_threshold: float = 0.25) -> BacktestResult:
    """Convenience function to run backtest."""
    backtester = PatternBacktester(csv_path, model_path)
    return backtester.run_backtest(test_periods, period_length, overlap, confidence_threshold)
