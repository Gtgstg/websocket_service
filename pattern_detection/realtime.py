import os
import time
import glob
from dataclasses import dataclass, field
from typing import Optional, Tuple, List

import cv2
import numpy as np
import mss  # type: ignore
from ultralytics import YOLO
from openpyxl import Workbook
import pandas as pd
from huggingface_hub import snapshot_download

from .historical_analysis import HistoricalAnalyzer, MarketAnalysis


SUPPORTED_CLASSES: List[str] = [
    'Head and shoulders bottom',
    'Head and shoulders top',
    'M_Head',
    'StockLine',
    'Triangle',
    'W_Bottom',
]


@dataclass
class CaptureRegion:
    top: int
    left: int
    width: int
    height: int

    def to_mss_monitor(self) -> dict:
        return {"top": self.top, "left": self.left, "width": self.width, "height": self.height}


@dataclass
class RuntimeConfig:
    model_repo_or_path: str = "foduucom/stockmarket-pattern-detection-yolov8"
    capture_region: CaptureRegion = field(default_factory=lambda: CaptureRegion(top=0, left=683, width=683, height=768))
    save_dir: str = os.path.join(os.path.expanduser("~"), "yolo_detection")
    screenshot_interval_sec: float = 60.0
    video_fps: float = 0.5
    confidence_threshold: float = 0.25
    device: Optional[str] = None  # e.g., "cpu" or "cuda:0"; YOLO auto-selects if None
    historical_data_path: Optional[str] = None  # Path to historical CSV data


class RealTimePatternDetector:
    def __init__(self, config: RuntimeConfig):
        self.config = config
        self.screenshots_path = os.path.join(self.config.save_dir, "screenshots")
        self.detect_path = os.path.join(self.config.save_dir, "runs", "detect")
        os.makedirs(self.screenshots_path, exist_ok=True)
        os.makedirs(self.detect_path, exist_ok=True)

        # Excel workbook setup
        self.excel_file = os.path.join(self.config.save_dir, "classification_results.xlsx")
        self.workbook = Workbook()
        self.worksheet = self.workbook.active
        self.worksheet.append(["Timestamp", "Image Path", "Label", "Confidence", "BBox[x1,y1,x2,y2]", "Historical Context"])

        # Video writer setup (lazy init when first annotated frame is produced)
        self.video_path = os.path.join(self.config.save_dir, "annotated_video.mp4")
        self.video_writer = None
        self.fourcc = cv2.VideoWriter_fourcc(*"mp4v")

        # Historical analysis setup
        self.historical_analysis = None
        if self.config.historical_data_path and os.path.exists(self.config.historical_data_path):
            print("Running historical market analysis...")
            analyzer = HistoricalAnalyzer(self.config.historical_data_path)
            self.historical_analysis = analyzer.run_full_analysis()
            analyzer.print_analysis_summary()

        # Resolve model weights path (local .pt or download from Hugging Face repo)
        weights_path = self._resolve_model_weights(self.config.model_repo_or_path)
        self.model = YOLO(weights_path)

    def _resolve_model_weights(self, model_repo_or_path: str) -> str:
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
        local_repo_dir = snapshot_download(repo_id=model_repo_or_path)
        candidates = sorted(glob.glob(os.path.join(local_repo_dir, "**", "*.pt"), recursive=True))
        if not candidates:
            raise FileNotFoundError(f"No .pt weights found in Hugging Face repo: {model_repo_or_path}")
        preferred = [p for p in candidates if os.path.basename(p).lower() in {"model.pt", "best.pt"}]
        return preferred[0] if preferred else candidates[0]

    def _init_video_writer(self, frame: np.ndarray) -> None:
        if self.video_writer is None:
            height, width = frame.shape[:2]
            self.video_writer = cv2.VideoWriter(self.video_path, self.fourcc, self.config.video_fps, (width, height))

    def _annotate_and_log(self, frame_bgr: np.ndarray, results) -> np.ndarray:
        annotated = frame_bgr.copy()
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

        best_label = "No pattern detected"
        best_conf = 0.0
        best_bbox = None

        if results and hasattr(results[0], 'boxes') and results[0].boxes is not None and len(results[0].boxes) > 0:
            boxes = results[0].boxes
            classes = boxes.cls.tolist()
            confs = boxes.conf.tolist()
            xyxy = boxes.xyxy.cpu().numpy()

            for i, cls_idx in enumerate(classes):
                label = SUPPORTED_CLASSES[int(cls_idx)] if int(cls_idx) < len(SUPPORTED_CLASSES) else str(int(cls_idx))
                conf = float(confs[i])
                x1, y1, x2, y2 = map(int, xyxy[i])

                if conf < self.config.confidence_threshold:
                    continue

                # Draw box and label
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 255), 2)
                cv2.putText(annotated, f"{label} {conf:.2f}", (x1, max(0, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)

                if conf > best_conf:
                    best_conf = conf
                    best_label = label
                    best_bbox = [x1, y1, x2, y2]

        # Timestamp banner
        cv2.putText(annotated, timestamp, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

        # Generate historical context
        historical_context = self._generate_historical_context(best_label, best_conf)

        # Log to worksheet
        bbox_str = "" if best_bbox is None else f"[{best_bbox[0]},{best_bbox[1]},{best_bbox[2]},{best_bbox[3]}]"
        self.worksheet.append([timestamp, "", best_label, round(best_conf, 4), bbox_str, historical_context])

        # Persist Excel incrementally to avoid data loss on crash
        self.workbook.save(self.excel_file)

        return annotated

    def _generate_historical_context(self, pattern_label: str, confidence: float) -> str:
        """Generate historical context for detected patterns."""
        if not self.historical_analysis:
            return "No historical data available"
        
        context_parts = []
        
        # Add trend context
        trend = self.historical_analysis.trend_analysis.get('overall_trend', 'Unknown')
        context_parts.append(f"Trend: {trend}")
        
        # Add volatility context
        vol = self.historical_analysis.volatility_metrics.get('annualized_volatility', 0)
        if vol > 0.3:
            context_parts.append("High Volatility")
        elif vol < 0.1:
            context_parts.append("Low Volatility")
        
        # Add pattern-specific context
        if pattern_label != "No pattern detected":
            if "Head and shoulders" in pattern_label:
                if "bottom" in pattern_label.lower():
                    context_parts.append("Reversal Pattern - Bullish")
                else:
                    context_parts.append("Reversal Pattern - Bearish")
            elif "Triangle" in pattern_label:
                context_parts.append("Continuation Pattern")
            elif "W_Bottom" in pattern_label:
                context_parts.append("Double Bottom - Bullish")
        
        # Add confidence context
        if confidence > 0.8:
            context_parts.append("High Confidence")
        elif confidence > 0.5:
            context_parts.append("Medium Confidence")
        else:
            context_parts.append("Low Confidence")
        
        return " | ".join(context_parts)

    def run(self, show_window: bool = True, capture_preview: bool = True) -> None:
        region = self.config.capture_region.to_mss_monitor()
        last_capture_time = 0.0
        frame_count = 0

        with mss.mss() as sct:
            while True:
                sct_img = sct.grab(region)
                frame = np.array(sct_img)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

                current_time = time.time()
                do_infer = (current_time - last_capture_time) >= self.config.screenshot_interval_sec

                annotated_for_preview = frame
                if do_infer:
                    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
                    image_name = f"frame_{timestamp}_{frame_count}.png"
                    image_path = os.path.join(self.screenshots_path, image_name)
                    cv2.imwrite(image_path, frame)

                    results = self.model(image_path, save=False, verbose=False)
                    annotated_for_preview = self._annotate_and_log(frame, results)

                    self._init_video_writer(annotated_for_preview)
                    self.video_writer.write(annotated_for_preview)

                    frame_count += 1
                    last_capture_time = current_time

                if show_window and capture_preview:
                    cv2.imshow("Pattern Detection", annotated_for_preview)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

        if self.video_writer is not None:
            self.video_writer.release()
        cv2.destroyAllWindows()


def run_realtime(
    model_repo_or_path: str = "foduucom/stockmarket-pattern-detection-yolov8",
    top: int = 0,
    left: int = 683,
    width: int = 683,
    height: int = 768,
    interval: float = 60.0,
    fps: float = 0.5,
    conf: float = 0.25,
    historical_data_path: Optional[str] = None,
) -> None:
    config = RuntimeConfig(
        model_repo_or_path=model_repo_or_path,
        capture_region=CaptureRegion(top=top, left=left, width=width, height=height),
        screenshot_interval_sec=interval,
        video_fps=fps,
        confidence_threshold=conf,
        historical_data_path=historical_data_path,
    )
    detector = RealTimePatternDetector(config)
    detector.run(show_window=True, capture_preview=True)


