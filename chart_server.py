from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import pandas as pd
import numpy as np
import json
from typing import Dict, List, Optional
import os
from datetime import datetime, timedelta

# Import your existing indicators
from indicators.TRIX import TRIXIndicator
from indicators.Bollinger import BollingerBands
from indicators.ema import EMA

app = FastAPI(title="Trading Chart Server")

# Mount static files
app.mount("/static", StaticFiles(directory="."), name="static")

class ChartDataProvider:
    """Provides chart data and calculates indicators for the frontend."""
    
    def __init__(self, csv_path: str = "BankNiftyFutures.csv"):
        self.csv_path = csv_path
        self.data = None
        self.indicators = {}
        self.load_data()
    
    def load_data(self):
        """Load and prepare CSV data for charting."""
        try:
            print(f"Loading data from: {self.csv_path}")
            df = pd.read_csv(self.csv_path)
            
            # Map common column names to standard OHLCV format
            column_mapping = {
                'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume',
                'OPEN': 'Open', 'HIGH': 'High', 'LOW': 'Low', 'CLOSE': 'Close', 'VOLUME': 'Volume',
                'Open': 'Open', 'High': 'High', 'Low': 'Low', 'Close': 'Close', 'Volume': 'Volume'
            }
            
            # Rename columns if needed
            for old_name, new_name in column_mapping.items():
                if old_name in df.columns:
                    df[new_name] = df[old_name]
            
            # Find date column
            date_columns = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()]
            if date_columns:
                df['Date'] = pd.to_datetime(df[date_columns[0]])
            else:
                # Create a simple index if no date column found
                df['Date'] = pd.to_datetime(range(len(df)))
            
            df.set_index('Date', inplace=True)
            
            # Ensure we have required columns
            required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            if not all(col in df.columns for col in required_columns):
                # Fill missing columns with reasonable defaults
                if 'Volume' not in df.columns:
                    df['Volume'] = 1000000  # Default volume
                if 'Open' not in df.columns and 'Close' in df.columns:
                    df['Open'] = df['Close'].shift(1).fillna(df['Close'])
                if 'High' not in df.columns:
                    df['High'] = df[['Open', 'Close']].max(axis=1)
                if 'Low' not in df.columns:
                    df['Low'] = df[['Open', 'Close']].min(axis=1)
            
            # Clean data
            df = df.dropna()
            
            # Convert to numeric
            for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            df = df.dropna()
            
            self.data = df
            print(f"Data loaded successfully. Shape: {df.shape}")
            print(f"Columns: {df.columns.tolist()}")
            print(f"Date range: {df.index.min()} to {df.index.max()}")
            
        except Exception as e:
            print(f"Error loading data: {e}")
            # Create sample data if CSV loading fails
            self.create_sample_data()
    
    def create_sample_data(self):
        """Create sample data if CSV loading fails."""
        print("Creating sample data...")
        dates = pd.date_range(start='2023-01-01', end='2024-01-01', freq='D')
        np.random.seed(42)
        
        base_price = 45000
        prices = []
        for i in range(len(dates)):
            change = np.random.normal(0, 100)
            base_price += change
            open_price = base_price
            close_price = base_price + np.random.normal(0, 50)
            high_price = max(open_price, close_price) + np.random.uniform(0, 100)
            low_price = min(open_price, close_price) - np.random.uniform(0, 100)
            volume = np.random.randint(100000, 1000000)
            
            prices.append({
                'Open': open_price,
                'High': high_price,
                'Low': low_price,
                'Close': close_price,
                'Volume': volume
            })
        
        self.data = pd.DataFrame(prices, index=dates)
        print("Sample data created successfully.")
    
    def calculate_indicators(self):
        """Calculate all technical indicators."""
        if self.data is None or len(self.data) == 0:
            return {}
        
        try:
            # Calculate basic indicators
            self.indicators = {}
            
            # Simple Moving Averages
            self.indicators['SMA_20'] = self.data['Close'].rolling(window=20).mean()
            self.indicators['SMA_50'] = self.data['Close'].rolling(window=50).mean()
            self.indicators['SMA_200'] = self.data['Close'].rolling(window=200).mean()
            
            # Exponential Moving Averages
            self.indicators['EMA_12'] = self.data['Close'].ewm(span=12, adjust=False).mean()
            self.indicators['EMA_26'] = self.data['Close'].ewm(span=26, adjust=False).mean()
            
            # RSI
            delta = self.data['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            self.indicators['RSI'] = 100 - (100 / (1 + rs))
            
            # MACD
            self.indicators['MACD'] = self.indicators['EMA_12'] - self.indicators['EMA_26']
            self.indicators['MACD_Signal'] = self.indicators['MACD'].ewm(span=9, adjust=False).mean()
            self.indicators['MACD_Histogram'] = self.indicators['MACD'] - self.indicators['MACD_Signal']
            
            # Bollinger Bands
            sma_20 = self.data['Close'].rolling(window=20).mean()
            std_20 = self.data['Close'].rolling(window=20).std()
            self.indicators['BB_Upper'] = sma_20 + (std_20 * 2)
            self.indicators['BB_Lower'] = sma_20 - (std_20 * 2)
            self.indicators['BB_Middle'] = sma_20
            
            # TRIX (if available)
            try:
                trix_indicator = TRIXIndicator()
                trix_data = trix_indicator.calculate(self.data['Close'])
                self.indicators['TRIX'] = trix_data['trix']
                self.indicators['TRIX_Signal'] = trix_data['signal']
            except Exception as e:
                print(f"TRIX calculation failed: {e}")
            
            # Bollinger Bands (if available)
            try:
                bb_indicator = BollingerBands()
                bb_data = bb_indicator.calculate(self.data['Close'])
                self.indicators['BB_Upper'] = bb_data['upper']
                self.indicators['BB_Lower'] = bb_data['lower']
                self.indicators['BB_Middle'] = bb_data['middle']
            except Exception as e:
                print(f"Bollinger Bands calculation failed: {e}")
            
            # EMA (if available)
            try:
                ema_indicator = EMA(period=12)
                ema_data = ema_indicator.calculate(self.data['Close'])
                self.indicators['EMA_12'] = ema_data['ema']
            except Exception as e:
                print(f"EMA calculation failed: {e}")
            
            print("Indicators calculated successfully.")
            
        except Exception as e:
            print(f"Error calculating indicators: {e}")
    
    def get_chart_data(self, period: int = 100) -> Dict:
        """Get formatted data for the chart."""
        if self.data is None:
            return {"error": "No data available"}
        
        # Calculate indicators if not already done
        if not self.indicators:
            self.calculate_indicators()
        
        # Get the last 'period' data points
        recent_data = self.data.tail(period)
        
        # Format data for lightweight-charts
        chart_data = {
            "candlesticks": [],
            "volume": [],
            "indicators": {}
        }
        
        for date, row in recent_data.iterrows():
            timestamp = int(date.timestamp())
            
            # Candlestick data
            chart_data["candlesticks"].append({
                "time": timestamp,
                "open": float(row['Open']),
                "high": float(row['High']),
                "low": float(row['Low']),
                "close": float(row['Close'])
            })
            
            # Volume data
            chart_data["volume"].append({
                "time": timestamp,
                "value": float(row['Volume']),
                "color": "#26a69a" if row['Close'] >= row['Open'] else "#ef5350"
            })
        
        # Add indicator data
        for name, values in self.indicators.items():
            if not values.empty:
                indicator_data = []
                for date, value in values.tail(period).items():
                    if pd.notna(value):
                        indicator_data.append({
                            "time": int(date.timestamp()),
                            "value": float(value)
                        })
                chart_data["indicators"][name] = indicator_data
        
        return chart_data
    
    def get_latest_values(self) -> Dict:
        """Get the latest indicator values for the dashboard."""
        if not self.indicators:
            self.calculate_indicators()
        
        latest = {}
        latest_price = self.data.iloc[-1]
        
        latest['price'] = {
            'open': float(latest_price['Open']),
            'high': float(latest_price['High']),
            'low': float(latest_price['Low']),
            'close': float(latest_price['Close']),
            'volume': float(latest_price['Volume'])
        }
        
        for name, values in self.indicators.items():
            if not values.empty and pd.notna(values.iloc[-1]):
                latest[name] = float(values.iloc[-1])
        
        return latest

# Initialize data provider
data_provider = ChartDataProvider()

@app.get("/", response_class=HTMLResponse)
async def get_chart_page():
    """Serve the main chart HTML page."""
    return FileResponse("trading_chart.html")

@app.get("/debug", response_class=HTMLResponse)
async def get_debug_chart_page():
    """Serve the debug chart HTML page."""
    return FileResponse("debug_chart.html")

@app.get("/api/data")
async def get_chart_data(period: int = 100):
    """Get chart data with indicators."""
    try:
        return data_provider.get_chart_data(period)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/latest")
async def get_latest_values():
    """Get latest indicator values."""
    try:
        return data_provider.get_latest_values()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/indicators")
async def get_available_indicators():
    """Get list of available indicators."""
    try:
        if not data_provider.indicators:
            data_provider.calculate_indicators()
        
        return {
            "indicators": list(data_provider.indicators.keys()),
            "data_points": len(data_provider.data) if data_provider.data is not None else 0
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/refresh")
async def refresh_data():
    """Refresh data and recalculate indicators."""
    try:
        data_provider.load_data()
        data_provider.calculate_indicators()
        return {"message": "Data refreshed successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    print("Starting Trading Chart Server...")
    print("Open http://localhost:8000 in your browser to view the chart")
    uvicorn.run(app, host="0.0.0.0", port=8000)
