import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import mplfinance as mpf
from typing import Dict, List, Optional, Tuple
import os

class TradingChart:
    """
    A comprehensive charting class for creating candlestick charts with technical indicators.
    Supports both Plotly (interactive) and MPLFinance (static) charting.
    """
    
    def __init__(self, data: pd.DataFrame):
        """
        Initialize the charting class with OHLCV data.
        
        Args:
            data (pd.DataFrame): DataFrame with columns ['Open', 'High', 'Low', 'Close', 'Volume']
                                and datetime index
        """
        self.data = data.copy()
        self.indicators = {}
        
        # Ensure data has the required columns
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        if not all(col in self.data.columns for col in required_columns):
            raise ValueError(f"Data must contain columns: {required_columns}")
        
        # Ensure index is datetime
        if not isinstance(self.data.index, pd.DatetimeIndex):
            self.data.index = pd.to_datetime(self.data.index)
    
    def add_indicator(self, name: str, values: pd.Series, color: str = 'blue', 
                     overlay: bool = False, panel: int = 1):
        """
        Add a technical indicator to the chart.
        
        Args:
            name (str): Name of the indicator
            values (pd.Series): Indicator values
            color (str): Color for the indicator
            overlay (bool): If True, overlay on main chart; if False, create separate panel
            panel (int): Panel number for non-overlay indicators
        """
        self.indicators[name] = {
            'values': values,
            'color': color,
            'overlay': overlay,
            'panel': panel
        }
    
    def create_plotly_chart(self, title: str = "Trading Chart", 
                           height: int = 800) -> go.Figure:
        """
        Create an interactive Plotly chart with candlesticks and indicators.
        
        Args:
            title (str): Chart title
            height (int): Chart height in pixels
            
        Returns:
            go.Figure: Plotly figure object
        """
        # Determine number of panels needed
        panels = set()
        for indicator in self.indicators.values():
            if not indicator['overlay']:
                panels.add(indicator['panel'])
        
        num_panels = max(panels) if panels else 1
        
        # Create subplots
        fig = make_subplots(
            rows=num_panels + 1,  # +1 for main candlestick chart
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            subplot_titles=[title] + [f"Panel {i}" for i in range(1, num_panels + 1)],
            row_heights=[0.6] + [0.4/num_panels] * num_panels if num_panels > 0 else [1]
        )
        
        # Add candlestick chart
        fig.add_trace(
            go.Candlestick(
                x=self.data.index,
                open=self.data['Open'],
                high=self.data['High'],
                low=self.data['Low'],
                close=self.data['Close'],
                name='OHLC',
                increasing_line_color='green',
                decreasing_line_color='red'
            ),
            row=1, col=1
        )
        
        # Add volume bars
        fig.add_trace(
            go.Bar(
                x=self.data.index,
                y=self.data['Volume'],
                name='Volume',
                marker_color='rgba(0, 0, 255, 0.3)',
                yaxis='y2'
            ),
            row=1, col=1
        )
        
        # Add overlay indicators
        for name, indicator in self.indicators.items():
            if indicator['overlay']:
                fig.add_trace(
                    go.Scatter(
                        x=self.data.index,
                        y=indicator['values'],
                        name=name,
                        line=dict(color=indicator['color']),
                        yaxis='y'
                    ),
                    row=1, col=1
                )
        
        # Add non-overlay indicators to separate panels
        for name, indicator in self.indicators.items():
            if not indicator['overlay']:
                panel_row = indicator['panel'] + 1
                fig.add_trace(
                    go.Scatter(
                        x=self.data.index,
                        y=indicator['values'],
                        name=name,
                        line=dict(color=indicator['color']),
                        yaxis=f'y{panel_row + 2}'  # +2 because main chart uses y and y2
                    ),
                    row=panel_row, col=1
                )
        
        # Update layout
        fig.update_layout(
            title=title,
            height=height,
            xaxis_rangeslider_visible=False,
            showlegend=True,
            template='plotly_dark'
        )
        
        # Update y-axes
        fig.update_yaxes(title_text="Price", row=1, col=1)
        fig.update_yaxes(title_text="Volume", row=1, col=1, secondary_y=True)
        
        for i in range(1, num_panels + 1):
            fig.update_yaxes(title_text=f"Indicator {i}", row=i+1, col=1)
        
        return fig
    
    def create_mplfinance_chart(self, title: str = "Trading Chart", 
                               save_path: Optional[str] = None,
                               style: str = 'charles') -> None:
        """
        Create a static MPLFinance chart and optionally save it.
        
        Args:
            title (str): Chart title
            save_path (str, optional): Path to save the chart image
            style (str): MPLFinance style ('charles', 'binance', 'blueskies', etc.)
        """
        # Prepare data for MPLFinance
        plot_data = self.data.copy()
        
        # Add overlay indicators
        for name, indicator in self.indicators.items():
            if indicator['overlay']:
                plot_data[name] = indicator['values']
        
        # Create additional plots for non-overlay indicators
        additional_plots = []
        for name, indicator in self.indicators.items():
            if not indicator['overlay']:
                additional_plots.append(
                    mpf.make_addplot(indicator['values'], panel=indicator['panel'], 
                                   color=indicator['color'], title=name)
                )
        
        # Create the chart
        mpf.plot(
            plot_data,
            type='candle',
            volume=True,
            title=title,
            style=style,
            addplot=additional_plots,
            savefig=save_path,
            figsize=(15, 10),
            panel_ratios=(6, 1)  # 6:1 ratio for price:volume
        )
    
    def save_plotly_html(self, filepath: str, title: str = "Trading Chart"):
        """
        Save the Plotly chart as an interactive HTML file.
        
        Args:
            filepath (str): Path to save the HTML file
            title (str): Chart title
        """
        fig = self.create_plotly_chart(title)
        fig.write_html(filepath)
        print(f"Chart saved to: {filepath}")
    
    def save_mplfinance_png(self, filepath: str, title: str = "Trading Chart"):
        """
        Save the MPLFinance chart as a PNG file.
        
        Args:
            filepath (str): Path to save the PNG file
            title (str): Chart title
        """
        self.create_mplfinance_chart(title, save_path=filepath)
        print(f"Chart saved to: {filepath}")


def create_sample_chart_with_indicators(data_path: str, output_dir: str = "charts"):
    """
    Create a sample chart with multiple indicators from CSV data.
    
    Args:
        data_path (str): Path to the CSV file with OHLCV data
        output_dir (str): Directory to save the output charts
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    print(f"Loading data from: {data_path}")
    df = pd.read_csv(data_path)
    
    # Ensure we have the required columns
    required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    if not all(col in df.columns for col in required_columns):
        print("Warning: CSV doesn't have all required columns. Attempting to map common names...")
        # Try to map common column names
        column_mapping = {
            'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume',
            'OPEN': 'Open', 'HIGH': 'High', 'LOW': 'Low', 'CLOSE': 'Close', 'VOLUME': 'Volume'
        }
        
        for old_name, new_name in column_mapping.items():
            if old_name in df.columns:
                df[new_name] = df[old_name]
    
    # Convert date column to datetime index
    date_columns = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()]
    if date_columns:
        try:
            # Try different date formats
            df['Date'] = pd.to_datetime(df[date_columns[0]], format='mixed', dayfirst=True)
        except:
            try:
                # Try with dayfirst=True for DD-MM-YYYY format
                df['Date'] = pd.to_datetime(df[date_columns[0]], dayfirst=True)
            except:
                # Fallback to automatic parsing
                df['Date'] = pd.to_datetime(df[date_columns[0]], errors='coerce')
        
        df.set_index('Date', inplace=True)
    else:
        # Create a simple index if no date column found
        df.index = pd.to_datetime(range(len(df)))
    
    # Clean data
    df = df.dropna()
    
    # Create chart instance
    chart = TradingChart(df)
    
    # Add some common indicators (you can customize these)
    try:
        # Simple Moving Averages
        chart.add_indicator('SMA 20', df['Close'].rolling(window=20).mean(), 'orange', overlay=True)
        chart.add_indicator('SMA 50', df['Close'].rolling(window=50).mean(), 'blue', overlay=True)
        
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        chart.add_indicator('RSI', rsi, 'purple', overlay=False, panel=1)
        
        # MACD
        exp1 = df['Close'].ewm(span=12, adjust=False).mean()
        exp2 = df['Close'].ewm(span=26, adjust=False).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=9, adjust=False).mean()
        chart.add_indicator('MACD', macd, 'blue', overlay=False, panel=2)
        chart.add_indicator('Signal', signal, 'red', overlay=False, panel=2)
        
        # Bollinger Bands
        sma_20 = df['Close'].rolling(window=20).mean()
        std_20 = df['Close'].rolling(window=20).std()
        upper_band = sma_20 + (std_20 * 2)
        lower_band = sma_20 - (std_20 * 2)
        chart.add_indicator('Upper BB', upper_band, 'gray', overlay=True)
        chart.add_indicator('Lower BB', lower_band, 'gray', overlay=True)
        
    except Exception as e:
        print(f"Warning: Could not calculate some indicators: {e}")
    
    # Save charts
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    
    # Save Plotly HTML
    html_path = os.path.join(output_dir, f"trading_chart_{timestamp}.html")
    chart.save_plotly_html(html_path, "Trading Chart with Indicators")
    
    # Save MPLFinance PNG
    png_path = os.path.join(output_dir, f"trading_chart_{timestamp}.png")
    chart.save_mplfinance_png(png_path, "Trading Chart with Indicators")
    
    print(f"Charts created successfully!")
    print(f"Interactive HTML: {html_path}")
    print(f"Static PNG: {png_path}")
    
    return chart


if __name__ == "__main__":
    # Example usage
    csv_path = "BankNiftyFutures.csv"
    if os.path.exists(csv_path):
        create_sample_chart_with_indicators(csv_path)
    else:
        print(f"CSV file not found: {csv_path}")
        print("Please ensure the CSV file exists in the current directory.")
