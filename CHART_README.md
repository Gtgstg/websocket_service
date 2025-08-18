# Trading Chart System with Lightweight Charts

This system provides interactive candlestick charts with technical indicators using the lightweight-charts library and your existing Python indicators.

## Features

- **Interactive Candlestick Charts**: Powered by lightweight-charts (JavaScript)
- **Technical Indicators**: SMA, EMA, RSI, MACD, Bollinger Bands, TRIX, and more
- **Real-time Data**: Loads data from your CSV files or live sources
- **Multiple Chart Types**: Both interactive HTML and static PNG charts
- **Responsive Design**: Works on desktop and mobile devices
- **Dark Theme**: Professional trading interface

## Files Overview

- `trading_chart.html` - Main chart interface (HTML + JavaScript)
- `chart_server.py` - Python FastAPI server for data and indicators
- `charting.py` - Python charting utilities (Plotly + MPLFinance)
- `test_chart.py` - Test script to verify the system works
- `package.json` - Node.js dependencies (lightweight-charts)

## Quick Start

### 1. Install Dependencies

```bash
# Install Python dependencies
pip install -r requirements.txt

# Install Node.js dependencies (if not already done)
npm install
```

### 2. Test the System

```bash
# Run the test script
python test_chart.py
```

### 3. Start the Chart Server

```bash
# Start the FastAPI server
python chart_server.py
```

### 4. Open in Browser

Navigate to `http://localhost:8000` to view the interactive chart.

## Usage

### Basic Chart Viewing

1. **Load Data**: Click "Load Data" to fetch data from your CSV files
2. **Adjust Period**: Change the period to view more or fewer data points
3. **Select Symbol**: Choose from available symbols (BANKNIFTY, NIFTY, SENSEX)
4. **Timeframe**: Select different time intervals

### Technical Indicators

The chart automatically displays:
- **Moving Averages**: SMA 20, SMA 50, SMA 200
- **RSI**: Relative Strength Index (14-period)
- **MACD**: Moving Average Convergence Divergence
- **Bollinger Bands**: 20-period with 2 standard deviations
- **TRIX**: Triple Exponential Average (if available)

### Chart Controls

- **Zoom**: Use mouse wheel or pinch gestures
- **Pan**: Click and drag to move around the chart
- **Crosshair**: Hover to see exact values
- **Legend**: Click indicator names to show/hide them

## API Endpoints

The chart server provides these REST endpoints:

- `GET /` - Main chart page
- `GET /api/data?period=100` - Get chart data with indicators
- `GET /api/latest` - Get latest indicator values
- `GET /api/indicators` - List available indicators
- `POST /api/refresh` - Refresh data and recalculate indicators

## Data Format

The system expects CSV files with these columns:
- `Open`, `High`, `Low`, `Close`, `Volume`
- Date column (automatically detected)

If columns are missing, the system will:
- Generate reasonable defaults
- Map common column names
- Create sample data if needed

## Customization

### Adding New Indicators

1. **Python Backend**: Add calculation logic in `chart_server.py`
2. **Frontend**: Update `trading_chart.html` to display the indicator
3. **Styling**: Modify colors and display options

### Chart Styling

Edit the CSS in `trading_chart.html` to customize:
- Colors and themes
- Layout and spacing
- Typography and fonts
- Responsive behavior

### Data Sources

Modify `ChartDataProvider` in `chart_server.py` to:
- Connect to live data feeds
- Support different file formats
- Add real-time updates
- Implement caching

## Troubleshooting

### Common Issues

1. **"No data available"**
   - Check if CSV file exists and is readable
   - Verify column names match expected format
   - Check file permissions

2. **"Indicators not loading"**
   - Ensure all Python dependencies are installed
   - Check for import errors in indicator modules
   - Verify data quality (no NaN values)

3. **"Chart not displaying"**
   - Check browser console for JavaScript errors
   - Verify lightweight-charts library is loaded
   - Check network requests in browser dev tools

### Debug Mode

Enable debug logging by modifying `chart_server.py`:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Performance Tips

- **Data Period**: Use smaller periods (50-200) for better performance
- **Caching**: Implement data caching for large datasets
- **Lazy Loading**: Load indicators on demand
- **Web Workers**: Use background threads for heavy calculations

## Advanced Features

### Real-time Updates

To add real-time data updates:

1. Implement WebSocket connection in `chart_server.py`
2. Add real-time data handling in `trading_chart.html`
3. Use `chart.update()` for live updates

### Multiple Timeframes

Support for different timeframes:

1. Add timeframe selection in the UI
2. Implement data aggregation logic
3. Update chart data based on selection

### Export Options

Export charts as:
- PNG images
- PDF documents
- CSV data
- JSON format

## Integration with Existing Systems

### Webhook Integration

The chart system can integrate with your existing webhook service:

1. Add chart endpoints to your main FastAPI app
2. Share data between services
3. Trigger chart updates on new data

### Database Integration

Connect to databases instead of CSV files:

1. Modify `ChartDataProvider.load_data()`
2. Add database connection logic
3. Implement query optimization

### External APIs

Connect to external data sources:

1. Add API client classes
2. Implement rate limiting
3. Handle authentication

## Contributing

To contribute to the chart system:

1. Follow the existing code structure
2. Add comprehensive tests
3. Update documentation
4. Ensure backward compatibility

## License

This charting system is part of your trading bot project. Use according to your project's license terms.

## Support

For issues and questions:
1. Check the troubleshooting section
2. Review error logs
3. Test with sample data
4. Verify dependencies are correctly installed

---

**Happy Trading! 📈**

