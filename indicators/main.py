import pandas as pd
import plotly.graph_objects as go

# Read only the first 5000 rows for performance
csv_path = 'websocket_service/BankNiftyFutures.csv'
df = pd.read_csv(csv_path, nrows=5000)

# Parse datetime for plotting
if 'Date' in df.columns and 'Time' in df.columns:
    df['datetime'] = pd.to_datetime(df['Date'].astype(str) + ' ' + df['Time'].astype(str))
else:
    df['datetime'] = pd.to_datetime(df['Date'])

# Create interactive bar chart
fig = go.Figure(data=[
    go.Bar(x=df['datetime'], y=df['Close 5'], name='Close Price', marker_color='blue')
])
fig.update_layout(
    title='BankNifty Futures Close Price (Bar Chart, First 5000 rows)',
    xaxis_title='Datetime',
    yaxis_title='Close Price',
    xaxis=dict(rangeslider=dict(visible=True), type='date'),
    template='plotly_white',
    height=600,
    width=1200
)

# Save as HTML for interactive viewing
fig.write_html('banknifty_close_bar_chart.html')
fig.show() 