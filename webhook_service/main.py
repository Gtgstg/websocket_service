from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
import os
import re
import pyotp
from SmartApi import SmartConnect

"""
FastAPI service that receives TradingView webhook alerts and places
orders via AngelOne SmartAPI. It expects an alert message similar to:

    BTC Scalping v9.29: ... order BUY @ 1 filled on BTCUSDT. New strategy position is 1

FastAPI will listen on /webhook and parse the alert, then submit a
MARKET order. Environment variables are used for SmartAPI credentials
and symbol mappings.

Required environment variables
-----------------------------
SMARTAPI_KEY           : API key issued by AngelOne/SmartAPI
SMARTAPI_CLIENT_ID     : Client/Username used to login
SMARTAPI_PASSWORD      : Password (2FA not included)
SMARTAPI_TOTP_SECRET   : TOTP secret for generating 2-FA pins

Optionally, you can add SYMBOL_MAP (JSON string) to override default
symbol-token mapping.

Run with:

    uvicorn webhook_service.main:app --reload --port 8000

"""

app = FastAPI(title="TradingView Webhook → SmartAPI bridge")

class Alert(BaseModel):
    """Payload schema for TradingView webhook."""
    version: str
    description: str
    parameters: list[float | int]
    order_action: str
    order_contracts: str | int | float  # Can be string, integer, or float
    ticker: str
    position_size: str | int | float  # Can be string, integer, or float
    message: str

# Action mapping for case-insensitive comparison
_ACTION_MAP = {"buy": "BUY", "sell": "SELL"}

# ---- SmartAPI helpers ------------------------------------------------------

def _build_smart_api() -> SmartConnect:
    """Login and return a ready-to-use SmartConnect instance."""
    api_key = os.getenv("SMARTAPI_KEY")
    client_id = os.getenv("SMARTAPI_CLIENT_ID")
    password = os.getenv("SMARTAPI_PASSWORD")
    totp_secret = os.getenv("SMARTAPI_TOTP_SECRET")

    if not all([api_key, client_id, password, totp_secret]):
        raise RuntimeError("Missing one or more SmartAPI env vars")

    smart = SmartConnect(api_key=api_key)

    # Generate TOTP-based 2FA pin
    otp = pyotp.TOTP(totp_secret).now()
    auth_data = smart.generateSession(client_id, password, otp)
    if auth_data.get("status") != True:
        raise RuntimeError(f"Unable to login: {auth_data}")

    # This call populates smart.feed_token
    smart.getfeedToken()
    return smart


def _symbol_token_map() -> dict:
    """Return mapping {TICKER: TOKEN, ...}."""
    # For production you probably want to load from DB or a config file.
    # Here we allow overriding via SYMBOL_MAP env var containing JSON.
    import json

    default_map = {
        "BTCUSDT": "12345", # TODO: replace with actual token
        "BANKNIFTY": "99926009",
    }

    env_val = os.getenv("SYMBOL_MAP")
    if env_val:
        try:
            default_map.update(json.loads(env_val))
        except json.JSONDecodeError as exc:
            raise RuntimeError("Invalid SYMBOL_MAP env var, expected JSON") from exc
    return default_map


# ---- Route -----------------------------------------------------------------

@app.post("/webhook", summary="TradingView alert handler")
async def tradingview_webhook(alert: Alert):
    """Receive TradingView alert and place SmartAPI order.
    
    Expected JSON format:
    {
        "version": "Strategy Name vX.XX",
        "description": "Strategy description",
        "parameters": [...],
        "order_action": "BUY" or "SELL",
        "order_contracts": "1.0",
        "ticker": "SYMBOL",
        "position_size": "1.0",
        "message": "Human readable message"
    }
    """
    try:
        action = _ACTION_MAP[alert.order_action.lower()]
        # Convert to float first to handle both string and numeric types, then to int for quantity
        quantity = int(float(alert.order_contracts))
        ticker = alert.ticker
        position_size = float(alert.position_size)  # Keep as float for position size
        
        # Print the received JSON for debugging
        print(f"Received webhook payload: {alert.dict()}")
        print(f"Converted order details - Action: {action}, Quantity: {quantity}, Ticker: {ticker}")
    except (ValueError, KeyError, TypeError) as e:
        raise HTTPException(status_code=400, detail=f"Invalid order data: {str(e)}")

    # Resolve SmartAPI symbol token
    token_map = _symbol_token_map()
    if ticker not in token_map:
        raise HTTPException(status_code=400, detail=f"Ticker {ticker} not mapped to SmartAPI token")

    symbol_token = token_map[ticker]

    # Place a market order
    try:
        smart = _build_smart_api()
        order_params = {
            "variety": "NORMAL",
            "tradingsymbol": ticker,
            "symboltoken": symbol_token,
            "transactiontype": action,  # BUY / SELL
            "exchange": "NSE",  # or 'NFO', etc. customise per instrument
            "ordertype": "MARKET",
            "producttype": "INTRADAY",
            "duration": "DAY",
            "price": "0",  # Market order
            "squareoff": "0",
            "stoploss": "0",  # Add stop-loss here if needed
            "quantity": str(int(quantity)),
        }
        order_id = smart.placeOrder(order_params)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Order placement failed: {exc}") from exc

    return {"status": "success", "order_id": order_id, "ticker": ticker, "action": action}
