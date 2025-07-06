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
    # TradingView sends the alert text inside the JSON field named `message`
    # if you check the "Send as JSON" checkbox. If user is not sending JSON
    # and just raw text, set contentType in webhook to 'text/plain' and
    # adjust logic accordingly.
    message: str

# Compile once at import time
_ALERT_REGEX = re.compile(
    r"order\s+(BUY|SELL|buy|sell)\s+@\s+([\d\.]+)\s+filled\s+on\s+([A-Z0-9_\-]+).*?position\s+is\s+([\-\d\.]+)",
    re.IGNORECASE,
)

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
    """Receive TradingView alert and place SmartAPI order."""
    match = _ALERT_REGEX.search(alert.message)
    if not match:
        raise HTTPException(status_code=400, detail="Alert text could not be parsed")

    action_str, contracts_str, ticker, position_size_str = match.groups()
    action = _ACTION_MAP[action_str.lower()]
    quantity = float(contracts_str)
    position_size = float(position_size_str)  # Currently unused, but captured for completeness

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
