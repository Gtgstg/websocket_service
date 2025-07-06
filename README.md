# TradingView → SmartAPI Webhook Service

This FastAPI micro-service receives webhook alerts from TradingView and converts them into live **AngelOne SmartAPI** orders.

## Features

* Simple `/webhook` POST endpoint
* Parses alert strings containing `order BUY/SELL @ <qty>`
* Generates TOTP on the fly and logs in to SmartAPI
* Market order execution (extend parameters for stop-loss, target, etc.)
* Symbol-token mapping via environment variable

## Quick Start

```bash
# 1. Install deps (preferably inside a venv)
python -m pip install -r requirements.txt

# 2. Export SmartAPI creds
set SMARTAPI_KEY=xxxxxx
set SMARTAPI_CLIENT_ID=YYYYY
set SMARTAPI_PASSWORD=ZZZZZ
set SMARTAPI_TOTP_SECRET=ABCDEF1234567890

# Optional custom symbol map (JSON)
set SYMBOL_MAP={"BTCUSDT":"12345"}

# 3. Run server (reload for dev)
uvicorn webhook_service.main:app --reload --port 8000
```

## TradingView Alert Setup

1. Create/modify your alert.
2. Tick **“Send webhook”** and point it to `http://<public-url>/webhook` (use a tunnel like [ngrok](https://ngrok.com/) if required).
3. Tick **“Send as JSON”** and set the message field to something like:

```json
{"message":"BTC Scalping v9.29: order {{strategy.order.action}} @ {{strategy.order.contracts}} filled on {{ticker}}. New strategy position is {{strategy.position_size}}"}
```

## Extending

* Adjust `_ALERT_REGEX` if your alert text changes.
* Add stop-loss/target params in `order_params`.
* Persist sessions or use SmartAPI’s refresh mechanism instead of logging in per request for production use.
