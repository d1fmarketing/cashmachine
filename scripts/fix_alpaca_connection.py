#\!/usr/bin/env python3
"""Quick fix for Alpaca API connection"""
import os
os.environ['APCA_API_KEY_ID'] = 'PKV1BCMHP5CRY8M5MW51'
os.environ['APCA_API_SECRET_KEY'] = 'gx1CjGcxlrUINFP9mxAKVxZ0p7TrfBMMzcdIPVYs'
os.environ['APCA_API_BASE_URL'] = 'https://paper-api.alpaca.markets'

import alpaca_trade_api as tradeapi

# Test connection
api = tradeapi.REST()
account = api.get_account()
print(f"✅ Connected\! Balance: ${account.cash}")
print(f"✅ Buying power: ${account.buying_power}")
print(f"✅ Portfolio value: ${account.portfolio_value}")
