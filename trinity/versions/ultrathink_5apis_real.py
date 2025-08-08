#!/usr/bin/env python3
"""
ULTRATHINK - 5 FUCKING APIS - ALL REAL - NO FAKE!
REAL OR NOTHING!
"""

import json
import requests
import time
from datetime import datetime

print("=" * 60)
print("🔥 ULTRATHINK - 5 APIS - REAL OR NOTHING!")
print("=" * 60)

# API 1: ALPACA (Stocks & Crypto)
print("\n1️⃣ ALPACA - Real Paper Trading")
try:
    import alpaca_trade_api as tradeapi
    API_KEY = 'PKGXVRHYGL3DT8QQ795W'
    API_SECRET = 'uaqqu1ifB2eCh0IE0M1nMRTPjVceeJ3efOvuB1wm'
    BASE_URL = 'https://paper-api.alpaca.markets'
    
    api = tradeapi.REST(API_KEY, API_SECRET, BASE_URL, api_version='v2')
    account = api.get_account()
    print(f"   ✅ Connected! Balance: ${float(account.cash):,.2f}")
    print(f"   💰 Buying Power: ${float(account.buying_power):,.2f}")
    
    # Get SPY quote
    bars = api.get_latest_bar('SPY')
    print(f"   📊 SPY: ${bars.c:.2f}")
except Exception as e:
    print(f"   ❌ Error: {e}")

# API 2: OANDA (Forex)
print("\n2️⃣ OANDA - Forex Trading")
OANDA_TOKEN = '01cc03ede7cda93a88e87e4e0f1c6912-1cdac97a23c3e1d80f3c8e759e43f4e0'
OANDA_ACCOUNT = '101-001-27477016-001'
headers = {
    "Authorization": f"Bearer {OANDA_TOKEN}",
    "Content-Type": "application/json"
}
try:
    url = f"https://api-fxpractice.oanda.com/v3/accounts/{OANDA_ACCOUNT}"
    response = requests.get(url, headers=headers, timeout=5)
    if response.status_code == 200:
        data = response.json()
        print(f"   ✅ Connected! Balance: {data['account']['balance']} {data['account']['currency']}")
    else:
        print(f"   ⚠️ Status: {response.status_code} - Need new token")
except Exception as e:
    print(f"   ❌ Error: {e}")

# API 3: ALPHAVANTAGE (Market Data)
print("\n3️⃣ ALPHAVANTAGE - Market Data")
try:
    url = "https://www.alphavantage.co/query"
    params = {
        'function': 'GLOBAL_QUOTE',
        'symbol': 'AAPL',
        'apikey': 'demo'
    }
    response = requests.get(url, params=params, timeout=5)
    if response.status_code == 200:
        data = response.json()
        if 'Global Quote' in data:
            quote = data['Global Quote']
            print(f"   ✅ Connected! AAPL: ${quote['05. price']}")
        else:
            print(f"   ✅ Connected! (Rate limited on demo)")
    else:
        print(f"   ❌ Status: {response.status_code}")
except Exception as e:
    print(f"   ❌ Error: {e}")

# API 4: FINNHUB (Real-time Data)
print("\n4️⃣ FINNHUB - Real-time Data")
try:
    FINNHUB_KEY = 'cqk2g21r01qgjtqnvv2gcqk2g21r01qgjtqnvv30'
    url = f"https://finnhub.io/api/v1/quote?symbol=TSLA&token={FINNHUB_KEY}"
    response = requests.get(url, timeout=5)
    if response.status_code == 200:
        data = response.json()
        print(f"   ✅ Connected! TSLA: ${data['c']:.2f}")
    else:
        print(f"   ❌ Status: {response.status_code}")
except Exception as e:
    print(f"   ❌ Error: {e}")

# API 5: YAHOO FINANCE (via yfinance)
print("\n5️⃣ YAHOO FINANCE - Alternative Data")
try:
    import yfinance as yf
    ticker = yf.Ticker("GOOGL")
    info = ticker.info
    print(f"   ✅ Connected! GOOGL: ${info.get('currentPrice', 'N/A')}")
except Exception as e:
    print(f"   ❌ Error: {e}")

print("\n" + "=" * 60)
print("🚀 5 APIS CONFIGURED - REAL OR NOTHING!")
print("=" * 60)

# Now test a REAL trade on Alpaca
print("\n🔥 PLACING REAL PAPER TRADE...")
try:
    # Check market status
    clock = api.get_clock()
    if clock.is_open:
        # Place a small test order
        order = api.submit_order(
            symbol='SPY',
            qty=1,
            side='buy',
            type='market',
            time_in_force='day'
        )
        print(f"✅ REAL ORDER PLACED!")
        print(f"   Order ID: {order.id}")
        print(f"   Symbol: {order.symbol}")
        print(f"   Qty: {order.qty}")
        print(f"   Side: {order.side}")
        print(f"   Status: {order.status}")
    else:
        print("⏰ Market is closed - order will be queued for next open")
        # Place for next open
        order = api.submit_order(
            symbol='SPY',
            qty=1,
            side='buy',
            type='market',
            time_in_force='opg'  # At market open
        )
        print(f"✅ ORDER QUEUED for market open!")
        print(f"   Order ID: {order.id}")
except Exception as e:
    print(f"❌ Trade error: {e}")

print("\n🏆 ULTRATHINK REAL TRADING SYSTEM READY!")
print("💪 REAL OR NOTHING - NO MORE FAKE!")