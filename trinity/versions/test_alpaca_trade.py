#!/usr/bin/env python3
"""
Test single Alpaca trade to verify API works
"""

import os
import requests
import uuid
from dotenv import load_dotenv

# Load environment
load_dotenv('/opt/cashmachine/trinity/.env')

# Get credentials
api_key = os.environ.get('ALPACA_API_KEY')
api_secret = os.environ.get('ALPACA_API_SECRET')

print(f"API Key: {api_key[:10]}...")
print(f"API Secret: {api_secret[:10]}...")

# Try account endpoint
headers = {
    'APCA-API-KEY-ID': api_key,
    'APCA-API-SECRET-KEY': api_secret
}

# Test connection
print("\n📡 Testing Alpaca connection...")
resp = requests.get(
    'https://paper-api.alpaca.markets/v2/account',
    headers=headers,
    timeout=10
)

print(f"Status: {resp.status_code}")
if resp.status_code == 200:
    account = resp.json()
    print(f"✅ Connected! Cash: ${account.get('cash', 0)}")
    print(f"Buying Power: ${account.get('buying_power', 0)}")
    
    # Try a tiny test trade
    print("\n🚀 Placing test trade: BUY 0.1 BTCUSD...")
    
    order_data = {
        'symbol': 'BTCUSD',
        'qty': '0.1',
        'side': 'buy',
        'type': 'market',
        'time_in_force': 'gtc'
    }
    
    resp = requests.post(
        'https://paper-api.alpaca.markets/v2/orders',
        json=order_data,
        headers={**headers, 'Idempotency-Key': f"test-{uuid.uuid4()}"},
        timeout=10
    )
    
    if resp.status_code in [200, 201]:
        order = resp.json()
        print(f"✅✅✅ REAL TRADE PLACED!")
        print(f"Order ID: {order['id']}")
        print(f"Symbol: {order['symbol']}")
        print(f"Status: {order['status']}")
    else:
        print(f"❌ Trade failed: {resp.status_code}")
        print(resp.text)
else:
    print(f"❌ Connection failed: {resp.text}")