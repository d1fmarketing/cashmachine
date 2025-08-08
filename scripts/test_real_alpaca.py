#!/usr/bin/env python3
"""
Test with REAL Alpaca credentials found in config
"""

import requests
import uuid

# REAL credentials from real_alpaca_config.py
api_key = 'PKCS3E7GZ5N9UDZEZ5BF'
api_secret = 'Y3BeJHStNv0CqonTONiPEfRkWlPdR4aokprHzmSP'

print(f"Using API Key: {api_key[:10]}...")

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
    print(f"✅ CONNECTED TO REAL ALPACA!")
    print(f"Cash: ${account.get('cash', 0)}")
    print(f"Buying Power: ${account.get('buying_power', 0)}")
    
    # Place a REAL test trade
    print("\n🚀 Placing REAL trade: BUY 0.1 BTCUSD...")
    
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
        headers={**headers, 'Idempotency-Key': f"ultra-{uuid.uuid4()}"},
        timeout=10
    )
    
    if resp.status_code in [200, 201]:
        order = resp.json()
        print(f"""
        ✅✅✅ REAL TRADE EXECUTED! ✅✅✅
        Order ID: {order['id']}
        Symbol: {order['symbol']}
        Qty: {order['qty']}
        Side: {order['side']}
        Status: {order['status']}
        """)
        
        # Get recent orders to verify
        print("\n📊 Verifying with broker...")
        resp = requests.get(
            'https://paper-api.alpaca.markets/v2/orders?limit=3',
            headers=headers,
            timeout=10
        )
        if resp.status_code == 200:
            orders = resp.json()
            print(f"Recent orders from broker: {len(orders)}")
            for o in orders:
                print(f"  - {o['side']} {o['qty']} {o['symbol']} - {o['status']}")
    else:
        print(f"❌ Trade failed: {resp.status_code}")
        print(resp.text)
else:
    print(f"❌ Connection failed: {resp.text}")