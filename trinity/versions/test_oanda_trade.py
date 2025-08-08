#!/usr/bin/env python3
"""
Test OANDA practice trade to verify API works
"""

import os
import requests
from dotenv import load_dotenv

# Load environment
load_dotenv('/opt/cashmachine/trinity/.env')

# Get credentials
api_token = os.environ.get('OANDA_API_TOKEN')
account_id = os.environ.get('OANDA_ACCOUNT_ID')

print(f"API Token: {api_token[:20]}...")
print(f"Account ID: {account_id}")

headers = {
    'Authorization': f'Bearer {api_token}',
    'Content-Type': 'application/json'
}

# Test connection
print("\n📡 Testing OANDA connection...")
resp = requests.get(
    f'https://api-fxpractice.oanda.com/v3/accounts/{account_id}',
    headers=headers,
    timeout=10
)

print(f"Status: {resp.status_code}")
if resp.status_code == 200:
    account = resp.json()['account']
    print(f"✅ Connected! Balance: {account.get('balance', 0)}")
    print(f"Currency: {account.get('currency', 'USD')}")
    
    # Try a tiny test trade
    print("\n🚀 Placing test trade: BUY 100 EUR_USD...")
    
    order_data = {
        'order': {
            'instrument': 'EUR_USD',
            'units': '100',  # Small positive for buy
            'type': 'MARKET',
            'timeInForce': 'FOK',
            'positionFill': 'DEFAULT'
        }
    }
    
    resp = requests.post(
        f'https://api-fxpractice.oanda.com/v3/accounts/{account_id}/orders',
        json=order_data,
        headers=headers,
        timeout=10
    )
    
    if resp.status_code in [200, 201]:
        result = resp.json()
        if 'orderFillTransaction' in result:
            fill = result['orderFillTransaction']
            print(f"✅✅✅ REAL TRADE EXECUTED!")
            print(f"Order ID: {fill['id']}")
            print(f"Instrument: {fill['instrument']}")
            print(f"Units: {fill['units']}")
            print(f"Price: {fill.get('price', 'N/A')}")
        else:
            print(f"Order response: {result}")
    else:
        print(f"❌ Trade failed: {resp.status_code}")
        print(resp.text)
else:
    print(f"❌ Connection failed: {resp.text}")