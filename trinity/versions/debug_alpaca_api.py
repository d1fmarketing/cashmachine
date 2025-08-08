#!/usr/bin/env python3
"""
Debug Alpaca API - find the real issue
"""

import os
import requests
import base64
from dotenv import load_dotenv

# Load environment
load_dotenv('/opt/cashmachine/trinity/.env')

# Try BOTH sets of credentials
creds = [
    {
        'key': os.environ.get('ALPACA_API_KEY'),
        'secret': os.environ.get('ALPACA_API_SECRET'),
        'source': '.env file'
    },
    {
        'key': 'PKCS3E7GZ5N9UDZEZ5BF',
        'secret': 'Y3BeJHStNv0CqonTONiPEfRkWlPdR4aokprHzmSP',
        'source': 'real_alpaca_config.py'
    }
]

# Try different endpoints and auth methods
endpoints = [
    'https://paper-api.alpaca.markets',
    'https://api.alpaca.markets',
    'https://data.alpaca.markets'
]

print("🔍 DEBUGGING ALPACA API CONNECTION")
print("=" * 60)

for cred in creds:
    print(f"\n📊 Testing credentials from: {cred['source']}")
    print(f"Key: {cred['key'][:10]}...")
    
    for endpoint in endpoints:
        print(f"\n  Testing endpoint: {endpoint}")
        
        # Method 1: Headers
        headers1 = {
            'APCA-API-KEY-ID': cred['key'],
            'APCA-API-SECRET-KEY': cred['secret']
        }
        
        # Method 2: Different header format
        headers2 = {
            'Apca-Api-Key-Id': cred['key'],
            'Apca-Api-Secret-Key': cred['secret']
        }
        
        # Method 3: Authorization header
        auth_string = f"{cred['key']}:{cred['secret']}"
        b64_auth = base64.b64encode(auth_string.encode()).decode()
        headers3 = {
            'Authorization': f'Basic {b64_auth}'
        }
        
        for i, headers in enumerate([headers1, headers2], 1):
            try:
                # Try account endpoint
                url = f"{endpoint}/v2/account"
                resp = requests.get(url, headers=headers, timeout=5)
                
                print(f"    Method {i}: Status {resp.status_code}")
                
                if resp.status_code == 200:
                    print(f"    ✅✅✅ SUCCESS with method {i}!")
                    account = resp.json()
                    print(f"    Cash: ${account.get('cash', 0)}")
                    print(f"    Pattern Day Trader: {account.get('pattern_day_trader', False)}")
                    print(f"    Trading Blocked: {account.get('trading_blocked', False)}")
                    print(f"    Account Blocked: {account.get('account_blocked', False)}")
                    
                    # Try to get positions
                    pos_resp = requests.get(
                        f"{endpoint}/v2/positions",
                        headers=headers,
                        timeout=5
                    )
                    if pos_resp.status_code == 200:
                        positions = pos_resp.json()
                        print(f"    Positions: {len(positions)}")
                    
                    # Try to get orders
                    ord_resp = requests.get(
                        f"{endpoint}/v2/orders?limit=5",
                        headers=headers,
                        timeout=5
                    )
                    if ord_resp.status_code == 200:
                        orders = ord_resp.json()
                        print(f"    Recent orders: {len(orders)}")
                    
                    print("\n    🎯 FOUND WORKING CONFIGURATION!")
                    print(f"    Endpoint: {endpoint}")
                    print(f"    Headers format: Method {i}")
                    print(f"    Key: {cred['key']}")
                    print(f"    Secret: {cred['secret']}")
                    exit(0)
                    
                elif resp.status_code == 403:
                    error = resp.json() if resp.text else {}
                    print(f"    ❌ Forbidden: {error.get('message', resp.text[:100])}")
                elif resp.status_code == 401:
                    print(f"    ❌ Unauthorized")
                else:
                    print(f"    ❌ Error: {resp.text[:100]}")
                    
            except requests.exceptions.ConnectionError:
                print(f"    ❌ Connection failed")
            except Exception as e:
                print(f"    ❌ Error: {str(e)[:100]}")

print("\n❌ No working configuration found")
print("\n📝 Possible issues:")
print("1. API keys may need to be regenerated")
print("2. Paper trading may need to be enabled in Alpaca dashboard")
print("3. Account may be restricted or need verification")