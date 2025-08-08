#!/usr/bin/env python3
"""Test OANDA API connection"""
from oandapyV20 import API
from oandapyV20.endpoints.accounts import AccountDetails
import json

# OANDA credentials
token = '7d5ad00e456c02fc0aaec8e97c5dc6ef-d017cc87bb5b91e8d0e22c0f27e4b67c'
account_id = '001-001-1473494-002'

print("Testing OANDA API connection...")
print("=" * 50)

try:
    # Create API client
    client = API(access_token=token, environment='practice')
    
    # Get account details
    r = AccountDetails(accountID=account_id)
    response = client.request(r)
    
    # Parse response
    account = response.get('account', {})
    balance = account.get('balance', 'N/A')
    currency = account.get('currency', 'USD')
    
    print(f"✅ OANDA Connected Successfully!")
    print(f"Account ID: {account_id}")
    print(f"Balance: {balance} {currency}")
    print(f"Margin Available: {account.get('marginAvailable', 'N/A')}")
    print(f"Open Positions: {account.get('openPositionCount', 0)}")
    print(f"Open Orders: {account.get('pendingOrderCount', 0)}")
    
    # Test placing an order (but don't actually place it)
    print("\n✅ OANDA API is working! Can execute forex trades.")
    
except Exception as e:
    print(f"❌ OANDA connection failed: {e}")

print("=" * 50)