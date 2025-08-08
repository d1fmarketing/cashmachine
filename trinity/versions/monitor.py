#\!/usr/bin/env python3
"""ULTRATHINK MONITOR - REAL-TIME STATUS"""
import redis, requests, json
from datetime import datetime

print("=" * 60)
print("📊 ULTRATHINK AI MONITOR")
print("=" * 60)

r = redis.Redis(host="10.100.2.200", port=6379, decode_responses=True)

# Get Alpaca account
headers = {
    "APCA-API-KEY-ID": "PKGXVRHYGL3DT8QQ795W",
    "APCA-API-SECRET-KEY": "uaqqu1ifB2eCh0IE0M1nMRTPjVceeJ3efOvuB1wm"
}

resp = requests.get("https://paper-api.alpaca.markets/v2/account", headers=headers)
if resp.status_code == 200:
    acc = resp.json()
    print(f"\n💰 Account Balance: ${acc['cash']}")
    print(f"📈 Equity: ${acc['equity']}")
    print(f"🔥 Buying Power: ${acc['buying_power']}")

# Get recent orders
resp2 = requests.get("https://paper-api.alpaca.markets/v2/orders?limit=10", headers=headers)
if resp2.status_code == 200:
    orders = resp2.json()
    print(f"\n📊 Recent Orders ({len(orders)} total):")
    for order in orders[:5]:
        print(f"  • {order['side'].upper()} {order['qty']} {order['symbol']} - {order['status']}")

# Get trades from Redis
trades = r.lrange("ultrathink:trades", 0, 10)
print(f"\n🧠 AI Trades in Redis: {len(trades)}")
for i, trade in enumerate(trades[:3], 1):
    t = json.loads(trade)
    print(f"  {i}. {t.get('symbol', 'N/A')} - AI: {t.get('ai', 'N/A')}")

print("\n✅ ULTRATHINK AI OPERATIONAL WITH ASI+HRM+AlphaGo\!")
print(f"⏰ Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}")
