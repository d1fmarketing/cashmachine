#\!/usr/bin/env python3
"""
ULTRATHINK SYSTEM MONITOR
Real-time monitoring of all components
"""

import redis
import time
from datetime import datetime
import os

def clear_screen():
    os.system('clear')

def main():
    r = redis.Redis(host='10.100.2.200', port=6379, decode_responses=True)
    
    while True:
        try:
            clear_screen()
            print("=" * 80)
            print("                    ULTRATHINK SYSTEM MONITOR")
            print("=" * 80)
            print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print()
            
            # Check ULTRATHINK signals
            ultra = r.hgetall('ultrathink:signals')
            if ultra:
                print("📊 ULTRATHINK SIGNALS")
                print(f"  Signal: {ultra.get('signal', 'none').upper()}")
                conf = ultra.get('confidence', '0')
                print(f"  Confidence: {float(conf):.2%}")
                print(f"  ASI: {ultra.get('asi', 'N/A')}")
                print(f"  HRM: {ultra.get('hrm', 'N/A')}")
                print(f"  MCTS: {ultra.get('mcts', 'N/A')}")
                print(f"  Last Update: {ultra.get('timestamp', 'unknown')[:19]}")
            else:
                print("❌ ULTRATHINK: No signals")
            
            print()
            
            # Check ML Farm
            ml = r.hgetall('ml_farm:unified')
            if ml:
                print("🧠 ML FARM BRAIN")
                print(f"  Signal: {ml.get('signal', 'none').upper()}")
                conf = ml.get('confidence', '0')
                print(f"  Confidence: {float(conf):.2%}")
                sacred = ml.get('sacred_alignment', '0')
                print(f"  Sacred Alignment: {float(sacred):.2%}")
                print(f"  Active Components: {ml.get('components_active', 0)}")
            else:
                print("❌ ML Farm: Inactive")
            
            print()
            
            # Check Trade Executor
            executor = r.hgetall('ultrathink:executor')
            if executor:
                print("💰 TRADE EXECUTOR")
                print(f"  Status: {executor.get('status', 'unknown')}")
                print(f"  Total Trades: {executor.get('total_trades', 0)}")
                print(f"  Last Trade: {executor.get('last_trade', 'none').upper()}")
                print(f"  Balance: ${executor.get('balance', 0)}")
            else:
                print("❌ Trade Executor: No data")
            
            print()
            
            # Check Market Data
            print("📈 MARKET DATA")
            symbols = ['SPY', 'AAPL', 'BTC', 'ETH']
            for symbol in symbols:
                data = r.hgetall(f'market:{symbol}')
                if data and 'price' in data:
                    price = float(data.get('price', 0))
                    sources = data.get('sources', 0)
                    print(f"  {symbol}: ${price:.2f} ({sources} sources)")
            
            print()
            
            # Count Redis keys
            all_keys = r.keys('*')
            market_keys = len([k for k in all_keys if k.startswith('market:')])
            trade_keys = len([k for k in all_keys if k.startswith('executed:')])
            
            print("📊 REDIS STATISTICS")
            print(f"  Total Keys: {len(all_keys)}")
            print(f"  Market Keys: {market_keys}")
            print(f"  Executed Trades: {trade_keys}")
            
            print()
            
            # Sacred moment indicator
            current_second = int(time.time()) % 100
            if current_second == 31:
                print("✨ PI MOMENT ACTIVE (3.14) ✨")
            elif current_second == 69:
                print("✨ SACRED 69 MOMENT ACTIVE ✨")
            
            print()
            print("Press Ctrl+C to exit")
            
            time.sleep(5)  # Update every 5 seconds
            
        except KeyboardInterrupt:
            print("\nMonitoring stopped.")
            break
        except Exception as e:
            print(f"Error: {e}")
            time.sleep(5)

if __name__ == "__main__":
    main()
