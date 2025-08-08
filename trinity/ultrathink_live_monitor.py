#!/usr/bin/env python3
"""
ULTRATHINK LIVE MONITORING
Shows real-time status of the trading system
"""
import redis
import json
import time
from datetime import datetime

def monitor_ultrathink():
    """Monitor ULTRATHINK in real-time"""
    
    # Connect to Redis
    r = redis.Redis(host='10.100.2.200', port=6379, decode_responses=True)
    
    print("=" * 70)
    print("🚀 ULTRATHINK LIVE MONITORING SYSTEM 🚀")
    print("=" * 70)
    
    while True:
        try:
            # Get current signals
            signals = r.hgetall('ultrathink:signals')
            
            # Get learning metrics
            metrics = r.hgetall('ultrathink:learning_metrics')
            
            # Get recent trades
            recent_trades = r.lrange('trinity:trades', 0, 4)
            
            # Get ML farm decision
            ml_decision = r.hgetall('ml_farm:decision')
            
            # Clear screen (optional)
            print("\033[H\033[J", end="")
            
            # Display header
            print("=" * 70)
            print(f"📊 ULTRATHINK STATUS - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print("=" * 70)
            
            # Display signals
            print("\n🧠 AI SIGNALS:")
            print(f"  Signal: {signals.get('signal', 'N/A').upper()}")
            print(f"  Confidence: {float(signals.get('confidence', 0)):.2%}")
            print(f"  ASI: {signals.get('asi', 'N/A')}")
            print(f"  HRM: {signals.get('hrm', 'N/A')}")
            print(f"  MCTS: {signals.get('mcts', 'N/A')}")
            
            # Display performance
            print("\n📈 PERFORMANCE:")
            total_trades = int(metrics.get('total_trades', 0))
            winning_trades = int(metrics.get('winning_trades', 0))
            win_rate = float(metrics.get('win_rate', 0))
            total_profit = float(metrics.get('total_profit', 0))
            
            print(f"  Total Trades: {total_trades}")
            print(f"  Winning Trades: {winning_trades}")
            print(f"  Win Rate: {win_rate:.1%}")
            print(f"  Total Profit: {total_profit:.4f} BTC equivalent")
            print(f"  Avg Profit/Trade: {total_profit/max(total_trades, 1):.6f}")
            
            # Display recent trades
            print("\n💰 RECENT TRADES:")
            for i, trade_json in enumerate(recent_trades[:3], 1):
                try:
                    trade = json.loads(trade_json)
                    symbol = trade.get('symbol', 'N/A')
                    side = trade.get('side', 'N/A').upper()
                    status = trade.get('status', 'N/A')
                    api = trade.get('api', 'N/A')
                    timestamp = trade.get('timestamp', '')[:19]
                    
                    print(f"  {i}. {side} {symbol} via {api} - {status} ({timestamp})")
                except:
                    pass
            
            # Display ML Farm decision
            if ml_decision:
                print("\n🎯 ML FARM DECISION:")
                print(f"  Signal: {ml_decision.get('signal', 'N/A')}")
                print(f"  Confidence: {float(ml_decision.get('confidence', 0)):.3f}")
                print(f"  Components Active: {ml_decision.get('components_active', 0)}/7")
            
            # Calculate stats
            if total_trades > 0:
                trades_per_hour = total_trades / 24  # Assuming 24 hours
                projected_daily = trades_per_hour * 24
                projected_profit = projected_daily * (total_profit / total_trades)
                
                print("\n📊 PROJECTIONS (24h):")
                print(f"  Projected Trades: {projected_daily:.0f}")
                print(f"  Projected Profit: {projected_profit:.4f} BTC")
                print(f"  Projected USD (@$50k/BTC): ${projected_profit * 50000:.2f}")
            
            print("\n" + "=" * 70)
            print("Press Ctrl+C to exit | Updates every 5 seconds")
            
            time.sleep(5)
            
        except KeyboardInterrupt:
            print("\n\n👋 Monitoring stopped")
            break
        except Exception as e:
            print(f"Error: {e}")
            time.sleep(5)

if __name__ == "__main__":
    monitor_ultrathink()