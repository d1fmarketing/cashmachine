#!/usr/bin/env python3
"""
ULTRATHINK Backtest Runner
Test the AI trading strategy with historical data
"""

import backtrader as bt
from datetime import datetime, timedelta
import sys
sys.path.insert(0, '/opt/cashmachine/trinity')
from ultrathink_strategy import UltrathinkStrategy

def run_backtest():
    """Run backtest with ULTRATHINK strategy"""
    
    cerebro = bt.Cerebro()
    
    # Add strategy
    cerebro.addstrategy(UltrathinkStrategy)
    
    # Set broker parameters
    cerebro.broker.setcash(100000.0)
    cerebro.broker.setcommission(commission=0.001)  # 0.1% commission
    
    # Add analyzers
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
    
    # Create sample data (replace with real data feed)
    data = bt.feeds.YahooFinanceData(
        dataname='SPY',
        fromdate=datetime.now() - timedelta(days=365),
        todate=datetime.now()
    )
    
    cerebro.adddata(data)
    
    print("🚀 Starting ULTRATHINK Backtest...")
    print(f"Initial Portfolio Value: ${cerebro.broker.getvalue():.2f}")
    
    # Run backtest
    results = cerebro.run()
    
    # Get results
    strat = results[0]
    sharpe = strat.analyzers.sharpe.get_analysis()
    drawdown = strat.analyzers.drawdown.get_analysis()
    returns = strat.analyzers.returns.get_analysis()
    trades = strat.analyzers.trades.get_analysis()
    
    print(f"\n📊 Backtest Results:")
    print(f"Sharpe Ratio: {sharpe.get('sharperatio', 'N/A')}")
    print(f"Max Drawdown: {drawdown.max.drawdown:.2f}%")
    print(f"Total Return: {returns.rtot:.2%}")
    print(f"Total Trades: {trades.total.total}")
    print(f"Win Rate: {(trades.won.total / trades.total.total * 100) if trades.total.total else 0:.2f}%")
    
    # Plot results
    cerebro.plot()

if __name__ == "__main__":
    run_backtest()
