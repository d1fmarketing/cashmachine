#!/usr/bin/env python3
"""
ULTRATHINK ULTIMATE INTEGRATED SYSTEM
Combines ALL revolutionary components:
- Real paper trading (Alpaca/OANDA)
- Microstructure analysis (order book)
- Multi-strategy ensemble
- ASI/HRM/MCTS signals
- Dynamic Kelly sizing
- Alternative data
"""

import os
import sys
import asyncio
import json
import logging
import numpy as np
import pandas as pd
from datetime import datetime
import redis.asyncio as redis
import alpaca_trade_api as tradeapi
from oandapyV20 import API as OandaAPI

# Import our revolutionary modules
sys.path.append('/opt/cashmachine/trinity')
from microstructure_genius import MicrostructureAnalyzer, HighFrequencyMicrostructure
from multi_strategy_ensemble import StrategyEnsemble
from ultrathink_revolution import (
    KellyCriterion, RegimeDetector, AlternativeDataCollector,
    SmartExecutor, RevolutionConfig
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - ULTIMATE - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class UltimateIntegratedSystem:
    """The ultimate ULTRATHINK system with everything integrated"""
    
    def __init__(self):
        self.config = RevolutionConfig()
        
        # Core components
        self.redis_client = None
        self.alpaca = None
        self.oanda = None
        
        # Revolutionary modules
        self.microstructure = MicrostructureAnalyzer()
        self.hft_micro = HighFrequencyMicrostructure()
        self.ensemble = StrategyEnsemble()
        self.regime_detector = RegimeDetector()
        self.kelly = KellyCriterion(self.config)
        self.alt_data = AlternativeDataCollector(self.config)
        self.smart_executor = None
        
        # Performance tracking
        self.total_trades = 0
        self.winning_trades = 0
        self.total_profit = 0
        self.start_balance = 0
        
    async def initialize(self):
        """Initialize all systems"""
        logger.info("""
        ╔══════════════════════════════════════════════════════════════╗
        ║         🚀 ULTRATHINK ULTIMATE INTEGRATED SYSTEM 🚀         ║
        ║                                                              ║
        ║  Components Integrated:                                      ║
        ║  ✅ ASI/HRM/MCTS AI Signals                                ║
        ║  ✅ Microstructure Analysis (Renaissance-style)             ║
        ║  ✅ Multi-Strategy Ensemble (7 strategies)                  ║
        ║  ✅ Real Paper Trading (Alpaca + OANDA)                    ║
        ║  ✅ Dynamic Kelly Sizing                                    ║
        ║  ✅ Alternative Data (Twitter, Blockchain)                  ║
        ║  ✅ Smart Order Execution (Iceberg, Limit)                  ║
        ║  ✅ Regime Detection (Trending/Ranging/Volatile)            ║
        ║                                                              ║
        ║  Target: 80% Win Rate, <1ms Execution                       ║
        ╚══════════════════════════════════════════════════════════════╝
        """)
        
        # Connect Redis
        try:
            self.redis_client = await redis.Redis(
                host=self.config.redis_host,
                port=self.config.redis_port,
                decode_responses=True
            )
            await self.redis_client.ping()
            logger.info(f"✅ Connected to Redis for ASI/HRM/MCTS signals")
        except Exception as e:
            logger.error(f"❌ Redis connection failed: {e}")
            return False
        
        # Initialize Alpaca
        try:
            self.alpaca = tradeapi.REST(
                self.config.alpaca_key,
                self.config.alpaca_secret,
                self.config.alpaca_base_url,
                api_version='v2'
            )
            account = self.alpaca.get_account()
            self.start_balance = float(account.cash)
            logger.info(f"✅ Alpaca Paper Trading - Balance: ${self.start_balance:,.2f}")
        except Exception as e:
            logger.error(f"❌ Alpaca connection failed: {e}")
            return False
        
        # Initialize OANDA
        try:
            self.oanda = OandaAPI(
                access_token=self.config.oanda_token,
                environment="practice"
            )
            logger.info("✅ OANDA Practice Account connected")
        except Exception as e:
            logger.error(f"❌ OANDA connection failed: {e}")
            return False
        
        # Initialize smart executor
        self.smart_executor = SmartExecutor(self.alpaca, self.oanda, self.config)
        
        logger.info("🎯 All systems initialized successfully!")
        return True
    
    async def get_all_signals(self) -> dict:
        """Gather signals from all sources"""
        signals = {}
        
        # 1. Get ASI/HRM/MCTS from Redis
        try:
            ultrathink = await self.redis_client.hgetall('ultrathink:signals')
            signals['asi'] = ultrathink.get('asi', 'hold:0.5')
            signals['hrm'] = ultrathink.get('hrm', 'hold:0.5')
            signals['mcts'] = ultrathink.get('mcts', 'hold:0.5')
            signals['ultrathink_signal'] = ultrathink.get('signal', 'hold')
            signals['ultrathink_confidence'] = float(ultrathink.get('confidence', 0.5))
        except Exception as e:
            logger.error(f"Error getting ULTRATHINK signals: {e}")
            signals['ultrathink_confidence'] = 0.0
        
        # 2. Get market data
        symbol = 'SPY'
        try:
            bars = self.alpaca.get_bars(symbol, '1Min', limit=100).df
            signals['market_data'] = bars
            
            # 3. Get microstructure analysis
            # In production, get real Level 2 data
            mock_book = {
                'bids': [[bars['close'].iloc[-1] - 0.01, 1000]],
                'asks': [[bars['close'].iloc[-1] + 0.01, 1000]]
            }
            mock_trades = [
                {'price': bars['close'].iloc[-1], 'volume': 100, 'side': 'buy'}
            ]
            
            from microstructure_genius import OrderBookSnapshot, OrderBookLevel
            book = OrderBookSnapshot(
                timestamp=datetime.now().timestamp(),
                bids=[OrderBookLevel(p, s, 1) for p, s in mock_book['bids']],
                asks=[OrderBookLevel(p, s, 1) for p, s in mock_book['asks']]
            )
            
            micro_prediction = self.microstructure.predict_price_movement(book, mock_trades)
            signals['microstructure'] = micro_prediction
            
            # 4. Get ensemble strategy votes
            ensemble_decision = await self.ensemble.get_ensemble_decision(
                bars,
                asi_signal=signals.get('asi', 'hold').split(':')[0],
                hrm_signal=signals.get('hrm', 'hold').split(':')[0],
                mcts_signal=signals.get('mcts', 'hold').split(':')[0],
                spread=book.spread / book.mid_price if book.mid_price > 0 else 0
            )
            signals['ensemble'] = ensemble_decision
            
            # 5. Get regime
            regime, regime_conf = await self.regime_detector.detect_regime(bars)
            signals['regime'] = regime
            signals['regime_confidence'] = regime_conf
            
            # 6. Get alternative data
            twitter = await self.alt_data.get_twitter_sentiment(symbol)
            blockchain = await self.alt_data.get_blockchain_metrics(symbol)
            signals['twitter_sentiment'] = twitter
            signals['blockchain_metrics'] = blockchain
            
        except Exception as e:
            logger.error(f"Error gathering signals: {e}")
        
        return signals
    
    def combine_all_signals(self, signals: dict) -> dict:
        """Combine all signals into final decision"""
        
        # Weight different signal sources
        weights = {
            'ultrathink': 0.25,  # ASI/HRM/MCTS
            'microstructure': 0.25,  # Order book analysis
            'ensemble': 0.3,  # Multi-strategy
            'alternative': 0.2  # Twitter, blockchain
        }
        
        final_score = 0
        total_confidence = 0
        
        # ULTRATHINK signal
        if 'ultrathink_signal' in signals:
            signal_map = {'buy': 1, 'sell': -1, 'hold': 0}
            ultra_score = signal_map.get(signals['ultrathink_signal'].lower(), 0)
            ultra_conf = signals.get('ultrathink_confidence', 0.5)
            final_score += ultra_score * weights['ultrathink']
            total_confidence += ultra_conf * weights['ultrathink']
        
        # Microstructure signal
        if 'microstructure' in signals:
            micro = signals['microstructure']
            micro_map = {'UP': 1, 'DOWN': -1, 'NEUTRAL': 0}
            micro_score = micro_map.get(micro.get('prediction', 'NEUTRAL'), 0)
            micro_conf = micro.get('confidence', 0.5)
            final_score += micro_score * weights['microstructure']
            total_confidence += micro_conf * weights['microstructure']
            
            # Bonus for strong microstructure signals
            if abs(micro.get('ofi', 0)) > 0.5:
                final_score += np.sign(micro['ofi']) * 0.1
                total_confidence += 0.1
        
        # Ensemble signal
        if 'ensemble' in signals:
            ensemble = signals['ensemble']
            ensemble_score = ensemble.get('score', 0)
            ensemble_conf = ensemble.get('confidence', 0.5)
            final_score += ensemble_score * weights['ensemble']
            total_confidence += ensemble_conf * weights['ensemble']
        
        # Alternative data
        twitter_sent = signals.get('twitter_sentiment', 0.5)
        alt_score = (twitter_sent - 0.5) * 2  # Convert to [-1, 1]
        final_score += alt_score * weights['alternative']
        total_confidence += abs(alt_score) * weights['alternative']
        
        # Regime adjustment
        regime = signals.get('regime')
        if regime:
            from multi_strategy_ensemble import MarketRegime
            if regime == MarketRegime.VOLATILE:
                total_confidence *= 0.7  # Less confident in volatile markets
            elif regime == MarketRegime.TRENDING_UP and final_score > 0:
                total_confidence *= 1.2  # More confident when aligned with trend
            elif regime == MarketRegime.TRENDING_DOWN and final_score < 0:
                total_confidence *= 1.2
        
        # Normalize confidence
        total_confidence = min(total_confidence, 0.95)
        
        # Determine action
        if final_score > 0.3:
            action = 'buy'
        elif final_score < -0.3:
            action = 'sell'
        else:
            action = 'hold'
        
        return {
            'action': action,
            'score': final_score,
            'confidence': total_confidence,
            'components': {
                'ultrathink': signals.get('ultrathink_signal', 'unknown'),
                'microstructure': signals.get('microstructure', {}).get('prediction', 'unknown'),
                'ensemble': signals.get('ensemble', {}).get('signal', 'unknown'),
                'regime': str(signals.get('regime', 'unknown'))
            }
        }
    
    async def execute_ultimate_trade(self, decision: dict, signals: dict):
        """Execute trade with all revolutionary features"""
        
        if decision['action'] == 'hold':
            logger.info("📊 Decision: HOLD - No trade executed")
            return
        
        if decision['confidence'] < self.config.min_confidence:
            logger.info(f"⚠️ Confidence {decision['confidence']:.2f} below threshold")
            return
        
        # Get account and calculate position size
        account = self.alpaca.get_account()
        capital = float(account.cash)
        
        # Use Kelly Criterion with all factors
        market_data = signals.get('market_data')
        if market_data is not None and len(market_data) > 0:
            volatility = market_data['close'].pct_change().std()
        else:
            volatility = 0.02  # Default 2% volatility
        
        position_size = self.kelly.calculate_position_size(
            capital,
            decision['confidence'],
            volatility,
            signals.get('regime')
        )
        
        # Calculate shares
        symbol = 'SPY'
        current_price = market_data['close'].iloc[-1] if market_data is not None else 400
        shares = int(position_size / current_price)
        
        if shares < 1:
            logger.info(f"Position too small: ${position_size:.2f}")
            return
        
        logger.info(f"""
        🎯 EXECUTING ULTIMATE TRADE:
        Signal: {decision['action'].upper()}
        Confidence: {decision['confidence']:.2%}
        Position Size: ${position_size:,.2f} ({shares} shares)
        Components:
        - ULTRATHINK: {decision['components']['ultrathink']}
        - Microstructure: {decision['components']['microstructure']}
        - Ensemble: {decision['components']['ensemble']}
        - Regime: {decision['components']['regime']}
        """)
        
        # Execute with smart routing
        try:
            if shares > 100 and self.config.use_iceberg:
                orders = await self.smart_executor.execute_with_iceberg(
                    symbol, decision['action'], shares
                )
                logger.info(f"✅ Iceberg order executed: {len(orders)} slices")
            else:
                order = await self.smart_executor.place_smart_order(
                    symbol, decision['action'], shares
                )
                logger.info(f"✅ Smart order executed: {order}")
            
            # Record trade
            self.total_trades += 1
            trade_record = {
                'timestamp': datetime.now().isoformat(),
                'action': decision['action'],
                'shares': shares,
                'price': current_price,
                'confidence': decision['confidence'],
                'signals': decision['components']
            }
            
            await self.redis_client.rpush(
                'ultimate:trades',
                json.dumps(trade_record)
            )
            
            # Update learning metrics
            await self.redis_client.hincrby('ultimate:stats', 'total_trades', 1)
            await self.redis_client.hset('ultimate:stats', 'last_trade', json.dumps(trade_record))
            
        except Exception as e:
            logger.error(f"❌ Trade execution failed: {e}")
    
    async def run(self):
        """Main loop"""
        if not await self.initialize():
            logger.error("Failed to initialize")
            return
        
        logger.info("🚀 Starting Ultimate Integrated Trading Loop")
        
        iteration = 0
        last_stats_time = datetime.now()
        
        while True:
            try:
                iteration += 1
                
                # 1. Gather all signals
                signals = await self.get_all_signals()
                
                # 2. Combine into final decision
                decision = self.combine_all_signals(signals)
                
                logger.info(f"📡 Iteration {iteration} | "
                          f"Decision: {decision['action'].upper()} | "
                          f"Score: {decision['score']:.3f} | "
                          f"Confidence: {decision['confidence']:.2%}")
                
                # 3. Execute if confident
                await self.execute_ultimate_trade(decision, signals)
                
                # 4. Log detailed stats every minute
                if (datetime.now() - last_stats_time).seconds > 60:
                    account = self.alpaca.get_account()
                    current_balance = float(account.cash)
                    profit = current_balance - self.start_balance
                    profit_pct = (profit / self.start_balance) * 100
                    
                    logger.info(f"""
                    📊 PERFORMANCE UPDATE:
                    Trades: {self.total_trades}
                    Starting Balance: ${self.start_balance:,.2f}
                    Current Balance: ${current_balance:,.2f}
                    Profit: ${profit:,.2f} ({profit_pct:.2f}%)
                    """)
                    
                    last_stats_time = datetime.now()
                
                # 5. Wait before next iteration
                await asyncio.sleep(5)
                
            except KeyboardInterrupt:
                logger.info("Shutdown requested")
                break
            except Exception as e:
                logger.error(f"Main loop error: {e}")
                await asyncio.sleep(10)
        
        logger.info("Ultimate system stopped")

def main():
    """Entry point"""
    system = UltimateIntegratedSystem()
    asyncio.run(system.run())

if __name__ == "__main__":
    main()