#!/usr/bin/env python3
"""
ULTRATHINK REAL BRAIN
Real AI making real trading decisions
No simulations - only reality!
"""

import json
import time
import redis
import logging
import threading
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - ULTRATHINK_REAL - %(levelname)s - %(message)s'
)

class UltrathinkRealBrain:
    """The REAL unified AI brain for trading"""
    
    def __init__(self):
        logging.info("🧠 Initializing ULTRATHINK REAL Brain...")
        
        # Redis connection
        self.redis_client = redis.Redis(
            host='10.100.2.200',
            port=6379,
            decode_responses=True
        )
        
        # Subscribe to real market data
        self.pubsub = self.redis_client.pubsub()
        self.pubsub.subscribe([
            'realmarket:forex',
            'realmarket:stocks',
            'realmarket:indicators',
            'realmarket:signals'
        ])
        
        # Trading parameters
        self.min_confidence = 0.70  # Only trade with high confidence
        self.max_positions = 5
        self.risk_per_trade = 0.02
        
        # Performance tracking
        self.decisions_made = 0
        self.signals_sent = 0
        self.current_positions = {}
        
        self.running = False
        
    def start(self):
        """Start the real AI brain"""
        logging.info("=" * 60)
        logging.info("🚀 ULTRATHINK REAL BRAIN STARTING")
        logging.info("Processing REAL market data")
        logging.info("Making REAL trading decisions")
        logging.info("=" * 60)
        
        self.running = True
        
        # Start processing threads
        threads = [
            threading.Thread(target=self._process_market_data, daemon=True),
            threading.Thread(target=self._make_decisions, daemon=True),
            threading.Thread(target=self._monitor_performance, daemon=True)
        ]
        
        for t in threads:
            t.start()
            
        logging.info("✅ Real AI Brain ACTIVE")
        
        # Main loop
        while self.running:
            try:
                time.sleep(1)
            except KeyboardInterrupt:
                self.stop()
                break
                
    def stop(self):
        """Stop the brain"""
        logging.info("Stopping ULTRATHINK Real Brain...")
        self.running = False
        self.pubsub.unsubscribe()
        
    def _process_market_data(self):
        """Process real market data"""
        
        while self.running:
            try:
                message = self.pubsub.get_message(timeout=1)
                
                if message and message['type'] == 'message':
                    channel = message['channel']
                    data = json.loads(message['data'])
                    
                    if channel == 'realmarket:indicators':
                        self._analyze_indicators(data)
                    elif channel == 'realmarket:signals':
                        self._process_signal(data)
                        
            except Exception as e:
                logging.error(f"Market data processing error: {e}")
                time.sleep(5)
                
    def _analyze_indicators(self, indicators: Dict):
        """Analyze technical indicators"""
        
        instrument = indicators.get('instrument')
        rsi = indicators.get('rsi', 50)
        trend = indicators.get('trend')
        momentum = indicators.get('momentum', 0)
        
        # Multi-factor analysis
        buy_signals = 0
        sell_signals = 0
        
        # RSI analysis
        if rsi < 30:
            buy_signals += 2
        elif rsi > 70:
            sell_signals += 2
        elif rsi < 40:
            buy_signals += 1
        elif rsi > 60:
            sell_signals += 1
            
        # Trend analysis
        if trend == 'up' and momentum > 0.5:
            buy_signals += 1
        elif trend == 'down' and momentum < -0.5:
            sell_signals += 1
            
        # Make decision
        if buy_signals >= 3:
            self._create_trade_signal(instrument, 'BUY', buy_signals / 5)
        elif sell_signals >= 3:
            self._create_trade_signal(instrument, 'SELL', sell_signals / 5)
            
    def _process_signal(self, signal: Dict):
        """Process incoming signal"""
        
        confidence = signal.get('confidence', 0)
        
        if confidence >= self.min_confidence:
            # Enhance signal with AI analysis
            enhanced_signal = self._enhance_signal(signal)
            
            # Send to Trinity for execution
            self._send_to_trinity(enhanced_signal)
            
    def _enhance_signal(self, signal: Dict) -> Dict:
        """Enhance signal with AI insights"""
        
        # Get current market state
        market_state = self.redis_client.hget('ai:current_market', 'state')
        
        if market_state:
            state = json.loads(market_state)
            
            # Add market context
            signal['market_context'] = {
                'forex_active': len(state.get('forex', {})),
                'stocks_active': len(state.get('stocks', {})),
                'timestamp': datetime.now().isoformat()
            }
            
            # Adjust confidence based on market conditions
            volatility = self._calculate_market_volatility(state)
            if volatility > 0.02:  # High volatility
                signal['confidence'] *= 0.9  # Reduce confidence
                signal['stop_loss'] = 0.015  # Tighter stop
            else:
                signal['confidence'] *= 1.05  # Increase confidence
                signal['stop_loss'] = 0.02  # Normal stop
                
            signal['take_profit'] = signal['stop_loss'] * 2  # 2:1 RR ratio
            
        return signal
        
    def _calculate_market_volatility(self, state: Dict) -> float:
        """Calculate overall market volatility"""
        
        volatilities = []
        
        # Forex volatility
        for pair, data in state.get('forex', {}).items():
            if 'volatility' in data:
                volatilities.append(data['volatility'])
                
        # Stock volatility (approximate from change %)
        for symbol, data in state.get('stocks', {}).items():
            if 'change_pct' in data:
                volatilities.append(abs(data['change_pct']) / 100)
                
        return np.mean(volatilities) if volatilities else 0.01
        
    def _create_trade_signal(self, instrument: str, action: str, confidence: float):
        """Create a trade signal"""
        
        if len(self.current_positions) >= self.max_positions:
            logging.info(f"⚠️ Max positions reached, skipping {instrument}")
            return
            
        signal = {
            'instrument': instrument,
            'symbol': instrument.replace('_', ''),  # For stocks
            'action': action,
            'confidence': min(confidence, 0.95),
            'risk_per_trade': self.risk_per_trade,
            'source': 'ULTRATHINK_REAL',
            'timestamp': datetime.now().isoformat()
        }
        
        # Send to Trinity
        self._send_to_trinity(signal)
        
    def _send_to_trinity(self, signal: Dict):
        """Send signal to Trinity for execution"""
        
        # Publish to execution channel
        self.redis_client.publish(
            'ultrathink:execute',
            json.dumps(signal)
        )
        
        # Store for tracking
        self.redis_client.lpush(
            'ultrathink:signals_sent',
            json.dumps(signal)
        )
        
        self.signals_sent += 1
        
        logging.info(f"🎯 Signal sent: {signal['action']} {signal.get('instrument', signal.get('symbol'))} (Confidence: {signal['confidence']:.2f})")
        
    def _make_decisions(self):
        """Main decision-making loop"""
        
        while self.running:
            try:
                # Get current market state
                market_state = self.redis_client.hget('ai:current_market', 'state')
                
                if market_state:
                    state = json.loads(market_state)
                    
                    # Analyze each instrument
                    for instrument, data in state.get('indicators', {}).items():
                        self._analyze_instrument(instrument, data)
                        
                    self.decisions_made += 1
                    
                    if self.decisions_made % 10 == 0:
                        logging.info(f"📊 Decisions made: {self.decisions_made}, Signals sent: {self.signals_sent}")
                        
                time.sleep(10)  # Make decisions every 10 seconds
                
            except Exception as e:
                logging.error(f"Decision making error: {e}")
                time.sleep(30)
                
    def _analyze_instrument(self, instrument: str, data: Dict):
        """Analyze individual instrument"""
        
        # Check if we already have a position
        if instrument in self.current_positions:
            # Monitor existing position
            self._monitor_position(instrument, data)
        else:
            # Look for entry opportunity
            self._find_entry(instrument, data)
            
    def _find_entry(self, instrument: str, data: Dict):
        """Find entry opportunity"""
        
        rsi = data.get('rsi', 50)
        trend = data.get('trend')
        momentum = data.get('momentum', 0)
        current_price = data.get('current_price', 0)
        bb_upper = data.get('bb_upper', 0)
        bb_lower = data.get('bb_lower', 0)
        
        confidence = 0.5
        action = None
        
        # Strong oversold
        if rsi < 25 and current_price <= bb_lower:
            action = 'BUY'
            confidence = 0.8
            
        # Strong overbought
        elif rsi > 75 and current_price >= bb_upper:
            action = 'SELL'
            confidence = 0.8
            
        # Trend following
        elif trend == 'up' and momentum > 2 and rsi < 60:
            action = 'BUY'
            confidence = 0.7
            
        elif trend == 'down' and momentum < -2 and rsi > 40:
            action = 'SELL'
            confidence = 0.7
            
        if action and confidence >= self.min_confidence:
            self._create_trade_signal(instrument, action, confidence)
            
            # Track position
            self.current_positions[instrument] = {
                'action': action,
                'entry_price': current_price,
                'timestamp': datetime.now().isoformat()
            }
            
    def _monitor_position(self, instrument: str, data: Dict):
        """Monitor existing position"""
        
        position = self.current_positions[instrument]
        current_price = data.get('current_price', 0)
        entry_price = position['entry_price']
        
        if entry_price == 0:
            return
            
        # Calculate P&L
        if position['action'] == 'BUY':
            pnl_pct = ((current_price - entry_price) / entry_price) * 100
        else:
            pnl_pct = ((entry_price - current_price) / entry_price) * 100
            
        # Check exit conditions
        if pnl_pct >= 2:  # 2% profit
            self._close_position(instrument, 'TAKE_PROFIT')
        elif pnl_pct <= -1:  # 1% loss
            self._close_position(instrument, 'STOP_LOSS')
            
    def _close_position(self, instrument: str, reason: str):
        """Close a position"""
        
        position = self.current_positions.get(instrument)
        if not position:
            return
            
        # Create close signal
        close_signal = {
            'instrument': instrument,
            'action': 'CLOSE',
            'reason': reason,
            'original_action': position['action'],
            'timestamp': datetime.now().isoformat()
        }
        
        # Send to Trinity
        self._send_to_trinity(close_signal)
        
        # Remove from tracking
        del self.current_positions[instrument]
        
        logging.info(f"📍 Position closed: {instrument} ({reason})")
        
    def _monitor_performance(self):
        """Monitor AI performance"""
        
        while self.running:
            try:
                stats = {
                    'decisions_made': self.decisions_made,
                    'signals_sent': self.signals_sent,
                    'open_positions': len(self.current_positions),
                    'positions': list(self.current_positions.keys()),
                    'timestamp': datetime.now().isoformat()
                }
                
                # Store stats
                self.redis_client.hset(
                    'ultrathink:real_stats',
                    'current',
                    json.dumps(stats)
                )
                
                logging.info(f"🧠 Brain Stats: Decisions={self.decisions_made}, Signals={self.signals_sent}, Positions={len(self.current_positions)}")
                
                time.sleep(30)  # Report every 30 seconds
                
            except Exception as e:
                logging.error(f"Performance monitor error: {e}")
                time.sleep(60)


def main():
    """Main entry point"""
    
    print("=" * 60)
    print("🧠 ULTRATHINK REAL BRAIN")
    print("Real AI | Real Decisions | Real Trading")
    print("=" * 60)
    
    brain = UltrathinkRealBrain()
    
    try:
        brain.start()
    except KeyboardInterrupt:
        print("\n⏹️ Shutting down Real Brain...")
        brain.stop()
        
    print("✅ Real Brain stopped")


if __name__ == "__main__":
    main()