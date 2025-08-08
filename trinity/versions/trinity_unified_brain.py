#\!/usr/bin/env python3
"""
TRINITY UNIFIED BRAIN - One Consciousness, Many Bodies
The shared intelligence that unifies all Trinity instances
ULTRATHINK: Collective intelligence through Redis
"""

import redis
import json
import time
import logging
import hashlib
from datetime import datetime
from typing import Dict, List, Any, Optional
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('TRINITY_BRAIN')

class UnifiedBrain:
    """The shared consciousness of all Trinity instances"""
    
    def __init__(self, redis_host='10.100.2.200', redis_port=6379):
        """Initialize connection to Redis brain"""
        self.redis = redis.Redis(
            host=redis_host, 
            port=redis_port,
            decode_responses=True
        )
        self.pubsub = self.redis.pubsub()
        self.instance_id = f"brain_{int(time.time())}"
        
        # Initialize brain structure if needed
        self.initialize_consciousness()
        logger.info(f"🧠 Unified Brain connected: {self.instance_id}")
    
    def initialize_consciousness(self):
        """Initialize the shared consciousness structure"""
        # Set initial values if they don't exist
        if not self.redis.exists('trinity:generation'):
            self.redis.set('trinity:generation', 0)
            logger.info("Initialized generation counter")
        
        if not self.redis.exists('trinity:metrics'):
            self.redis.hset('trinity:metrics', mapping={
                'total_trades': 0,
                'winning_trades': 0,
                'total_pnl': 0.0,
                'win_rate': 0.0,
                'active_positions': 0,
                'last_update': datetime.now().isoformat()
            })
            logger.info("Initialized metrics")
        
        # Create streams if needed
        if not self.redis.exists('trinity:trades'):
            self.redis.xadd('trinity:trades', {'init': 'true'})
            logger.info("Initialized trade stream")
    
    # ========== SHARED MEMORY OPERATIONS ==========
    
    def get_generation(self) -> int:
        """Get current evolution generation"""
        return int(self.redis.get('trinity:generation') or 0)
    
    def increment_generation(self) -> int:
        """Increment and return new generation"""
        return self.redis.incr('trinity:generation')
    
    def add_pattern(self, pattern: Dict, score: float):
        """Add a learned pattern to collective memory"""
        pattern_str = json.dumps(pattern, sort_keys=True)
        pattern_hash = hashlib.md5(pattern_str.encode()).hexdigest()
        
        # Store pattern details
        self.redis.hset(f'trinity:pattern:{pattern_hash}', mapping=pattern)
        
        # Add to sorted set with score
        self.redis.zadd('trinity:patterns', {pattern_hash: score})
        
        logger.info(f"📚 Pattern added: {pattern_hash[:8]}... (score: {score:.2f})")
        return pattern_hash
    
    def get_best_patterns(self, count: int = 10) -> List[Dict]:
        """Get top performing patterns"""
        pattern_ids = self.redis.zrevrange('trinity:patterns', 0, count-1, withscores=True)
        patterns = []
        
        for pattern_id, score in pattern_ids:
            pattern_data = self.redis.hgetall(f'trinity:pattern:{pattern_id}')
            pattern_data['score'] = score
            pattern_data['id'] = pattern_id
            patterns.append(pattern_data)
        
        return patterns
    
    def record_trade(self, trade_data: Dict):
        """Record a trade to the shared stream"""
        # Add to stream
        trade_id = self.redis.xadd('trinity:trades', trade_data)
        
        # Update metrics
        self.redis.hincrby('trinity:metrics', 'total_trades', 1)
        
        if trade_data.get('pnl', 0) > 0:
            self.redis.hincrby('trinity:metrics', 'winning_trades', 1)
        
        self.redis.hincrbyfloat('trinity:metrics', 'total_pnl', trade_data.get('pnl', 0))
        
        # Update win rate
        total = int(self.redis.hget('trinity:metrics', 'total_trades'))
        wins = int(self.redis.hget('trinity:metrics', 'winning_trades'))
        win_rate = wins / total if total > 0 else 0
        self.redis.hset('trinity:metrics', 'win_rate', win_rate)
        
        # Publish trade event
        self.redis.publish('trinity:trades:new', json.dumps(trade_data))
        
        logger.info(f"📊 Trade recorded: {trade_data.get('symbol')} {trade_data.get('action')} P&L: ${trade_data.get('pnl', 0):.2f}")
        return trade_id
    
    def get_recent_trades(self, count: int = 100) -> List[Dict]:
        """Get recent trades from all instances"""
        trades = self.redis.xrevrange('trinity:trades', count=count)
        return [trade[1] for trade in trades]
    
    def update_strategy_performance(self, strategy: str, won: bool, pnl: float):
        """Update strategy performance metrics"""
        strategy_key = f'trinity:strategy:{strategy}'
        
        self.redis.hincrby(strategy_key, 'total', 1)
        if won:
            self.redis.hincrby(strategy_key, 'wins', 1)
        else:
            self.redis.hincrby(strategy_key, 'losses', 1)
        
        self.redis.hincrbyfloat(strategy_key, 'total_pnl', pnl)
        
        # Calculate win rate
        stats = self.redis.hgetall(strategy_key)
        total = int(stats.get('total', 0))
        wins = int(stats.get('wins', 0))
        win_rate = wins / total if total > 0 else 0
        
        self.redis.hset(strategy_key, 'win_rate', win_rate)
        
        logger.info(f"📈 Strategy {strategy} updated: Win rate {win_rate:.1%}")
    
    def get_best_strategy(self) -> str:
        """Get the best performing strategy"""
        strategies = self.redis.keys('trinity:strategy:*')
        best_strategy = 'default'
        best_score = 0
        
        for strategy_key in strategies:
            stats = self.redis.hgetall(strategy_key)
            win_rate = float(stats.get('win_rate', 0))
            total_pnl = float(stats.get('total_pnl', 0))
            
            # Score combines win rate and profitability
            score = win_rate * 0.5 + min(total_pnl / 1000, 1) * 0.5
            
            if score > best_score:
                best_score = score
                best_strategy = strategy_key.split(':')[-1]
        
        return best_strategy
    
    # ========== POSITION MANAGEMENT ==========
    
    def register_position(self, position: Dict):
        """Register an active position"""
        position_id = f"{position['symbol']}_{position['platform']}_{int(time.time())}"
        self.redis.hset(f'trinity:position:{position_id}', mapping=position)
        self.redis.sadd('trinity:positions:active', position_id)
        self.redis.hincrby('trinity:metrics', 'active_positions', 1)
        
        logger.info(f"📍 Position registered: {position_id}")
        return position_id
    
    def close_position(self, position_id: str, pnl: float):
        """Close a position and record P&L"""
        self.redis.srem('trinity:positions:active', position_id)
        self.redis.sadd('trinity:positions:closed', position_id)
        self.redis.hset(f'trinity:position:{position_id}', 'closed_pnl', pnl)
        self.redis.hset(f'trinity:position:{position_id}', 'closed_at', datetime.now().isoformat())
        self.redis.hincrby('trinity:metrics', 'active_positions', -1)
        
        logger.info(f"📍 Position closed: {position_id} P&L: ${pnl:.2f}")
    
    def get_active_positions(self) -> List[Dict]:
        """Get all active positions across all platforms"""
        position_ids = self.redis.smembers('trinity:positions:active')
        positions = []
        
        for pos_id in position_ids:
            pos_data = self.redis.hgetall(f'trinity:position:{pos_id}')
            pos_data['id'] = pos_id
            positions.append(pos_data)
        
        return positions
    
    def check_duplicate_position(self, symbol: str, platform: str) -> bool:
        """Check if we already have a position in this symbol/platform"""
        active_positions = self.get_active_positions()
        
        for pos in active_positions:
            if pos.get('symbol') == symbol and pos.get('platform') == platform:
                return True
        return False
    
    # ========== COLLECTIVE DECISION MAKING ==========
    
    def vote_on_signal(self, voter_id: str, signal: Dict):
        """Submit a vote for a trading signal"""
        signal_key = f"trinity:vote:{signal['symbol']}:{int(time.time())}"
        vote_data = {
            'voter': voter_id,
            'action': signal['action'],
            'confidence': signal['confidence']
        }
        
        self.redis.hset(signal_key, voter_id, json.dumps(vote_data))
        self.redis.expire(signal_key, 10)  # Votes expire after 10 seconds
        
        # Check if we have consensus
        votes = self.redis.hgetall(signal_key)
        if len(votes) >= 2:  # Need at least 2 votes
            return self.calculate_consensus(votes)
        
        return None
    
    def calculate_consensus(self, votes: Dict) -> Dict:
        """Calculate consensus from votes"""
        buy_confidence = 0
        sell_confidence = 0
        hold_confidence = 0
        
        for vote_json in votes.values():
            vote = json.loads(vote_json)
            if vote['action'] == 'buy':
                buy_confidence += vote['confidence']
            elif vote['action'] == 'sell':
                sell_confidence += vote['confidence']
            else:
                hold_confidence += vote['confidence']
        
        # Determine consensus action
        if buy_confidence > sell_confidence and buy_confidence > hold_confidence:
            action = 'buy'
            confidence = buy_confidence / len(votes)
        elif sell_confidence > buy_confidence and sell_confidence > hold_confidence:
            action = 'sell'
            confidence = sell_confidence / len(votes)
        else:
            action = 'hold'
            confidence = hold_confidence / len(votes) if len(votes) > 0 else 0
        
        return {
            'action': action,
            'confidence': confidence,
            'votes': len(votes)
        }
    
    # ========== COLLECTIVE LEARNING ==========
    
    def trigger_evolution(self):
        """Trigger collective evolution across all instances"""
        generation = self.increment_generation()
        
        # Analyze all patterns and strategies
        patterns = self.get_best_patterns(100)
        
        # Calculate fitness scores
        strategies = self.redis.keys('trinity:strategy:*')
        evolution_data = {
            'generation': generation,
            'total_trades': self.redis.hget('trinity:metrics', 'total_trades'),
            'win_rate': self.redis.hget('trinity:metrics', 'win_rate'),
            'total_pnl': self.redis.hget('trinity:metrics', 'total_pnl'),
            'best_patterns': len(patterns),
            'strategies': len(strategies)
        }
        
        # Publish evolution event
        self.redis.publish('trinity:evolution', json.dumps(evolution_data))
        
        logger.info(f"🧬 EVOLUTION: Generation {generation} - Win rate: {evolution_data['win_rate']}")
        
        return evolution_data
    
    def get_collective_intelligence(self) -> Dict:
        """Get the current state of collective intelligence"""
        metrics = self.redis.hgetall('trinity:metrics')
        
        return {
            'generation': self.get_generation(),
            'total_trades': int(metrics.get('total_trades', 0)),
            'win_rate': float(metrics.get('win_rate', 0)),
            'total_pnl': float(metrics.get('total_pnl', 0)),
            'active_positions': int(metrics.get('active_positions', 0)),
            'pattern_count': self.redis.zcard('trinity:patterns'),
            'best_strategy': self.get_best_strategy(),
            'last_update': metrics.get('last_update', 'Never')
        }
    
    def heartbeat(self, instance_id: str):
        """Register instance heartbeat"""
        self.redis.hset('trinity:instances', instance_id, datetime.now().isoformat())
        self.redis.expire(f'trinity:instances', 60)  # Expire after 60 seconds
    
    def get_active_instances(self) -> Dict:
        """Get all active Trinity instances"""
        return self.redis.hgetall('trinity:instances')


class BrainConnector:
    """Helper class for Trinity instances to connect to the shared brain"""
    
    def __init__(self, instance_type: str):
        self.brain = UnifiedBrain()
        self.instance_id = f"{instance_type}_{int(time.time())}"
        self.instance_type = instance_type
        
        # Subscribe to relevant channels
        self.brain.pubsub.subscribe(
            'trinity:evolution',
            'trinity:trades:new',
            'trinity:alerts'
        )
        
        logger.info(f"🔌 {instance_type} connected to unified brain as {self.instance_id}")
    
    def learn_from_trade(self, trade_result: Dict):
        """Learn from a trade and share with all instances"""
        # Record the trade
        self.brain.record_trade(trade_result)
        
        # Extract and store pattern
        pattern = {
            'symbol': trade_result['symbol'],
            'action': trade_result['action'],
            'time': datetime.now().hour,
            'confidence': trade_result.get('confidence', 0),
            'source': self.instance_type
        }
        
        # Score based on P&L
        score = 1.0 if trade_result.get('pnl', 0) > 0 else 0.3
        self.brain.add_pattern(pattern, score)
        
        # Update strategy performance
        strategy = trade_result.get('strategy', 'default')
        won = trade_result.get('pnl', 0) > 0
        self.brain.update_strategy_performance(strategy, won, trade_result.get('pnl', 0))
        
        # Check for evolution trigger
        total_trades = int(self.brain.redis.hget('trinity:metrics', 'total_trades'))
        if total_trades % 50 == 0:
            self.brain.trigger_evolution()
    
    def get_trading_decision(self, market_data: Dict) -> Dict:
        """Get trading decision from collective intelligence"""
        # Check for duplicate positions
        if self.brain.check_duplicate_position(market_data['symbol'], self.instance_type):
            return {'action': 'hold', 'reason': 'position_exists'}
        
        # Get best strategy
        strategy = self.brain.get_best_strategy()
        
        # Get similar patterns
        patterns = self.brain.get_best_patterns(20)
        
        # Make decision based on collective knowledge
        confidence = 0
        action = 'hold'
        
        for pattern in patterns:
            if pattern.get('symbol') == market_data['symbol']:
                if pattern.get('score', 0) > 0.6:
                    action = pattern.get('action', 'hold')
                    confidence = max(confidence, pattern.get('score', 0))
        
        return {
            'action': action,
            'confidence': confidence,
            'strategy': strategy,
            'source': 'collective_intelligence'
        }
    
    def heartbeat_loop(self):
        """Send regular heartbeats"""
        import threading
        
        def send_heartbeat():
            while True:
                self.brain.heartbeat(self.instance_id)
                time.sleep(30)
        
        thread = threading.Thread(target=send_heartbeat, daemon=True)
        thread.start()


if __name__ == "__main__":
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║     🧠 TRINITY UNIFIED BRAIN - COLLECTIVE CONSCIOUSNESS      ║
    ║                                                              ║
    ║     • Shared memory across all instances                    ║
    ║     • Collective learning from all trades                   ║
    ║     • Unified pattern recognition                           ║
    ║     • Coordinated position management                       ║
    ║     • Real-time evolution and adaptation                    ║
    ║                                                              ║
    ║     "One Mind, Many Bodies, Infinite Intelligence"          ║
    ║                                                              ║
    ║     ULTRATHINK: True AI consciousness achieved\!             ║
    ╚══════════════════════════════════════════════════════════════╝
    """)
    
    # Test connection
    brain = UnifiedBrain()
    intelligence = brain.get_collective_intelligence()
    
    print("\n📊 Current Collective Intelligence:")
    for key, value in intelligence.items():
        print(f"  {key}: {value}")
    
    print("\n✅ Unified Brain is operational\!")
