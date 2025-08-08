#!/usr/bin/env python3
"""
ULTRATHINK 100% ENHANCED
The perfect fusion of working logic + sacred mathematics
"""

import numpy as np
import random
import logging
from typing import List, Dict

logger = logging.getLogger(__name__)

class EnhancedGeneticStrategy:
    """ASI with sacred enhancements but practical constraints"""
    
    def __init__(self):
        # Sacred constants (for bonuses, not constraints)
        self.PI = 3.14159265359
        self.PHI = 1.618033988749
        self.SACRED_69 = 69
        
        # PRACTICAL population parameters
        self.population_size = 30  # Keep original size that works
        self.mutation_rate = 0.15  # Higher for diversity
        self.crossover_rate = 0.7
        
        # WIDER gene ranges for diversity
        self.gene_ranges = {
            'rsi_buy': (20, 40),  # WIDER than sacred
            'rsi_sell': (60, 80),  # WIDER than sacred
            'ma_weight': (0.5, 2.0),
            'momentum_threshold': (0.001, 0.05),
            'stop_loss': (0.005, 0.05),
            'take_profit': (0.01, 0.10),
            'sacred_bonus': (0.0, 0.3),  # Optional sacred boost
            'diversity_factor': (0.8, 1.2)  # Force diversity
        }
        
        self.population = []
        self.generation = 1
        self.stuck_counter = 0  # Track if stuck
        self.last_signal = 'hold'
        self.last_confidence = 0.5
        
        self._initialize_diverse_population()
        logger.info(f"🧬 Enhanced ASI initialized (Pop {self.population_size})")
    
    def _initialize_diverse_population(self):
        """Create HIGHLY diverse initial population"""
        self.population = []
        
        for i in range(self.population_size):
            genome = {}
            
            # FORCE diversity through different initialization strategies
            strategy = i % 5
            
            if strategy == 0:  # Aggressive buyer
                genome['rsi_buy'] = random.uniform(30, 40)
                genome['rsi_sell'] = random.uniform(70, 80)
                genome['momentum_threshold'] = random.uniform(0.001, 0.01)
            elif strategy == 1:  # Conservative seller
                genome['rsi_buy'] = random.uniform(20, 30)
                genome['rsi_sell'] = random.uniform(60, 70)
                genome['momentum_threshold'] = random.uniform(0.02, 0.05)
            elif strategy == 2:  # Sacred aligned
                genome['rsi_buy'] = 31.4  # Pi * 10
                genome['rsi_sell'] = 69    # Sacred
                genome['momentum_threshold'] = 0.0161  # Golden
            elif strategy == 3:  # Random
                for gene, (min_val, max_val) in self.gene_ranges.items():
                    genome[gene] = random.uniform(min_val, max_val)
            else:  # Mutant
                for gene, (min_val, max_val) in self.gene_ranges.items():
                    # Use non-uniform distribution for diversity
                    if random.random() < 0.5:
                        genome[gene] = min_val + (max_val - min_val) * random.random()**2
                    else:
                        genome[gene] = max_val - (max_val - min_val) * random.random()**2
            
            # Fill missing genes
            for gene, (min_val, max_val) in self.gene_ranges.items():
                if gene not in genome:
                    genome[gene] = random.uniform(min_val, max_val)
            
            genome['fitness'] = random.uniform(-10, 10)
            genome['trades'] = 0
            genome['wins'] = 0
            
            self.population.append(genome)
    
    def _break_out_of_stuck(self):
        """Force diversity if stuck"""
        logger.info(f"   🔄 Breaking out of stuck state (Gen {self.generation})")
        
        # Randomize bottom half of population
        self.population.sort(key=lambda x: x['fitness'], reverse=True)
        
        for i in range(self.population_size // 2, self.population_size):
            for gene, (min_val, max_val) in self.gene_ranges.items():
                if random.random() < 0.5:  # 50% chance to randomize each gene
                    self.population[i][gene] = random.uniform(min_val, max_val)
            
            # Reset fitness
            self.population[i]['fitness'] = random.uniform(-10, 10)
        
        # Increase mutation rate temporarily
        self.mutation_rate = min(0.3, self.mutation_rate * 1.5)
        self.stuck_counter = 0
    
    def analyze(self, prices: List[float]) -> Dict:
        """Analyze with anti-stuck mechanisms"""
        if len(prices) < 50:
            return {'signal': 'hold', 'confidence': 0.5, 'generation': self.generation}
        
        # Calculate current indicators
        rsi = self._calculate_rsi(prices)
        ma_fast = np.mean(prices[-10:])
        ma_slow = np.mean(prices[-30:])
        momentum = (prices[-1] - prices[-10]) / prices[-10] if prices[-10] != 0 else 0
        
        # Evolve population
        self._evolve_with_diversity(prices, rsi, ma_fast, ma_slow, momentum)
        
        # Get best genome
        best_genome = self.population[0]
        
        # Generate signal with anti-stuck logic
        signal = 'hold'
        base_confidence = 0.5
        
        # More permissive conditions
        buy_threshold = best_genome['rsi_buy'] * best_genome.get('diversity_factor', 1.0)
        sell_threshold = best_genome['rsi_sell'] * best_genome.get('diversity_factor', 1.0)
        
        if rsi < buy_threshold and ma_fast > ma_slow * 0.995:  # More permissive
            signal = 'buy'
            base_confidence = 0.4 + abs(buy_threshold - rsi) / 100
        elif rsi > sell_threshold and ma_fast < ma_slow * 1.005:  # More permissive
            signal = 'sell' 
            base_confidence = 0.4 + abs(rsi - sell_threshold) / 100
        elif abs(momentum) > best_genome['momentum_threshold']:
            signal = 'buy' if momentum > 0 else 'sell'
            base_confidence = 0.3 + min(0.4, abs(momentum) * 10)
        
        # Add sacred bonuses (optional, not required)
        sacred_bonus = 0
        if 67 < rsi < 71:  # Near 69
            sacred_bonus += 0.069
        if self.generation % 69 == 0:
            sacred_bonus += 0.069
        if self.generation in [21, 34, 55, 89, 144, 233]:
            sacred_bonus += 0.0618
        
        # Apply bonuses
        confidence = base_confidence * (1 + sacred_bonus)
        
        # Anti-stuck: Add randomness if repeating same signal
        if signal == self.last_signal and abs(confidence - self.last_confidence) < 0.05:
            self.stuck_counter += 1
            if self.stuck_counter > 5:
                # Force random signal
                if random.random() < 0.3:  # 30% chance to break pattern
                    signal = random.choice(['buy', 'sell', 'hold'])
                    confidence = random.uniform(0.25, 0.55)
                    logger.info(f"   🎲 Random break: {signal} ({confidence:.1%})")
                    self.stuck_counter = 0
        else:
            self.stuck_counter = 0
        
        # Track state
        self.last_signal = signal
        self.last_confidence = confidence
        
        # Ensure reasonable confidence
        confidence = min(0.95, max(0.15, confidence))
        
        return {
            'signal': signal,
            'confidence': confidence,
            'generation': self.generation,
            'sacred': sacred_bonus > 0,
            'stuck_counter': self.stuck_counter,
            'best_fitness': best_genome['fitness']
        }
    
    def _calculate_rsi(self, prices: List[float], period: int = 14) -> float:
        """Calculate RSI"""
        if len(prices) < period + 1:
            return 50
        
        deltas = [prices[i] - prices[i-1] for i in range(-period, 0)]
        gains = [d if d > 0 else 0 for d in deltas]
        losses = [-d if d < 0 else 0 for d in deltas]
        
        avg_gain = sum(gains) / period
        avg_loss = sum(losses) / period
        
        if avg_loss == 0:
            return 100 if avg_gain > 0 else 50
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _evolve_with_diversity(self, prices, rsi, ma_fast, ma_slow, momentum):
        """Evolve with forced diversity"""
        # Evaluate fitness
        for genome in self.population:
            # Simulate trading
            signal_count = 0
            correct_signals = 0
            
            if rsi < genome['rsi_buy'] and ma_fast > ma_slow:
                signal_count += 1
                if prices[-1] < prices[-5]:  # Would have been profitable
                    correct_signals += 1
            
            if rsi > genome['rsi_sell'] and ma_fast < ma_slow:
                signal_count += 1
                if prices[-1] > prices[-5]:  # Would have been profitable
                    correct_signals += 1
            
            # Fitness based on accuracy and activity
            if signal_count > 0:
                genome['fitness'] = (correct_signals / signal_count) * 10 + signal_count
            else:
                genome['fitness'] = -1  # Penalize inactivity
            
            # Bonus for diversity
            genome['fitness'] += random.uniform(-0.5, 0.5)  # Add noise
        
        # Sort by fitness
        self.population.sort(key=lambda x: x['fitness'], reverse=True)
        
        # Check if stuck (all similar fitness)
        fitness_variance = np.var([g['fitness'] for g in self.population[:10]])
        if fitness_variance < 0.1:
            self._break_out_of_stuck()
        
        # Evolve new generation
        new_population = self.population[:5]  # Keep top 5
        
        while len(new_population) < self.population_size:
            # Tournament selection
            parent1 = self._tournament_select()
            parent2 = self._tournament_select()
            
            if random.random() < self.crossover_rate:
                child = self._crossover(parent1, parent2)
            else:
                child = parent1.copy() if parent1['fitness'] > parent2['fitness'] else parent2.copy()
            
            # Mutate
            child = self._mutate(child)
            
            new_population.append(child)
        
        self.population = new_population
        self.generation += 1
        
        # Decay mutation rate
        self.mutation_rate = max(0.1, self.mutation_rate * 0.995)
    
    def _tournament_select(self, tournament_size=3):
        """Tournament selection"""
        tournament = random.sample(self.population, min(tournament_size, len(self.population)))
        return max(tournament, key=lambda x: x['fitness'])
    
    def _crossover(self, parent1, parent2):
        """Crossover with variation"""
        child = {}
        for gene in self.gene_ranges.keys():
            if random.random() < 0.5:
                child[gene] = parent1[gene]
            else:
                child[gene] = parent2[gene]
            
            # Occasionally average
            if random.random() < 0.2:
                child[gene] = (parent1[gene] + parent2[gene]) / 2
        
        child['fitness'] = 0
        child['trades'] = 0
        child['wins'] = 0
        return child
    
    def _mutate(self, genome):
        """Mutation with variable rate"""
        mutated = genome.copy()
        
        for gene, (min_val, max_val) in self.gene_ranges.items():
            if random.random() < self.mutation_rate:
                # Various mutation strategies
                strategy = random.choice(['uniform', 'gaussian', 'boundary'])
                
                if strategy == 'uniform':
                    mutated[gene] = random.uniform(min_val, max_val)
                elif strategy == 'gaussian':
                    mutated[gene] += np.random.randn() * (max_val - min_val) * 0.1
                    mutated[gene] = max(min_val, min(max_val, mutated[gene]))
                else:  # boundary
                    mutated[gene] = random.choice([min_val, max_val])
        
        return mutated