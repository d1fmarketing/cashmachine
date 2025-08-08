#!/usr/bin/env python3
"""
Configuration Manager for CASHMACHINE/ULTRATHINK
Centralizes all configuration loading and management
"""

import os
import json
import logging
from typing import Dict, Any, Optional, List
from pathlib import Path
from dataclasses import dataclass
from dotenv import load_dotenv

logger = logging.getLogger(__name__)


@dataclass
class APIConfig:
    """API configuration container"""
    alphavantage_key: str
    polygon_key: str
    finnhub_key: str
    coingecko_key: str
    twelvedata_key: str
    oanda_token: str
    oanda_account: str
    alpaca_key: str
    alpaca_secret: str
    alpaca_base_url: str


@dataclass
class NetworkConfig:
    """Network configuration container"""
    redis_host: str
    redis_port: int
    redis_db: int
    redis_password: Optional[str]
    trinity_host: str
    data_collector_host: str
    ml_farm_host: str
    bridge_host: str
    proxy_host: str
    superset_host: str


@dataclass
class PathConfig:
    """Path configuration container"""
    log_dir: Path
    config_dir: Path
    trinity_dir: Path
    data_dir: Path
    model_dir: Path
    ultrathink_log: Path
    trinity_log: Path
    circuit_breaker_log: Path
    learning_log: Path


@dataclass
class TradingConfig:
    """Trading configuration container"""
    max_position_size: int
    max_daily_trades: int
    max_loss_percent: float
    min_confidence_threshold: float
    target_buy_ratio: float
    target_sell_ratio: float
    target_hold_ratio: float
    balance_check_interval: int
    force_balance_threshold: float


@dataclass
class LearningConfig:
    """Learning configuration container"""
    learning_rate: float
    exploration_rate: float
    min_exploration_rate: float
    model_save_interval: int
    memory_size: int


@dataclass
class FeatureFlags:
    """Feature flags container"""
    enable_learning: bool
    enable_paper_trading: bool
    enable_circuit_breaker: bool
    enable_balance_enforcement: bool
    debug_mode: bool
    enable_alerts: bool
    enable_metrics: bool


class ConfigManager:
    """
    Central configuration manager for the entire system
    """
    
    def __init__(self, env_file: Optional[str] = None):
        """
        Initialize configuration manager
        
        Args:
            env_file: Path to .env file (defaults to .env in project root)
        """
        # Determine env file path
        if env_file:
            self.env_file = Path(env_file)
        else:
            # Try multiple locations
            possible_paths = [
                Path('/opt/cashmachine/.env'),
                Path('/tmp/cashmachine/.env'),
                Path.cwd() / '.env',
                Path.home() / '.cashmachine' / '.env'
            ]
            
            for path in possible_paths:
                if path.exists():
                    self.env_file = path
                    break
            else:
                # Use default location
                self.env_file = Path('/opt/cashmachine/.env')
        
        # Load environment variables
        if self.env_file.exists():
            load_dotenv(self.env_file)
            logger.info(f"Loaded configuration from {self.env_file}")
        else:
            logger.warning(f"No .env file found at {self.env_file}, using environment variables")
        
        # Initialize configuration sections
        self.api = self._load_api_config()
        self.network = self._load_network_config()
        self.paths = self._load_path_config()
        self.trading = self._load_trading_config()
        self.learning = self._load_learning_config()
        self.features = self._load_feature_flags()
        
        # Additional dynamic configuration
        self._custom_config = {}
    
    def _load_api_config(self) -> APIConfig:
        """Load API configuration from environment"""
        return APIConfig(
            alphavantage_key=os.getenv('ALPHAVANTAGE_API_KEY', ''),
            polygon_key=os.getenv('POLYGON_API_KEY', ''),
            finnhub_key=os.getenv('FINNHUB_API_KEY', ''),
            coingecko_key=os.getenv('COINGECKO_API_KEY', ''),
            twelvedata_key=os.getenv('TWELVEDATA_API_KEY', ''),
            oanda_token=os.getenv('OANDA_API_TOKEN', ''),
            oanda_account=os.getenv('OANDA_ACCOUNT_ID', ''),
            alpaca_key=os.getenv('ALPACA_API_KEY', ''),
            alpaca_secret=os.getenv('ALPACA_API_SECRET', ''),
            alpaca_base_url=os.getenv('ALPACA_BASE_URL', 'https://paper-api.alpaca.markets/v2')
        )
    
    def _load_network_config(self) -> NetworkConfig:
        """Load network configuration from environment"""
        return NetworkConfig(
            redis_host=os.getenv('REDIS_HOST', 'localhost'),
            redis_port=int(os.getenv('REDIS_PORT', 6379)),
            redis_db=int(os.getenv('REDIS_DB', 0)),
            redis_password=os.getenv('REDIS_PASSWORD') or None,
            trinity_host=os.getenv('TRINITY_HOST', 'localhost'),
            data_collector_host=os.getenv('DATA_COLLECTOR_HOST', 'localhost'),
            ml_farm_host=os.getenv('ML_FARM_HOST', 'localhost'),
            bridge_host=os.getenv('BRIDGE_HOST', 'localhost'),
            proxy_host=os.getenv('PROXY_HOST', 'localhost'),
            superset_host=os.getenv('SUPERSET_HOST', 'localhost')
        )
    
    def _load_path_config(self) -> PathConfig:
        """Load path configuration from environment"""
        return PathConfig(
            log_dir=Path(os.getenv('LOG_DIR', '/tmp')),
            config_dir=Path(os.getenv('CONFIG_DIR', '/opt/cashmachine/config')),
            trinity_dir=Path(os.getenv('TRINITY_DIR', '/opt/cashmachine/trinity')),
            data_dir=Path(os.getenv('DATA_DIR', '/opt/cashmachine/data')),
            model_dir=Path(os.getenv('MODEL_DIR', '/opt/cashmachine/models')),
            ultrathink_log=Path(os.getenv('ULTRATHINK_LOG', '/tmp/ultrathink.log')),
            trinity_log=Path(os.getenv('TRINITY_LOG', '/tmp/trinity_integrated.log')),
            circuit_breaker_log=Path(os.getenv('CIRCUIT_BREAKER_LOG', '/tmp/circuit_breaker.log')),
            learning_log=Path(os.getenv('LEARNING_LOG', '/tmp/ultrathink_learning.log'))
        )
    
    def _load_trading_config(self) -> TradingConfig:
        """Load trading configuration from environment"""
        return TradingConfig(
            max_position_size=int(os.getenv('MAX_POSITION_SIZE', 1000)),
            max_daily_trades=int(os.getenv('MAX_DAILY_TRADES', 100)),
            max_loss_percent=float(os.getenv('MAX_LOSS_PERCENT', 2.0)),
            min_confidence_threshold=float(os.getenv('MIN_CONFIDENCE_THRESHOLD', 0.6)),
            target_buy_ratio=float(os.getenv('TARGET_BUY_RATIO', 0.3333)),
            target_sell_ratio=float(os.getenv('TARGET_SELL_RATIO', 0.3333)),
            target_hold_ratio=float(os.getenv('TARGET_HOLD_RATIO', 0.3334)),
            balance_check_interval=int(os.getenv('BALANCE_CHECK_INTERVAL', 10)),
            force_balance_threshold=float(os.getenv('FORCE_BALANCE_THRESHOLD', 0.37))
        )
    
    def _load_learning_config(self) -> LearningConfig:
        """Load learning configuration from environment"""
        return LearningConfig(
            learning_rate=float(os.getenv('LEARNING_RATE', 0.01)),
            exploration_rate=float(os.getenv('EXPLORATION_RATE', 0.3)),
            min_exploration_rate=float(os.getenv('MIN_EXPLORATION_RATE', 0.05)),
            model_save_interval=int(os.getenv('MODEL_SAVE_INTERVAL', 10)),
            memory_size=int(os.getenv('MEMORY_SIZE', 1000))
        )
    
    def _load_feature_flags(self) -> FeatureFlags:
        """Load feature flags from environment"""
        return FeatureFlags(
            enable_learning=os.getenv('ENABLE_LEARNING', 'true').lower() == 'true',
            enable_paper_trading=os.getenv('ENABLE_PAPER_TRADING', 'true').lower() == 'true',
            enable_circuit_breaker=os.getenv('ENABLE_CIRCUIT_BREAKER', 'true').lower() == 'true',
            enable_balance_enforcement=os.getenv('ENABLE_BALANCE_ENFORCEMENT', 'true').lower() == 'true',
            debug_mode=os.getenv('DEBUG_MODE', 'false').lower() == 'true',
            enable_alerts=os.getenv('ENABLE_ALERTS', 'false').lower() == 'true',
            enable_metrics=os.getenv('ENABLE_METRICS', 'true').lower() == 'true'
        )
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value by key
        
        Args:
            key: Configuration key (can use dot notation, e.g., 'api.alphavantage_key')
            default: Default value if key not found
        
        Returns:
            Configuration value or default
        """
        parts = key.split('.')
        
        if len(parts) == 1:
            # Check custom config first
            if key in self._custom_config:
                return self._custom_config[key]
            # Then check environment
            return os.getenv(key.upper(), default)
        
        # Navigate nested configuration
        obj = self
        for part in parts:
            if hasattr(obj, part):
                obj = getattr(obj, part)
            else:
                return default
        
        return obj
    
    def set(self, key: str, value: Any):
        """
        Set custom configuration value
        
        Args:
            key: Configuration key
            value: Configuration value
        """
        self._custom_config[key] = value
    
    def get_redis_url(self) -> str:
        """Get Redis connection URL"""
        if self.network.redis_password:
            return f"redis://:{self.network.redis_password}@{self.network.redis_host}:{self.network.redis_port}/{self.network.redis_db}"
        return f"redis://{self.network.redis_host}:{self.network.redis_port}/{self.network.redis_db}"
    
    def get_api_config(self, api_name: str) -> Dict[str, Any]:
        """
        Get configuration for specific API
        
        Args:
            api_name: Name of the API (e.g., 'alphavantage', 'polygon')
        
        Returns:
            API configuration dictionary
        """
        api_configs = {
            'alphavantage': {
                'key': self.api.alphavantage_key,
                'base_url': 'https://www.alphavantage.co/query',
                'rate_limit': 5 if not self.api.alphavantage_key else 500  # Premium vs free
            },
            'polygon': {
                'key': self.api.polygon_key,
                'base_url': 'https://api.polygon.io',
                'rate_limit': 5
            },
            'finnhub': {
                'key': self.api.finnhub_key,
                'base_url': 'https://finnhub.io/api/v1',
                'rate_limit': 60
            },
            'coingecko': {
                'key': self.api.coingecko_key,
                'base_url': 'https://api.coingecko.com/api/v3',
                'rate_limit': 50
            },
            'oanda': {
                'token': self.api.oanda_token,
                'account': self.api.oanda_account,
                'base_url': 'https://api-fxpractice.oanda.com/v3',
                'stream_url': 'https://stream-fxpractice.oanda.com/v3'
            },
            'alpaca': {
                'key': self.api.alpaca_key,
                'secret': self.api.alpaca_secret,
                'base_url': self.api.alpaca_base_url
            }
        }
        
        return api_configs.get(api_name, {})
    
    def validate(self) -> Dict[str, List[str]]:
        """
        Validate configuration
        
        Returns:
            Dictionary of validation errors by section
        """
        errors = {}
        
        # Check required API keys
        api_errors = []
        if not self.api.alphavantage_key and not self.api.polygon_key:
            api_errors.append("At least one market data API key required")
        
        if api_errors:
            errors['api'] = api_errors
        
        # Check network configuration
        network_errors = []
        if not self.network.redis_host:
            network_errors.append("Redis host is required")
        
        if network_errors:
            errors['network'] = network_errors
        
        # Check paths exist
        path_errors = []
        for dir_path in [self.paths.log_dir, self.paths.config_dir]:
            if not dir_path.exists():
                path_errors.append(f"Directory does not exist: {dir_path}")
        
        if path_errors:
            errors['paths'] = path_errors
        
        # Check trading parameters
        trading_errors = []
        if self.trading.target_buy_ratio + self.trading.target_sell_ratio + self.trading.target_hold_ratio != 1.0:
            trading_errors.append("Target ratios must sum to 1.0")
        
        if self.trading.max_loss_percent <= 0:
            trading_errors.append("Max loss percent must be positive")
        
        if trading_errors:
            errors['trading'] = trading_errors
        
        return errors
    
    def save_to_file(self, file_path: Optional[str] = None):
        """
        Save current configuration to JSON file
        
        Args:
            file_path: Path to save configuration (defaults to config_dir/config.json)
        """
        if not file_path:
            file_path = self.paths.config_dir / 'config.json'
        
        config_dict = {
            'api': self.api.__dict__,
            'network': self.network.__dict__,
            'paths': {k: str(v) for k, v in self.paths.__dict__.items()},
            'trading': self.trading.__dict__,
            'learning': self.learning.__dict__,
            'features': self.features.__dict__,
            'custom': self._custom_config
        }
        
        with open(file_path, 'w') as f:
            json.dump(config_dict, f, indent=2)
        
        logger.info(f"Configuration saved to {file_path}")
    
    def __repr__(self) -> str:
        return f"ConfigManager(env_file={self.env_file})"


# Singleton instance
_config_instance: Optional[ConfigManager] = None


def get_config() -> ConfigManager:
    """
    Get singleton configuration instance
    
    Returns:
        ConfigManager instance
    """
    global _config_instance
    if _config_instance is None:
        _config_instance = ConfigManager()
    return _config_instance


def reset_config(env_file: Optional[str] = None):
    """
    Reset configuration with new env file
    
    Args:
        env_file: Path to new .env file
    """
    global _config_instance
    _config_instance = ConfigManager(env_file)
    return _config_instance


# Example usage
if __name__ == "__main__":
    # Load configuration
    config = get_config()
    
    # Validate configuration
    errors = config.validate()
    if errors:
        print("Configuration errors:")
        for section, section_errors in errors.items():
            print(f"  {section}:")
            for error in section_errors:
                print(f"    - {error}")
    else:
        print("Configuration valid!")
    
    # Access configuration
    print(f"\nRedis Host: {config.network.redis_host}")
    print(f"AlphaVantage Key: {config.api.alphavantage_key[:10]}..." if config.api.alphavantage_key else "AlphaVantage Key: Not set")
    print(f"Learning Enabled: {config.features.enable_learning}")
    print(f"Max Position Size: {config.trading.max_position_size}")
    
    # Get specific API config
    alpaca_config = config.get_api_config('alpaca')
    print(f"\nAlpaca Config: {alpaca_config}")
    
    # Save configuration
    # config.save_to_file()