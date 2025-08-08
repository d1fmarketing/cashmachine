# OANDA Trinity Integration Guide

## 🚀 Complete OANDA Trading API Integration for CashMachine Trinity System

This comprehensive integration provides secure, robust, and production-ready OANDA API connectivity for the Trinity trading system with advanced error handling, security hardening, and blackbox environment support.

## 📋 Table of Contents

1. [Quick Start](#quick-start)
2. [Architecture Overview](#architecture-overview)
3. [Components](#components)
4. [Security Features](#security-features)
5. [Installation & Setup](#installation--setup)
6. [Configuration](#configuration)
7. [Usage Examples](#usage-examples)
8. [Monitoring & Debugging](#monitoring--debugging)
9. [Production Deployment](#production-deployment)
10. [Troubleshooting](#troubleshooting)

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- OANDA API account (practice or live)
- Required packages: `requests`, `cryptography`, `pandas`, `numpy`

### Installation

```bash
# Install required packages
pip install requests cryptography pandas numpy

# Run the setup script
python setup_oanda_trinity.py --environment practice
```

### Basic Usage

```python
from trinity_oanda_integration import TrinityOANDAIntegration, TradingSignal

# Initialize integration
trinity = TrinityOANDAIntegration()

# Create trading signal
signal = TradingSignal(
    instrument="EUR_USD",
    action="BUY",
    confidence=0.85,
    stop_loss=1.0500,
    take_profit=1.0600
)

# Execute signal
result = await trinity.execute_signal(signal)
print(f"Trade executed: {result}")
```

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    Trinity Trading System                    │
├─────────────────────────────────────────────────────────────┤
│  TrinityOANDAIntegration (Main Interface)                   │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐ ┌──────────────┐ ┌─────────────────────┐   │
│  │   Auth      │ │    Proxy     │ │   Error Handler     │   │
│  │  Manager    │ │   Config     │ │   & Retry Logic     │   │
│  └─────────────┘ └──────────────┘ └─────────────────────┘   │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐ ┌──────────────┐ ┌─────────────────────┐   │
│  │  Security   │ │ Connection   │ │   Monitoring &      │   │
│  │ Hardening   │ │   Tester     │ │   Logging           │   │
│  └─────────────┘ └──────────────┘ └─────────────────────┘   │
├─────────────────────────────────────────────────────────────┤
│                      OANDA API                              │
└─────────────────────────────────────────────────────────────┘
```

## 🔧 Components

### 1. Authentication Manager (`oanda_auth_manager.py`)
- **Purpose**: Secure credential storage and API authentication
- **Features**:
  - AES-256 encryption for credentials
  - PBKDF2 key derivation
  - Session management
  - Credential validation

### 2. Proxy Configuration (`oanda_proxy_config.py`)
- **Purpose**: Proxy and network configuration for blackbox environments
- **Features**:
  - VPC endpoint support
  - Circuit breaker pattern
  - Rate limiting
  - Connection health monitoring

### 3. Trinity Integration (`trinity_oanda_integration.py`)
- **Purpose**: Main trading interface and signal execution
- **Features**:
  - Trading signal processing
  - Position management
  - Risk management
  - Performance tracking

### 4. Connection Tester (`oanda_connection_tester.py`)
- **Purpose**: Comprehensive connectivity validation
- **Features**:
  - Multi-endpoint testing
  - Latency benchmarking  
  - Trading workflow validation
  - Detailed reporting

### 5. Security Hardening (`oanda_security_hardening.py`)
- **Purpose**: Advanced security measures and monitoring
- **Features**:
  - IP whitelisting
  - SSL/TLS validation
  - Security event logging
  - Audit trail creation

### 6. Error Handler (`oanda_error_handler.py`)
- **Purpose**: Robust error handling and retry mechanisms
- **Features**:
  - Exponential backoff retry
  - Circuit breaker pattern
  - Error classification
  - Trading-specific error handling

## 🔒 Security Features

### Encryption & Storage
- **AES-256-GCM** encryption for credentials
- **PBKDF2** key derivation with 100,000 iterations
- **Secure file permissions** (0600)
- **Encrypted session management**

### Network Security
- **IP whitelisting** support
- **SSL/TLS validation** with minimum version enforcement
- **Certificate verification**
- **Proxy support** for secure environments

### Trading Security
- **Position size limits**
- **Daily risk limits**
- **Trading request validation**
- **Emergency stop functionality**

### Monitoring & Auditing
- **Security event logging**
- **Audit trail creation**
- **Suspicious activity detection**
- **Comprehensive security scanning**

## ⚙️ Installation & Setup

### Automated Setup

```bash
# Complete setup with interactive prompts
python setup_oanda_trinity.py

# Setup with parameters
python setup_oanda_trinity.py \
  --api-key "your-oanda-api-key" \
  --account-id "your-account-id" \
  --environment practice \
  --verbose
```

### Manual Setup

1. **Install Dependencies**
```bash
pip install requests cryptography pandas numpy
```

2. **Setup Authentication**
```python
from oanda_auth_manager import OANDAAuthManager

auth = OANDAAuthManager()
auth.store_credentials(
    api_key="your-oanda-api-key",
    account_id="your-account-id",
    environment="practice"
)
```

3. **Configure Security**
```python
from oanda_security_hardening import OANDASecurityHardening

security = OANDASecurityHardening()
scan_results = security.perform_security_scan()
```

4. **Test Connectivity**
```python
from oanda_connection_tester import OANDAConnectionTester

tester = OANDAConnectionTester()
results = await tester.run_comprehensive_test()
```

## 🔧 Configuration

### Security Configuration

```json
{
  "encryption": {
    "algorithm": "AES-256-GCM",
    "key_derivation": "PBKDF2",
    "iterations": 100000
  },
  "authentication": {
    "max_failed_attempts": 3,
    "lockout_duration": 300,
    "session_timeout": 3600
  },
  "network_security": {
    "require_ssl": true,
    "min_tls_version": "1.2",
    "verify_certificates": true
  },
  "trading_security": {
    "max_position_size": 0.05,
    "require_confirmation": true,
    "emergency_stop_enabled": true
  }
}
```

### Proxy Configuration

```json
{
  "proxy_enabled": false,
  "vpc_endpoints": {
    "enabled": true,
    "s3_endpoint": "vpce-xxxxx.s3.region.vpce.amazonaws.com"
  },
  "connection_settings": {
    "timeout": 30,
    "retries": 3,
    "verify_ssl": true
  },
  "circuit_breaker": {
    "enabled": true,
    "failure_threshold": 5,
    "recovery_timeout": 60
  }
}
```

## 💻 Usage Examples

### Basic Trading

```python
import asyncio
from trinity_oanda_integration import TrinityOANDAIntegration, TradingSignal

async def basic_trading_example():
    # Initialize Trinity integration
    trinity = TrinityOANDAIntegration()
    
    # Create trading signal
    signal = TradingSignal(
        instrument="EUR_USD",
        action="BUY",
        confidence=0.85,
        stop_loss=1.0500,
        take_profit=1.0600
    )
    
    # Execute signal with risk checks
    risk_approved, reason = await trinity.risk_check(signal)
    if risk_approved:
        result = await trinity.execute_signal(signal)
        print(f"Trade executed: {result}")
    else:
        print(f"Trade rejected: {reason}")

# Run example
asyncio.run(basic_trading_example())
```

### Market Data Analysis

```python
async def market_data_example():
    trinity = TrinityOANDAIntegration()
    
    # Get historical data
    instruments = ['EUR_USD', 'GBP_USD', 'USD_JPY']
    market_data = await trinity.get_market_data(instruments, count=100, granularity="H1")
    
    for instrument, df in market_data.items():
        print(f"\n{instrument} Data:")
        print(f"Latest Close: {df['close'].iloc[-1]}")
        print(f"24h Change: {((df['close'].iloc[-1] / df['close'].iloc[-25]) - 1) * 100:.2f}%")
        
    # Get current prices
    current_prices = await trinity.get_current_prices(instruments)
    for instrument, price_data in current_prices.items():
        print(f"{instrument}: Bid={price_data['bid']}, Ask={price_data['ask']}")

asyncio.run(market_data_example())
```

### Portfolio Management

```python
async def portfolio_example():
    trinity = TrinityOANDAIntegration()
    
    # Get current positions
    positions = await trinity.get_positions()
    print(f"Open Positions: {len(positions)}")
    
    for instrument, position in positions.items():
        print(f"{instrument}: {position['long_units']} long, {position['short_units']} short")
        print(f"  Unrealized P&L: {position['unrealized_pl']}")
    
    # Get performance metrics
    metrics = trinity.get_performance_metrics()
    print("\nPerformance Metrics:")
    print(f"Account Balance: ${metrics['account_balance']}")
    print(f"Total Trades: {metrics['total_trades']}")
    print(f"Win Rate: {metrics['win_rate']:.1%}")

asyncio.run(portfolio_example())
```

### Error Handling Example

```python
from oanda_error_handler import handle_oanda_errors, OANDAErrorHandler

# Initialize error handler
error_handler = OANDAErrorHandler()

@handle_oanda_errors(error_handler, "get_account_balance", "/v3/accounts")
async def get_account_balance():
    trinity = TrinityOANDAIntegration()
    account_info = trinity.get_account_info()
    return float(account_info['account']['balance'])

async def error_handling_example():
    try:
        balance = await get_account_balance()
        print(f"Account Balance: ${balance}")
    except Exception as e:
        print(f"Operation failed: {e}")
        
        # Get error statistics
        stats = error_handler.get_error_statistics()
        print(f"Total Errors Today: {stats['recent_errors_24h']}")

asyncio.run(error_handling_example())
```

## 📊 Monitoring & Debugging

### Connection Testing

```bash
# Test all components
python oanda_connection_tester.py --test all

# Test specific component
python oanda_connection_tester.py --test auth
python oanda_connection_tester.py --test market
python oanda_connection_tester.py --test latency --iterations 50
```

### Security Monitoring

```bash
# Run security scan
python oanda_security_hardening.py --action scan

# Validate SSL connections
python oanda_security_hardening.py --action validate-ssl --hostname api-fxpractice.oanda.com

# Emergency lockdown
python oanda_security_hardening.py --action lockdown --reason "Security incident"
```

### Error Analysis

```python
from oanda_error_handler import OANDAErrorHandler

error_handler = OANDAErrorHandler()

# Get error statistics
stats = error_handler.get_error_statistics()
print(f"Error Rate: {stats['error_rate_24h']:.2f} errors/hour")

# Export detailed report
report_file = error_handler.export_error_report()
print(f"Error report: {report_file}")
```

### Performance Monitoring

```python
async def monitoring_example():
    trinity = TrinityOANDAIntegration()
    
    # Performance metrics
    metrics = trinity.get_performance_metrics()
    
    # Connection health
    from oanda_proxy_config import OANDAProxyConfig
    proxy = OANDAProxyConfig()
    health = proxy.get_health_status()
    
    print("System Health:")
    print(f"  Account Balance: ${metrics['account_balance']}")
    print(f"  Open Positions: {metrics['open_positions']}")
    print(f"  Circuit Breaker: {health['circuit_breaker_state']}")
    print(f"  Rate Limiting: {'Enabled' if health['rate_limiting_enabled'] else 'Disabled'}")

asyncio.run(monitoring_example())
```

## 🚀 Production Deployment

### Blackbox Environment Setup

```bash
# Configure for blackbox environment
python setup_oanda_trinity.py --environment live --verbose

# Test in blackbox mode
python oanda_connection_tester.py --test all
```

### Environment Variables

```bash
# Set environment variables for production
export OANDA_ENVIRONMENT=live
export CASHMACHINE_LOG_LEVEL=INFO
export CASHMACHINE_SECURITY_LEVEL=high
```

### Systemd Service (Optional)

```ini
[Unit]
Description=OANDA Trinity Trading System
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/opt/cashmachine
ExecStart=/usr/bin/python3 trinity_trading_daemon.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

### Monitoring Setup

```python
# monitoring_daemon.py
import asyncio
import logging
from datetime import datetime, timedelta

async def monitoring_loop():
    while True:
        try:
            # Health checks
            trinity = TrinityOANDAIntegration()
            metrics = trinity.get_performance_metrics()
            
            # Alert on issues
            if metrics.get('unrealized_pl', 0) < -1000:  # $1000 loss
                # Send alert (email, slack, etc.)
                pass
                
            # Log metrics
            logging.info(f"System healthy - Balance: ${metrics['account_balance']}")
            
        except Exception as e:
            logging.error(f"Monitoring error: {e}")
        
        await asyncio.sleep(60)  # Check every minute

if __name__ == '__main__':
    asyncio.run(monitoring_loop())
```

## 🔧 Troubleshooting

### Common Issues

1. **Authentication Failures**
```bash
# Re-validate credentials
python oanda_auth_manager.py --action validate

# Check credential format
python oanda_auth_manager.py --action info
```

2. **Network Connectivity Issues**
```bash
# Test connectivity
python oanda_connection_tester.py --test all

# Check proxy configuration
python oanda_proxy_config.py --action status
```

3. **Trading Errors**
```python
# Check account status
trinity = TrinityOANDAIntegration()
account_info = trinity.account_info
print(f"Margin Available: {account_info['account']['marginAvailable']}")

# Review error logs
error_handler = OANDAErrorHandler()
stats = error_handler.get_error_statistics()
```

### Debug Mode

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Enable verbose error reporting
from oanda_error_handler import OANDAErrorHandler
error_handler = OANDAErrorHandler()
error_handler.logger.setLevel(logging.DEBUG)
```

### Emergency Procedures

```python
# Emergency stop all trading
trinity = TrinityOANDAIntegration()
await trinity.emergency_stop()

# Security lockdown
from oanda_security_hardening import OANDASecurityHardening
security = OANDASecurityHardening()
security.emergency_lockdown("Manual intervention")
```

## 📞 Support & Resources

### OANDA Resources
- [OANDA API Documentation](https://developer.oanda.com/rest-live-v20/introduction/)
- [OANDA Developer Portal](https://developer.oanda.com/)
- [OANDA Support](https://www.oanda.com/us-en/support/)

### CashMachine Resources
- Project Repository: `/Users/d1f/Desktop/777/Projeto CashMachine/`
- Setup Script: `setup_oanda_trinity.py`
- Configuration: `~/.cashmachine/`
- Logs: `~/.cashmachine/security_events.log`

### File Structure

```
~/.cashmachine/
├── oanda_credentials.enc     # Encrypted credentials
├── proxy_config.json         # Proxy configuration
├── security_config.json      # Security settings
├── security_events.log       # Security audit log
├── setup.log                 # Setup log
└── reports/                  # Test and error reports
    ├── connection_test_*.json
    └── error_report_*.json
```

---

## ⚡ Quick Reference

### Key Commands

```bash
# Complete setup
python setup_oanda_trinity.py

# Test connectivity
python oanda_connection_tester.py --test all

# Security scan
python oanda_security_hardening.py --action scan

# Validate credentials
python oanda_auth_manager.py --action validate
```

### Key Classes

- `TrinityOANDAIntegration`: Main trading interface
- `OANDAAuthManager`: Credential management
- `OANDAProxyConfig`: Network configuration
- `OANDASecurityHardening`: Security features
- `OANDAConnectionTester`: Connectivity validation
- `OANDAErrorHandler`: Error handling & retry

### Environment Files

- Production: `~/.cashmachine/`
- Credentials: `~/.cashmachine/oanda_credentials.enc`
- Logs: `~/.cashmachine/security_events.log`

---

**Status**: ✅ Production Ready  
**Security Level**: 🔒 High  
**Environment**: 🏗️ Blackbox Compatible  
**Testing**: 🧪 Comprehensive Suite  

*This integration provides enterprise-grade security, reliability, and performance for OANDA API trading operations within the Trinity system.*