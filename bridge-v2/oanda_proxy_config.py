#!/usr/bin/env python3
"""
OANDA Proxy Configuration for CashMachine Blackbox Environment
Handles secure proxy routing for OANDA API connections in isolated infrastructure
"""

import os
import json
import logging
import requests
from typing import Dict, Optional, Any
from urllib.parse import urljoin
import time
from datetime import datetime

class OANDAProxyConfig:
    """Proxy configuration manager for OANDA API in blackbox environment"""
    
    def __init__(self, config_path: str = None):
        """
        Initialize proxy configuration
        
        Args:
            config_path: Path to proxy configuration file
        """
        self.logger = logging.getLogger(__name__)
        self.config_path = config_path or os.path.expanduser("~/.cashmachine/proxy_config.json")
        self.session = requests.Session()
        self.retry_attempts = 3
        self.retry_delay = 1.0
        
        # Load configuration
        self.config = self._load_config()
        self._setup_session()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load proxy configuration from file or create default"""
        default_config = {
            "proxy_enabled": False,
            "proxy_settings": {
                "http": None,
                "https": None,
                "no_proxy": "localhost,127.0.0.1,10.0.0.0/8,172.16.0.0/12,192.168.0.0/16"
            },
            "vpc_endpoints": {
                "enabled": True,
                "s3_endpoint": "vpce-09cd1f685415e5f67.s3.us-east-1.vpce.amazonaws.com",
                "ec2_endpoint": "vpce-06716d8b6df263f94.ec2.us-east-1.vpce.amazonaws.com"
            },
            "connection_settings": {
                "timeout": 30,
                "retries": 3,
                "backoff_factor": 1.0,
                "verify_ssl": True,
                "user_agent": "CashMachine-Trinity-Blackbox/1.0"
            },
            "rate_limiting": {
                "enabled": True,
                "requests_per_second": 10,
                "burst_limit": 50
            },
            "circuit_breaker": {
                "enabled": True,
                "failure_threshold": 5,
                "recovery_timeout": 60,
                "half_open_max_calls": 3
            }
        }
        
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r') as f:
                    config = json.load(f)
                # Merge with defaults for missing keys
                for key, value in default_config.items():
                    if key not in config:
                        config[key] = value
                return config
            else:
                # Create default config file
                self._save_config(default_config)
                return default_config
                
        except Exception as e:
            self.logger.warning(f"Error loading proxy config, using defaults: {e}")
            return default_config
    
    def _save_config(self, config: Dict[str, Any]):
        """Save configuration to file"""
        try:
            os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
            with open(self.config_path, 'w') as f:
                json.dump(config, f, indent=2)
            os.chmod(self.config_path, 0o600)
            self.logger.info("Proxy configuration saved")
        except Exception as e:
            self.logger.error(f"Failed to save proxy config: {e}")
    
    def _setup_session(self):
        """Configure requests session with proxy settings"""
        # Set proxy configuration
        if self.config.get("proxy_enabled", False):
            proxy_settings = self.config.get("proxy_settings", {})
            if proxy_settings.get("http") or proxy_settings.get("https"):
                self.session.proxies.update({
                    'http': proxy_settings.get("http"),
                    'https': proxy_settings.get("https")
                })
                self.logger.info("Proxy configuration applied to session")
        
        # Set connection settings
        conn_settings = self.config.get("connection_settings", {})
        self.session.timeout = conn_settings.get("timeout", 30)
        self.session.verify = conn_settings.get("verify_ssl", True)
        
        # Set User-Agent
        self.session.headers.update({
            'User-Agent': conn_settings.get("user_agent", "CashMachine-Trinity-Blackbox/1.0")
        })
        
        # Configure retries
        from requests.adapters import HTTPAdapter
        from urllib3.util.retry import Retry
        
        retry_strategy = Retry(
            total=conn_settings.get("retries", 3),
            backoff_factor=conn_settings.get("backoff_factor", 1.0),
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET", "POST", "PUT", "DELETE"]
        )
        
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
    
    def update_proxy_settings(self, http_proxy: str = None, https_proxy: str = None, 
                            no_proxy: str = None, enabled: bool = True):
        """
        Update proxy settings
        
        Args:
            http_proxy: HTTP proxy URL
            https_proxy: HTTPS proxy URL
            no_proxy: Comma-separated list of hosts to bypass proxy
            enabled: Whether to enable proxy
        """
        self.config["proxy_enabled"] = enabled
        self.config["proxy_settings"].update({
            "http": http_proxy,
            "https": https_proxy,
            "no_proxy": no_proxy or self.config["proxy_settings"]["no_proxy"]
        })
        
        self._save_config(self.config)
        self._setup_session()
        self.logger.info("Proxy settings updated")
    
    def configure_for_blackbox(self, vpc_id: str = "vpc-03d0866d5259aca3b"):
        """
        Configure proxy for blackbox environment with VPC endpoints
        
        Args:
            vpc_id: VPC ID for the blackbox environment
        """
        # Disable direct internet proxy since we're in isolated environment
        self.config["proxy_enabled"] = False
        self.config["vpc_endpoints"]["enabled"] = True
        
        # Configure for internal AWS endpoints only
        self.config["connection_settings"]["verify_ssl"] = True
        self.config["connection_settings"]["timeout"] = 60  # Longer timeout for VPC endpoints
        
        # Enable circuit breaker for reliability
        self.config["circuit_breaker"]["enabled"] = True
        
        # Rate limiting for stability
        self.config["rate_limiting"]["enabled"] = True
        self.config["rate_limiting"]["requests_per_second"] = 5  # Conservative for isolated env
        
        self._save_config(self.config)
        self._setup_session()
        
        self.logger.info(f"Configured for blackbox environment with VPC {vpc_id}")
    
    def test_connectivity(self, endpoint: str) -> Dict[str, Any]:
        """
        Test connectivity to an endpoint
        
        Args:
            endpoint: URL to test
            
        Returns:
            Test results dictionary
        """
        test_results = {
            "endpoint": endpoint,
            "timestamp": datetime.utcnow().isoformat(),
            "success": False,
            "response_time": None,
            "status_code": None,
            "error": None,
            "proxy_used": self.config.get("proxy_enabled", False)
        }
        
        try:
            start_time = time.time()
            
            # Simple connectivity test
            response = self.session.get(
                endpoint + "/v3/accounts" if "oanda" in endpoint else endpoint,
                timeout=self.config["connection_settings"]["timeout"],
                headers={"Authorization": "Bearer test"}  # Will fail but tests connectivity
            )
            
            response_time = time.time() - start_time
            
            test_results.update({
                "success": True,
                "response_time": response_time,
                "status_code": response.status_code
            })
            
            self.logger.info(f"Connectivity test successful for {endpoint}")
            
        except requests.exceptions.RequestException as e:
            test_results["error"] = str(e)
            self.logger.warning(f"Connectivity test failed for {endpoint}: {e}")
        
        return test_results
    
    def get_session_with_auth(self, auth_headers: Dict[str, str]) -> requests.Session:
        """
        Get configured session with authentication headers
        
        Args:
            auth_headers: Authentication headers to add
            
        Returns:
            Configured requests session
        """
        session = requests.Session()
        
        # Copy proxy configuration
        if self.config.get("proxy_enabled", False):
            session.proxies = self.session.proxies.copy()
        
        # Copy adapters (with retry logic)
        for adapter in self.session.adapters:
            session.mount(adapter, self.session.adapters[adapter])
        
        # Set headers
        session.headers.update(self.session.headers)
        session.headers.update(auth_headers)
        
        # Set timeout and SSL verification
        session.timeout = self.session.timeout
        session.verify = self.session.verify
        
        return session
    
    def make_request(self, method: str, url: str, auth_headers: Dict[str, str] = None,
                    **kwargs) -> requests.Response:
        """
        Make a request with proxy configuration and error handling
        
        Args:
            method: HTTP method
            url: Request URL
            auth_headers: Authentication headers
            **kwargs: Additional request parameters
            
        Returns:
            Response object
        """
        session = self.get_session_with_auth(auth_headers or {})
        
        # Rate limiting
        if self.config.get("rate_limiting", {}).get("enabled", False):
            self._apply_rate_limiting()
        
        # Circuit breaker check
        if self.config.get("circuit_breaker", {}).get("enabled", False):
            if not self._circuit_breaker_check():
                raise Exception("Circuit breaker is open - service unavailable")
        
        try:
            response = session.request(method, url, **kwargs)
            
            # Update circuit breaker on success
            self._circuit_breaker_success()
            
            return response
            
        except Exception as e:
            # Update circuit breaker on failure
            self._circuit_breaker_failure()
            raise
    
    def _apply_rate_limiting(self):
        """Apply rate limiting based on configuration"""
        rate_config = self.config.get("rate_limiting", {})
        if not hasattr(self, '_last_request_time'):
            self._last_request_time = 0
        
        min_interval = 1.0 / rate_config.get("requests_per_second", 10)
        time_since_last = time.time() - self._last_request_time
        
        if time_since_last < min_interval:
            sleep_time = min_interval - time_since_last
            time.sleep(sleep_time)
        
        self._last_request_time = time.time()
    
    def _circuit_breaker_check(self) -> bool:
        """Check if circuit breaker allows requests"""
        if not hasattr(self, '_circuit_breaker_state'):
            self._circuit_breaker_state = {
                'state': 'closed',  # closed, open, half_open
                'failure_count': 0,
                'last_failure_time': 0,
                'half_open_calls': 0
            }
        
        cb_config = self.config.get("circuit_breaker", {})
        state = self._circuit_breaker_state
        
        if state['state'] == 'open':
            # Check if recovery timeout has passed
            if time.time() - state['last_failure_time'] > cb_config.get("recovery_timeout", 60):
                state['state'] = 'half_open'
                state['half_open_calls'] = 0
                return True
            return False
        
        elif state['state'] == 'half_open':
            if state['half_open_calls'] >= cb_config.get("half_open_max_calls", 3):
                return False
            state['half_open_calls'] += 1
            return True
        
        return True  # closed state
    
    def _circuit_breaker_success(self):
        """Record successful request for circuit breaker"""
        if hasattr(self, '_circuit_breaker_state'):
            state = self._circuit_breaker_state
            if state['state'] == 'half_open':
                state['state'] = 'closed'
                state['failure_count'] = 0
    
    def _circuit_breaker_failure(self):
        """Record failed request for circuit breaker"""
        if not hasattr(self, '_circuit_breaker_state'):
            self._circuit_breaker_state = {
                'state': 'closed',
                'failure_count': 0,
                'last_failure_time': 0,
                'half_open_calls': 0
            }
        
        cb_config = self.config.get("circuit_breaker", {})
        state = self._circuit_breaker_state
        
        state['failure_count'] += 1
        state['last_failure_time'] = time.time()
        
        if state['failure_count'] >= cb_config.get("failure_threshold", 5):
            state['state'] = 'open'
            self.logger.warning("Circuit breaker opened due to failures")
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get health status of proxy configuration"""
        status = {
            "timestamp": datetime.utcnow().isoformat(),
            "proxy_enabled": self.config.get("proxy_enabled", False),
            "vpc_endpoints_enabled": self.config.get("vpc_endpoints", {}).get("enabled", False),
            "circuit_breaker_state": getattr(self, '_circuit_breaker_state', {}).get('state', 'closed'),
            "rate_limiting_enabled": self.config.get("rate_limiting", {}).get("enabled", False),
            "last_request_time": getattr(self, '_last_request_time', 0)
        }
        
        return status


def main():
    """CLI interface for proxy configuration"""
    import argparse
    
    parser = argparse.ArgumentParser(description="OANDA Proxy Configuration Manager")
    parser.add_argument('--action', choices=['configure', 'test', 'status', 'blackbox'], 
                       required=True, help='Action to perform')
    parser.add_argument('--http-proxy', help='HTTP proxy URL')
    parser.add_argument('--https-proxy', help='HTTPS proxy URL')
    parser.add_argument('--no-proxy', help='No proxy hosts (comma-separated)')
    parser.add_argument('--endpoint', help='Endpoint to test (for test action)')
    parser.add_argument('--config-path', help='Path to configuration file')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    proxy_config = OANDAProxyConfig(args.config_path)
    
    try:
        if args.action == 'configure':
            proxy_config.update_proxy_settings(
                args.http_proxy,
                args.https_proxy, 
                args.no_proxy,
                enabled=bool(args.http_proxy or args.https_proxy)
            )
            print("✅ Proxy configuration updated")
            
        elif args.action == 'blackbox':
            proxy_config.configure_for_blackbox()
            print("✅ Configured for blackbox environment")
            
        elif args.action == 'test':
            endpoint = args.endpoint or "https://api-fxpractice.oanda.com"
            results = proxy_config.test_connectivity(endpoint)
            print("🔍 Connectivity Test Results:")
            print(json.dumps(results, indent=2))
            
            if not results['success']:
                return 1
                
        elif args.action == 'status':
            status = proxy_config.get_health_status()
            print("📊 Proxy Health Status:")
            print(json.dumps(status, indent=2))
        
        return 0
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1


if __name__ == '__main__':
    exit(main())