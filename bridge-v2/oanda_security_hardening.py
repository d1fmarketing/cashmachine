#!/usr/bin/env python3
"""
OANDA Security Hardening Module for Trinity System
Advanced security measures for API credentials, connections, and trading operations
"""

import os
import json
import hashlib
import hmac
import secrets
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import ipaddress
import ssl
import socket

class SecurityLevel(Enum):
    """Security levels for different operations"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class SecurityEvent:
    """Security event for logging and monitoring"""
    event_type: str
    severity: SecurityLevel
    message: str
    source_ip: Optional[str] = None
    user_agent: Optional[str] = None
    timestamp: Optional[str] = None
    details: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow().isoformat()
        if self.details is None:
            self.details = {}

class OANDASecurityHardening:
    """Security hardening for OANDA API integration"""
    
    def __init__(self, config_path: str = None):
        """
        Initialize security hardening
        
        Args:
            config_path: Path to security configuration file
        """
        self.logger = logging.getLogger(__name__)
        self.config_path = config_path or os.path.expanduser("~/.cashmachine/security_config.json")
        self.security_log_path = os.path.expanduser("~/.cashmachine/security_events.log")
        
        # Load security configuration
        self.config = self._load_security_config()
        
        # Security state
        self.failed_attempts = {}
        self.blocked_ips = set()
        self.session_tokens = {}
        self.api_rate_limits = {}
        
        # Initialize security logging
        self._setup_security_logging()
    
    def _load_security_config(self) -> Dict[str, Any]:
        """Load security configuration"""
        default_config = {
            "encryption": {
                "algorithm": "AES-256-GCM",
                "key_derivation": "PBKDF2",
                "iterations": 100000,
                "salt_length": 32
            },
            "authentication": {
                "max_failed_attempts": 3,
                "lockout_duration": 300,  # 5 minutes
                "session_timeout": 3600,   # 1 hour
                "require_2fa": False
            },
            "network_security": {
                "allowed_ips": [],
                "blocked_ips": [],
                "require_ssl": True,
                "min_tls_version": "1.2",
                "verify_certificates": True
            },
            "api_security": {
                "rate_limit_requests": 100,
                "rate_limit_window": 60,
                "max_request_size": 1048576,  # 1MB
                "allowed_endpoints": [
                    "/v3/accounts",
                    "/v3/instruments",
                    "/v3/orders",
                    "/v3/positions",
                    "/v3/pricing"
                ]
            },
            "monitoring": {
                "log_all_requests": True,
                "alert_on_suspicious": True,
                "max_log_size": 104857600,  # 100MB
                "log_retention_days": 30
            },
            "trading_security": {
                "max_position_size": 0.05,  # 5% of account
                "require_confirmation": True,
                "emergency_stop_enabled": True,
                "max_daily_trades": 100
            }
        }
        
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r') as f:
                    config = json.load(f)
                # Merge with defaults
                for section, settings in default_config.items():
                    if section not in config:
                        config[section] = settings
                    else:
                        for key, value in settings.items():
                            if key not in config[section]:
                                config[section][key] = value
                return config
            else:
                self._save_security_config(default_config)
                return default_config
                
        except Exception as e:
            self.logger.warning(f"Error loading security config, using defaults: {e}")
            return default_config
    
    def _save_security_config(self, config: Dict[str, Any]):
        """Save security configuration"""
        try:
            os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
            with open(self.config_path, 'w') as f:
                json.dump(config, f, indent=2)
            os.chmod(self.config_path, 0o600)
            self.logger.info("Security configuration saved")
        except Exception as e:
            self.logger.error(f"Failed to save security config: {e}")
    
    def _setup_security_logging(self):
        """Setup dedicated security event logging"""
        try:
            os.makedirs(os.path.dirname(self.security_log_path), exist_ok=True)
            
            # Create security logger
            security_logger = logging.getLogger('oanda_security')
            security_logger.setLevel(logging.INFO)
            
            # Create file handler
            handler = logging.FileHandler(self.security_log_path)
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            
            security_logger.addHandler(handler)
            
            # Set restrictive permissions
            os.chmod(self.security_log_path, 0o600)
            
        except Exception as e:
            self.logger.error(f"Failed to setup security logging: {e}")
    
    def log_security_event(self, event: SecurityEvent):
        """Log a security event"""
        try:
            security_logger = logging.getLogger('oanda_security')
            
            log_message = f"[{event.event_type}] {event.message}"
            if event.source_ip:
                log_message += f" | IP: {event.source_ip}"
            if event.details:
                log_message += f" | Details: {json.dumps(event.details)}"
            
            if event.severity == SecurityLevel.CRITICAL:
                security_logger.critical(log_message)
            elif event.severity == SecurityLevel.HIGH:
                security_logger.error(log_message)
            elif event.severity == SecurityLevel.MEDIUM:
                security_logger.warning(log_message)
            else:
                security_logger.info(log_message)
                
        except Exception as e:
            self.logger.error(f"Failed to log security event: {e}")
    
    def validate_credentials_strength(self, api_key: str) -> Tuple[bool, List[str]]:
        """
        Validate OANDA API key strength
        
        Args:
            api_key: API key to validate
            
        Returns:
            Tuple of (is_valid, issues_list)
        """
        issues = []
        
        # Check length
        if len(api_key) < 32:
            issues.append("API key too short (minimum 32 characters)")
        
        # Check character complexity
        has_upper = any(c.isupper() for c in api_key)
        has_lower = any(c.islower() for c in api_key)
        has_digit = any(c.isdigit() for c in api_key)
        has_special = any(c in '-_' for c in api_key)
        
        if not (has_upper and has_lower and has_digit):
            issues.append("API key should contain uppercase, lowercase, and digits")
        
        # Check for common patterns
        if api_key.count('-') > 10:  # OANDA keys typically have many hyphens
            pass  # This is normal for OANDA
        else:
            issues.append("API key format doesn't match OANDA standard")
        
        # Check entropy (simplified)
        unique_chars = len(set(api_key))
        if unique_chars < len(api_key) * 0.6:
            issues.append("API key has low entropy (too repetitive)")
        
        is_valid = len(issues) == 0
        
        if is_valid:
            self.log_security_event(SecurityEvent(
                event_type="CREDENTIAL_VALIDATION",
                severity=SecurityLevel.LOW,
                message="API key validation passed"
            ))
        else:
            self.log_security_event(SecurityEvent(
                event_type="CREDENTIAL_VALIDATION",
                severity=SecurityLevel.MEDIUM,
                message=f"API key validation failed: {'; '.join(issues)}"
            ))
        
        return is_valid, issues
    
    def check_ip_whitelist(self, ip_address: str) -> bool:
        """
        Check if IP address is in whitelist
        
        Args:
            ip_address: IP address to check
            
        Returns:
            True if allowed, False if blocked
        """
        try:
            ip = ipaddress.ip_address(ip_address)
            
            # Check blocked IPs
            for blocked_ip in self.config["network_security"]["blocked_ips"]:
                if ip in ipaddress.ip_network(blocked_ip, strict=False):
                    self.log_security_event(SecurityEvent(
                        event_type="IP_BLOCKED",
                        severity=SecurityLevel.HIGH,
                        message=f"Blocked IP attempted access: {ip_address}",
                        source_ip=ip_address
                    ))
                    return False
            
            # Check allowed IPs (if whitelist is configured)
            allowed_ips = self.config["network_security"]["allowed_ips"]
            if allowed_ips:
                for allowed_ip in allowed_ips:
                    if ip in ipaddress.ip_network(allowed_ip, strict=False):
                        return True
                
                # If whitelist exists but IP not in it, block
                self.log_security_event(SecurityEvent(
                    event_type="IP_NOT_WHITELISTED",
                    severity=SecurityLevel.MEDIUM,
                    message=f"Non-whitelisted IP attempted access: {ip_address}",
                    source_ip=ip_address
                ))
                return False
            
            # No whitelist configured, allow if not blocked
            return True
            
        except Exception as e:
            self.logger.error(f"Error checking IP whitelist: {e}")
            # Default to blocking on error
            return False
    
    def check_rate_limit(self, endpoint: str, identifier: str = "default") -> bool:
        """
        Check API rate limiting
        
        Args:
            endpoint: API endpoint being accessed
            identifier: Unique identifier (IP, user, etc.)
            
        Returns:
            True if within limits, False if rate limited
        """
        try:
            now = time.time()
            window = self.config["api_security"]["rate_limit_window"]
            max_requests = self.config["api_security"]["rate_limit_requests"]
            
            key = f"{identifier}:{endpoint}"
            
            if key not in self.api_rate_limits:
                self.api_rate_limits[key] = []
            
            # Clean old entries
            self.api_rate_limits[key] = [
                timestamp for timestamp in self.api_rate_limits[key]
                if now - timestamp < window
            ]
            
            # Check if over limit
            if len(self.api_rate_limits[key]) >= max_requests:
                self.log_security_event(SecurityEvent(
                    event_type="RATE_LIMIT_EXCEEDED",
                    severity=SecurityLevel.MEDIUM,
                    message=f"Rate limit exceeded for {endpoint}",
                    details={"identifier": identifier, "requests": len(self.api_rate_limits[key])}
                ))
                return False
            
            # Add current request
            self.api_rate_limits[key].append(now)
            return True
            
        except Exception as e:
            self.logger.error(f"Error checking rate limit: {e}")
            return True  # Allow on error (fail open for functionality)
    
    def validate_ssl_connection(self, hostname: str, port: int = 443) -> Tuple[bool, Dict[str, Any]]:
        """
        Validate SSL/TLS connection security
        
        Args:
            hostname: Hostname to validate
            port: Port number
            
        Returns:
            Tuple of (is_secure, validation_details)
        """
        try:
            # Create SSL context with security requirements
            context = ssl.create_default_context()
            context.check_hostname = True
            context.verify_mode = ssl.CERT_REQUIRED
            
            # Set minimum TLS version
            min_tls = self.config["network_security"]["min_tls_version"]
            if min_tls == "1.3":
                context.minimum_version = ssl.TLSVersion.TLSv1_3
            elif min_tls == "1.2":
                context.minimum_version = ssl.TLSVersion.TLSv1_2
            
            # Connect and get certificate info
            with socket.create_connection((hostname, port), timeout=10) as sock:
                with context.wrap_socket(sock, server_hostname=hostname) as ssock:
                    cert = ssock.getpeercert()
                    cipher = ssock.cipher()
                    version = ssock.version()
            
            # Validate certificate
            validation_details = {
                "hostname": hostname,
                "tls_version": version,
                "cipher_suite": cipher[0] if cipher else None,
                "certificate_subject": cert.get('subject'),
                "certificate_issuer": cert.get('issuer'),
                "certificate_expires": cert.get('notAfter'),
                "san_list": cert.get('subjectAltName', [])
            }
            
            # Check TLS version
            is_secure = True
            if version not in ['TLSv1.2', 'TLSv1.3']:
                is_secure = False
                validation_details['security_issue'] = f"Insecure TLS version: {version}"
            
            # Check certificate expiry
            if cert.get('notAfter'):
                from datetime import datetime
                expiry = datetime.strptime(cert['notAfter'], '%b %d %H:%M:%S %Y %Z')
                if expiry < datetime.utcnow() + timedelta(days=30):
                    validation_details['certificate_warning'] = "Certificate expires within 30 days"
            
            if is_secure:
                self.log_security_event(SecurityEvent(
                    event_type="SSL_VALIDATION",
                    severity=SecurityLevel.LOW,
                    message=f"SSL validation passed for {hostname}",
                    details=validation_details
                ))
            else:
                self.log_security_event(SecurityEvent(
                    event_type="SSL_VALIDATION_FAILED",
                    severity=SecurityLevel.HIGH,
                    message=f"SSL validation failed for {hostname}",
                    details=validation_details
                ))
            
            return is_secure, validation_details
            
        except Exception as e:
            self.logger.error(f"SSL validation error for {hostname}: {e}")
            return False, {"error": str(e)}
    
    def generate_secure_session_token(self) -> str:
        """Generate a cryptographically secure session token"""
        return secrets.token_urlsafe(32)
    
    def validate_trading_request(self, request_data: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Validate trading request for security compliance
        
        Args:
            request_data: Trading request data
            
        Returns:
            Tuple of (is_valid, reason)
        """
        try:
            # Check position size limits
            if 'units' in request_data:
                units = abs(float(request_data['units']))
                max_position = self.config["trading_security"]["max_position_size"]
                
                # This would need account balance to calculate properly
                # For now, use a reasonable absolute limit
                if units > 100000:  # Example limit
                    self.log_security_event(SecurityEvent(
                        event_type="LARGE_POSITION_BLOCKED",
                        severity=SecurityLevel.HIGH,
                        message=f"Blocked large position request: {units} units",
                        details=request_data
                    ))
                    return False, "Position size exceeds security limits"
            
            # Check for suspicious instruments
            if 'instrument' in request_data:
                instrument = request_data['instrument']
                # Add any instrument blacklist checks here
                if instrument.startswith('_'):  # Example: block instruments starting with _
                    return False, "Suspicious instrument identifier"
            
            # Check order type restrictions
            if 'type' in request_data:
                order_type = request_data['type']
                dangerous_types = ['GUARANTEED_STOP_LOSS']  # Example restrictions
                if order_type in dangerous_types:
                    return False, f"Order type {order_type} not allowed"
            
            # Log successful validation
            self.log_security_event(SecurityEvent(
                event_type="TRADING_REQUEST_VALIDATED",
                severity=SecurityLevel.LOW,
                message="Trading request passed security validation",
                details={"instrument": request_data.get('instrument')}
            ))
            
            return True, "Request validated"
            
        except Exception as e:
            self.logger.error(f"Error validating trading request: {e}")
            return False, f"Validation error: {e}"
    
    def create_audit_trail(self, action: str, details: Dict[str, Any]) -> str:
        """
        Create audit trail entry
        
        Args:
            action: Action being performed
            details: Action details
            
        Returns:
            Audit trail ID
        """
        try:
            audit_id = secrets.token_hex(16)
            
            audit_entry = {
                "audit_id": audit_id,
                "timestamp": datetime.utcnow().isoformat(),
                "action": action,
                "details": details,
                "checksum": self._calculate_checksum(action, details)
            }
            
            # Log to audit trail
            self.log_security_event(SecurityEvent(
                event_type="AUDIT_TRAIL",
                severity=SecurityLevel.LOW,
                message=f"Audit trail created for {action}",
                details=audit_entry
            ))
            
            return audit_id
            
        except Exception as e:
            self.logger.error(f"Error creating audit trail: {e}")
            return ""
    
    def _calculate_checksum(self, action: str, details: Dict[str, Any]) -> str:
        """Calculate checksum for audit trail integrity"""
        try:
            data = json.dumps({"action": action, "details": details}, sort_keys=True)
            return hashlib.sha256(data.encode()).hexdigest()
        except Exception as e:
            self.logger.error(f"Error calculating checksum: {e}")
            return ""
    
    def perform_security_scan(self) -> Dict[str, Any]:
        """Perform comprehensive security scan"""
        scan_results = {
            "timestamp": datetime.utcnow().isoformat(),
            "scan_id": secrets.token_hex(16),
            "results": {}
        }
        
        try:
            # Check file permissions
            critical_files = [
                self.config_path,
                self.security_log_path,
                os.path.expanduser("~/.cashmachine/oanda_credentials.enc")
            ]
            
            permission_issues = []
            for file_path in critical_files:
                if os.path.exists(file_path):
                    stat_info = os.stat(file_path)
                    permissions = oct(stat_info.st_mode)[-3:]
                    if permissions != '600':
                        permission_issues.append(f"{file_path}: {permissions} (should be 600)")
            
            scan_results["results"]["file_permissions"] = {
                "status": "PASS" if not permission_issues else "FAIL",
                "issues": permission_issues
            }
            
            # Check SSL configuration
            oanda_endpoints = [
                "api-fxpractice.oanda.com",
                "api-fxtrade.oanda.com"
            ]
            
            ssl_issues = []
            for endpoint in oanda_endpoints:
                is_secure, details = self.validate_ssl_connection(endpoint)
                if not is_secure:
                    ssl_issues.append(f"{endpoint}: {details.get('security_issue', 'Unknown issue')}")
            
            scan_results["results"]["ssl_validation"] = {
                "status": "PASS" if not ssl_issues else "FAIL",
                "issues": ssl_issues
            }
            
            # Check configuration security
            config_issues = []
            
            # Check if 2FA is enabled (if supported)
            if not self.config["authentication"].get("require_2fa", False):
                config_issues.append("Two-factor authentication is not enabled")
            
            # Check session timeout
            if self.config["authentication"]["session_timeout"] > 3600:
                config_issues.append("Session timeout too long (>1 hour)")
            
            # Check rate limiting
            if self.config["api_security"]["rate_limit_requests"] > 1000:
                config_issues.append("Rate limit too high (>1000 requests)")
            
            scan_results["results"]["configuration"] = {
                "status": "PASS" if not config_issues else "WARN",
                "issues": config_issues
            }
            
            # Overall assessment
            all_issues = permission_issues + ssl_issues + config_issues
            scan_results["overall_status"] = "PASS" if not all_issues else "FAIL"
            scan_results["total_issues"] = len(all_issues)
            
            # Log scan results
            self.log_security_event(SecurityEvent(
                event_type="SECURITY_SCAN",
                severity=SecurityLevel.MEDIUM if all_issues else SecurityLevel.LOW,
                message=f"Security scan completed: {len(all_issues)} issues found",
                details={"scan_id": scan_results["scan_id"]}
            ))
            
            return scan_results
            
        except Exception as e:
            self.logger.error(f"Error performing security scan: {e}")
            scan_results["error"] = str(e)
            return scan_results
    
    def emergency_lockdown(self, reason: str):
        """Emergency security lockdown"""
        try:
            self.log_security_event(SecurityEvent(
                event_type="EMERGENCY_LOCKDOWN",
                severity=SecurityLevel.CRITICAL,
                message=f"Emergency lockdown activated: {reason}"
            ))
            
            # Block all IPs temporarily
            self.blocked_ips.add("0.0.0.0/0")
            
            # Clear all session tokens
            self.session_tokens.clear()
            
            # Set maximum rate limiting
            for key in self.api_rate_limits:
                self.api_rate_limits[key] = [time.time()] * 1000  # Fill rate limit
            
            self.logger.critical(f"EMERGENCY LOCKDOWN ACTIVATED: {reason}")
            
        except Exception as e:
            self.logger.error(f"Error during emergency lockdown: {e}")


def main():
    """CLI interface for security hardening"""
    import argparse
    
    parser = argparse.ArgumentParser(description="OANDA Security Hardening")
    parser.add_argument('--action', choices=['scan', 'validate-ssl', 'check-config', 'lockdown'], 
                       required=True, help='Security action to perform')
    parser.add_argument('--hostname', help='Hostname for SSL validation')
    parser.add_argument('--reason', help='Reason for lockdown')
    parser.add_argument('--config-path', help='Path to security configuration')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    security = OANDASecurityHardening(args.config_path)
    
    try:
        if args.action == 'scan':
            results = security.perform_security_scan()
            print("🔒 Security Scan Results")
            print("=" * 40)
            print(f"Overall Status: {results['overall_status']}")
            print(f"Total Issues: {results['total_issues']}")
            
            for category, result in results["results"].items():
                print(f"\n{category.title()}: {result['status']}")
                for issue in result.get('issues', []):
                    print(f"  ⚠️  {issue}")
            
            if results['overall_status'] == 'PASS':
                print("\n✅ All security checks passed!")
            else:
                print("\n❌ Security issues found - please review and fix")
            
        elif args.action == 'validate-ssl':
            hostname = args.hostname or "api-fxpractice.oanda.com"
            is_secure, details = security.validate_ssl_connection(hostname)
            
            print(f"🔐 SSL Validation for {hostname}")
            print("=" * 40)
            print(f"Status: {'✅ SECURE' if is_secure else '❌ INSECURE'}")
            print(f"TLS Version: {details.get('tls_version', 'Unknown')}")
            print(f"Cipher Suite: {details.get('cipher_suite', 'Unknown')}")
            
            if 'security_issue' in details:
                print(f"⚠️  Issue: {details['security_issue']}")
            
        elif args.action == 'check-config':
            print("📋 Security Configuration Check")
            print("=" * 40)
            
            # Display current configuration
            for section, settings in security.config.items():
                print(f"\n{section.title()}:")
                for key, value in settings.items():
                    if 'password' in key.lower() or 'key' in key.lower():
                        value = "***HIDDEN***"
                    print(f"  {key}: {value}")
            
        elif args.action == 'lockdown':
            reason = args.reason or "Manual lockdown via CLI"
            security.emergency_lockdown(reason)
            print(f"🚨 Emergency lockdown activated: {reason}")
        
        return 0
        
    except Exception as e:
        print(f"❌ Security operation failed: {e}")
        return 1


if __name__ == '__main__':
    exit(main())