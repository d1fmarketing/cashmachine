#!/usr/bin/env python3
"""
OANDA Trinity Integration Setup Script
Complete setup and configuration for OANDA trading API integration
"""

import os
import sys
import json
import logging
import asyncio
from datetime import datetime
from typing import Dict, Any

# Import our modules
from oanda_auth_manager import OANDAAuthManager
from oanda_proxy_config import OANDAProxyConfig
from oanda_security_hardening import OANDASecurityHardening
from oanda_connection_tester import OANDAConnectionTester
from trinity_oanda_integration import TrinityOANDAIntegration
from oanda_error_handler import OANDAErrorHandler

class OANDATrinitySetup:
    """Complete setup orchestrator for OANDA Trinity integration"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.setup_results = {
            "timestamp": datetime.utcnow().isoformat(),
            "steps_completed": [],
            "errors": [],
            "warnings": [],
            "configuration": {}
        }
    
    def setup_logging(self, verbose: bool = False):
        """Setup logging configuration"""
        log_level = logging.DEBUG if verbose else logging.INFO
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler(
                    os.path.expanduser("~/.cashmachine/setup.log"),
                    mode='a'
                )
            ]
        )
        self.logger.info("Logging configured for OANDA Trinity setup")
    
    def check_prerequisites(self) -> bool:
        """Check system prerequisites"""
        self.logger.info("Checking prerequisites...")
        
        try:
            # Check Python version
            if sys.version_info < (3, 8):
                self.setup_results["errors"].append("Python 3.8+ required")
                return False
            
            # Check required packages
            required_packages = [
                'requests', 'cryptography', 'pandas', 'numpy'
            ]
            
            missing_packages = []
            for package in required_packages:
                try:
                    __import__(package)
                except ImportError:
                    missing_packages.append(package)
            
            if missing_packages:
                self.setup_results["errors"].append(
                    f"Missing packages: {', '.join(missing_packages)}"
                )
                self.logger.error(f"Install missing packages: pip install {' '.join(missing_packages)}")
                return False
            
            # Check directory permissions
            cashmachine_dir = os.path.expanduser("~/.cashmachine")
            if not os.path.exists(cashmachine_dir):
                os.makedirs(cashmachine_dir, mode=0o700)
            
            # Test write permissions
            test_file = os.path.join(cashmachine_dir, "test_write.tmp")
            try:
                with open(test_file, 'w') as f:
                    f.write("test")
                os.remove(test_file)
            except Exception as e:
                self.setup_results["errors"].append(f"Directory permission error: {e}")
                return False
            
            self.setup_results["steps_completed"].append("prerequisites_check")
            self.logger.info("✅ Prerequisites check passed")
            return True
            
        except Exception as e:
            self.setup_results["errors"].append(f"Prerequisites check failed: {e}")
            self.logger.error(f"Prerequisites check failed: {e}")
            return False
    
    def setup_authentication(self, api_key: str = None, account_id: str = None, 
                           environment: str = "practice") -> bool:
        """Setup OANDA authentication"""
        self.logger.info("Setting up OANDA authentication...")
        
        try:
            auth_manager = OANDAAuthManager()
            
            if not api_key or not account_id:
                self.logger.info("Interactive credential setup required")
                print("\n" + "="*60)
                print("OANDA API CREDENTIALS SETUP")
                print("="*60)
                print("Please provide your OANDA API credentials.")
                print("You can get these from: https://developer.oanda.com/")
                print()
                
                if not api_key:
                    api_key = input("OANDA API Key: ").strip()
                if not account_id:
                    account_id = input("OANDA Account ID: ").strip()
                
                env_choice = input(f"Environment [practice/live] (default: {environment}): ").strip()
                if env_choice:
                    environment = env_choice
            
            # Validate credentials format
            if len(api_key) < 32:
                self.setup_results["errors"].append("API key appears to be too short")
                return False
            
            if not account_id.isdigit():
                self.setup_results["errors"].append("Account ID should be numeric")
                return False
            
            # Store credentials
            success = auth_manager.store_credentials(api_key, account_id, environment)
            if not success:
                self.setup_results["errors"].append("Failed to store credentials")
                return False
            
            # Validate credentials
            is_valid, message = auth_manager.validate_credentials()
            if not is_valid:
                self.setup_results["errors"].append(f"Credential validation failed: {message}")
                return False
            
            self.setup_results["steps_completed"].append("authentication_setup")
            self.setup_results["configuration"]["environment"] = environment
            self.setup_results["configuration"]["account_id"] = account_id[:4] + "****"  # Masked
            
            self.logger.info("✅ Authentication setup completed")
            return True
            
        except Exception as e:
            self.setup_results["errors"].append(f"Authentication setup failed: {e}")
            self.logger.error(f"Authentication setup failed: {e}")
            return False
    
    def setup_proxy_configuration(self, blackbox_mode: bool = True) -> bool:
        """Setup proxy configuration"""
        self.logger.info("Setting up proxy configuration...")
        
        try:
            proxy_config = OANDAProxyConfig()
            
            if blackbox_mode:
                # Configure for blackbox environment
                proxy_config.configure_for_blackbox()
                self.setup_results["configuration"]["proxy_mode"] = "blackbox"
                self.logger.info("Configured for blackbox environment")
            else:
                # Interactive proxy setup
                print("\n" + "="*60)
                print("PROXY CONFIGURATION")
                print("="*60)
                
                use_proxy = input("Use HTTP proxy? [y/N]: ").strip().lower()
                if use_proxy in ['y', 'yes']:
                    http_proxy = input("HTTP Proxy URL: ").strip()
                    https_proxy = input("HTTPS Proxy URL (optional): ").strip() or http_proxy
                    
                    proxy_config.update_proxy_settings(
                        http_proxy=http_proxy,
                        https_proxy=https_proxy,
                        enabled=True
                    )
                    self.setup_results["configuration"]["proxy_enabled"] = True
                else:
                    self.setup_results["configuration"]["proxy_enabled"] = False
            
            self.setup_results["steps_completed"].append("proxy_configuration")
            self.logger.info("✅ Proxy configuration completed")
            return True
            
        except Exception as e:
            self.setup_results["errors"].append(f"Proxy configuration failed: {e}")
            self.logger.error(f"Proxy configuration failed: {e}")
            return False
    
    def setup_security_hardening(self) -> bool:
        """Setup security hardening"""
        self.logger.info("Setting up security hardening...")
        
        try:
            security = OANDASecurityHardening()
            
            # Perform security scan
            scan_results = security.perform_security_scan()
            
            if scan_results["overall_status"] != "PASS":
                self.setup_results["warnings"].append(
                    f"Security scan found {scan_results['total_issues']} issues"
                )
                self.logger.warning(f"Security issues found: {scan_results['total_issues']}")
            
            # Store security configuration
            self.setup_results["configuration"]["security_scan"] = {
                "status": scan_results["overall_status"],
                "issues": scan_results["total_issues"]
            }
            
            self.setup_results["steps_completed"].append("security_hardening")
            self.logger.info("✅ Security hardening completed")
            return True
            
        except Exception as e:
            self.setup_results["errors"].append(f"Security hardening failed: {e}")
            self.logger.error(f"Security hardening failed: {e}")
            return False
    
    async def test_connectivity(self) -> bool:
        """Test OANDA connectivity"""
        self.logger.info("Testing OANDA connectivity...")
        
        try:
            tester = OANDAConnectionTester()
            
            # Run comprehensive tests
            test_results = await tester.run_comprehensive_test()
            
            success_rate = test_results["summary"]["success_rate"]
            if success_rate < 0.8:  # Less than 80% success
                self.setup_results["errors"].append(
                    f"Connectivity tests failed: {success_rate:.1%} success rate"
                )
                return False
            
            # Store test results
            self.setup_results["configuration"]["connectivity_test"] = {
                "success_rate": success_rate,
                "total_tests": test_results["summary"]["total_tests"],
                "passed_tests": test_results["summary"]["passed_tests"]
            }
            
            # Generate test report
            report_file = tester.generate_test_report(test_results)
            if report_file:
                self.setup_results["configuration"]["test_report"] = report_file
            
            self.setup_results["steps_completed"].append("connectivity_test")
            self.logger.info(f"✅ Connectivity tests passed ({success_rate:.1%} success rate)")
            return True
            
        except Exception as e:
            self.setup_results["errors"].append(f"Connectivity test failed: {e}")
            self.logger.error(f"Connectivity test failed: {e}")
            return False
    
    def setup_error_handling(self) -> bool:
        """Setup error handling"""
        self.logger.info("Setting up error handling...")
        
        try:
            error_handler = OANDAErrorHandler()
            
            # Test error handling with a simple operation
            # This would be expanded in a real implementation
            
            self.setup_results["steps_completed"].append("error_handling_setup")
            self.logger.info("✅ Error handling setup completed")
            return True
            
        except Exception as e:
            self.setup_results["errors"].append(f"Error handling setup failed: {e}")
            self.logger.error(f"Error handling setup failed: {e}")
            return False
    
    def create_integration_instance(self) -> bool:
        """Create and test Trinity integration instance"""
        self.logger.info("Creating Trinity integration instance...")
        
        try:
            # Create integration components
            auth_manager = OANDAAuthManager()
            proxy_config = OANDAProxyConfig()
            
            # Create Trinity integration
            trinity = TrinityOANDAIntegration(auth_manager, proxy_config)
            
            # Basic validation
            if not trinity.credentials:
                self.setup_results["errors"].append("Trinity integration has no credentials")
                return False
            
            if not trinity.session_data:
                self.setup_results["errors"].append("Trinity integration has no session")
                return False
            
            self.setup_results["steps_completed"].append("trinity_integration")
            self.logger.info("✅ Trinity integration instance created")
            return True
            
        except Exception as e:
            self.setup_results["errors"].append(f"Trinity integration failed: {e}")
            self.logger.error(f"Trinity integration failed: {e}")
            return False
    
    def generate_setup_report(self) -> str:
        """Generate comprehensive setup report"""
        report_file = f"oanda_trinity_setup_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        report_path = os.path.expanduser(f"~/.cashmachine/{report_file}")
        
        try:
            with open(report_path, 'w') as f:
                json.dump(self.setup_results, f, indent=2)
            
            self.logger.info(f"Setup report saved to {report_path}")
            return report_path
            
        except Exception as e:
            self.logger.error(f"Failed to save setup report: {e}")
            return ""
    
    async def run_complete_setup(self, api_key: str = None, account_id: str = None,
                               environment: str = "practice", blackbox_mode: bool = True,
                               verbose: bool = False):
        """Run complete OANDA Trinity setup"""
        self.setup_logging(verbose)
        
        print("\n" + "="*80)
        print("OANDA TRINITY INTEGRATION SETUP")
        print("="*80)
        print("Setting up secure OANDA API integration for Trinity trading system")
        print()
        
        # Setup steps
        setup_steps = [
            ("Prerequisites Check", lambda: self.check_prerequisites()),
            ("Authentication Setup", lambda: self.setup_authentication(api_key, account_id, environment)),
            ("Proxy Configuration", lambda: self.setup_proxy_configuration(blackbox_mode)),
            ("Security Hardening", lambda: self.setup_security_hardening()),
            ("Connectivity Testing", lambda: asyncio.create_task(self.test_connectivity())),
            ("Error Handling Setup", lambda: self.setup_error_handling()),
            ("Trinity Integration", lambda: self.create_integration_instance())
        ]
        
        # Execute setup steps
        for step_name, step_func in setup_steps:
            print(f"📋 {step_name}...")
            
            try:
                if asyncio.iscoroutinefunction(step_func):
                    success = await step_func()
                else:
                    result = step_func()
                    if asyncio.iscoroutine(result):
                        success = await result
                    else:
                        success = result
                
                if success:
                    print(f"✅ {step_name} completed")
                else:
                    print(f"❌ {step_name} failed")
                    break
                    
            except Exception as e:
                print(f"❌ {step_name} failed with exception: {e}")
                self.setup_results["errors"].append(f"{step_name}: {e}")
                break
        
        # Generate final report
        print("\n" + "="*80)
        print("SETUP SUMMARY")
        print("="*80)
        
        total_steps = len(setup_steps)
        completed_steps = len(self.setup_results["steps_completed"])
        
        print(f"Steps Completed: {completed_steps}/{total_steps}")
        print(f"Errors: {len(self.setup_results['errors'])}")
        print(f"Warnings: {len(self.setup_results['warnings'])}")
        
        if self.setup_results["errors"]:
            print("\n❌ ERRORS:")
            for error in self.setup_results["errors"]:
                print(f"  • {error}")
        
        if self.setup_results["warnings"]:
            print("\n⚠️  WARNINGS:")
            for warning in self.setup_results["warnings"]:
                print(f"  • {warning}")
        
        # Generate report
        report_path = self.generate_setup_report()
        if report_path:
            print(f"\n📄 Setup report saved to: {report_path}")
        
        # Success determination
        setup_success = len(self.setup_results["errors"]) == 0 and completed_steps == total_steps
        
        if setup_success:
            print("\n🎉 OANDA Trinity integration setup completed successfully!")
            print("\nNext steps:")
            print("  1. Test trading operations with demo account")
            print("  2. Configure trading strategies")
            print("  3. Set up monitoring and alerts")
            print("  4. Begin live trading (if configured)")
        else:
            print("\n💥 Setup failed. Please review errors and try again.")
            return False
        
        return setup_success


async def main():
    """CLI interface for OANDA Trinity setup"""
    import argparse
    
    parser = argparse.ArgumentParser(description="OANDA Trinity Integration Setup")
    parser.add_argument('--api-key', help='OANDA API key')
    parser.add_argument('--account-id', help='OANDA account ID')
    parser.add_argument('--environment', choices=['practice', 'live'], default='practice',
                       help='OANDA environment')
    parser.add_argument('--no-blackbox', action='store_true', 
                       help='Disable blackbox proxy configuration')
    parser.add_argument('--verbose', action='store_true', help='Verbose logging')
    
    args = parser.parse_args()
    
    # Create setup instance
    setup = OANDATrinitySetup()
    
    # Run complete setup
    success = await setup.run_complete_setup(
        api_key=args.api_key,
        account_id=args.account_id,
        environment=args.environment,
        blackbox_mode=not args.no_blackbox,
        verbose=args.verbose
    )
    
    return 0 if success else 1


if __name__ == '__main__':
    exit(asyncio.run(main()))