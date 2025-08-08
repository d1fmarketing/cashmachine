#!/usr/bin/env python3
"""
OANDA Authentication Manager for Trinity System
Secure credential storage and API authentication for CashMachine trading platform
"""

import os
import json
import hashlib
import base64
from typing import Dict, Optional, Tuple
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import logging
from datetime import datetime, timedelta

class OANDAAuthManager:
    """Secure authentication manager for OANDA trading API"""
    
    def __init__(self, credentials_path: str = None):
        """
        Initialize the OANDA authentication manager
        
        Args:
            credentials_path: Path to encrypted credentials file
        """
        self.logger = logging.getLogger(__name__)
        self.credentials_path = credentials_path or os.path.expanduser("~/.cashmachine/oanda_credentials.enc")
        self.session_file = os.path.expanduser("~/.cashmachine/oanda_session.json")
        self._ensure_credentials_dir()
        
        # OANDA API endpoints
        self.endpoints = {
            'practice': 'https://api-fxpractice.oanda.com',
            'live': 'https://api-fxtrade.oanda.com',
            'stream_practice': 'https://stream-fxpractice.oanda.com',
            'stream_live': 'https://stream-fxtrade.oanda.com'
        }
        
    def _ensure_credentials_dir(self):
        """Ensure credentials directory exists with proper permissions"""
        cred_dir = os.path.dirname(self.credentials_path)
        if not os.path.exists(cred_dir):
            os.makedirs(cred_dir, mode=0o700)
            self.logger.info(f"Created secure credentials directory: {cred_dir}")
    
    def _derive_key(self, password: str, salt: bytes) -> bytes:
        """Derive encryption key from password using PBKDF2"""
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(password.encode()))
        return key
    
    def store_credentials(self, api_key: str, account_id: str, environment: str = 'practice', password: str = None):
        """
        Securely store OANDA API credentials
        
        Args:
            api_key: OANDA API key
            account_id: OANDA account ID
            environment: 'practice' or 'live'
            password: Encryption password (will prompt if not provided)
        """
        if not password:
            import getpass
            password = getpass.getpass("Enter encryption password for OANDA credentials: ")
        
        # Generate salt and derive key
        salt = os.urandom(16)
        key = self._derive_key(password, salt)
        fernet = Fernet(key)
        
        # Prepare credentials data
        credentials_data = {
            'api_key': api_key,
            'account_id': account_id,
            'environment': environment,
            'created_at': datetime.utcnow().isoformat(),
            'last_used': None
        }
        
        # Encrypt credentials
        encrypted_data = fernet.encrypt(json.dumps(credentials_data).encode())
        
        # Store with salt
        with open(self.credentials_path, 'wb') as f:
            f.write(salt + encrypted_data)
        
        # Set restrictive permissions
        os.chmod(self.credentials_path, 0o600)
        
        self.logger.info(f"OANDA credentials stored securely for {environment} environment")
        return True
    
    def load_credentials(self, password: str = None) -> Dict[str, str]:
        """
        Load and decrypt OANDA API credentials
        
        Args:
            password: Decryption password (will prompt if not provided)
            
        Returns:
            Dictionary containing decrypted credentials
        """
        if not os.path.exists(self.credentials_path):
            raise FileNotFoundError("OANDA credentials file not found. Run store_credentials first.")
        
        if not password:
            import getpass
            password = getpass.getpass("Enter decryption password for OANDA credentials: ")
        
        try:
            # Read encrypted file
            with open(self.credentials_path, 'rb') as f:
                file_data = f.read()
            
            # Extract salt and encrypted data
            salt = file_data[:16]
            encrypted_data = file_data[16:]
            
            # Derive key and decrypt
            key = self._derive_key(password, salt)
            fernet = Fernet(key)
            decrypted_data = fernet.decrypt(encrypted_data).decode()
            
            # Parse credentials
            credentials = json.loads(decrypted_data)
            
            # Update last used timestamp
            credentials['last_used'] = datetime.utcnow().isoformat()
            self._update_last_used(credentials, password)
            
            return credentials
            
        except Exception as e:
            self.logger.error(f"Failed to decrypt OANDA credentials: {e}")
            raise ValueError("Invalid password or corrupted credentials file")
    
    def _update_last_used(self, credentials: Dict, password: str):
        """Update the last used timestamp in encrypted file"""
        try:
            salt = os.urandom(16)
            key = self._derive_key(password, salt)
            fernet = Fernet(key)
            
            encrypted_data = fernet.encrypt(json.dumps(credentials).encode())
            
            with open(self.credentials_path, 'wb') as f:
                f.write(salt + encrypted_data)
                
        except Exception as e:
            self.logger.warning(f"Failed to update last used timestamp: {e}")
    
    def get_auth_headers(self, credentials: Dict = None, password: str = None) -> Dict[str, str]:
        """
        Get authentication headers for OANDA API requests
        
        Args:
            credentials: Pre-loaded credentials (optional)
            password: Password for credential decryption (optional)
            
        Returns:
            Dictionary containing authentication headers
        """
        if not credentials:
            credentials = self.load_credentials(password)
        
        headers = {
            'Authorization': f"Bearer {credentials['api_key']}",
            'Content-Type': 'application/json',
            'Accept-Datetime-Format': 'RFC3339',
            'User-Agent': 'CashMachine-Trinity-v1.0'
        }
        
        return headers
    
    def get_api_endpoint(self, environment: str = None, stream: bool = False) -> str:
        """
        Get appropriate OANDA API endpoint
        
        Args:
            environment: 'practice' or 'live' (will use stored if not provided)
            stream: Whether to use streaming endpoint
            
        Returns:
            API endpoint URL
        """
        if not environment:
            try:
                credentials = self.load_credentials()
                environment = credentials['environment']
            except:
                environment = 'practice'  # Default to practice
        
        if stream:
            return self.endpoints.get(f'stream_{environment}', self.endpoints['stream_practice'])
        else:
            return self.endpoints.get(environment, self.endpoints['practice'])
    
    def validate_credentials(self, credentials: Dict = None, password: str = None) -> Tuple[bool, str]:
        """
        Validate OANDA API credentials by making a test request
        
        Args:
            credentials: Pre-loaded credentials (optional)
            password: Password for credential decryption (optional)
            
        Returns:
            Tuple of (is_valid, message)
        """
        try:
            import requests
            
            if not credentials:
                credentials = self.load_credentials(password)
            
            headers = self.get_auth_headers(credentials)
            endpoint = self.get_api_endpoint(credentials['environment'])
            
            # Test with accounts endpoint
            response = requests.get(
                f"{endpoint}/v3/accounts",
                headers=headers,
                timeout=10
            )
            
            if response.status_code == 200:
                accounts_data = response.json()
                if 'accounts' in accounts_data and accounts_data['accounts']:
                    return True, "Credentials valid and authenticated successfully"
                else:
                    return False, "No accounts found for the provided credentials"
            elif response.status_code == 401:
                return False, "Invalid API key or unauthorized access"
            elif response.status_code == 403:
                return False, "Forbidden - check account permissions"
            else:
                return False, f"API request failed with status {response.status_code}"
                
        except requests.exceptions.RequestException as e:
            return False, f"Network error during validation: {e}"
        except Exception as e:
            return False, f"Validation error: {e}"
    
    def get_account_info(self, credentials: Dict = None, password: str = None) -> Dict:
        """
        Get account information from OANDA
        
        Args:
            credentials: Pre-loaded credentials (optional)
            password: Password for credential decryption (optional)
            
        Returns:
            Account information dictionary
        """
        try:
            import requests
            
            if not credentials:
                credentials = self.load_credentials(password)
            
            headers = self.get_auth_headers(credentials)
            endpoint = self.get_api_endpoint(credentials['environment'])
            account_id = credentials['account_id']
            
            # Get account details
            response = requests.get(
                f"{endpoint}/v3/accounts/{account_id}",
                headers=headers,
                timeout=10
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                raise Exception(f"Failed to get account info: {response.status_code}")
                
        except Exception as e:
            self.logger.error(f"Error getting account info: {e}")
            raise
    
    def create_session(self, credentials: Dict = None, password: str = None) -> Dict:
        """
        Create and store a session for OANDA API
        
        Args:
            credentials: Pre-loaded credentials (optional)
            password: Password for credential decryption (optional)
            
        Returns:
            Session information
        """
        if not credentials:
            credentials = self.load_credentials(password)
        
        # Validate credentials first
        is_valid, message = self.validate_credentials(credentials)
        if not is_valid:
            raise ValueError(f"Cannot create session: {message}")
        
        # Create session data
        session_data = {
            'environment': credentials['environment'],
            'account_id': credentials['account_id'],
            'created_at': datetime.utcnow().isoformat(),
            'expires_at': (datetime.utcnow() + timedelta(hours=24)).isoformat(),
            'endpoint': self.get_api_endpoint(credentials['environment']),
            'stream_endpoint': self.get_api_endpoint(credentials['environment'], stream=True)
        }
        
        # Store session
        with open(self.session_file, 'w') as f:
            json.dump(session_data, f, indent=2)
        
        os.chmod(self.session_file, 0o600)
        
        self.logger.info(f"OANDA session created for {credentials['environment']} environment")
        return session_data
    
    def get_active_session(self) -> Optional[Dict]:
        """
        Get active OANDA session if it exists and is valid
        
        Returns:
            Session data if valid, None otherwise
        """
        try:
            if not os.path.exists(self.session_file):
                return None
            
            with open(self.session_file, 'r') as f:
                session_data = json.load(f)
            
            # Check if session is expired
            expires_at = datetime.fromisoformat(session_data['expires_at'])
            if datetime.utcnow() > expires_at:
                self.logger.info("OANDA session expired")
                os.remove(self.session_file)
                return None
            
            return session_data
            
        except Exception as e:
            self.logger.warning(f"Error checking active session: {e}")
            return None
    
    def cleanup_credentials(self):
        """Remove stored credentials and session files"""
        try:
            if os.path.exists(self.credentials_path):
                os.remove(self.credentials_path)
                self.logger.info("OANDA credentials removed")
            
            if os.path.exists(self.session_file):
                os.remove(self.session_file)
                self.logger.info("OANDA session removed")
                
            return True
            
        except Exception as e:
            self.logger.error(f"Error cleaning up credentials: {e}")
            return False


def main():
    """CLI interface for OANDA authentication manager"""
    import argparse
    
    parser = argparse.ArgumentParser(description="OANDA Authentication Manager")
    parser.add_argument('--action', choices=['store', 'validate', 'info', 'session', 'cleanup'], 
                       required=True, help='Action to perform')
    parser.add_argument('--api-key', help='OANDA API key (for store action)')
    parser.add_argument('--account-id', help='OANDA account ID (for store action)')
    parser.add_argument('--environment', choices=['practice', 'live'], default='practice',
                       help='OANDA environment')
    parser.add_argument('--credentials-path', help='Path to credentials file')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    auth_manager = OANDAAuthManager(args.credentials_path)
    
    try:
        if args.action == 'store':
            if not args.api_key or not args.account_id:
                print("Error: --api-key and --account-id are required for store action")
                return 1
            
            success = auth_manager.store_credentials(
                args.api_key, 
                args.account_id, 
                args.environment
            )
            if success:
                print(f"✅ OANDA credentials stored successfully for {args.environment} environment")
            else:
                print("❌ Failed to store credentials")
                return 1
                
        elif args.action == 'validate':
            is_valid, message = auth_manager.validate_credentials()
            if is_valid:
                print(f"✅ {message}")
            else:
                print(f"❌ {message}")
                return 1
                
        elif args.action == 'info':
            account_info = auth_manager.get_account_info()
            print("📊 Account Information:")
            print(json.dumps(account_info, indent=2))
            
        elif args.action == 'session':
            session_data = auth_manager.create_session()
            print("🔒 Session created:")
            print(json.dumps(session_data, indent=2))
            
        elif args.action == 'cleanup':
            success = auth_manager.cleanup_credentials()
            if success:
                print("🧹 Credentials cleaned up successfully")
            else:
                print("❌ Failed to cleanup credentials")
                return 1
        
        return 0
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1


if __name__ == '__main__':
    exit(main())