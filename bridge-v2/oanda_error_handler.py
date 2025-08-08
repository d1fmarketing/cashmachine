#!/usr/bin/env python3
"""
OANDA Error Handler and Retry Mechanisms for Trinity System
Comprehensive error handling, circuit breakers, and retry strategies
"""

import asyncio
import json
import logging
import time
import random
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
import functools
import traceback

class ErrorType(Enum):
    """Types of errors that can occur"""
    NETWORK_ERROR = "network_error"
    API_ERROR = "api_error"
    AUTHENTICATION_ERROR = "authentication_error"
    RATE_LIMIT_ERROR = "rate_limit_error"
    VALIDATION_ERROR = "validation_error"
    TRADING_ERROR = "trading_error"
    TIMEOUT_ERROR = "timeout_error"
    UNKNOWN_ERROR = "unknown_error"

class ErrorSeverity(Enum):
    """Error severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class ErrorContext:
    """Context information for error handling"""
    operation: str
    endpoint: Optional[str] = None
    request_data: Optional[Dict[str, Any]] = None
    user_agent: Optional[str] = None
    timestamp: Optional[str] = None
    correlation_id: Optional[str] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow().isoformat()
        if self.correlation_id is None:
            import uuid
            self.correlation_id = str(uuid.uuid4())

@dataclass
class ErrorRecord:
    """Record of an error occurrence"""
    error_type: ErrorType
    severity: ErrorSeverity
    message: str
    context: ErrorContext
    exception: Optional[Exception] = None
    stack_trace: Optional[str] = None
    retry_count: int = 0
    resolved: bool = False
    resolution_time: Optional[str] = None
    
    def __post_init__(self):
        if self.exception and not self.stack_trace:
            self.stack_trace = traceback.format_exc()

@dataclass
class RetryConfig:
    """Configuration for retry mechanisms"""
    max_attempts: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    exponential_backoff: bool = True
    jitter: bool = True
    retry_on: List[ErrorType] = field(default_factory=lambda: [
        ErrorType.NETWORK_ERROR,
        ErrorType.TIMEOUT_ERROR,
        ErrorType.RATE_LIMIT_ERROR
    ])

class CircuitBreakerState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"

@dataclass
class CircuitBreaker:
    """Circuit breaker for fault tolerance"""
    failure_threshold: int = 5
    recovery_timeout: float = 60.0
    half_open_max_calls: int = 3
    
    # State tracking
    state: CircuitBreakerState = CircuitBreakerState.CLOSED
    failure_count: int = 0
    last_failure_time: float = 0.0
    half_open_calls: int = 0
    
    def should_allow_request(self) -> bool:
        """Check if request should be allowed"""
        if self.state == CircuitBreakerState.CLOSED:
            return True
        elif self.state == CircuitBreakerState.OPEN:
            if time.time() - self.last_failure_time > self.recovery_timeout:
                self.state = CircuitBreakerState.HALF_OPEN
                self.half_open_calls = 0
                return True
            return False
        else:  # HALF_OPEN
            return self.half_open_calls < self.half_open_max_calls
    
    def record_success(self):
        """Record successful operation"""
        if self.state == CircuitBreakerState.HALF_OPEN:
            self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
    
    def record_failure(self):
        """Record failed operation"""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.state == CircuitBreakerState.HALF_OPEN:
            self.state = CircuitBreakerState.OPEN
        elif self.failure_count >= self.failure_threshold:
            self.state = CircuitBreakerState.OPEN

class OANDAErrorHandler:
    """Comprehensive error handling for OANDA API operations"""
    
    def __init__(self, config_path: str = None):
        """
        Initialize error handler
        
        Args:
            config_path: Path to error handling configuration
        """
        self.logger = logging.getLogger(__name__)
        self.config_path = config_path
        
        # Error tracking
        self.error_history: List[ErrorRecord] = []
        self.error_stats = {
            "total_errors": 0,
            "errors_by_type": {},
            "errors_by_endpoint": {},
            "avg_resolution_time": 0.0
        }
        
        # Circuit breakers by endpoint
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        
        # Retry configurations
        self.default_retry_config = RetryConfig()
        self.endpoint_retry_configs: Dict[str, RetryConfig] = {}
        
        # Load configuration
        self._load_configuration()
    
    def _load_configuration(self):
        """Load error handling configuration"""
        # This would load from config file in a real implementation
        # For now, use sensible defaults
        
        # Endpoint-specific retry configs
        self.endpoint_retry_configs = {
            "/v3/accounts": RetryConfig(max_attempts=5, base_delay=0.5),
            "/v3/orders": RetryConfig(max_attempts=2, base_delay=1.0),  # Less aggressive for trading
            "/v3/pricing": RetryConfig(max_attempts=3, base_delay=0.1),
            "/v3/positions": RetryConfig(max_attempts=3, base_delay=0.5),
        }
        
        self.logger.info("Error handling configuration loaded")
    
    def classify_error(self, exception: Exception, context: ErrorContext) -> Tuple[ErrorType, ErrorSeverity]:
        """
        Classify an error by type and severity
        
        Args:
            exception: The exception that occurred
            context: Error context
            
        Returns:
            Tuple of (error_type, severity)
        """
        error_msg = str(exception).lower()
        
        # Network-related errors
        if any(keyword in error_msg for keyword in [
            'connection', 'timeout', 'network', 'unreachable', 'dns'
        ]):
            return ErrorType.NETWORK_ERROR, ErrorSeverity.MEDIUM
        
        # Authentication errors
        if any(keyword in error_msg for keyword in [
            'unauthorized', 'forbidden', 'authentication', 'invalid token'
        ]):
            return ErrorType.AUTHENTICATION_ERROR, ErrorSeverity.HIGH
        
        # Rate limiting
        if any(keyword in error_msg for keyword in [
            'rate limit', 'too many requests', '429'
        ]):
            return ErrorType.RATE_LIMIT_ERROR, ErrorSeverity.LOW
        
        # API-specific errors
        if any(keyword in error_msg for keyword in [
            'bad request', '400', 'invalid parameter'
        ]):
            return ErrorType.VALIDATION_ERROR, ErrorSeverity.MEDIUM
        
        # Trading-specific errors
        if any(keyword in error_msg for keyword in [
            'insufficient margin', 'position', 'order', 'trading'
        ]):
            return ErrorType.TRADING_ERROR, ErrorSeverity.HIGH
        
        # Default classification
        return ErrorType.UNKNOWN_ERROR, ErrorSeverity.MEDIUM
    
    def record_error(self, exception: Exception, context: ErrorContext) -> ErrorRecord:
        """
        Record an error for tracking and analysis
        
        Args:
            exception: The exception that occurred
            context: Error context
            
        Returns:
            Error record
        """
        error_type, severity = self.classify_error(exception, context)
        
        error_record = ErrorRecord(
            error_type=error_type,
            severity=severity,
            message=str(exception),
            context=context,
            exception=exception
        )
        
        # Add to history
        self.error_history.append(error_record)
        
        # Update statistics
        self.error_stats["total_errors"] += 1
        
        type_key = error_type.value
        if type_key not in self.error_stats["errors_by_type"]:
            self.error_stats["errors_by_type"][type_key] = 0
        self.error_stats["errors_by_type"][type_key] += 1
        
        if context.endpoint:
            if context.endpoint not in self.error_stats["errors_by_endpoint"]:
                self.error_stats["errors_by_endpoint"][context.endpoint] = 0
            self.error_stats["errors_by_endpoint"][context.endpoint] += 1
        
        # Log error
        self.logger.error(
            f"Error recorded: {error_type.value} - {severity.value} - {exception} "
            f"(Operation: {context.operation}, Correlation: {context.correlation_id})"
        )
        
        return error_record
    
    def get_circuit_breaker(self, endpoint: str) -> CircuitBreaker:
        """Get or create circuit breaker for endpoint"""
        if endpoint not in self.circuit_breakers:
            self.circuit_breakers[endpoint] = CircuitBreaker()
        return self.circuit_breakers[endpoint]
    
    def get_retry_config(self, endpoint: str = None) -> RetryConfig:
        """Get retry configuration for endpoint"""
        if endpoint and endpoint in self.endpoint_retry_configs:
            return self.endpoint_retry_configs[endpoint]
        return self.default_retry_config
    
    def calculate_retry_delay(self, attempt: int, config: RetryConfig) -> float:
        """
        Calculate delay before next retry attempt
        
        Args:
            attempt: Current attempt number (1-based)
            config: Retry configuration
            
        Returns:
            Delay in seconds
        """
        if config.exponential_backoff:
            delay = config.base_delay * (2 ** (attempt - 1))
        else:
            delay = config.base_delay
        
        # Apply maximum delay limit
        delay = min(delay, config.max_delay)
        
        # Add jitter to prevent thundering herd
        if config.jitter:
            jitter_range = delay * 0.1  # 10% jitter
            delay += random.uniform(-jitter_range, jitter_range)
        
        return max(0, delay)
    
    def should_retry(self, error_record: ErrorRecord, attempt: int, config: RetryConfig) -> bool:
        """
        Determine if operation should be retried
        
        Args:
            error_record: The error that occurred
            attempt: Current attempt number
            config: Retry configuration
            
        Returns:
            True if should retry, False otherwise
        """
        # Check max attempts
        if attempt >= config.max_attempts:
            return False
        
        # Check if error type is retryable
        if error_record.error_type not in config.retry_on:
            return False
        
        # Don't retry authentication errors
        if error_record.error_type == ErrorType.AUTHENTICATION_ERROR:
            return False
        
        # Don't retry validation errors
        if error_record.error_type == ErrorType.VALIDATION_ERROR:
            return False
        
        # Special handling for trading errors
        if error_record.error_type == ErrorType.TRADING_ERROR:
            # Only retry certain trading errors
            trading_retryable = [
                'temporary', 'busy', 'service unavailable'
            ]
            if not any(keyword in error_record.message.lower() for keyword in trading_retryable):
                return False
        
        return True
    
    async def execute_with_retry(self, 
                                operation: Callable,
                                context: ErrorContext,
                                *args, **kwargs) -> Any:
        """
        Execute operation with retry logic
        
        Args:
            operation: Async function to execute
            context: Error context
            *args, **kwargs: Arguments for operation
            
        Returns:
            Operation result
            
        Raises:
            Exception: If all retry attempts fail
        """
        endpoint = context.endpoint or "unknown"
        circuit_breaker = self.get_circuit_breaker(endpoint)
        retry_config = self.get_retry_config(endpoint)
        
        last_error = None
        
        for attempt in range(1, retry_config.max_attempts + 1):
            # Check circuit breaker
            if not circuit_breaker.should_allow_request():
                raise Exception(f"Circuit breaker is open for endpoint {endpoint}")
            
            try:
                # Update context with attempt info
                context.correlation_id = f"{context.correlation_id}-attempt-{attempt}"
                
                # Execute operation
                result = await operation(*args, **kwargs)
                
                # Success - update circuit breaker
                circuit_breaker.record_success()
                
                # If this was a retry, log success
                if attempt > 1:
                    self.logger.info(
                        f"Operation succeeded after {attempt} attempts: {context.operation}"
                    )
                
                return result
                
            except Exception as e:
                last_error = e
                
                # Record error
                error_record = self.record_error(e, context)
                error_record.retry_count = attempt - 1
                
                # Update circuit breaker
                circuit_breaker.record_failure()
                
                # Check if should retry
                if not self.should_retry(error_record, attempt, retry_config):
                    self.logger.error(
                        f"Operation failed permanently after {attempt} attempts: {context.operation}"
                    )
                    raise e
                
                # Calculate delay for next attempt
                if attempt < retry_config.max_attempts:
                    delay = self.calculate_retry_delay(attempt, retry_config)
                    
                    self.logger.warning(
                        f"Operation failed (attempt {attempt}/{retry_config.max_attempts}), "
                        f"retrying in {delay:.2f}s: {e}"
                    )
                    
                    await asyncio.sleep(delay)
        
        # All attempts failed
        if last_error:
            raise last_error
        else:
            raise Exception("Operation failed with unknown error")
    
    def handle_trading_error(self, error: Exception, operation: str, 
                           order_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Handle trading-specific errors with specialized logic
        
        Args:
            error: Trading error that occurred
            operation: Trading operation (e.g., 'place_order', 'close_position')
            order_data: Order data that caused the error
            
        Returns:
            Error handling result with recommendations
        """
        error_msg = str(error).lower()
        
        handling_result = {
            "error_type": "trading_error",
            "operation": operation,
            "message": str(error),
            "severity": "high",
            "should_retry": False,
            "recommendations": [],
            "safe_to_continue": True
        }
        
        # Insufficient margin
        if "insufficient margin" in error_msg or "margin" in error_msg:
            handling_result.update({
                "error_subtype": "insufficient_margin",
                "recommendations": [
                    "Reduce position size",
                    "Check account balance",
                    "Close existing positions to free margin"
                ],
                "safe_to_continue": True
            })
        
        # Position limits
        elif "position limit" in error_msg or "max positions" in error_msg:
            handling_result.update({
                "error_subtype": "position_limit",
                "recommendations": [
                    "Close existing positions",
                    "Reduce position size",
                    "Check position limits configuration"
                ],
                "safe_to_continue": True
            })
        
        # Market closed
        elif "market closed" in error_msg or "trading disabled" in error_msg:
            handling_result.update({
                "error_subtype": "market_closed",
                "recommendations": [
                    "Wait for market to open",
                    "Check trading hours",
                    "Queue order for next trading session"
                ],
                "safe_to_continue": True,
                "should_retry": True  # Can retry when market opens
            })
        
        # Invalid price
        elif "invalid price" in error_msg or "price" in error_msg:
            handling_result.update({
                "error_subtype": "invalid_price",
                "recommendations": [
                    "Check current market price",
                    "Adjust order price",
                    "Use market order instead of limit order"
                ],
                "safe_to_continue": True
            })
        
        # Order rejected
        elif "rejected" in error_msg:
            handling_result.update({
                "error_subtype": "order_rejected",
                "severity": "critical",
                "recommendations": [
                    "Review order parameters",
                    "Check account status",
                    "Contact support if persistent"
                ],
                "safe_to_continue": False
            })
        
        # Log trading error with specialized handling
        self.logger.error(
            f"Trading error handled: {handling_result['error_subtype']} - {operation} - {error}"
        )
        
        return handling_result
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get comprehensive error statistics"""
        recent_errors = [
            error for error in self.error_history
            if datetime.fromisoformat(error.context.timestamp) > 
               datetime.utcnow() - timedelta(hours=24)
        ]
        
        stats = {
            "total_errors": len(self.error_history),
            "recent_errors_24h": len(recent_errors),
            "error_rate_24h": len(recent_errors) / 24.0,  # Errors per hour
            "errors_by_type": self.error_stats["errors_by_type"].copy(),
            "errors_by_endpoint": self.error_stats["errors_by_endpoint"].copy(),
            "circuit_breaker_states": {
                endpoint: breaker.state.value 
                for endpoint, breaker in self.circuit_breakers.items()
            },
            "top_error_endpoints": [],
            "error_trends": []
        }
        
        # Calculate top error endpoints
        if self.error_stats["errors_by_endpoint"]:
            sorted_endpoints = sorted(
                self.error_stats["errors_by_endpoint"].items(),
                key=lambda x: x[1],
                reverse=True
            )
            stats["top_error_endpoints"] = sorted_endpoints[:5]
        
        return stats
    
    def cleanup_old_errors(self, days: int = 7):
        """Clean up old error records"""
        cutoff_time = datetime.utcnow() - timedelta(days=days)
        
        initial_count = len(self.error_history)
        self.error_history = [
            error for error in self.error_history
            if datetime.fromisoformat(error.context.timestamp) > cutoff_time
        ]
        
        cleaned_count = initial_count - len(self.error_history)
        if cleaned_count > 0:
            self.logger.info(f"Cleaned up {cleaned_count} old error records")
    
    def export_error_report(self, output_file: str = None) -> str:
        """Export comprehensive error report"""
        if output_file is None:
            output_file = f"oanda_error_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        report = {
            "report_timestamp": datetime.utcnow().isoformat(),
            "summary": self.get_error_statistics(),
            "recent_errors": [
                {
                    "timestamp": error.context.timestamp,
                    "type": error.error_type.value,
                    "severity": error.severity.value,
                    "message": error.message,
                    "operation": error.context.operation,
                    "endpoint": error.context.endpoint,
                    "retry_count": error.retry_count,
                    "resolved": error.resolved
                }
                for error in self.error_history[-100:]  # Last 100 errors
            ],
            "circuit_breaker_status": {
                endpoint: {
                    "state": breaker.state.value,
                    "failure_count": breaker.failure_count,
                    "last_failure_time": breaker.last_failure_time
                }
                for endpoint, breaker in self.circuit_breakers.items()
            }
        }
        
        try:
            with open(output_file, 'w') as f:
                json.dump(report, f, indent=2)
            
            self.logger.info(f"Error report exported to {output_file}")
            return output_file
            
        except Exception as e:
            self.logger.error(f"Failed to export error report: {e}")
            return ""


# Decorator for automatic error handling
def handle_oanda_errors(error_handler: OANDAErrorHandler = None, 
                       operation_name: str = None,
                       endpoint: str = None):
    """
    Decorator for automatic OANDA error handling
    
    Args:
        error_handler: Error handler instance
        operation_name: Name of the operation
        endpoint: API endpoint being accessed
    """
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            nonlocal error_handler, operation_name
            
            if error_handler is None:
                error_handler = OANDAErrorHandler()
            
            if operation_name is None:
                operation_name = func.__name__
            
            context = ErrorContext(
                operation=operation_name,
                endpoint=endpoint
            )
            
            return await error_handler.execute_with_retry(
                func, context, *args, **kwargs
            )
        
        return wrapper
    return decorator


async def main():
    """Example usage and testing"""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    error_handler = OANDAErrorHandler()
    
    # Example: Function that might fail
    @handle_oanda_errors(error_handler, "test_operation", "/v3/test")
    async def failing_operation():
        import random
        if random.random() < 0.7:  # 70% chance of failure
            raise Exception("Simulated network error")
        return "Success!"
    
    try:
        result = await failing_operation()
        print(f"Operation result: {result}")
        
        # Get error statistics
        stats = error_handler.get_error_statistics()
        print(f"Error statistics: {json.dumps(stats, indent=2)}")
        
    except Exception as e:
        print(f"Operation ultimately failed: {e}")


if __name__ == '__main__':
    asyncio.run(main())