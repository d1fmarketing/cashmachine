#!/usr/bin/env python3
"""
OANDA Connection Tester for Trinity System
Comprehensive testing and validation of OANDA API endpoints and connectivity
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
import statistics

from oanda_auth_manager import OANDAAuthManager
from oanda_proxy_config import OANDAProxyConfig
from trinity_oanda_integration import TrinityOANDAIntegration

@dataclass
class ConnectionTestResult:
    """Result of a connection test"""
    test_name: str
    endpoint: str
    success: bool
    response_time: float
    status_code: Optional[int] = None
    error_message: Optional[str] = None
    timestamp: str = None
    details: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow().isoformat()
        if self.details is None:
            self.details = {}

class OANDAConnectionTester:
    """Comprehensive connection testing for OANDA API"""
    
    def __init__(self, auth_manager: OANDAAuthManager = None,
                 proxy_config: OANDAProxyConfig = None):
        """
        Initialize connection tester
        
        Args:
            auth_manager: OANDA authentication manager
            proxy_config: Proxy configuration manager
        """
        self.logger = logging.getLogger(__name__)
        self.auth_manager = auth_manager or OANDAAuthManager()
        self.proxy_config = proxy_config or OANDAProxyConfig()
        
        # Test configuration
        self.test_instruments = ['EUR_USD', 'GBP_USD', 'USD_JPY', 'AUD_USD']
        self.test_results = []
        
        # Performance metrics
        self.latency_stats = {
            'min': float('inf'),
            'max': 0.0,
            'avg': 0.0,
            'samples': []
        }
    
    async def test_authentication(self) -> ConnectionTestResult:
        """Test OANDA API authentication"""
        test_name = "Authentication Test"
        
        try:
            start_time = time.time()
            
            # Load credentials
            credentials = self.auth_manager.load_credentials()
            
            # Validate credentials
            is_valid, message = self.auth_manager.validate_credentials(credentials)
            
            response_time = time.time() - start_time
            
            result = ConnectionTestResult(
                test_name=test_name,
                endpoint="Authentication",
                success=is_valid,
                response_time=response_time,
                error_message=None if is_valid else message,
                details={'message': message}
            )
            
            self.logger.info(f"Authentication test: {'PASSED' if is_valid else 'FAILED'}")
            return result
            
        except Exception as e:
            return ConnectionTestResult(
                test_name=test_name,
                endpoint="Authentication",
                success=False,
                response_time=0.0,
                error_message=str(e)
            )
    
    async def test_account_access(self) -> ConnectionTestResult:
        """Test account information access"""
        test_name = "Account Access Test"
        
        try:
            start_time = time.time()
            
            credentials = self.auth_manager.load_credentials()
            account_info = self.auth_manager.get_account_info(credentials)
            
            response_time = time.time() - start_time
            
            # Validate account info structure
            required_fields = ['account', 'lastTransactionID']
            account_fields = ['id', 'currency', 'balance', 'NAV']
            
            success = (
                all(field in account_info for field in required_fields) and
                all(field in account_info['account'] for field in account_fields)
            )
            
            result = ConnectionTestResult(
                test_name=test_name,
                endpoint="Account Info",
                success=success,
                response_time=response_time,
                details={
                    'account_id': account_info.get('account', {}).get('id'),
                    'currency': account_info.get('account', {}).get('currency'),
                    'balance': account_info.get('account', {}).get('balance')
                }
            )
            
            self.logger.info(f"Account access test: {'PASSED' if success else 'FAILED'}")
            return result
            
        except Exception as e:
            return ConnectionTestResult(
                test_name=test_name,
                endpoint="Account Info",
                success=False,
                response_time=0.0,
                error_message=str(e)
            )
    
    async def test_market_data_access(self) -> List[ConnectionTestResult]:
        """Test market data access for multiple instruments"""
        results = []
        
        for instrument in self.test_instruments:
            test_name = f"Market Data Test - {instrument}"
            
            try:
                start_time = time.time()
                
                trinity = TrinityOANDAIntegration(self.auth_manager, self.proxy_config)
                market_data = await trinity.get_market_data([instrument], count=10)
                
                response_time = time.time() - start_time
                
                success = (
                    instrument in market_data and
                    len(market_data[instrument]) > 0 and
                    all(col in market_data[instrument].columns 
                        for col in ['open', 'high', 'low', 'close', 'volume'])
                )
                
                result = ConnectionTestResult(
                    test_name=test_name,
                    endpoint=f"Market Data - {instrument}",
                    success=success,
                    response_time=response_time,
                    details={
                        'instrument': instrument,
                        'candles_received': len(market_data.get(instrument, [])),
                        'columns': list(market_data.get(instrument, {}).columns) if success else None
                    }
                )
                
                self.logger.info(f"Market data test for {instrument}: {'PASSED' if success else 'FAILED'}")
                results.append(result)
                
                # Update latency stats
                self._update_latency_stats(response_time)
                
            except Exception as e:
                result = ConnectionTestResult(
                    test_name=test_name,
                    endpoint=f"Market Data - {instrument}",
                    success=False,
                    response_time=0.0,
                    error_message=str(e)
                )
                results.append(result)
                
            # Rate limiting between requests
            await asyncio.sleep(0.2)
        
        return results
    
    async def test_pricing_stream(self) -> ConnectionTestResult:
        """Test real-time pricing stream"""
        test_name = "Pricing Stream Test"
        
        try:
            start_time = time.time()
            
            trinity = TrinityOANDAIntegration(self.auth_manager, self.proxy_config)
            prices = await trinity.get_current_prices(self.test_instruments[:2])  # Test with 2 instruments
            
            response_time = time.time() - start_time
            
            success = (
                len(prices) > 0 and
                all(instrument in prices for instrument in self.test_instruments[:2]) and
                all('bid' in price_data and 'ask' in price_data 
                    for price_data in prices.values())
            )
            
            result = ConnectionTestResult(
                test_name=test_name,
                endpoint="Pricing Stream",
                success=success,
                response_time=response_time,
                details={
                    'instruments_tested': self.test_instruments[:2],
                    'prices_received': len(prices),
                    'sample_spread': prices.get('EUR_USD', {}).get('spread') if 'EUR_USD' in prices else None
                }
            )
            
            self.logger.info(f"Pricing stream test: {'PASSED' if success else 'FAILED'}")
            return result
            
        except Exception as e:
            return ConnectionTestResult(
                test_name=test_name,
                endpoint="Pricing Stream",
                success=False,
                response_time=0.0,
                error_message=str(e)
            )
    
    async def test_order_simulation(self) -> ConnectionTestResult:
        """Test order placement (simulation only)"""
        test_name = "Order Simulation Test"
        
        try:
            start_time = time.time()
            
            credentials = self.auth_manager.load_credentials()
            headers = self.auth_manager.get_auth_headers(credentials)
            endpoint = self.auth_manager.get_api_endpoint(credentials['environment'])
            
            # Test with a very small position that won't execute
            order_data = {
                "order": {
                    "type": "LIMIT",
                    "instrument": "EUR_USD",
                    "units": "1",
                    "price": "0.9000",  # Very low price that won't execute
                    "timeInForce": "GTC"
                }
            }
            
            url = f"{endpoint}/v3/accounts/{credentials['account_id']}/orders"
            response = self.proxy_config.make_request('POST', url, headers, json=order_data)
            
            response_time = time.time() - start_time
            
            # For testing, we expect this to succeed in creating the order
            success = response.status_code == 201
            
            # Cancel the order immediately if it was created
            if success:
                result_data = response.json()
                order_id = result_data.get('orderCreateTransaction', {}).get('id')
                if order_id:
                    cancel_url = f"{endpoint}/v3/accounts/{credentials['account_id']}/orders/{order_id}/cancel"
                    self.proxy_config.make_request('PUT', cancel_url, headers)
            
            result = ConnectionTestResult(
                test_name=test_name,
                endpoint="Order Placement",
                success=success,
                response_time=response_time,
                status_code=response.status_code,
                details={
                    'order_type': 'LIMIT',
                    'test_instrument': 'EUR_USD',
                    'cancelled_immediately': success
                }
            )
            
            self.logger.info(f"Order simulation test: {'PASSED' if success else 'FAILED'}")
            return result
            
        except Exception as e:
            return ConnectionTestResult(
                test_name=test_name,
                endpoint="Order Placement",
                success=False,
                response_time=0.0,
                error_message=str(e)
            )
    
    async def test_positions_access(self) -> ConnectionTestResult:
        """Test positions access"""
        test_name = "Positions Access Test"
        
        try:
            start_time = time.time()
            
            trinity = TrinityOANDAIntegration(self.auth_manager, self.proxy_config)
            positions = await trinity.get_positions()
            
            response_time = time.time() - start_time
            
            # Success if we can retrieve positions (even if empty)
            success = isinstance(positions, dict)
            
            result = ConnectionTestResult(
                test_name=test_name,
                endpoint="Positions",
                success=success,
                response_time=response_time,
                details={
                    'open_positions': len(positions),
                    'instruments': list(positions.keys()) if positions else []
                }
            )
            
            self.logger.info(f"Positions access test: {'PASSED' if success else 'FAILED'}")
            return result
            
        except Exception as e:
            return ConnectionTestResult(
                test_name=test_name,
                endpoint="Positions",
                success=False,
                response_time=0.0,
                error_message=str(e)
            )
    
    async def test_latency_benchmark(self, iterations: int = 10) -> ConnectionTestResult:
        """Benchmark API latency with multiple requests"""
        test_name = f"Latency Benchmark ({iterations} iterations)"
        
        try:
            latencies = []
            
            trinity = TrinityOANDAIntegration(self.auth_manager, self.proxy_config)
            
            for i in range(iterations):
                start_time = time.time()
                
                # Quick pricing request
                await trinity.get_current_prices(['EUR_USD'])
                
                latency = time.time() - start_time
                latencies.append(latency)
                
                # Small delay between requests
                await asyncio.sleep(0.1)
            
            # Calculate statistics
            min_latency = min(latencies)
            max_latency = max(latencies)
            avg_latency = statistics.mean(latencies)
            median_latency = statistics.median(latencies)
            std_dev = statistics.stdev(latencies) if len(latencies) > 1 else 0
            
            # Update global latency stats
            self.latency_stats['samples'].extend(latencies)
            self.latency_stats['min'] = min(self.latency_stats['min'], min_latency)
            self.latency_stats['max'] = max(self.latency_stats['max'], max_latency)
            self.latency_stats['avg'] = statistics.mean(self.latency_stats['samples'])
            
            result = ConnectionTestResult(
                test_name=test_name,
                endpoint="Latency Benchmark",
                success=True,
                response_time=avg_latency,
                details={
                    'iterations': iterations,
                    'min_latency': min_latency,
                    'max_latency': max_latency,
                    'avg_latency': avg_latency,
                    'median_latency': median_latency,
                    'std_deviation': std_dev,
                    'p95_latency': sorted(latencies)[int(0.95 * len(latencies))] if len(latencies) > 1 else avg_latency
                }
            )
            
            self.logger.info(f"Latency benchmark: avg={avg_latency:.3f}s, p95={result.details['p95_latency']:.3f}s")
            return result
            
        except Exception as e:
            return ConnectionTestResult(
                test_name=test_name,
                endpoint="Latency Benchmark",
                success=False,
                response_time=0.0,
                error_message=str(e)
            )
    
    def _update_latency_stats(self, latency: float):
        """Update latency statistics"""
        self.latency_stats['samples'].append(latency)
        self.latency_stats['min'] = min(self.latency_stats['min'], latency)
        self.latency_stats['max'] = max(self.latency_stats['max'], latency)
        if self.latency_stats['samples']:
            self.latency_stats['avg'] = statistics.mean(self.latency_stats['samples'])
    
    async def run_comprehensive_test(self) -> Dict[str, Any]:
        """Run all connection tests"""
        self.logger.info("Starting comprehensive OANDA connection test suite")
        start_time = time.time()
        
        all_results = []
        
        # Run individual tests
        tests = [
            self.test_authentication(),
            self.test_account_access(),
            self.test_pricing_stream(),
            self.test_positions_access(),
            self.test_order_simulation(),
            self.test_latency_benchmark()
        ]
        
        # Add market data tests
        market_data_results = await self.test_market_data_access()
        
        # Execute main tests
        for test_coro in tests:
            try:
                result = await test_coro
                all_results.append(result)
            except Exception as e:
                self.logger.error(f"Test failed with exception: {e}")
                all_results.append(ConnectionTestResult(
                    test_name="Unknown Test",
                    endpoint="Unknown",
                    success=False,
                    response_time=0.0,
                    error_message=str(e)
                ))
        
        # Add market data results
        all_results.extend(market_data_results)
        
        total_time = time.time() - start_time
        
        # Calculate summary statistics
        total_tests = len(all_results)
        passed_tests = sum(1 for result in all_results if result.success)
        failed_tests = total_tests - passed_tests
        success_rate = passed_tests / total_tests if total_tests > 0 else 0
        
        summary = {
            "test_suite": "OANDA Connection Test Suite",
            "timestamp": datetime.utcnow().isoformat(),
            "total_time": total_time,
            "summary": {
                "total_tests": total_tests,
                "passed_tests": passed_tests,
                "failed_tests": failed_tests,
                "success_rate": success_rate
            },
            "latency_stats": self.latency_stats,
            "test_results": [asdict(result) for result in all_results],
            "overall_status": "PASS" if success_rate >= 0.8 else "FAIL"
        }
        
        self.logger.info(f"Test suite completed: {passed_tests}/{total_tests} tests passed ({success_rate:.1%})")
        return summary
    
    async def test_trading_workflow(self) -> ConnectionTestResult:
        """Test complete trading workflow (simulation)"""
        test_name = "Trading Workflow Test"
        
        try:
            start_time = time.time()
            
            trinity = TrinityOANDAIntegration(self.auth_manager, self.proxy_config)
            
            # Step 1: Get market data
            market_data = await trinity.get_market_data(['EUR_USD'], count=5)
            
            # Step 2: Get current prices
            prices = await trinity.get_current_prices(['EUR_USD'])
            
            # Step 3: Check positions
            positions = await trinity.get_positions()
            
            # Step 4: Get performance metrics
            metrics = trinity.get_performance_metrics()
            
            response_time = time.time() - start_time
            
            success = (
                'EUR_USD' in market_data and
                'EUR_USD' in prices and
                isinstance(positions, dict) and
                'account_balance' in metrics
            )
            
            result = ConnectionTestResult(
                test_name=test_name,
                endpoint="Trading Workflow",
                success=success,
                response_time=response_time,
                details={
                    'market_data_candles': len(market_data.get('EUR_USD', [])),
                    'current_price': prices.get('EUR_USD', {}).get('mid'),
                    'open_positions': len(positions),
                    'account_balance': metrics.get('account_balance')
                }
            )
            
            self.logger.info(f"Trading workflow test: {'PASSED' if success else 'FAILED'}")
            return result
            
        except Exception as e:
            return ConnectionTestResult(
                test_name=test_name,
                endpoint="Trading Workflow",
                success=False,
                response_time=0.0,
                error_message=str(e)
            )
    
    def generate_test_report(self, test_results: Dict[str, Any], 
                           output_file: str = None) -> str:
        """Generate a detailed test report"""
        if output_file is None:
            output_file = f"oanda_connection_test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        try:
            with open(output_file, 'w') as f:
                json.dump(test_results, f, indent=2)
            
            self.logger.info(f"Test report saved to {output_file}")
            return output_file
            
        except Exception as e:
            self.logger.error(f"Failed to save test report: {e}")
            return ""


async def main():
    """CLI interface for connection testing"""
    import argparse
    
    parser = argparse.ArgumentParser(description="OANDA Connection Tester")
    parser.add_argument('--test', choices=['auth', 'account', 'market', 'pricing', 'orders', 'positions', 'latency', 'workflow', 'all'], 
                       default='all', help='Test to run')
    parser.add_argument('--iterations', type=int, default=10, help='Iterations for latency test')
    parser.add_argument('--output', help='Output file for test report')
    parser.add_argument('--verbose', action='store_true', help='Verbose logging')
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format='%(asctime)s - %(levelname)s - %(message)s')
    
    tester = OANDAConnectionTester()
    
    try:
        if args.test == 'all':
            results = await tester.run_comprehensive_test()
            print("\n" + "="*60)
            print("OANDA CONNECTION TEST SUMMARY")
            print("="*60)
            print(f"Overall Status: {results['overall_status']}")
            print(f"Tests Passed: {results['summary']['passed_tests']}/{results['summary']['total_tests']}")
            print(f"Success Rate: {results['summary']['success_rate']:.1%}")
            print(f"Total Time: {results['total_time']:.2f} seconds")
            
            if results['latency_stats']['samples']:
                print(f"Average Latency: {results['latency_stats']['avg']:.3f} seconds")
            
            # Generate report
            report_file = tester.generate_test_report(results, args.output)
            if report_file:
                print(f"Detailed report: {report_file}")
            
        else:
            # Run individual test
            test_map = {
                'auth': tester.test_authentication,
                'account': tester.test_account_access,
                'market': lambda: tester.test_market_data_access(),
                'pricing': tester.test_pricing_stream,
                'orders': tester.test_order_simulation,
                'positions': tester.test_positions_access,
                'latency': lambda: tester.test_latency_benchmark(args.iterations),
                'workflow': tester.test_trading_workflow
            }
            
            if args.test in test_map:
                result = await test_map[args.test]()
                if isinstance(result, list):
                    for r in result:
                        print(f"{r.test_name}: {'PASSED' if r.success else 'FAILED'}")
                        if not r.success and r.error_message:
                            print(f"  Error: {r.error_message}")
                else:
                    print(f"{result.test_name}: {'PASSED' if result.success else 'FAILED'}")
                    if not result.success and result.error_message:
                        print(f"Error: {result.error_message}")
            else:
                print(f"Unknown test: {args.test}")
                return 1
        
        return 0
        
    except Exception as e:
        print(f"❌ Test execution failed: {e}")
        return 1


if __name__ == '__main__':
    exit(asyncio.run(main()))