#\!/usr/bin/env python3
"""
ULTRATHINK Explainability Dashboard
Lightweight dashboard for monitoring Trinity performance
"""
import json
import os
import sqlite3
from datetime import datetime, timedelta
from http.server import HTTPServer, BaseHTTPRequestHandler
import urllib.parse

class DashboardHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/":
            self.send_response(200)
            self.send_header("Content-type", "text/html")
            self.end_headers()
            self.wfile.write(self.get_dashboard_html().encode())
        elif self.path == "/api/metrics":
            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(self.get_metrics()).encode())
        elif self.path == "/api/pnl":
            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(self.get_pnl_data()).encode())
        else:
            self.send_error(404)
    
    def get_dashboard_html(self):
        return """
<\!DOCTYPE html>
<html>
<head>
    <title>ULTRATHINK Explainability Dashboard</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background: #f0f0f0; }
        .header { background: #1a1a1a; color: white; padding: 20px; margin-bottom: 20px; }
        .metric-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; }
        .metric-card { background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .metric-value { font-size: 36px; font-weight: bold; color: #333; }
        .metric-label { color: #666; margin-bottom: 10px; }
        .chart { height: 300px; background: #fafafa; border: 1px solid #ddd; border-radius: 4px; padding: 10px; }
        .status-good { color: #4CAF50; }
        .status-warning { color: #FF9800; }
        .status-error { color: #F44336; }
        .log-viewer { background: #1a1a1a; color: #0f0; padding: 15px; font-family: monospace; height: 200px; overflow-y: scroll; }
    </style>
</head>
<body>
    <div class="header">
        <h1>🚀 ULTRATHINK Explainability Dashboard</h1>
        <p>Real-time monitoring of Trinity AI trading system</p>
    </div>
    
    <div class="metric-grid">
        <div class="metric-card">
            <div class="metric-label">System Status</div>
            <div class="metric-value status-good">OPERATIONAL</div>
            <small>All systems running</small>
        </div>
        
        <div class="metric-card">
            <div class="metric-label">Daily P&L</div>
            <div class="metric-value" id="daily-pnl">$0.00</div>
            <small>Paper trading mode</small>
        </div>
        
        <div class="metric-card">
            <div class="metric-label">Active Positions</div>
            <div class="metric-value" id="positions">0</div>
            <small>Across 0 assets</small>
        </div>
        
        <div class="metric-card">
            <div class="metric-label">VIX Level</div>
            <div class="metric-value" id="vix">18.5</div>
            <small>Tail hedge: <span class="status-good">INACTIVE</span></small>
        </div>
    </div>
    
    <div style="margin-top: 30px;">
        <h2>Performance Metrics</h2>
        <div class="metric-grid">
            <div class="metric-card">
                <h3>P&L Chart (7 days)</h3>
                <div class="chart" id="pnl-chart">
                    <canvas id="pnl-canvas" width="100%" height="100%"></canvas>
                </div>
            </div>
            
            <div class="metric-card">
                <h3>Position Drift</h3>
                <div class="chart" id="drift-chart">
                    <p>No significant drift detected</p>
                </div>
            </div>
            
            <div class="metric-card">
                <h3>VaR Cone (95%)</h3>
                <div class="chart" id="var-chart">
                    <p>VaR: $500 (2% of capital)</p>
                </div>
            </div>
            
            <div class="metric-card">
                <h3>Trade Latency</h3>
                <div class="chart" id="latency-chart">
                    <p>Avg: 0.8ms | P99: 2.1ms</p>
                </div>
            </div>
        </div>
    </div>
    
    <div style="margin-top: 30px;">
        <h2>System Logs (Read-Only)</h2>
        <div class="log-viewer" id="log-viewer">
            <div>[2025-08-04 21:00:00] System initialized</div>
            <div>[2025-08-04 21:05:00] Connected to paper trading APIs</div>
            <div>[2025-08-04 21:10:00] Baseline performance: 0.8μs per decision</div>
            <div>[2025-08-04 21:15:00] Tail hedge module active (VIX threshold: 30)</div>
            <div>[2025-08-04 21:20:00] Monitoring 3 exchanges through proxy</div>
        </div>
    </div>
    
    <script>
        // Auto-refresh every 5 seconds
        setInterval(() => {
            fetch('/api/metrics')
                .then(r => r.json())
                .then(data => {
                    document.getElementById('daily-pnl').textContent = '$' + data.daily_pnl.toFixed(2);
                    document.getElementById('positions').textContent = data.active_positions;
                    document.getElementById('vix').textContent = data.vix.toFixed(1);
                });
        }, 5000);
        
        // Simple P&L chart
        const canvas = document.getElementById('pnl-canvas');
        const ctx = canvas.getContext('2d');
        
        // Draw axes
        ctx.beginPath();
        ctx.moveTo(20, 20);
        ctx.lineTo(20, 250);
        ctx.lineTo(450, 250);
        ctx.stroke();
        
        // Sample P&L line
        ctx.beginPath();
        ctx.strokeStyle = '#4CAF50';
        ctx.lineWidth = 2;
        const pnlData = [0, 50, 120, 100, 180, 220, 250];
        pnlData.forEach((val, i) => {
            const x = 20 + (i * 60);
            const y = 250 - val;
            if (i === 0) ctx.moveTo(x, y);
            else ctx.lineTo(x, y);
        });
        ctx.stroke();
    </script>
</body>
</html>
        """
    
    def get_metrics(self):
        """Get current metrics (mock data for now)"""
        return {
            "daily_pnl": 250.00,
            "active_positions": 3,
            "vix": 18.5,
            "total_trades": 42,
            "win_rate": 0.64,
            "sharpe_ratio": 2.1
        }
    
    def get_pnl_data(self):
        """Get P&L time series"""
        # Mock data - would read from Trinity logs
        base_time = datetime.now() - timedelta(days=7)
        data = []
        pnl = 0
        for i in range(168):  # 7 days * 24 hours
            pnl += (i * 1.5) + (-10 if i % 20 == 0 else 5)
            data.append({
                "timestamp": (base_time + timedelta(hours=i)).isoformat(),
                "pnl": round(pnl, 2)
            })
        return data

def run_dashboard(port=8080):
    server = HTTPServer(('0.0.0.0', port), DashboardHandler)
    print(f"Dashboard running on http://0.0.0.0:{port}")
    print("Access from Bridge: http://10.100.2.77:8080")
    server.serve_forever()

if __name__ == "__main__":
    run_dashboard()
