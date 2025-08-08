#!/bin/bash
# ULTRATHINK CashMachine Black Box Initialization
# NO INTERNET ACCESS - Local packages only

set -e

# Create ultrathink user
useradd -m -s /bin/bash ultrathink
usermod -aG sudo ultrathink

# Create directory structure
mkdir -p /opt/cashmachine/{data,logs,config,strategies,backtest}
chown -R ultrathink:ultrathink /opt/cashmachine

# System optimization for low latency
echo "net.core.rmem_max = 134217728" >> /etc/sysctl.conf
echo "net.core.wmem_max = 134217728" >> /etc/sysctl.conf
echo "net.ipv4.tcp_rmem = 4096 87380 134217728" >> /etc/sysctl.conf
echo "net.ipv4.tcp_wmem = 4096 65536 134217728" >> /etc/sysctl.conf
sysctl -p

# CPU frequency scaling - maximum performance
echo "performance" | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor

# Create circuit breaker script
cat > /opt/cashmachine/circuit_breaker.sh << 'EOF'
#!/bin/bash
# Emergency kill switch
echo "EMERGENCY STOP ACTIVATED"
pkill -9 -f "cashmachine"
pkill -9 -f "trading"
pkill -9 -f "python"
echo "All trading processes terminated"
EOF
chmod +x /opt/cashmachine/circuit_breaker.sh

# Create startup marker
echo "BLACK BOX INITIALIZED: $(date)" > /opt/cashmachine/init.log
echo "NO INTERNET ACCESS - MAXIMUM SECURITY" >> /opt/cashmachine/init.log