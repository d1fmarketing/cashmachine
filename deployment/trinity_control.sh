#!/bin/bash
# Trinity Control Script

case "$1" in
    start)
        echo "🚀 Starting Trinity..."
        sudo systemctl start trinity.service
        echo "✅ Trinity is awakening..."
        sleep 2
        sudo systemctl status trinity.service --no-pager
        ;;
    stop)
        echo "🛑 Stopping Trinity..."
        sudo systemctl stop trinity.service
        echo "💤 Trinity is sleeping..."
        ;;
    restart)
        echo "🔄 Restarting Trinity..."
        sudo systemctl restart trinity.service
        echo "✅ Trinity reborn..."
        sleep 2
        sudo systemctl status trinity.service --no-pager
        ;;
    status)
        echo "📊 Trinity Status:"
        sudo systemctl status trinity.service --no-pager
        ;;
    logs)
        echo "📜 Trinity Logs (last 100 lines):"
        sudo journalctl -u trinity.service -n 100 --no-pager
        ;;
    follow)
        echo "👁️ Following Trinity consciousness (Ctrl+C to exit):"
        sudo journalctl -u trinity.service -f
        ;;
    memory)
        echo "🧠 Trinity Memory:"
        if [ -f /opt/cashmachine/trinity/data/trinity_memory.json ]; then
            cat /opt/cashmachine/trinity/data/trinity_memory.json | python3 -m json.tool
        else
            echo "No memory file found yet"
        fi
        ;;
    performance)
        echo "📈 Trinity Performance:"
        echo "---"
        echo "CPU & Memory Usage:"
        ps aux | grep trinity_daemon | grep -v grep
        echo "---"
        echo "Recent Trades:"
        tail -20 /opt/cashmachine/trinity/logs/trinity_daemon.log | grep -E "BUY|SELL|profit"
        ;;
    *)
        echo "Usage: $0 {start|stop|restart|status|logs|follow|memory|performance}"
        exit 1
        ;;
esac
