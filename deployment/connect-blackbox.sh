#!/bin/bash
# ULTRATHINK CashMachine Black Box Connection Script
# SECURE ACCESS - NO INTERNET

echo "========================================="
echo "   CASHMACHINE BLACK BOX CONNECTION"
echo "   ULTRATHINK - MAXIMUM SECURITY"
echo "========================================="
echo ""

INSTANCE_ID="i-0fdf66e20fe5c1bdb"

# Check if SSM plugin is installed
if ! command -v session-manager-plugin &> /dev/null; then
    echo "ERROR: AWS SSM Session Manager plugin not installed"
    echo "Install: https://docs.aws.amazon.com/systems-manager/latest/userguide/session-manager-working-with-install-plugin.html"
    exit 1
fi

echo "Connecting to BLACK BOX instance..."
echo "Instance: $INSTANCE_ID"
echo "Type: ISOLATED - NO INTERNET ACCESS"
echo ""

# Start SSM session
aws ssm start-session --target $INSTANCE_ID --region us-east-1

# Alternative SSH connection (requires bastion)
# ssh -i ~/.ssh/cashmachine-blackbox-key.pem ubuntu@10.0.1.208