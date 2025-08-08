#!/bin/bash
# ULTRATHINK SSH BRIDGE - Executa TODOS comandos na EC2!
# Respeita a REGRA VERMELHA: Trabalho APENAS na EC2

BRIDGE_IP="34.199.119.53"
KEY_PATH="$HOME/.ssh/cashmachine-blackbox-key.pem"

# Se nenhum comando fornecido, conecta interativamente
if [ $# -eq 0 ]; then
    echo "🌉 Conectando na ULTRATHINK Bridge..."
    ssh -i "$KEY_PATH" -o StrictHostKeyChecking=no ubuntu@$BRIDGE_IP
else
    # Executa comando remotamente
    ssh -i "$KEY_PATH" -o StrictHostKeyChecking=no ubuntu@$BRIDGE_IP "$@"
fi