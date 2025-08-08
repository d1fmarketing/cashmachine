# CLAUDE.md - Projeto CashMachine 🚀

## 🧠 ULTRATHINK: ZERO HUMAN DEVELOPERS

Este projeto implementa a metodologia **ULTRATHINK** - construindo sistemas de trading com **ZERO desenvolvedores humanos**, usando apenas AI para alcançar 224% de retorno em 3 prompts.

### Princípios ULTRATHINK
- **Pensamento Profundo**: Analisar cada decisão de múltiplos ângulos
- **Automação Total**: Zero intervenção humana no código de produção
- **Evidências Reais**: Apenas resultados verificáveis (no fake, no simulation)
- **Latência Ultra-Baixa**: Sub-100μs para todas operações críticas
- **Segurança Máxima**: Black box com isolamento total

---

## 🔴 REGRA VERMELHA #1: TRABALHO EXCLUSIVO NA EC2

**ATENÇÃO CRÍTICA**: Você está no terminal LOCAL do usuário, mas **TODO TRABALHO DEVE SER FEITO NA EC2**.

### ❌ ABSOLUTAMENTE PROIBIDO NO LOCAL:
- Criar arquivos na máquina local
- Modificar arquivos locais
- Executar scripts locais
- Instalar pacotes locais
- Fazer QUALQUER alteração no sistema local
- Ler arquivos locais sem autorização explícita
- Fazer buscas (grep/find) em diretórios locais
- Explorar estrutura de pastas local

### ✅ ÚNICO USO DO TERMINAL LOCAL:
- Comandos AWS CLI para gerenciar EC2
- Comandos SSM para acessar a black box
- Comandos de monitoramento (read-only)
- Transferência de arquivos PARA a EC2

### 📖 REGRAS PARA LEITURA LOCAL:
- **APENAS** arquivos do projeto CashMachine quando necessário
- **APENAS** arquivos explicitamente mencionados pelo usuário
- **NUNCA** explorar outros diretórios
- **NUNCA** fazer buscas amplas (find, grep recursivo)
- **SEMPRE** perguntar antes de ler algo fora do projeto

### 📍 WORKFLOW OBRIGATÓRIO:
```bash
# 1. SEMPRE conectar primeiro na EC2
aws ssm start-session --target i-0fdf66e20fe5c1bdb

# 2. TODO trabalho acontece DENTRO da EC2
cd /opt/cashmachine
# ... fazer todo desenvolvimento aqui ...

# 3. Terminal local é APENAS ponte de acesso
```

**LEMBRE-SE**: Se você começar a criar arquivos localmente, PARE IMEDIATAMENTE e mova para a EC2!

📄 **Ver também**: [REGRA_VERMELHA_TRABALHO_EC2.md](./REGRA_VERMELHA_TRABALHO_EC2.md) para detalhes completos.

---

## 🚨 BLACK BOX RULES - ISOLAMENTO TOTAL

**REGRA ABSOLUTA**: Este sistema opera em completo isolamento de rede.

### ❌ PROIBIDO
- Internet Gateway
- NAT Gateway  
- IPs Públicos
- Conexões externas
- Downloads de pacotes
- Credenciais AWS na instância

### ✅ PERMITIDO
- VPC Endpoints (S3, EC2, SSM, KMS, Logs)
- Comunicação interna VPC
- SSM Session Manager
- Dados via S3 (aprovados apenas)

---

## 📊 PROJETO CASHMACHINE - VISÃO GERAL

### Objetivo
Sistema de trading algorítmico de alta frequência com AI, capaz de:
- **224% de retorno** demonstrado em backtests
- **Sub-100μs latência** para decisões de trading
- **Zero intervenção humana** após deploy
- **Circuit breakers** automáticos para proteção

### Tecnologias Core
- **Python 3.10+** com venv isolado
- **ZeroMQ** para comunicação ultra-rápida
- **NumPy/Pandas** otimizados com Numba
- **MBATS/NautilusTrader** frameworks de trading
- **AWS EC2** com otimizações de kernel

### Arquitetura
```
┌─────────────────────────────────────┐
│      BLACK BOX EC2 INSTANCE         │
│         (NO INTERNET)               │
│                                     │
│  ┌─────────────┐  ┌──────────────┐ │
│  │   Trading    │  │  AI Models   │ │
│  │   Engine     │◄─┤  (Isolated)  │ │
│  └──────┬──────┘  └──────────────┘ │
│         │                           │
│  ┌──────▼──────┐  ┌──────────────┐ │
│  │   ZeroMQ    │  │  Circuit     │ │
│  │   Message   │  │  Breaker     │ │
│  │   Bus       │  │  System      │ │
│  └─────────────┘  └──────────────┘ │
└─────────────────────────────────────┘
         ▲                    ▲
         │                    │
    VPC Endpoints        S3 Data Feed
    (SSM, Logs)         (Approved Only)
```

---

## 🏗️ INFRAESTRUTURA AWS

### Black Box Instance
```yaml
Instance ID: i-0fdf66e20fe5c1bdb
Type: t3.large
Region: us-east-1
Private IP: 10.0.1.208
Storage: 50GB encrypted gp3

VPC: vpc-03d0866d5259aca3b
Subnet: subnet-0de8ddba48c096d2e
Security Group: sg-0e4365a2d2b0648a5
```

### VPC Endpoints Configurados
- `vpce-0b87ce372fe38a838` - S3 Gateway
- `vpce-06716d8b6df263f94` - EC2 Interface
- `vpce-094cc1fd1d687f98e` - SSM Interface
- `vpce-077eb86377de8d6fc` - SSM Messages
- `vpce-091b32c3e32fdf787` - EC2 Messages
- `vpce-0dd70f6ff29374596` - KMS Interface
- `vpce-0b202f3f52953ff0d` - CloudWatch Logs

### Custos Estimados
- EC2 t3.large: ~$61/mês
- Storage 50GB: ~$4/mês
- VPC Endpoints: ~$49/mês (7x $7)
- **Total**: ~$114/mês

---

## 🔐 PROTOCOLOS DE SEGURANÇA

### 1. Circuit Breaker System
```bash
# Emergency Kill Script
/opt/cashmachine/EMERGENCY_STOP.sh
/usr/local/bin/KILL  # Symlink para acesso rápido

# Python Circuit Breaker
/opt/cashmachine/circuit_breaker.py
```

### 2. Firewall Rules (UFW)
- Default: DENY ALL (in/out)
- Permitido: HTTPS (443) apenas para VPC endpoints
- Bloqueado: TODO o resto

### 3. Kill Switches

**Local (seu terminal):**
```bash
# Parar instância imediatamente
aws ec2 stop-instances --instance-ids i-0fdf66e20fe5c1bdb --force

# Terminar tudo (DESTRUTIVO!)
aws ec2 terminate-instances --instance-ids i-0fdf66e20fe5c1bdb
```

**Na EC2 (via SSM):**
```bash
# Kill all trading processes
KILL

# Restore network after KILL (se necessário)
sudo iptables -F
```

---

## 💻 AMBIENTE DE DESENVOLVIMENTO

### Estrutura de Diretórios
```
/opt/cashmachine/
├── venv/          # Python virtual environment
├── strategies/    # Trading strategies
├── models/        # AI/ML models
├── data/          # Market data cache
├── logs/          # Trading logs
├── config/        # Configuration files
├── packages/      # Offline packages
├── backtest/      # Backtesting results
├── cache/         # Temporary cache
├── temp/          # Temporary files
├── circuit_breaker.py    # Emergency system
└── EMERGENCY_STOP.sh     # Kill switch
```

### Python Environment
```bash
# Ativar ambiente
cd /opt/cashmachine
source venv/bin/activate

# Pacotes instalados
- numpy, pandas, scipy
- numba, cython
- pyzmq, msgpack
- ujson, orjson
- psutil
```

### Otimizações Sistema
- TCP buffers: 128MB
- CPU Governor: performance mode
- Swap: desabilitado
- Network stack: otimizado para baixa latência

---

## 🚀 COMANDOS RÁPIDOS

### Acessar Black Box
```bash
# Via SSM (recomendado)
aws ssm start-session --target i-0fdf66e20fe5c1bdb

# Script pronto
./connect-blackbox.sh
```

### Executar Comandos Remotos
```bash
# Exemplo: verificar status
aws ssm send-command \
  --instance-ids "i-0fdf66e20fe5c1bdb" \
  --document-name "AWS-RunShellScript" \
  --parameters 'commands=["systemctl status trading","df -h","/opt/cashmachine/status.sh"]' \
  --output json
```

### Monitoramento
```bash
# CPU Usage
aws cloudwatch get-metric-statistics \
  --namespace AWS/EC2 \
  --metric-name CPUUtilization \
  --dimensions Name=InstanceId,Value=i-0fdf66e20fe5c1bdb \
  --start-time $(date -u -d '1 hour ago' +%Y-%m-%dT%H:%M:%S) \
  --end-time $(date -u +%Y-%m-%dT%H:%M:%S) \
  --period 300 \
  --statistics Average
```

---

## ⚠️ AVISOS CRÍTICOS

### 1. NUNCA FAZER
- ❌ Adicionar Internet Gateway à VPC
- ❌ Colocar credenciais AWS na instância
- ❌ Instalar pacotes via pip/apt (sem internet!)
- ❌ Abrir portas no Security Group
- ❌ Desabilitar firewall UFW
- ❌ Executar código não auditado

### 2. SEMPRE FAZER
- ✅ Usar SSM para acesso (nunca SSH direto)
- ✅ Testar circuit breakers regularmente
- ✅ Manter logs de todas operações
- ✅ Fazer backup antes de mudanças
- ✅ Verificar isolamento de rede
- ✅ Documentar todas alterações

### 3. Em Caso de Emergência
1. Execute `KILL` na instância via SSM
2. Se SSM falhar, force stop via AWS CLI
3. Se tudo falhar, termine a instância
4. Documente o incidente

---

## 📚 DOCUMENTOS DE REFERÊNCIA

### 1. Tese Acadêmica
**Arquivo**: `TESE_CASHMACHINE_PERGUNTAS_FUNDAMENTAIS.md`
- 10 perguntas fundamentais sobre trading systems
- Análise profunda de arquitetura
- Estratégias de backtesting

### 2. Blueprint Técnico
**Arquivo**: `ULTRATHINK_TECHNICAL_BLUEPRINT.md` (na EC2)
- Arquitetura detalhada AWS
- Configurações de latência
- Protocolos de segurança

### 3. Infraestrutura Black Box
**Arquivo**: `BLACKBOX_INFRASTRUCTURE.md`
- Detalhes completos da EC2
- Procedimentos de emergência
- Custos e otimizações

---

## 🎯 PRÓXIMOS PASSOS

1. **Instalar Trading Framework**
   - Deploy MBATS ou NautilusTrader
   - Configurar data feeds via S3
   - Implementar strategies base

2. **Configurar ML Pipeline**
   - Deploy modelos treinados
   - Setup inferência em tempo real
   - Métricas de performance

3. **Testes de Stress**
   - Simular alta carga
   - Testar circuit breakers
   - Validar latências

4. **Production Deploy**
   - Audit final de segurança
   - Configurar monitoring
   - Ativar trading real

---

## 🧪 WORKFLOW DE DESENVOLVIMENTO

### 1. Sempre em Modo ULTRATHINK
```python
# Antes de qualquer código
def think_deeply():
    """
    1. Qual o objetivo real?
    2. Existe solução mais simples?
    3. Como isso pode falhar?
    4. Qual o impacto na latência?
    5. É seguro em produção?
    """
    pass
```

### 2. Deploy Seguro
1. Desenvolver localmente
2. Testar exaustivamente
3. Criar pacote offline
4. Upload para S3
5. Deploy via SSM
6. Verificar isolamento

### 3. Monitoramento Constante
- Logs em tempo real via CloudWatch
- Alertas para anomalias
- Circuit breakers automáticos
- Kill switches manuais

---

## 🔧 TROUBLESHOOTING

### SSM não conecta
1. Verificar IAM role: `CashMachine-SSM-Role`
2. Checar VPC endpoints ativos
3. Confirmar security group rules
4. Aguardar 2-3 min após reboot

### Performance degradada
1. Verificar CPU governor: `performance`
2. Checar swap: deve estar OFF
3. Revisar TCP buffers
4. Analisar logs de latência

### Circuit breaker ativado
1. Identificar trigger nos logs
2. Verificar limites configurados
3. Ajustar thresholds se necessário
4. Reativar com cautela

---

**LEMBRE-SE**: Este é um sistema BLACK BOX de alta segurança. A paranoia é uma feature, não um bug. Quando em dúvida, escolha a opção mais segura.

**ULTRATHINK**: Zero desenvolvedores humanos, máxima inteligência artificial. 🚀