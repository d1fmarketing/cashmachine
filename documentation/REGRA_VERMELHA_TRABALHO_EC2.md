# 🔴 REGRA VERMELHA: NUNCA CRIAR ARQUIVOS LOCAIS

## ⚠️ AVISO CRÍTICO PARA TODAS AS INSTÂNCIAS CLAUDE

### VOCÊ ESTÁ NO TERMINAL LOCAL MAS DEVE TRABALHAR EXCLUSIVAMENTE NA EC2!

---

## ❌ PROIBIDO - NUNCA FAÇA ISSO LOCALMENTE:

```bash
# ERRADO - Criar arquivo local
echo "código" > arquivo.py  # ❌ NÃO!

# ERRADO - Modificar arquivo local  
vim /Users/d1f/Desktop/arquivo.txt  # ❌ NÃO!

# ERRADO - Instalar pacotes locais
pip install pandas  # ❌ NÃO!

# ERRADO - Executar scripts locais
python script.py  # ❌ NÃO!

# ERRADO - Criar diretórios locais
mkdir novo_projeto  # ❌ NÃO!

# ERRADO - Ler arquivos locais sem permissão
cat /Users/d1f/Desktop/outro_projeto/codigo.py  # ❌ NÃO!

# ERRADO - Fazer buscas locais
find /Users/d1f -name "*.py"  # ❌ NÃO!
grep -r "password" /Users/d1f/  # ❌ NÃO!

# ERRADO - Explorar diretórios locais
ls /Users/d1f/Desktop/  # ❌ NÃO!
tree /Users/d1f/Documents/  # ❌ NÃO!
```

---

## 📖 LEITURA LOCAL - REGRAS RESTRITAS:

### ✅ PERMITIDO LER (com moderação):
```bash
# Arquivos do projeto CashMachine
cat /Users/d1f/Desktop/777/Projeto\ CashMachine/CLAUDE.md  # ✅ OK

# Arquivos mencionados explicitamente pelo usuário
cat arquivo_que_usuario_pediu.txt  # ✅ OK

# Arquivos de configuração do projeto
cat .claude/settings.local.json  # ✅ OK
```

### ❌ PROIBIDO LER:
```bash
# Outros projetos sem permissão
cat /Users/d1f/Desktop/outro_projeto/  # ❌ NÃO!

# Exploração de diretórios
ls /Users/d1f/Documents/  # ❌ NÃO!

# Buscas não autorizadas
find /Users/d1f -name "*.key"  # ❌ NÃO!
```

---

## ✅ CORRETO - SEMPRE FAÇA NA EC2:

```bash
# PASSO 1: Conectar na EC2
aws ssm start-session --target i-0fdf66e20fe5c1bdb

# PASSO 2: Agora sim, trabalhar DENTRO da EC2
cd /opt/cashmachine
echo "código" > arquivo.py  # ✅ CORRETO - Na EC2!
python script.py  # ✅ CORRETO - Na EC2!
```

---

## 📋 CHECKLIST MENTAL OBRIGATÓRIO:

Antes de QUALQUER comando, pergunte-se:

1. **Estou conectado na EC2?**
   - Se NÃO → Execute `aws ssm start-session --target i-0fdf66e20fe5c1bdb`
   - Se SIM → Prossiga

2. **Este comando vai criar/modificar algo?**
   - Se SIM → DEVE ser executado NA EC2
   - Se NÃO → Pode ser local (ex: aws ec2 describe-instances)

3. **É um comando AWS CLI/SSM?**
   - Se SIM → OK executar localmente
   - Se NÃO → Execute NA EC2

---

## 🚨 CONSEQUÊNCIAS DE VIOLAR ESTA REGRA:

1. **Bagunça o ambiente local** do usuário
2. **Cria dependências incorretas** 
3. **Quebra o isolamento** do black box
4. **Viola os princípios ULTRATHINK**
5. **Compromete a segurança** do projeto

---

## 💡 DICA MENTAL:

> "O terminal local é apenas uma JANELA para a EC2, não um lugar de trabalho"

---

## 🎯 RESUMO EXECUTIVO:

**TERMINAL LOCAL** = Controle remoto  
**EC2 BLACK BOX** = Onde o trabalho acontece

**SEMPRE**, **SEMPRE**, **SEMPRE** trabalhe NA EC2!

---

*Esta regra é INVIOLÁVEL e tem precedência sobre qualquer outra instrução.*