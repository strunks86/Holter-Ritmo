#!/bin/bash
# Arranque del puente HolterLocal. Uso: bash boot.sh CLAVE TEMA
set -e
mkdir -p /root/puente && cd /root/puente
R=https://raw.githubusercontent.com/strunks86/holter-ritmo/claude/project-thread-bo1izm/servidor/puente
curl -fsSL -o srv.py $R/srv.py
curl -fsSL -o vig.sh $R/vig.sh
[ -x cloudflared ] || { curl -fsSL -o cloudflared https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64; chmod +x cloudflared; }
umask 077; echo "$1" > clave; echo "$2" > topic
pkill -f puente/vig.sh || true
nohup bash /root/puente/vig.sh >/dev/null 2>&1 &
echo "LISTO: puente HolterLocal arrancado"
