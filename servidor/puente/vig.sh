#!/bin/bash
# Mantiene vivos el servicio y el tunel; cuando el tunel cambia de direccion la avisa por ntfy.
cd /root/puente
TOPIC=$(cat topic)
while true; do
  pgrep -f "puente/srv.py" >/dev/null || nohup python3 /root/puente/srv.py /root/puente/clave >srv.log 2>&1 &
  if ! pgrep -f "cloudflared tunnel" >/dev/null; then
    rm -f cf.log
    nohup ./cloudflared tunnel --no-autoupdate --url http://127.0.0.1:8765 >cf.log 2>&1 &
    for i in $(seq 30); do U=$(grep -o 'https://[a-z0-9-]*\.trycloudflare\.com' cf.log | head -1); [ -n "$U" ] && break; sleep 2; done
    [ -n "$U" ] && curl -s -d "$U" "https://ntfy.sh/$TOPIC" >/dev/null
  fi
  sleep 20
done
