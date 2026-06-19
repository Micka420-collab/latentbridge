#!/usr/bin/env bash
# Runs ON the Proxmox container (112). Clones the project, restores the Hermès
# config, rewrites Windows paths to Linux, and builds the project venv.
set -e
export PATH="$HOME/.local/bin:$PATH"
cd /root

echo "[provision] clone project"
rm -rf latentbridge
git clone -q lb.bundle latentbridge
cp -f lb.env latentbridge/.env
mkdir -p latentbridge/mydata && cp -f today.yaml latentbridge/mydata/today.yaml

echo "[provision] restore Hermès config -> ~/.hermes"
[ -d "$HOME/.hermes" ] && mv "$HOME/.hermes" "$HOME/.hermes.bak.$$"
mkdir -p "$HOME/.hermes"
rm -rf /tmp/hx && mkdir -p /tmp/hx && tar -xzf hermes.tgz -C /tmp/hx
cp -a /tmp/hx/hermes/. "$HOME/.hermes/"
rm -f "$HOME/.hermes/"*.lock "$HOME/.hermes/"*.pid "$HOME/.hermes/gateway_state.json" 2>/dev/null || true

echo "[provision] rewrite Windows paths -> Linux"
grep -rIl 'F:' "$HOME/.hermes" 2>/dev/null | while read -r f; do
  sed -i -e 's#F:/newtechno/.venv/Scripts/python.exe#/root/latentbridge/.venv/bin/python#g' \
         -e 's#F:/newtechno#/root/latentbridge#g' \
         -e 's#F:\\newtechno#/root/latentbridge#g' "$f"
done
grep -rIl 'venv/Scripts/python.exe' "$HOME/.hermes" 2>/dev/null | xargs -r \
  sed -i 's#venv/Scripts/python\.exe#venv/bin/python#g'

echo "[provision] project venv + deps (CPU torch)"
cd /root/latentbridge
uv venv .venv --python 3.11 >/dev/null
uv pip install -q --python .venv/bin/python torch --index-url https://download.pytorch.org/whl/cpu
uv pip install -q --python .venv/bin/python numpy pyyaml openai python-dotenv fastapi "uvicorn[standard]"

echo "[provision] DONE"
echo "  model:   $(grep -m1 '^model:' "$HOME/.hermes/config.yaml" 2>/dev/null)"
echo "  skills:  $(ls "$HOME/.hermes/skills" 2>/dev/null | tr '\n' ' ')"
echo "  wa-sess: $([ -d "$HOME/.hermes/platforms/whatsapp/session" ] && echo present || echo none)"
echo "  torch:   $(.venv/bin/python -c 'import torch;print(torch.__version__)' 2>/dev/null)"
