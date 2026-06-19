# Migrate LatentBridge + Hermès from this Windows PC to the Proxmox container 112.
# RUN THIS YOURSELF (your action authorizes the transfer the agent's harness blocks):
#   powershell -ExecutionPolicy Bypass -File F:\newtechno\scripts\migrate-to-proxmox.ps1
#
# Prereq: stop the Windows Hermès gateway first (Ctrl+C) so the WhatsApp session
# can move cleanly — only ONE machine may run it.

$ErrorActionPreference = "Stop"
$C    = "192.168.1.112"                 # container
$PROJ = "F:\newtechno"
$LA   = $env:LOCALAPPDATA
$TMP  = Join-Path $env:TEMP "hermes-migrate"
New-Item -Force -ItemType Directory $TMP | Out-Null

Write-Host "1/5  bundle project (git)..."
git -C $PROJ bundle create "$TMP\lb.bundle" main

Write-Host "2/5  pack Hermès config (skip logs/caches)..."
tar -C $LA `
  --exclude="hermes/logs" --exclude="hermes/audio_cache" --exclude="hermes/image_cache" `
  --exclude="hermes/cache" --exclude="hermes/*.lock" --exclude="hermes/*.pid" `
  -czf "$TMP\hermes.tgz" hermes

Write-Host "3/5  copy everything to the container..."
scp "$TMP\lb.bundle"                 "root@${C}:/root/lb.bundle"
scp "$TMP\hermes.tgz"                "root@${C}:/root/hermes.tgz"
scp "$PROJ\.env"                     "root@${C}:/root/lb.env"
scp "$PROJ\mydata\today.yaml"        "root@${C}:/root/today.yaml"
scp "$PROJ\scripts\_provision-container.sh" "root@${C}:/root/_provision.sh"

Write-Host "4/5  provision the container..."
ssh "root@$C" "bash /root/_provision.sh"

Write-Host "5/5  done. To start Hermès on the container (always-on):"
Write-Host "     ssh root@$C '/root/stack/hermes-agent/.venv/bin/hermes gateway'"
Write-Host ""
Write-Host "Then message yourself on WhatsApp — it now runs from Proxmox (192.168.1.112)."
Write-Host "Make sure winagent\run.cmd is running on THIS PC so Hermès can control it."
