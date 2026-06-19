# Run ONCE as Administrator (right-click > Run with PowerShell as admin, or:
#   Start-Process powershell -Verb RunAs -ArgumentList '-File F:\newtechno\winagent\setup-windows-admin.ps1')
# Sets up: winagent firewall (Hermès only) + Windows Remote Desktop (RDP).

$hermes = "192.168.1.112"   # the Hermès container; only it may reach the agent

Write-Host "1) Firewall: allow winagent :8765 from $hermes only..."
Remove-NetFirewallRule -DisplayName "winagent-hermes" -ErrorAction SilentlyContinue
New-NetFirewallRule -DisplayName "winagent-hermes" -Direction Inbound -Protocol TCP `
  -LocalPort 8765 -RemoteAddress $hermes -Action Allow | Out-Null

Write-Host "2) Enable Remote Desktop (RDP)..."
Set-ItemProperty -Path 'HKLM:\System\CurrentControlSet\Control\Terminal Server' `
  -Name fDenyTSConnections -Value 0
Enable-NetFirewallRule -DisplayGroup "Remote Desktop"
Set-ItemProperty -Path 'HKLM:\System\CurrentControlSet\Control\Terminal Server\WinStations\RDP-Tcp' `
  -Name UserAuthentication -Value 1

Write-Host ""
Write-Host "DONE. winagent reachable from $hermes only; RDP enabled for your LAN."
Write-Host "Tip: connect to this PC's desktop with:  mstsc /v:192.168.1.48"
