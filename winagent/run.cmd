@echo off
REM Launch the Hermès computer-use agent (port 8765) in this desktop session.
REM Optional auto-start: copy this file into your Startup folder yourself ->
REM   open  shell:startup  (Win+R) and drop a shortcut to this file there.
cd /d "%~dp0.."
start "" /min ".venv\Scripts\pythonw.exe" "winagent\server.py"
echo winagent started on port 8765 (token in winagent\.token).
