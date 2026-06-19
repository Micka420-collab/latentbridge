@echo off
REM Launcher for the locally-installed Hermes Agent (uv venv, Python 3.11).
REM Force UTF-8 so reading the Node WhatsApp bridge output doesn't crash on cp1252.
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
"%~dp0external\hermes-agent\.venv\Scripts\hermes.exe" %*
