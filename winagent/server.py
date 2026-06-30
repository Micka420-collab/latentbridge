# Démarrage de WinAgent avec logging
import logging
import os
import secrets
import subprocess
import uvicorn
# Pour le serveur FastAPI
from fastapi import FastAPI, Header, HTTPException, Request, Response

# Configurer le logging
logging.basicConfig(level=logging.INFO, filename='winagent.log',
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Variables nécessaires
ROOT = os.path.dirname(os.path.abspath(__file__))
TOKEN_FILE = os.path.join(ROOT, ".token")
PORT = int(os.environ.get("WINAGENT_PORT", "8765"))
ALLOW = set(filter(None, os.environ.get(
    "WINAGENT_ALLOW", "192.168.1.112,127.0.0.1,::1").split(",")))

# Fonction de génération de token
if os.path.exists(TOKEN_FILE):
    TOKEN = open(TOKEN_FILE, encoding="utf-8").read().strip()
else:
    TOKEN = secrets.token_urlsafe(32)
    with open(TOKEN_FILE, "w", encoding="utf-8") as f:
        f.write(TOKEN)

logging.info(f"[winagent] token: {TOKEN}")
logging.info(f"[winagent] allow: {sorted(ALLOW)}  port: {PORT}")

# Créer l'application FastAPI
app = FastAPI(title="winagent")

# Lancer le serveur
if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=PORT)
