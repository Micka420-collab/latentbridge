"""Client Hermès -> agent computer-use Windows. Voir & contrôler le PC.

    python scripts/pc.py health
    python scripts/pc.py screenshot screen.png
    python scripts/pc.py click 640 480
    python scripts/pc.py type "bonjour"
    python scripts/pc.py key ctrl c
    python scripts/pc.py launch chrome

Lit WINAGENT_URL et WINAGENT_TOKEN depuis l'environnement / .env.
Flux typique pour Hermès (vision) : screenshot -> regarde l'image -> click/type.
"""
from __future__ import annotations

import json
import os
import sys
import urllib.request

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

URL = os.environ.get("WINAGENT_URL", "http://192.168.1.48:8765")
TOK = os.environ.get("WINAGENT_TOKEN", "")


def call(method, path, data=None):
    body = json.dumps(data).encode() if data is not None else None
    req = urllib.request.Request(
        URL + path, data=body, method=method,
        headers={"Authorization": f"Bearer {TOK}", "Content-Type": "application/json"})
    return urllib.request.urlopen(req, timeout=20)


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return
    cmd = sys.argv[1]
    if cmd == "health":
        print(call("GET", "/health").read().decode())
    elif cmd == "screenshot":
        out = sys.argv[2] if len(sys.argv) > 2 else "screen.png"
        open(out, "wb").write(call("GET", "/screenshot").read())
        print(f"saved {out}")
    elif cmd == "click":
        call("POST", "/click", {"x": int(sys.argv[2]), "y": int(sys.argv[3])})
        print("clicked")
    elif cmd == "type":
        call("POST", "/type", {"text": sys.argv[2]})
        print("typed")
    elif cmd == "key":
        call("POST", "/key", {"keys": sys.argv[2:]})
        print("key sent")
    elif cmd == "launch":
        call("POST", "/launch", {"app": sys.argv[2]})
        print(f"launched {sys.argv[2]}")
    else:
        print(f"unknown: {cmd}\n{__doc__}")


if __name__ == "__main__":
    main()
