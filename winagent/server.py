"""Windows computer-use agent — lets Hermès (on Proxmox) see & control THIS PC.

SECURITY (read this):
  - Every request needs the bearer token (winagent/.token, auto-generated).
  - Only IPs in WINAGENT_ALLOW (default: Hermès 192.168.1.112 + localhost) are served.
  - GUI primitives + launch-app only. NO arbitrary shell endpoint.
  - Must run in YOUR interactive desktop session (not as a SYSTEM service), or
    mouse/keyboard input won't reach the screen.

Run:  .venv\Scripts\python.exe winagent\server.py     (then it listens on :8765)
"""
from __future__ import annotations

import io
import os
import secrets
import subprocess

import uvicorn
from fastapi import FastAPI, Header, HTTPException, Request, Response
from pydantic import BaseModel

import mss
import mss.tools
import pyautogui

pyautogui.FAILSAFE = False

ROOT = os.path.dirname(os.path.abspath(__file__))
TOKEN_FILE = os.path.join(ROOT, ".token")
PORT = int(os.environ.get("WINAGENT_PORT", "8765"))
ALLOW = set(filter(None, os.environ.get(
    "WINAGENT_ALLOW", "192.168.1.112,127.0.0.1,::1").split(",")))


def _token() -> str:
    if os.path.exists(TOKEN_FILE):
        return open(TOKEN_FILE, encoding="utf-8").read().strip()
    tok = secrets.token_urlsafe(32)
    with open(TOKEN_FILE, "w", encoding="utf-8") as f:
        f.write(tok)
    return tok


TOKEN = _token()
app = FastAPI(title="winagent")


def _guard(request: Request, authorization: str | None):
    ip = request.client.host if request.client else "?"
    if ip not in ALLOW:
        raise HTTPException(403, f"ip {ip} not allowed")
    if authorization != f"Bearer {TOKEN}":
        raise HTTPException(401, "bad token")


class Click(BaseModel):
    x: int
    y: int
    button: str = "left"
    double: bool = False


class Move(BaseModel):
    x: int
    y: int


class Type(BaseModel):
    text: str
    interval: float = 0.0


class Keys(BaseModel):
    keys: list[str]


class Scroll(BaseModel):
    amount: int


class Launch(BaseModel):
    app: str            # e.g. "chrome", "notepad", a path, or a URL


@app.get("/health")
def health(request: Request, authorization: str = Header(None)):
    _guard(request, authorization)
    w, h = pyautogui.size()
    return {"ok": True, "screen": [w, h]}


@app.get("/screenshot")
def screenshot(request: Request, authorization: str = Header(None)):
    _guard(request, authorization)
    with mss.mss() as sct:
        shot = sct.grab(sct.monitors[1])  # primary monitor
        png = mss.tools.to_png(shot.rgb, shot.size)
    return Response(content=png, media_type="image/png")


@app.post("/click")
def click(body: Click, request: Request, authorization: str = Header(None)):
    _guard(request, authorization)
    pyautogui.click(body.x, body.y, button=body.button, clicks=2 if body.double else 1)
    return {"ok": True}


@app.post("/move")
def move(body: Move, request: Request, authorization: str = Header(None)):
    _guard(request, authorization)
    pyautogui.moveTo(body.x, body.y)
    return {"ok": True}


@app.post("/type")
def type_text(body: Type, request: Request, authorization: str = Header(None)):
    _guard(request, authorization)
    pyautogui.write(body.text, interval=body.interval)
    return {"ok": True}


@app.post("/key")
def key(body: Keys, request: Request, authorization: str = Header(None)):
    _guard(request, authorization)
    pyautogui.hotkey(*body.keys)
    return {"ok": True}


@app.post("/scroll")
def scroll(body: Scroll, request: Request, authorization: str = Header(None)):
    _guard(request, authorization)
    pyautogui.scroll(body.amount)
    return {"ok": True}


@app.post("/launch")
def launch(body: Launch, request: Request, authorization: str = Header(None)):
    _guard(request, authorization)
    # constrained: start an app/URL via the shell's "start" (no arbitrary args chain)
    subprocess.Popen(["cmd", "/c", "start", "", body.app], shell=False)
    return {"ok": True, "launched": body.app}


if __name__ == "__main__":
    print(f"[winagent] token: {TOKEN}")
    print(f"[winagent] allow: {sorted(ALLOW)}  port: {PORT}")
    uvicorn.run(app, host="0.0.0.0", port=PORT)
