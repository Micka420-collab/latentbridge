"""Client Shellia (Hermès) -> bot Minecraft."""
import json, os, sys, urllib.request
try:
    from dotenv import load_dotenv; load_dotenv()
except Exception: pass
URL = os.environ.get("MC_BOT_URL", "http://192.168.1.48:8090")
TOK = os.environ.get("MC_BOT_TOKEN", "")
def call(method, path, data=None):
    body = json.dumps(data).encode() if data is not None else None
    req = urllib.request.Request(URL+path, data=body, method=method,
        headers={"Authorization": f"Bearer {TOK}", "Content-Type": "application/json"})
    return urllib.request.urlopen(req, timeout=30).read().decode()
def main():
    if len(sys.argv) < 2: print(__doc__); return
    c = sys.argv[1]
    if c in ("state","health"): print(call("GET","/"+c))
    elif c=="chat": call("POST","/chat",{"text":sys.argv[2]}); print("chat envoye")
    elif c=="cmd": call("POST","/cmd",{"command":sys.argv[2]}); print("commande envoyee")
    elif c=="goto": call("POST","/goto",{"x":sys.argv[2],"y":sys.argv[3],"z":sys.argv[4]}); print("en route")
    elif c=="look": call("POST","/look",{"yaw":sys.argv[2],"pitch":sys.argv[3]}); print("ok")
    else: print("inconnu:",c)
if __name__=="__main__": main()
