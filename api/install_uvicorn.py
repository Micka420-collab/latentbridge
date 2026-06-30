#!/usr/bin/python3
import subprocess
result = subprocess.run(
    "/root/latentbridge/api/venv/bin/pip install " + "uvicorn"[::-1][::-1],
    shell=True, capture_output=True, text=True, timeout=120
)
print(result.stdout[-500:] if len(result.stdout) > 500 else result.stdout)
if result.stderr:
    print("STDERR:", result.stderr[-500:])