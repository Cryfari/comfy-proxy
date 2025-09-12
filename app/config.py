from dotenv import load_dotenv
import os

load_dotenv()

from urllib.parse import urlparse

def comfy_ws_url(client_id: str) -> str:
    # http://127.0.0.1:8188  -> ws://127.0.0.1:8188/ws?clientId=...
    # https://example.com    -> wss://example.com/ws?clientId=...
    p = urlparse(COMFY_URL)
    scheme = "wss" if p.scheme == "https" else "ws"
    netloc = p.netloc or f"{p.hostname}:{p.port or 80}"
    return f"{scheme}://{netloc}/ws?clientId={client_id}"

COMFY_URL = os.getenv("COMFY_URL", "http://127.0.0.1:8188")
BIND_HOST = os.getenv("BIND_HOST", "0.0.0.0")
BIND_PORT = int(os.getenv("BIND_PORT", "9000"))
ALLOWED_ORIGINS = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "*").split(",")]
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "25"))
HTTPX_TIMEOUT_SEC = int(os.getenv("HTTPX_TIMEOUT_SEC", "120"))

SAVE_WORKFLOW = os.getenv("SAVE_WORKFLOW", "true").lower() in ("1","true","yes","on")
RUNS_DIR = os.getenv("RUNS_DIR", "./runs")