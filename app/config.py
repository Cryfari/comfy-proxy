from dotenv import load_dotenv
import os

load_dotenv()

COMFY_URL = os.getenv("COMFY_URL", "http://127.0.0.1:8188")
BIND_HOST = os.getenv("BIND_HOST", "0.0.0.0")
BIND_PORT = int(os.getenv("BIND_PORT", "9000"))
ALLOWED_ORIGINS = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "*").split(",")]
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "25"))
HTTPX_TIMEOUT_SEC = int(os.getenv("HTTPX_TIMEOUT_SEC", "120"))