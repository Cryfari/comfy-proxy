import httpx
from .config import COMFY_URL, HTTPX_TIMEOUT_SEC

async def comfy_post(path: str, json=None, files=None):
    async with httpx.AsyncClient(timeout=HTTPX_TIMEOUT_SEC) as c:
        r = await c.post(f"{COMFY_URL}{path}", json=json, files=files)
        r.raise_for_status()
        return r

async def comfy_get(path: str, params=None):
    async with httpx.AsyncClient(timeout=HTTPX_TIMEOUT_SEC) as c:
        print(f"{COMFY_URL}{path}")
        r = await c.get(f"{COMFY_URL}{path}", params=params)
        r.raise_for_status()
        return r