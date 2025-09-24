import mimetypes
from uuid import uuid4
import uuid
from fastapi import FastAPI, UploadFile, File, HTTPException, Response, WebSocket, WebSocketDisconnect, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from typing import Literal
from pathlib import Path
from datetime import datetime

from .config import ALLOWED_ORIGINS, COMFY_URL, SAVE_WORKFLOW, RUNS_DIR
from .schemas import GenerateRequest
from . import comfy_client
import json, pathlib
import time
import asyncio
import websockets
from websockets.exceptions import ConnectionClosedOK, ConnectionClosedError
from .config import comfy_ws_url

from .utils import save_json_atomic, now_jkt_iso
import os


app = FastAPI(title="ComfyUI Proxy")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS if ALLOWED_ORIGINS != ["*"] else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class GenerateParams:
    width: int = 512
    height: int = 512
    steps: int = 20
    cfg_scale: float = 7.0
    sampler_name: str = "Euler a"

class generateRequest:
    preset: str
    params: dict = GenerateParams

async def _pump_client_to_upstream(client_ws: WebSocket, upstream_ws: websockets.WebSocketClientProtocol):
    try:
        while True:
            msg = await client_ws.receive()
            if "text" in msg:
                await upstream_ws.send(msg["text"])
            elif "bytes" in msg and msg["bytes"] is not None:
                await upstream_ws.send(msg["bytes"])
            elif msg.get("type") == "websocket.disconnect":
                try:
                    await upstream_ws.close()
                finally:
                    break
    except WebSocketDisconnect:
        try:
            await upstream_ws.close()
        except Exception:
            pass

async def _pump_upstream_to_client(client_ws: WebSocket, upstream_ws: websockets.WebSocketClientProtocol):
    try:
        async for message in upstream_ws:
            if isinstance(message, (bytes, bytearray)):
                await client_ws.send_bytes(message)
            else:
                await client_ws.send_text(message)
    except (ConnectionClosedOK, ConnectionClosedError):
        # upstream menutup koneksi
        try:
            await client_ws.close()
        except Exception:
            pass

# Root folder output (bisa override lewat ENV jika mau)
OUTPUT_ROOT = Path(os.getenv("OUTPUT_ROOT", "./output")).resolve()
IMAGE_DIR = (OUTPUT_ROOT / "image")
VIDEO_DIR = (OUTPUT_ROOT / "video")
AUDIO_DIR = (OUTPUT_ROOT / "audio")

# Pastikan folder ada
for d in (IMAGE_DIR, VIDEO_DIR, AUDIO_DIR):
    d.mkdir(parents=True, exist_ok=True)

# Ekstensi yang diizinkan per kategori
ALLOWED_EXT = {
    "image": {".png", ".jpg", ".jpeg", ".webp"},  # utama: .png
    "video": {".mp4", ".mkv", ".avi"},            # utama: .mp4
    "audio": {".mp3", ".wav"}                     # utama: .mp3
}

def _list_media(dirpath: Path, allowed: set[str], limit: int, newest_first: bool):
    items = []
    if not dirpath.exists():
        return items
    for p in dirpath.iterdir():
        if not p.is_file():
            continue
        if p.suffix.lower() not in allowed:
            continue
        stat = p.stat()
        items.append({
            "filename": p.name,
            "size": stat.st_size,
            "mtime": stat.st_mtime,  # epoch
        })
    # sort by mtime
    items.sort(key=lambda x: x["mtime"], reverse=newest_first)
    if limit > 0:
        items = items[:limit]
    # format readable time
    for it in items:
        it["modified_at"] = datetime.fromtimestamp(it["mtime"]).isoformat(timespec="seconds")
    return items

def _build_url(request: Request, category: str, filename: str) -> str:
    # URL ke endpoint file langsung
    return str(request.url_for("get_media_file", category=category, filename=filename))

@app.get('/favicon.ico', include_in_schema=False)
async def favicon():
    # Pastikan Anda membuat folder assets/icon di dalam folder static
    return FileResponse('static/assets/icon/favicon.ico')

@app.get("/")
async def image():
    # return html page
    return FileResponse("static/index.html", media_type="text/html")

@app.get("/video")
async def video():
    return FileResponse("static/video_ui.html", media_type="text/html")

@app.get("/audio")
async def audio():
    return FileResponse("static/audio_ui.html", media_type="text/html")

@app.get("/faceswap")
async def face_swap():
    return FileResponse("static/faceswap.html", media_type="text/html")

@app.get("/uno")
async def uno():
    return FileResponse("static/flux_uno.html", media_type="text/html")

@app.get("/gallery")
async def gallery():
    return FileResponse("static/gallery.html", media_type="text/html")

@app.websocket("/ws")
async def ws_proxy(
    client_ws: WebSocket,
    client_id: str = Query("comfy-proxy"),   # FE bisa override ?client_id=...
):
    # (Opsional) Validasi Origin di sini jika perlu
    # origin = client_ws.headers.get("origin", "")
    # if origin not in ALLOWED_ORIGINS and "*" not in ALLOWED_ORIGINS:
    #     await client_ws.close(code=1008)  # Policy Violation
    #     return

    await client_ws.accept()

    upstream_url = comfy_ws_url(client_id)

    # Tambahkan header ekstra jika diperlukan (mis. X-Forwarded-For)
    extra_headers = []
    try:
        async with websockets.connect(
            upstream_url,
            extra_headers=extra_headers,
            max_size=None,  # biar payload besar (preview image) aman
            ping_interval=20,
            ping_timeout=20,
        ) as upstream_ws:

            # Jalankan relay dua arah
            t1 = asyncio.create_task(_pump_client_to_upstream(client_ws, upstream_ws))
            t2 = asyncio.create_task(_pump_upstream_to_client(client_ws, upstream_ws))

            done, pending = await asyncio.wait({t1, t2}, return_when=asyncio.FIRST_COMPLETED)
            for t in pending:
                t.cancel()

    except Exception as e:
        # Kirim alasan ke klien (opsional)
        try:
            await client_ws.send_text(f'{{"type":"ws_error","message":"{str(e)}"}}')
        except Exception:
            pass
        finally:
            try:
                await client_ws.close()
            except Exception:
                pass

@app.get("/history/media")
def list_all_media(
    request: Request,
    limit: int = Query(50, ge=0, le=1000, description="Maksimal item per kategori"),
    newest_first: bool = Query(True, description="Urutkan terbaru dulu"),
):
    images = _list_media(IMAGE_DIR, ALLOWED_EXT["image"], limit, newest_first)
    videos = _list_media(VIDEO_DIR, ALLOWED_EXT["video"], limit, newest_first)
    audios = _list_media(AUDIO_DIR, ALLOWED_EXT["audio"], limit, newest_first)

    # lengkapi dengan URL akses file
    for it in images:
        it["url"] = _build_url(request, "image", it["filename"])
    for it in videos:
        it["url"] = _build_url(request, "video", it["filename"])
    for it in audios:
        it["url"] = _build_url(request, "audio", it["filename"])

    return {
        "image": images,
        "video": videos,
        "audio": audios,
        "root": str(OUTPUT_ROOT),
    }

@app.get("/history/media/{category}")
def list_media_by_category(
    request: Request,
    category: Literal["image", "video", "audio"],
    limit: int = Query(50, ge=0, le=1000),
    newest_first: bool = Query(True),
):
    dirmap = {"image": IMAGE_DIR, "video": VIDEO_DIR, "audio": AUDIO_DIR}
    items = _list_media(dirmap[category], ALLOWED_EXT[category], limit, newest_first)
    for it in items:
        it["url"] = _build_url(request, category, it["filename"])
    return {"category": category, "items": items}

@app.get("/media/{category}/{filename}", name="get_media_file")
def get_media_file(category: Literal["image","video","audio"], filename: str):
    # Sanitasi nama file untuk mencegah path traversal
    if "/" in filename or "\\" in filename:
        raise HTTPException(status_code=400, detail="Invalid filename")
    base = {"image": IMAGE_DIR, "video": VIDEO_DIR, "audio": AUDIO_DIR}[category]
    file_path = (base / filename).resolve()

    # Pastikan file masih di dalam direktori kategori
    if not str(file_path).startswith(str(base.resolve())):
        raise HTTPException(status_code=400, detail="Invalid path")
    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(status_code=404, detail="File not found")
    if file_path.suffix.lower() not in ALLOWED_EXT[category]:
        raise HTTPException(status_code=415, detail="Unsupported media type")

    # Tentukan content-type
    ctype, _ = mimetypes.guess_type(str(file_path))
    return FileResponse(path=str(file_path), media_type=ctype or "application/octet-stream")

@app.get("/health")
async def health():
    r = await comfy_client.comfy_get("/queue")
    return {"ok": True, "queue": r.json()}

@app.get("/object-info")
async def object_info():
    r = await comfy_client.comfy_get("/object_info")
    return r.json()

@app.post("/upload/image")
async def upload_image(file: UploadFile = File(...)):
    # (opsional) validasi mime & ukuran via file.spool_max_size
    timestamp = time.time()
    filename = f"{timestamp}-{file.filename}"
    files = {"image": (filename, await file.read(), file.content_type)}
    r = await comfy_client.comfy_post("/upload/image", files=files)
    return r.json()

@app.post("/upload/audio")
async def upload_audio(file: UploadFile = File(...)):
    timestamp = time.time()
    filename = f"{timestamp}-{file.filename}"
    files = {"image": (filename, await file.read(), file.content_type)}
    r = await comfy_client.comfy_post("/upload/image", files=files)
    return r.json()

@app.post("/interupt")
async def interupt():
    r = await comfy_client.comfy_post("/interupt")
    return r.json()


@app.post("/generate")
async def generate(req: GenerateRequest):
    preset_path = pathlib.Path(__file__).parent / "presets" / f"{req.preset}.json"
    if not preset_path.exists():
        raise HTTPException(404, f"Preset {req.preset} not found")
    workflow = json.loads(preset_path.read_text(encoding="utf-8"))

    # panggil patcher sesuai preset
    if req.preset.startswith("t2i"):
        from .patchers import t2i
        wf = t2i.apply(workflow, req.params)
    elif req.preset == "face_swap_only":
        from .patchers import face_swap_only
        wf = face_swap_only.apply(workflow, req.params)
    elif req.preset == "t2a":
        from .patchers import t2a
        wf = t2a.apply(workflow, req.params)
    elif req.preset == "t2a_repaint":
        from .patchers import t2a_repaint
        wf = t2a_repaint.apply(workflow, req.params)
    elif req.preset == "t2v":
        from .patchers import t2v
        wf = t2v.apply(workflow, req.params)
    elif req.preset == "flux-uno":
        from .patchers import flux_uno
        wf = flux_uno.apply(workflow, req.params)

    CLIENT_ID = "comfy-proxy"

    prompt_id = str(uuid.uuid4())
    r = await comfy_client.comfy_post("/prompt", json={"prompt": wf, "client_id": CLIENT_ID})

    resp = r.json()
    prompt_id = resp.get("prompt_id") or resp.get("promptId") or str(uuid4())
    want_save = SAVE_WORKFLOW or bool(req.params.get("save_workflow", False))
    if want_save:
        day = now_jkt_iso()[:10]  # YYYY-MM-DD
        path = os.path.join(RUNS_DIR, day, f"{req.preset}-{prompt_id}.workflow.json")
        record = {
            "meta": {
                "preset": req.preset,
                "timestamp": now_jkt_iso(),
                "client_id": CLIENT_ID,
                "comfy_url": COMFY_URL,
                "prompt_id": prompt_id,
                "version": 1
            },
            "params": req.params,   # simpan payload params untuk audit/repro
            "workflow": wf          # exactly what we sent to /prompt
        }
        save_json_atomic(path, record)
        # Tampilkan path di respons (biar FE bisa download/lihat)
        # resp["workflow_path"] = path

    return {"prompt_id": prompt_id, "saved_workflow": path}
    return {"prompt_id": prompt_id}

@app.get("/history")
async def history():
    r = await comfy_client.comfy_get("/history")
    return r.json()

@app.get("/history/{prompt_id}")
async def history(prompt_id: str):
    r = await comfy_client.comfy_get(f"/history/{prompt_id}")
    return r.json()

@app.get("/image")
async def image(filename: str, subfolder: str = Query("")):
    # Sanitasi: tolak path traversal
    if ".." in filename or filename.startswith("/"):
        raise HTTPException(400, "Invalid filename")
    r = await comfy_client.comfy_get("/view", params={"filename": filename, "subfolder": subfolder})
    return Response(content=r.content, media_type=r.headers.get("Content-Type", "image/png"))


@app.get("/models")
async def models():
    r = await comfy_client.comfy_get("/models")
    return r.json()

@app.get("/models/{folder}")
async def models_in_folder(folder: str):
    r = await comfy_client.comfy_get(f"/models/{folder}")
    return r.json()
