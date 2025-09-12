# app/routers/framepack.py
import asyncio
from fastapi import APIRouter, UploadFile, File, Form, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import JSONResponse
from pathlib import Path
import os, json, base64

from app.integrations.framepack_runner import start_job, stop_job, get_status, get_progress_queue

router = APIRouter(prefix="/framepack", tags=["framepack"])

UPLOAD_DIR = Path("./uploads/framepack").resolve()
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

@router.post("/generate")
async def framepack_generate(
    prompt: str = Form(""),
    negative: str = Form(""),
    seed: int = Form(31337),
    seconds: float = Form(5.0),
    steps: int = Form(25),
    gs: float = Form(10.0),
    cfg: float = Form(1.0),
    rs: float = Form(0.0),
    gpu_memory_preservation: float = Form(6.0),
    use_teacache: bool = Form(True),
    file: UploadFile = File(...),
):
    # simpan file image
    ext = os.path.splitext(file.filename)[1].lower() or ".png"
    image_path = str(UPLOAD_DIR / f"{os.getpid()}_{file.filename}")
    with open(image_path, "wb") as f:
        f.write(await file.read())

    params = {
        "prompt": prompt,
        "negative": negative,
        "seed": seed,
        "seconds": seconds,
        "latent_window_size": 9,  # sesuai rekomendasi demo (jangan diubah)
        "steps": steps,
        "cfg": cfg,
        "gs": gs,
        "rs": rs,
        "gpu_memory_preservation": gpu_memory_preservation,
        "use_teacache": use_teacache,
    }
    job_id = start_job(image_path, params)
    return {"job_id": job_id}

@router.post("/stop/{job_id}")
async def framepack_stop(job_id: str):
    ok = stop_job(job_id)
    if not ok:
        raise HTTPException(404, "Job not found")
    return {"stopping": True}

@router.get("/status/{job_id}")
async def framepack_status(job_id: str):
    return get_status(job_id)

@router.websocket("/ws/{job_id}")
async def framepack_ws(websocket: WebSocket, job_id: str):
    await websocket.accept()
    stream = get_progress_queue(job_id)
    if not stream:
        await websocket.send_json({"type":"error","message":"job not found"})
        await websocket.close()
        return
    try:
        while True:
            flag, data = stream.output_queue.next()
            if flag is None:
                await asyncio.sleep(0.05)
                continue
            if flag == "progress":
                preview_np, desc, html = data
                # kirim preview sebagai PNG base64 kecil (opsional)
                preview_b64 = None
                if preview_np is not None:
                    from PIL import Image
                    from io import BytesIO
                    im = Image.fromarray(preview_np)
                    buf = BytesIO()
                    im.save(buf, format="PNG")
                    preview_b64 = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")
                await websocket.send_json({"type":"progress","desc":desc,"hint":html,"preview":preview_b64})
            elif flag == "file":
                await websocket.send_json({"type":"file","path":data})
            elif flag == "end":
                await websocket.send_json({"type":"end"})
                await websocket.close()
                break
    except WebSocketDisconnect:
        # client putus; biarkan job lanjut di server
        return
