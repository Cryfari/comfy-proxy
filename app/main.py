from fastapi import FastAPI, UploadFile, File, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from .config import ALLOWED_ORIGINS
from .schemas import GenerateRequest
from . import comfy_client
import json, pathlib

app = FastAPI(title="ComfyUI Proxy")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS if ALLOWED_ORIGINS != ["*"] else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
    files = {"image": (file.filename, await file.read(), file.content_type)}
    r = await comfy_client.comfy_post("/upload/image", files=files)
    return r.json()

@app.post("/generate")
async def generate(req: GenerateRequest):
    preset_path = pathlib.Path(__file__).parent / "presets" / f"{req.preset}.json"
    if not preset_path.exists():
        raise HTTPException(404, f"Preset {req.preset} not found")
    workflow = json.loads(preset_path.read_text(encoding="utf-8"))

    # panggil patcher sesuai preset
    from .patchers import t2i_lora  # contoh; map preset→module
    wf = t2i_lora.apply(workflow, req.params)

    r = await comfy_client.comfy_post("/prompt", json={"prompt": wf, "client_id": "comfy-proxy"})
    return r.json()

@app.get("/history/{prompt_id}")
async def history(prompt_id: str):
    r = await comfy_client.comfy_get(f"/history/{prompt_id}")
    return r.json()

@app.get("/image")
async def image(filename: str):
    # Sanitasi: tolak path traversal
    if ".." in filename or filename.startswith("/"):
        raise HTTPException(400, "Invalid filename")
    r = await comfy_client.comfy_get("/view", params={"filename": filename})
    return Response(content=r.content, media_type="image/png")
