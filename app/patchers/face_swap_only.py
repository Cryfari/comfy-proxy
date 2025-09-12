# app/patchers/face_swap_only.py
import copy

def _next_id(wf): 
    return str(max(map(int, wf.keys())) + 1 if wf else 1)

def _find_nodes(wf, class_type):
    return [(nid, n) for nid, n in wf.items() if n.get("class_type") == class_type]

def _first_node(wf, class_type):
    xs = _find_nodes(wf, class_type)
    return xs[0] if xs else (None, None)

def _out(nid, idx=0): 
    return [nid, idx]

def _default_upscale_model(scale: int | None) -> str:
    if scale == 1: return "1x-ESRGAN.safetensors"
    if scale == 2: return "2x-ESRGAN.safetensors"
    if scale == 4: return "4x-ESRGAN.safetensors" 
    if scale == 8: return "8x-ESRGAN.safetensors"
    return "2x-ESRGAN.safetensors"


def _get_faceswap_output(wf):
    swap_id, swap_node = _first_node(wf, "ReActorFaceSwap")
    if not swap_node:
        raise RuntimeError("ReActorFaceSwap not found in workflow")
    return _out(swap_id, 0)

def _rewire_sinks_to(wf, image_ref):
    for _, node in wf.items():
        if node.get("class_type") in ("SaveImage", "ETN_SendImageWebSocket"):
            if "images" in node.get("inputs", {}):
                node["inputs"]["images"] = image_ref

def _ensure_loader_b64(wf, existing_id, image_b64, title):
    """Clone loader jika perlu, dan set base64."""
    if existing_id is not None:
        # clone node bawaan agar aman dimodifikasi
        new_id = _next_id(wf)
        wf[new_id] = copy.deepcopy(wf[existing_id])
    else:
        new_id = _next_id(wf)
        wf[new_id] = {"class_type": "ETN_LoadImageBase64", "inputs": {}, "_meta": {"title": title}}
    wf[new_id]["inputs"]["image"] = image_b64 or ""
    return new_id


def _ensure_upscaler_nodes(wf, image_ref, model_name):
    """Kembalikan node_id dari ImageUpscaleWithModel yang siap dipakai."""
    import copy
    # coba gunakan template jika preset sudah punya:
    loader_id, loader_node = _first_node(wf, "UpscaleModelLoader")
    up_id, up_node         = _first_node(wf, "ImageUpscaleWithModel")

    if loader_node and up_node:
        # clone agar node asli tidak tertimpa
        new_loader = _next_id(wf); wf[new_loader] = copy.deepcopy(loader_node)
        new_up     = _next_id(wf); wf[new_up]     = copy.deepcopy(up_node)
        wf[new_loader]["inputs"]["model_name"] = model_name
        wf[new_up]["inputs"]["image"] = image_ref
        wf[new_up]["inputs"]["upscale_model"] = _out(new_loader, 0)
        return new_up

    # kalau tidak ada template, buat dari nol
    new_loader = _next_id(wf)
    wf[new_loader] = {
        "class_type": "UpscaleModelLoader",
        "inputs": {"model_name": model_name},
        "_meta": {"title": "Load Upscale Model (dyn)"}
    }
    new_up = _next_id(wf)
    wf[new_up] = {
        "class_type": "ImageUpscaleWithModel",
        "inputs": {"image": image_ref, "upscale_model": _out(new_loader, 0)},
        "_meta": {"title": "Upscale Image (dyn)"}
    }
    return new_up

# app/patchers/face_swap_only.py

def _find_nodes(wf, class_type):
    return [(nid, n) for nid, n in wf.items() if n.get("class_type") == class_type]

def _first_node(wf, class_type):
    xs = _find_nodes(wf, class_type)
    return xs[0] if xs else (None, None)

def _out(nid, idx=0):
    return [nid, idx]

def _rewire_sinks_to(wf, image_ref):
    for _, node in wf.items():
        if node.get("class_type") in ("SaveImage", "ETN_SendImageWebSocket"):
            if "images" in node.get("inputs", {}):
                node["inputs"]["images"] = image_ref

def apply(workflow: dict, params: dict) -> dict:
    # 0) pakai workflow apa adanya (jangan clone ReActor)
    wf = workflow  # <-- tidak deepcopy; kita memang ingin menulis di node yang ada

    # 1) ambil node inti dari preset
    swap_id, swap_node = _first_node(wf, "ReActorFaceSwap")
    if not swap_node:
        raise RuntimeError("ReActorFaceSwap not found in preset")

    # dapatkan referensi input/source/booster dari koneksi yang sudah ada
    inp_ref  = swap_node["inputs"].get("input_image")   # e.g. [loader_in_id, 0]
    src_ref  = swap_node["inputs"].get("source_image")  # e.g. [loader_src_id, 0]
    boost_ref = swap_node["inputs"].get("face_boost")   # e.g. [boost_id, 0] atau None

    if not (isinstance(inp_ref, list) and isinstance(src_ref, list)):
        raise RuntimeError("ReActorFaceSwap inputs are not wired to loaders as expected")

    loader_in_id  = str(inp_ref[0])
    loader_src_id = str(src_ref[0])

    # 2) isi ulang base64 pada loader yang sudah ada (tanpa membuat loader baru)
    input_b64  = params.get("input_image_b64") or ""
    source_b64 = params.get("source_image_b64") or ""
    if not input_b64 or not source_b64:
        raise RuntimeError("Both 'input_image_b64' and 'source_image_b64' are required")

    if wf.get(loader_in_id, {}).get("class_type") != "ETN_LoadImageBase64":
        raise RuntimeError("Preset's input loader is not ETN_LoadImageBase64")
    if wf.get(loader_src_id, {}).get("class_type") != "ETN_LoadImageBase64":
        raise RuntimeError("Preset's source loader is not ETN_LoadImageBase64")

    wf[loader_in_id]["inputs"]["image"]  = input_b64
    wf[loader_src_id]["inputs"]["image"] = source_b64

    # 3) booster: gunakan node yang sudah ada (jika memang terhubung), hanya ganti param
    use_boost = False
    boost_id = None
    face_boost_cfg = (params.get("face_boost") or {})
    if boost_ref and isinstance(boost_ref, list):
        boost_id = str(boost_ref[0])
        if wf.get(boost_id, {}).get("class_type") == "ReActorFaceBoost":
            # kalau payload tidak menyentuh booster, tetap pertahankan default preset
            enabled = bool(face_boost_cfg.get("enabled", True))
            wf[boost_id]["inputs"]["enabled"] = enabled
            wf[boost_id]["inputs"]["boost_model"] = face_boost_cfg.get("boost_model", wf[boost_id]["inputs"].get("boost_model", "GFPGANv1.4.pth"))
            wf[boost_id]["inputs"]["interpolation"] = face_boost_cfg.get("interpolation", wf[boost_id]["inputs"].get("interpolation", "Bilinear"))
            wf[boost_id]["inputs"]["visibility"] = float(face_boost_cfg.get("visibility", wf[boost_id]["inputs"].get("visibility", 1.0)))
            wf[boost_id]["inputs"]["codeformer_weight"] = float(face_boost_cfg.get("codeformer_weight", wf[boost_id]["inputs"].get("codeformer_weight", 0.5)))
            wf[boost_id]["inputs"]["restore_with_main_after"] = bool(face_boost_cfg.get("restore_with_main_after", wf[boost_id]["inputs"].get("restore_with_main_after", False)))
            use_boost = enabled

    # 4) set seluruh parameter ReActorFaceSwap (tanpa membuat node baru)
    s = swap_node["inputs"]
    s["enabled"] = True
    s["swap_model"] = params.get("swap_model", s.get("swap_model", "inswapper_128.onnx"))
    s["facedetection"] = params.get("facedetection", s.get("facedetection", "retinaface_resnet50"))
    s["face_restore_model"] = params.get("face_restore_model", s.get("face_restore_model", "none"))
    s["face_restore_visibility"] = float(params.get("face_restore_visibility", s.get("face_restore_visibility", 1.0)))
    s["codeformer_weight"] = float(params.get("codeformer_weight", s.get("codeformer_weight", 0.5)))
    s["detect_gender_input"] = params.get("detect_gender_input", s.get("detect_gender_input", "auto"))
    s["detect_gender_source"] = params.get("detect_gender_source", s.get("detect_gender_source", "auto"))
    s["input_faces_index"] = str(params.get("input_faces_index", s.get("input_faces_index", "0")))
    s["source_faces_index"] = str(params.get("source_faces_index", s.get("source_faces_index", "0")))
    s["console_log_level"] = int(params.get("console_log_level", s.get("console_log_level", 1)))

    # rewire input/source/boost tetap ke node yang sama (loader & booster preset)
    s["input_image"]  = _out(loader_in_id, 0)
    s["source_image"] = _out(loader_src_id, 0)
    if boost_id:
        s["face_boost"] = _out(boost_id, 0)
        # kalau user mematikan booster → biarkan terhubung tapi enabled=False (aman)
    # jika preset tidak punya booster, biarkan apa adanya (field boleh tak ada)

    # 5) pastikan sink menyimpan hasil swap (tanpa bikin node baru)
    _rewire_sinks_to(wf, _out(swap_id, 0))

    # 6) (opsional) upscale untuk face_swap_only → panggil helper kamu yang sudah ada
    #    ini boleh clone/buat node baru (permintaanmu hanya untuk TIDAK membuat ReActor baru)
    up_cfg = params.get("upscale")
    if up_cfg:
        wf = apply_upscale_for_faceswap(wf, up_cfg)

    return wf

def apply_upscale_for_faceswap(wf: dict, upscale_cfg: dict) -> dict:
    """Sisipkan upscaler setelah FaceSwap dan rewire Save/Send ke hasil upscale."""
    up = upscale_cfg or {}
    enabled = bool(up.get("enabled", False))
    base_ref = _get_faceswap_output(wf)

    print("apply_upscale_for_faceswap:", enabled, up)
    if not enabled:
        # tanpa upscale → arahkan sink ke hasil FaceSwap langsung
        _rewire_sinks_to(wf, base_ref)
        print("  no upscale, rewired to faceswap output")
        return wf

    scale = up.get("scale")
    model_name = up.get("model_name") or _default_upscale_model(scale)

    up_node_id = _ensure_upscaler_nodes(wf, base_ref, model_name)
    up_ref = _out(up_node_id, 0)

    # (opsional) jika ingin batasi dimensi akhir, di sini bisa tambahkan node resize
    # contoh placeholder (aktifkan jika kamu punya node resize yang pasti ada):
    # if up.get("target_max"):
    #     resize_id = _next_id(wf)
    #     wf[resize_id] = {
    #         "class_type": "ImageResize",
    #         "inputs": {"image": up_ref, "max_side": int(up["target_max"]), "mode": "lanczos"},
    #     }
    #     up_ref = _out(resize_id, 0)

    _rewire_sinks_to(wf, up_ref)
    
    return wf
