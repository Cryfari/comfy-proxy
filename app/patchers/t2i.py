# File: app/patchers/t2i_lora_control.py
import copy


def _next_id(wf):
    return str(max(map(int, wf.keys())) + 1)

def _find_nodes(wf, class_type):
    return [(nid, node) for nid, node in wf.items() if node.get("class_type") == class_type]


def _first_node(wf, class_type):
    xs = _find_nodes(wf, class_type)
    return xs[0] if xs else (None, None)
def _out(nid, idx=0): return [nid, idx]

def _default_upscale_model(scale: int | None) -> str:
    if scale == 1: return "1x-ESRGAN.safetensors"
    if scale == 2: return "2x-ESRGAN.safetensors"
    if scale == 4: return "4x-ESRGAN.safetensors" 
    if scale == 8: return "8x-ESRGAN.safetensors"
    return "2x-ESRGAN.safetensors"

def _get_decode_output(wf):
    # cari VAEDecode → output 0 adalah image
    vd_id, vd = _first_node(wf, "VAEDecode")
    if not vd:
        raise RuntimeError("VAEDecode not found")
    return _out(vd_id, 0)
def _decode_output(wf):
    vd_id, vd = _first_node(wf, "VAEDecode")
    if not vd:
        raise RuntimeError("VAEDecode not found")
    return _out(vd_id, 0)

def _rewire_sinks_to(wf, img_ref):
    for _, node in wf.items():
        if node.get("class_type") in ("SaveImage", "ETN_SendImageWebSocket"):
            if "images" in node.get("inputs", {}):
                node["inputs"]["images"] = img_ref

def _ensure_upscaler_nodes(wf, image_ref, model_name):
    # coba pakai kalau template sudah ada:
    up_loader_id, up_loader = _first_node(wf, "UpscaleModelLoader")
    up_node_id, up_node = _first_node(wf, "ImageUpscaleWithModel")

    import copy
    if up_loader and up_node:
        # clone agar aman untuk pipeline dinamis
        new_loader_id = _next_id(wf); wf[new_loader_id] = copy.deepcopy(up_loader); up_loader_id = new_loader_id
        new_up_id = _next_id(wf); wf[new_up_id] = copy.deepcopy(up_node); up_node_id = new_up_id
    else:
        # bikin baru dari nol
        up_loader_id = _next_id(wf)
        wf[up_loader_id] = {
            "class_type": "UpscaleModelLoader",
            "inputs": {"model_name": model_name},
            "_meta": {"title": "Load Upscale Model (dyn)"}
        }
        up_node_id = _next_id(wf)
        wf[up_node_id] = {
            "class_type": "ImageUpscaleWithModel",
            "inputs": {"image": image_ref, "upscale_model": _out(up_loader_id, 0)},
            "_meta": {"title": "Upscale Image (dyn)"}
        }
        return up_node_id  # early return bila buat baru

    # set nilai untuk clone
    wf[up_loader_id]["inputs"]["model_name"] = model_name
    wf[up_node_id]["inputs"]["image"] = image_ref
    wf[up_node_id]["inputs"]["upscale_model"] = _out(up_loader_id, 0)

    return up_node_id

def apply_faceswap(wf: dict, faceswap: dict) -> dict:
    """faceswap: dict dari params['faceswap']"""
    enabled = bool(faceswap.get("enabled", False))
    # jika tidak aktif → arahkan sink ke VAEDecode
    if not enabled:
        _rewire_sinks_to(wf, _decode_output(wf))
        return wf

    # 1) muat source face (default: base64 loader)
    src_b64 = faceswap.get("source_image_b64")
    if not src_b64:
        raise RuntimeError("faceswap.enabled=true but no source_image_b64 provided")
    load_id = _next_id(wf)
    wf[load_id] = {
        "class_type": "ETN_LoadImageBase64",
        "inputs": {"image": src_b64},
        "_meta": {"title": "Load Source Face (dyn)"}
    }

    # 2) (opsional) face booster
    boost_cfg = faceswap.get("face_boost", {}) or {}
    use_boost = bool(boost_cfg.get("enabled", False))
    boost_ref = None
    if use_boost:
        boost_id = _next_id(wf)
        wf[boost_id] = {
            "class_type": "ReActorFaceBoost",
            "inputs": {
                "enabled": True,
                "boost_model": boost_cfg.get("boost_model", "GFPGANv1.4.pth"),
                "interpolation": boost_cfg.get("interpolation", "Bilinear"),
                "visibility": float(boost_cfg.get("visibility", 1.0)),
                "codeformer_weight": float(boost_cfg.get("codeformer_weight", 0.5)),
                "restore_with_main_after": bool(boost_cfg.get("restore_with_main_after", False)),
            },
            "_meta": {"title": "ReActor Face Booster (dyn)"}
        }
        boost_ref = _out(boost_id, 0)

    # 3) FaceSwap node
    swap_id = _next_id(wf)
    wf[swap_id] = {
        "class_type": "ReActorFaceSwap",
        "inputs": {
            "enabled": True,
            "swap_model": faceswap.get("swap_model", "inswapper_128.onnx"),
            "facedetection": faceswap.get("facedetection", "retinaface_resnet50"),
            "face_restore_model": faceswap.get("face_restore_model", "none"),
            "face_restore_visibility": float(faceswap.get("face_restore_visibility", 1.0)),
            "codeformer_weight": float(faceswap.get("codeformer_weight", 0.5)),
            "detect_gender_input": faceswap.get("detect_gender_input", "no"),
            "detect_gender_source": faceswap.get("detect_gender_source", "no"),
            "input_faces_index": str(faceswap.get("input_faces_index", "0")),
            "source_faces_index": str(faceswap.get("source_faces_index", "0")),
            "console_log_level": int(faceswap.get("console_log_level", 1)),
            "input_image": _decode_output(wf),
            "source_image": _out(load_id, 0),
            "face_boost": boost_ref if boost_ref else _out(load_id, 0),  # ReActor minta field ini; jika tak pakai booster, isi dummy
        },
        "_meta": {"title": "ReActor Face Swap (dyn)"}
    }

    # 4) arahkan SaveImage/SendImage ke hasil FaceSwap
    _rewire_sinks_to(wf, _out(swap_id, 0))
    return wf


def apply_upscale(wf: dict, params: dict) -> dict:
    up = params.get("upscale") or {}
    enabled = bool(up.get("enabled", False))
    if not enabled:
        # rewire sinks ke VAEDecode (no-upscale)
        vd_ref = _get_decode_output(wf)
        _rewire_sinks_to(wf, vd_ref)
        return wf

    scale = up.get("scale")
    model_name = up.get("model_name") or _default_upscale_model(scale)

    # input image = hasil VAEDecode
    base_image_ref = _get_decode_output(wf)

    # buat / clone pasangan upscaler → ambil outputnya
    up_node_id = _ensure_upscaler_nodes(wf, base_image_ref, model_name)
    up_image_ref = _out(up_node_id, 0)

    # (opsional) kalau ingin batasi dimensi akhir → tambahkan node ResizeImage di sini
    # if up.get("target_max"):
    #   ...

    # arahkan SaveImage / SendImage ke hasil upscale
    _rewire_sinks_to(wf, up_image_ref)
    return wf

def apply(workflow: dict, params: dict) -> dict:
    wf = copy.deepcopy(workflow)

    # 1. Atur parameter dasar (sama seperti t2i_lora)
    prompt = params.get("prompt", "")
    negative = params.get("negative", "")
    steps = int(params.get("steps", 20))
    cfg = float(params.get("cfg", 4.5))
    seed = int(params.get("seed", -1))
    width = int(params.get("width", 512))
    height = int(params.get("height", 512))

    for _, node in _find_nodes(wf, "CLIPTextEncode"):
        title = node.get("_meta", {}).get("title", "").lower()
        if "positive" in title:
            node["inputs"]["text"] = prompt
        if "negative" in title:
            node["inputs"]["text"] = negative

    ks_nodes = _find_nodes(wf, "KSampler")
    if not ks_nodes:
        raise RuntimeError("KSampler not found")
    ks_id, ks_node = ks_nodes[0]
    ks_node["inputs"].update({"steps": steps, "cfg": cfg, "seed": seed})
    if "sampler" in params:   ks_node["inputs"]["sampler_name"] = params["sampler"]
    if "scheduler" in params: ks_node["inputs"]["scheduler"] = params["scheduler"]
    if "denoise" in params:   ks_node["inputs"]["denoise"] = float(params["denoise"])

    for _, node in _find_nodes(wf, "EmptySD3LatentImage"):
        node["inputs"].update({"width": width, "height": height})

    # 2. Temukan titik awal untuk model dan clip
    ckpt_nodes = _find_nodes(wf, "CheckpointLoaderSimple")
    if not ckpt_nodes:
        raise RuntimeError("CheckpointLoaderSimple not found")
    ckpt_id, _ = ckpt_nodes[0]
    
    # 3. Sisipkan LoRA (sama seperti t2i_lora)
    loras = params.get("loras", []) or []
    last_model, last_clip = [ckpt_id, 0], [ckpt_id, 1]

    for l in loras:
        lora_loader_id = _next_id(wf)
        wf[lora_loader_id] = {
            "inputs": {
                "lora_name": l.get("lora_name"),
                "strength_model": float(l.get("strength_model", 0.8)),
                "strength_clip": float(l.get("strength_clip", 0.8)),
                "model": last_model,
                "clip": last_clip,
            },
            "class_type": "LoraLoader",
        }
        last_model, last_clip = [lora_loader_id, 0], [lora_loader_id, 1]
    
    # 4. Hubungkan ulang clip ke semua CLIPTextEncode
    for _, node in _find_nodes(wf, "CLIPTextEncode"):
        node["inputs"]["clip"] = last_clip

    # 5. Sisipkan ControlNets secara dinamis
    controls = params.get("controls", []) or []
    
    # Temukan sumber positive dan negative conditioning awal
    pos_prompt_id = _find_nodes(wf, "CLIPTextEncode")[0][0] # Asumsi pertama positif
    neg_prompt_id = _find_nodes(wf, "CLIPTextEncode")[1][0] # Asumsi kedua negatif
    
    curr_cond = [[pos_prompt_id, 0], [neg_prompt_id, 0]] # [positive, negative]

    for ctrl in controls:
        if not ctrl.get("image"): continue

        # Node A: Muat gambar (Base64)
        load_image_id = _next_id(wf)
        wf[load_image_id] = {
            "inputs": {"image": ctrl["image"]},
            "class_type": "ETN_LoadImageBase64",
        }

        # Node B: Preprocessor
        preprocessor_name = ctrl.get("preprocessor", "none").strip()
        if preprocessor_name == "none": continue
        
        pre_id = _next_id(wf)
        wf[pre_id] = {
            "inputs": {
                "preprocessor": preprocessor_name,
                "resolution": int(ctrl.get("resolution", 512)),
                "image": [load_image_id, 0],
            },
            "class_type": "AIO_Preprocessor",
        }
        
        # Node C: Muat model ControlNet
        cnl_id = _next_id(wf)
        model_name = "ConntrolnetUnionPro2.safetensors"
        wf[cnl_id] = {
            "inputs": {"control_net_name": model_name},
            "class_type": "ControlNetLoader",
        }
        
        # Node D: Terapkan ControlNet (Advanced)
        app_id = _next_id(wf)
        # ambil checkpoint id
        cp_id = _find_nodes(wf, "CheckpointLoaderSimple")[0][0]
        
        wf[app_id] = {
            "inputs": {
                "strength": float(ctrl.get("strength", 1.0)),
                "start_percent": float(ctrl.get("start_percent", 0.0)),
                "end_percent": float(ctrl.get("end_percent", 1.0)),
                "positive": curr_cond[0], # Ambil dari iterasi sebelumnya
                "negative": curr_cond[1], # Ambil dari iterasi sebelumnya
                "control_net": [cnl_id, 0],
                "image": [pre_id, 0],
                "vae": [cp_id, 2],
            },
            "class_type": "ControlNetApplyAdvanced",
        }
        # Perbarui conditioning untuk iterasi berikutnya
        curr_cond = [[app_id, 0], [app_id, 1]]

    # 6. Hubungkan kembali KSampler ke sumber akhir (setelah LoRA dan ControlNet)
    ks_node["inputs"]["model"] = last_model
    ks_node["inputs"]["positive"] = curr_cond[0]
    ks_node["inputs"]["negative"] = curr_cond[1]


    wf = apply_upscale(wf, params)
    wf = apply_faceswap(wf, params.get("faceswap", {}))
    return wf