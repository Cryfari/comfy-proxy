# File: app/patchers/t2i.py
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
    # Anda bisa sesuaikan nama model default di sini
    return "4x-ESRGAN.safetensors"

def _get_decode_output(wf):
    # cari VAEDecode → output 0 adalah image
    vd_id, vd = _first_node(wf, "VAEDecode")
    if not vd:
        raise RuntimeError("VAEDecode not found")
    return _out(vd_id, 0)

def _rewire_sinks_to(wf, img_ref):
    for _, node in wf.items():
        if node.get("class_type") in ("SaveImage", "ETN_SendImageWebSocket"):
            if "images" in node.get("inputs", {}):
                node["inputs"]["images"] = img_ref

def apply_faceswap(wf: dict, faceswap: dict) -> dict:
    """faceswap: dict dari params['faceswap']"""
    enabled = bool(faceswap.get("enabled", False))
    if not enabled:
        return wf # Jangan rewire jika tidak aktif, biarkan upscale yang mengatur

    # 1) muat source face
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
            },
            "_meta": {"title": "ReActor Face Booster (dyn)"}
        }
        boost_ref = _out(boost_id, 0)

    # 3) FaceSwap node
    swap_id = _next_id(wf)
    
    # Ambil output dari upscale atau VAE decode sebagai input
    up_mix_id, _ = _first_node(wf, "JWImageMix")
    input_image_ref = _out(up_mix_id, 0) if up_mix_id else _get_decode_output(wf)

    wf[swap_id] = {
        "class_type": "ReActorFaceSwap",
        "inputs": {
            "enabled": True,
            "swap_model": faceswap.get("swap_model", "inswapper_128.onnx"),
            "facedetection": faceswap.get("facedetection", "retinaface_resnet50"),
            "face_restore_model": faceswap.get("face_restore_model", "none"),
            "face_restore_visibility": float(faceswap.get("face_restore_visibility", 1.0)),
            "codeformer_weight": float(faceswap.get("codeformer_weight", 0.5)),
            "input_faces_index": str(faceswap.get("input_faces_index", "0")),
            "source_faces_index": str(faceswap.get("source_faces_index", "0")),
            "input_image": input_image_ref,
            "source_image": _out(load_id, 0),
            "face_boost": boost_ref,
        },
        "_meta": {"title": "ReActor Face Swap (dyn)"}
    }

    # 4) arahkan SaveImage/SendImage ke hasil FaceSwap
    _rewire_sinks_to(wf, _out(swap_id, 0))
    return wf

def apply_upscale(wf: dict, params: dict) -> dict:
    up = params.get("upscale") or {}
    enabled = bool(up.get("enabled", False))
    
    base_image_ref = _get_decode_output(wf)

    if not enabled:
        # Tanpa upscale, arahkan sink ke output VAE Decode
        _rewire_sinks_to(wf, base_image_ref)
        return wf

    # --- Implementasi Alur Kerja Upscale dengan Blending ---

    scale_factor = int(up.get("scale", 2))
    model_name = up.get("model_name") or _default_upscale_model(scale_factor)
    blend_factor = float(up.get("blend", 0.5))
    method = up.get("method", "lanczos")

    # 1. Dapatkan ukuran gambar asli
    get_size_id = _next_id(wf)
    wf[get_size_id] = {
        "class_type": "DF_Get_image_size",
        "inputs": {"image": base_image_ref},
        "_meta": {"title": "Get image size (dyn)"}
    }

    # 2. Buat node untuk faktor upscale
    factor_id = _next_id(wf)
    wf[factor_id] = {
        "class_type": "Int Literal",
        "inputs": {"int": scale_factor},
        "_meta": {"title": "Upscale Factor (dyn)"}
    }

    # 3. Hitung lebar dan tinggi target
    mul_w_id = _next_id(wf)
    wf[mul_w_id] = {
        "class_type": "JWIntegerMul",
        "inputs": {"a": _out(get_size_id, 0), "b": _out(factor_id, 0)},
        "_meta": {"title": "Width Multiplier (dyn)"}
    }
    mul_h_id = _next_id(wf)
    wf[mul_h_id] = {
        "class_type": "JWIntegerMul",
        "inputs": {"a": _out(get_size_id, 1), "b": _out(factor_id, 0)},
        "_meta": {"title": "Height Multiplier (dyn)"}
    }

    # 4. Muat model upscaler
    loader_id = _next_id(wf)
    wf[loader_id] = {
        "class_type": "UpscaleModelLoader",
        "inputs": {"model_name": model_name},
        "_meta": {"title": "Load Upscale Model (dyn)"}
    }

    # 5. Lakukan upscale pada gambar asli
    upscale_node_id = _next_id(wf)
    wf[upscale_node_id] = {
        "class_type": "ImageUpscaleWithModel",
        "inputs": {"upscale_model": _out(loader_id, 0), "image": base_image_ref},
        "_meta": {"title": "Upscale with Model (dyn)"}
    }

    # 6. Skalakan gambar yang sudah di-upscale ke dimensi target (untuk konsistensi)
    scale_upscaled_id = _next_id(wf)
    wf[scale_upscaled_id] = {
        "class_type": "ImageScale",
        "inputs": {
            "upscale_method": method,
            "width": _out(mul_w_id, 0), "height": _out(mul_h_id, 0),
            "crop": "disabled", "image": _out(upscale_node_id, 0)
        },
        "_meta": {"title": "Scale Upscaled Image (dyn)"}
    }

    # 7. Skalakan gambar asli ke dimensi target
    scale_original_id = _next_id(wf)
    wf[scale_original_id] = {
        "class_type": "ImageScale",
        "inputs": {
            "upscale_method": method,
            "width": _out(mul_w_id, 0), "height": _out(mul_h_id, 0),
            "crop": "disabled", "image": base_image_ref
        },
        "_meta": {"title": "Scale Original Image (dyn)"}
    }

    # 8. Campurkan (blend) kedua gambar yang sudah diskalakan
    mix_id = _next_id(wf)
    wf[mix_id] = {
        "class_type": "JWImageMix",
        "inputs": {
            "blend_type": "mix", "factor": blend_factor,
            "image_a": _out(scale_upscaled_id, 0),
            "image_b": _out(scale_original_id, 0)
        },
        "_meta": {"title": "Image Mix (dyn)"}
    }

    # 9. Arahkan sink (SaveImage, dll) ke hasil akhir dari mix
    _rewire_sinks_to(wf, _out(mix_id, 0))
    
    return wf


def apply(workflow: dict, params: dict) -> dict:
    wf = copy.deepcopy(workflow)

    # 1. Atur parameter dasar
    prompt = params.get("prompt", "")
    negative = params.get("negative", "")
    steps = int(params.get("steps", 20))
    cfg = float(params.get("cfg", 4.5))
    seed = int(params.get("seed", -1))
    width = int(params.get("width", 512))
    height = int(params.get("height", 512))

    for _, node in _find_nodes(wf, "CLIPTextEncode"):
        title = node.get("_meta", {}).get("title", "").lower()
        if "positive" in title: node["inputs"]["text"] = prompt
        if "negative" in title: node["inputs"]["text"] = negative

    ks_id, ks_node = _first_node(wf, "KSampler")
    if not ks_node: raise RuntimeError("KSampler not found")
    ks_node["inputs"].update({"steps": steps, "cfg": cfg, "seed": seed})
    if "sampler" in params:   ks_node["inputs"]["sampler_name"] = params["sampler"]
    if "scheduler" in params: ks_node["inputs"]["scheduler"] = params["scheduler"]
    if "denoise" in params:   ks_node["inputs"]["denoise"] = float(params["denoise"])

    empty_latent_id, empty_latent_node = _first_node(wf, "EmptySD3LatentImage")
    if empty_latent_node:
        empty_latent_node["inputs"].update({"width": width, "height": height})

    # 2. Temukan titik awal model dan clip
    ckpt_id, _ = _first_node(wf, "CheckpointLoaderSimple")
    if not ckpt_id: raise RuntimeError("CheckpointLoaderSimple not found")
    
    # 3. Sisipkan LoRA
    loras = params.get("loras", []) or []
    last_model, last_clip = _out(ckpt_id, 0), _out(ckpt_id, 1)

    for l in loras:
        lora_loader_id = _next_id(wf)
        wf[lora_loader_id] = {
            "inputs": {
                "lora_name": l.get("lora_name"),
                "strength_model": float(l.get("strength_model", 0.8)),
                "strength_clip": float(l.get("strength_clip", 0.8)),
                "model": last_model, "clip": last_clip,
            },
            "class_type": "LoraLoader",
        }
        last_model, last_clip = _out(lora_loader_id, 0), _out(lora_loader_id, 1)
    
    # 4. Hubungkan ulang clip ke semua CLIPTextEncode
    for _, node in _find_nodes(wf, "CLIPTextEncode"):
        node["inputs"]["clip"] = last_clip

    # 5. Sisipkan ControlNets secara dinamis
    controls = params.get("controls", []) or []
    
    pos_prompt_id, _ = [n for n in _find_nodes(wf, "CLIPTextEncode") if "positive" in n[1].get("_meta",{}).get("title","").lower()][0]
    neg_prompt_id, _ = [n for n in _find_nodes(wf, "CLIPTextEncode") if "negative" in n[1].get("_meta",{}).get("title","").lower()][0]
    
    curr_cond = [_out(pos_prompt_id, 0), _out(neg_prompt_id, 0)]

    for ctrl in controls:
        if not ctrl.get("image"): continue

        load_image_id = _next_id(wf)
        wf[load_image_id] = { "inputs": {"image": ctrl["image"]}, "class_type": "ETN_LoadImageBase64" }

        preprocessor_name = ctrl.get("preprocessor", "none").strip()
        if preprocessor_name == "none": continue
        
        pre_id = _next_id(wf)
        wf[pre_id] = {
            "inputs": { "preprocessor": preprocessor_name, "resolution": int(ctrl.get("resolution", 512)), "image": _out(load_image_id, 0) },
            "class_type": "AIO_Preprocessor",
        }
        
        cnl_id = _next_id(wf)
        wf[cnl_id] = { "inputs": {"control_net_name": "ControlnetUnionPro2.safetensors"}, "class_type": "ControlNetLoader" }
        
        app_id = _next_id(wf)
        wf[app_id] = {
            "inputs": {
                "strength": float(ctrl.get("strength", 1.0)), "start_percent": float(ctrl.get("start_percent", 0.0)), "end_percent": float(ctrl.get("end_percent", 1.0)),
                "positive": curr_cond[0], "negative": curr_cond[1],
                "control_net": _out(cnl_id, 0), "image": _out(pre_id, 0),
                "vae": _out(ckpt_id, 2),
            },
            "class_type": "ControlNetApplyAdvanced",
        }
        curr_cond = [_out(app_id, 0), _out(app_id, 1)]

    # 6. Hubungkan KSampler ke sumber akhir (setelah LoRA dan ControlNet)
    ks_node["inputs"]["model"] = last_model
    ks_node["inputs"]["positive"] = curr_cond[0]
    ks_node["inputs"]["negative"] = curr_cond[1]

    # 7. Terapkan upscale dan faceswap (dalam urutan ini)
    wf = apply_upscale(wf, params)
    wf = apply_faceswap(wf, params.get("faceswap", {}))
    return wf
