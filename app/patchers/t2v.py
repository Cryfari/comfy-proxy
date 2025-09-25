# app/patchers/t2v.py
import copy
from math import ceil

def _find(wf, t): return [(nid, n) for nid, n in wf.items() if n.get("class_type") == t]
def _first(wf, t):
    xs = _find(wf, t); return xs[0] if xs else (None, None)

def _next_id(wf): return str(max(map(int, wf.keys())) + 1 if wf else 1)
def _out(nid, idx=0): return [nid, idx]

def _ensure_load_image(wf, image_name=None, image_b64=None):
    """Tanpa resize: pakai LoadImage (path) atau loader base64 (opsional)."""
    if image_name:
        nid, node = _first(wf, "LoadImage")
        if nid and "image" in wf[nid]["inputs"]:
            # pakai yang ada
            wf[nid]["inputs"]["image"] = image_name
            return nid
        # buat baru bila tidak ada
        nid = _next_id(wf)
        wf[nid] = {
            "class_type": "LoadImage",
            "inputs": {"image": image_name},
            "_meta": {"title": "T2I2V: LoadImage"}
        }
        return nid

def _ensure_vae_encode(wf, image_ref):
    """Encode image ke latent pakai VAE yang sudah diload di preset."""
    # pakai VAELoader yang ada
    vae_id, vae = _first(wf, "VAELoader")
    if not vae:
        # fallback: beberapa preset pakai "VAELoaderWAN" / dll—silakan tambah di sini sesuai presetmu
        vae_id, vae = _first(wf, "VAELoaderWAN")
    if not vae:
        raise RuntimeError("VAELoader not found for I2V encode")

    enc_id = _next_id(wf)
    wf[enc_id] = {
        "class_type": "VAEEncode",
        "inputs": {"pixels": image_ref, "vae": _out(vae_id, 0)},
        "_meta": {"title": "I2V: VAEEncode"}
    }
    return enc_id

def _wire_i2v_into_latent(wf, lat_node_id, init_image_ref=None, init_latent_ref=None):
    """Sambungkan ke Wan22ImageToVideoLatent: dukung berbagai nama field yang mungkin."""
    lat_inputs = wf[lat_node_id]["inputs"]
    # beberapa varian field yang sering dipakai
    # prioritas: latent jika disediakan
    if init_latent_ref:
        for k in ("latent", "init_latent", "latent_image", "image_latent"):
            if k in lat_inputs:
                lat_inputs[k] = init_latent_ref
                break
            else:
                # jika tidak ada fieldnya, tambahkan (aman untuk ComfyUI)
                lat_inputs["init_latent"] = init_latent_ref
    elif init_image_ref:
        for k in ("image", "images", "first_frame", "init_image"):
            if k in lat_inputs:
                lat_inputs[k] = init_image_ref
                break
        else:
            lat_inputs["init_image"] = init_image_ref

def _ensure_clip_vision(wf, image_ref):
    # Pakai CLIPVisionLoader/Encode yang sudah ada kalau ada; kalau tidak, buat baru.
    cvl_id, cvl = _first(wf, "CLIPVisionLoader")
    if not cvl:
        cvl_id = _next_id(wf)
        wf[cvl_id] = {"class_type": "CLIPVisionLoader", "inputs": {"clip_name": "clip_vision_h.safetensors"}, "_meta": {"title": "CLIP Vision"}}
    cve_id, cve = _first(wf, "CLIPVisionEncode")
    if not cve:
        cve_id = _next_id(wf)
        wf[cve_id] = {"class_type": "CLIPVisionEncode", "inputs": {"clip_vision": _out(cvl_id, 0), "image": image_ref, "crop": "none"}, "_meta": {"title": "CLIP Vision Encode"}}
    else:
        wf[cve_id]["inputs"]["clip_vision"] = _out(cvl_id, 0)
        wf[cve_id]["inputs"]["image"] = image_ref
    return cve_id

def _wire_createvideo_savevideo(wf, images_ref):
    cv_id, cv = _first(wf, "CreateVideo")
    sv_id, sv = _first(wf, "SaveVideo")
    if not cv or not sv:
        raise RuntimeError("CreateVideo/SaveVideo tidak ditemukan")
    # beberapa node pakai 'images' utk CreateVideo
    cv["inputs"]["images"] = images_ref
    # SaveVideo prioritas 'video', fallback 'images'
    if "video" in sv["inputs"]:
        sv["inputs"]["video"] = _out(cv_id, 0)
    else:
        sv["inputs"]["images"] = images_ref


def apply(workflow: dict, params: dict) -> dict:
    wf = copy.deepcopy(workflow)


    unet_id, unet = _first(wf, "UNETLoader")
    if unet and "model" in params:
        unet["inputs"]["unet_name"] = params["model"]
        
    # (1) Prompts
    for nid, node in _find(wf, "CLIPTextEncode"):
        title = node.get("_meta", {}).get("title","").lower()
        if "positive" in title:
            node["inputs"]["text"] = params.get("prompt", node["inputs"].get("text",""))
        elif "negative" in title:
            node["inputs"]["text"] = params.get("negative", node["inputs"].get("text",""))

    # (2) WanImageToVideo (latent config)
    lat_id, lat = _first(wf, "WanImageToVideo")
    if not lat:
        raise RuntimeError("WanImageToVideo not found")

    if "width" in params:  lat["inputs"]["width"]  = int(params["width"])
    if "height" in params: lat["inputs"]["height"] = int(params["height"])
    fps = int(params.get("fps", 24))
    length = params.get("length")
    seconds = params.get("seconds")
    if length is None and seconds is not None:
        length = int(fps * float(seconds)) + 1
    if length is not None:
        lat["inputs"]["length"] = int(length)
    if "batch_size" in params:
        lat["inputs"]["batch_size"] = int(params["batch_size"])

    # (3) KSampler
    ks_id, ks = _first(wf, "KSampler")
    if not ks:
        raise RuntimeError("KSampler not found")
    if "steps" in params:     ks["inputs"]["steps"] = int(params["steps"])
    if "cfg" in params:       ks["inputs"]["cfg"] = float(params["cfg"])
    if "seed" in params:      ks["inputs"]["seed"] = int(params["seed"])
    if "sampler" in params:   ks["inputs"]["sampler_name"] = params["sampler"]
    if "scheduler" in params: ks["inputs"]["scheduler"] = params["scheduler"]
    if "denoise" in params:   ks["inputs"]["denoise"] = float(params["denoise"])

    # (4) ModelSamplingSD3 (shift)
    ms_id, ms = _first(wf, "ModelSamplingSD3")
    if ms and "model_shift" in params:
        ms["inputs"]["shift"] = float(params["model_shift"])

    # (5) I2V (opsional) — perhatikan: HARUS isi start_image DAN clip_vision_output bila enabled
    i2v = params.get("i2v") or {}
    if i2v.get("enabled"):
        # siapkan path/b64
        img_name = i2v.get("image_ref")

        loader_id = _ensure_load_image(wf, image_name=img_name)
        start_ref = _out(loader_id, 0)
        lat["inputs"]["start_image"] = start_ref

        # pastikan CLIP-Vision encode terhubung → set clip_vision_output
        cve_id = _ensure_clip_vision(wf, start_ref)
        lat["inputs"]["clip_vision_output"] = _out(cve_id, 0)
        ul_id, ul = _first(wf, "UNETLoader")
        if ul and "model" not in params:
            ul["inputs"]["unet_name"] = "wan2.1_i2v_720p_14B_fp16.safetensors"
    else:
        # T2V murni → HAPUS keys ini agar tidak [null,0]
        lat["inputs"].pop("start_image", None)
        lat["inputs"].pop("clip_vision_output", None)

    # (6) Rewire output video
    # Jika kamu pakai upscale dinamis: tentukan source CreateVideo.images lewat fungsi upscale-mu
    # a) tanpa upscale → VAEDecode(8) langsung
    vd_id, vd = _first(wf, "VAEDecode")
    base_frames = _out(vd_id, 0) if vd else None

    # Panggil util upscale-mu (yang memakai ImageUpscaleWithModel → ImageScale → JWImageMix):
    wf = _apply_video_upscale(wf, params)   # <- fungsi yang sudah kamu punya
    # Jika belum, minimal pastikan CreateVideo → SaveVideo terhubung:
    # mix_id, mix = _first(wf, "JWImageMix")
    # if mix:
    #     _wire_createvideo_savevideo(wf, _out(mix_id, 0))
    # elif base_frames:
    #     _wire_createvideo_savevideo(wf, base_frames)

    # (7) Set fps di CreateVideo kalau diubah
    cv_id, cv = _first(wf, "CreateVideo")
    if cv:
        cv["inputs"]["fps"] = fps
    
    

    return wf
def _apply_video_upscale(wf, params):
    up = (params.get("upscale") or {})
    if not up.get("enabled"): 
        # bypass: CreateVideo ambil dari VAEDecode
        vd_id, vd = _first(wf, "VAEDecode")
        cv_id, cv = _first(wf, "CreateVideo")
        sv_id, sv = _first(wf, "SaveVideo")
        if not (vd and cv and sv): return wf
        cv["inputs"]["images"] = _out(vd_id, 0)
        sv["inputs"]["video"] = _out(cv_id, 0) if "video" in sv["inputs"] else sv["inputs"].get("images", _out(cv_id,0))
        return wf

    # 0) node wajib (sudah ada di t2v)
    vd_id, vd = _first(wf, "VAEDecode")
    cv_id, cv = _first(wf, "CreateVideo")
    sv_id, sv = _first(wf, "SaveVideo")
    if not (vd and cv and sv):
        raise RuntimeError("VAEDecode/CreateVideo/SaveVideo tidak ditemukan")

    # 1) muat model upscaler
    loader_id, loader = _first(wf, "UpscaleModelLoader")
    if not loader:
        loader_id = _next_id(wf)
        wf[loader_id] = {
            "class_type": "UpscaleModelLoader",
            "inputs": {"model_name": up.get("model_name","4x-ESRGAN.safetensors")},
            "_meta": {"title": "Load Upscale Model (dyn)"}
        }
    else:
        wf[loader_id]["inputs"]["model_name"] = up.get("model_name", wf[loader_id]["inputs"].get("model_name"))

    # 2) jalankan upscaler pada frame dari VAEDecode
    up_node_id = _next_id(wf)
    wf[up_node_id] = {
        "class_type": "ImageUpscaleWithModel",
        "inputs": {"upscale_model": _out(loader_id, 0), "image": _out(vd_id, 0)},
        "_meta": {"title": "Upscale Image (dyn)"}
    }

    # 3) hitung target width/height = size_asli * factor
    get_size_id = _next_id(wf)
    wf[get_size_id] = {
        "class_type": "DF_Get_image_size",
        "inputs": {"image": _out(vd_id, 0)},
        "_meta": {"title": "Get image size (dyn)"}
    }
    factor_id = _next_id(wf)
    wf[factor_id] = {
        "class_type": "Int Literal",
        "inputs": {"int": int(up.get("factor", 2))},
        "_meta": {"title": "Upscale Factor (dyn)"}
    }
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

    # 4) samakan resolusi kedua jalur: upscaled & original → target (agar bisa di-mix)
    up_scaled_id = _next_id(wf)
    wf[up_scaled_id] = {
        "class_type": "ImageScale",
        "inputs": {
            "upscale_method": up.get("method","lanczos"),
            "width": _out(mul_w_id, 0),
            "height": _out(mul_h_id, 0),
            "crop": "disabled",
            "image": _out(up_node_id, 0)
        },
        "_meta": {"title": "Upscale Image (to target) (dyn)"}
    }
    base_scaled_id = _next_id(wf)
    wf[base_scaled_id] = {
        "class_type": "ImageScale",
        "inputs": {
            "upscale_method": up.get("method","lanczos"),
            "width": _out(mul_w_id, 0),
            "height": _out(mul_h_id, 0),
            "crop": "disabled",
            "image": _out(vd_id, 0)
        },
        "_meta": {"title": "Base Image (to target) (dyn)"}
    }

    # 5) blend dua jalur (sesuai contoh)
    mix_id = _next_id(wf)
    wf[mix_id] = {
        "class_type": "JWImageMix",
        "inputs": {
            "blend_type": "mix",
            "factor": float(up.get("blend", 0.5)),   # 0..1
            "image_a": _out(up_scaled_id, 0),
            "image_b": _out(base_scaled_id, 0)
        },
        "_meta": {"title": "Image Mix (dyn)"}
    }

    print('mix id', mix_id)

    # 6) rewire output video: CreateVideo ← mix; SaveVideo ← CreateVideo
    cv["inputs"]["fps"] = cv["inputs"].get("fps", 24)
    cv["inputs"]["images"] = _out(mix_id, 0)
    print('cv', cv['inputs'])
    if "video" in sv["inputs"]:
        sv["inputs"]["video"] = _out(cv_id, 0)
    else:
        sv["inputs"]["images"] = _out(mix_id, 0)
    print('sv input', sv['inputs'])
    return wf