# app/patchers/t2a.py
import copy

def _next_id(wf):
    return str(max(map(int, wf.keys())) + 1)

def _find_nodes(wf, class_type):
    return [(nid, n) for nid, n in wf.items() if n.get("class_type") == class_type]

def _first(wf, class_type):
    xs = _find_nodes(wf, class_type)
    return xs[0] if xs else (None, None)

def _out(nid, idx=0): return [nid, idx]

def apply(workflow: dict, params: dict) -> dict:
    wf = copy.deepcopy(workflow)

    # --- Start LoRA Application ---
    ckpt_id, ckpt_node = _first(wf, "CheckpointLoaderSimple")
    if not ckpt_node:
        raise RuntimeError("CheckpointLoaderSimple not found")
        
    loras = params.get("loras", []) or []
    last_model, last_clip = _out(ckpt_id, 0), _out(ckpt_id, 1)

    for lora in loras:
        lora_loader_id = _next_id(wf)
        wf[lora_loader_id] = {
            "inputs": {
                "lora_name": lora.get("lora_name"),
                "strength_model": float(lora.get("strength_model", 0.8)),
                "strength_clip": float(lora.get("strength_clip", 0.8)),
                "model": last_model,
                "clip": last_clip,
            },
            "class_type": "LoraLoader",
            "_meta": {"title": f"LoRA: {lora.get('lora_name')}"}
        }
        last_model = _out(lora_loader_id, 0)
        last_clip = _out(lora_loader_id, 1)
    # --- End LoRA Application ---

    # 1) Text encoder (tags / lyrics / strength) - Connect to final clip output
    enc_id, enc = _first(wf, "TextEncodeAceStepAudio")
    if not enc:
        raise RuntimeError("TextEncodeAceStepAudio not found")
    enc["inputs"]["tags"] = params.get("tags", enc["inputs"].get("tags", ""))
    enc["inputs"]["lyrics"] = params.get("lyrics", enc["inputs"].get("lyrics", ""))
    if "lyrics_strength" in params:
        enc["inputs"]["lyrics_strength"] = float(params["lyrics_strength"])
    enc["inputs"]["clip"] = last_clip 

    # 2) Durasi (latent audio)
    la_id, la = _first(wf, "EmptyAceStepLatentAudio")
    if not la:
        raise RuntimeError("EmptyAceStepLatentAudio not found")
    if "seconds" in params:
        la["inputs"]["seconds"] = int(params["seconds"])

    # 3) KSampler (sampling audio)
    ks_id, ks = _first(wf, "KSampler")
    if not ks:
        raise RuntimeError("KSampler not found")
    if "steps" in params:      ks["inputs"]["steps"] = int(params["steps"])
    if "cfg" in params:        ks["inputs"]["cfg"] = float(params["cfg"])
    if "seed" in params:       ks["inputs"]["seed"] = int(params["seed"])
    if "sampler" in params:    ks["inputs"]["sampler_name"] = params["sampler"]
    if "scheduler" in params:  ks["inputs"]["scheduler"] = params["scheduler"]
    if "denoise" in params:    ks["inputs"]["denoise"] = float(params["denoise"])

    # 4) Opsional: latent ops and ModelSampling (connect to final model)
    ms_id, ms = _first(wf, "ModelSamplingSD3")
    if ms:
        ms["inputs"]["model"] = last_model
        if "model_sampling_shift" in params:
            ms["inputs"]["shift"] = float(params["model_sampling_shift"])

    tonemap_id, tonemap = _first(wf, "LatentOperationTonemapReinhard")
    if tonemap and "tonemap_reinhard_multiplier" in params:
        tonemap["inputs"]["multiplier"] = float(params["tonemap_reinhard_multiplier"])

    # 5) Output format & nama file
    save_mp3_id, save_mp3 = _first(wf, "SaveAudioMP3")
    if not save_mp3:
        raise RuntimeError("SaveAudioMP3 not found")

    fmt = params.get("format", "mp3").lower()
    if fmt == "mp3":
        if "mp3_quality" in params:
            save_mp3["inputs"]["quality"] = params["mp3_quality"]
        if "filename_prefix" in params:
            save_mp3["inputs"]["filename_prefix"] = params["filename_prefix"]
    elif fmt == "wav":
        wav_id, wav_node = _first(wf, "SaveAudioWAV")
        if wav_node:
            if "filename_prefix" in params:
                wav_node["inputs"]["filename_prefix"] = params["filename_prefix"]

    return wf
