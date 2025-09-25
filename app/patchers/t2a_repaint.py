# app/patchers/t2a_repaint.py
import copy, os

def _next_id(wf):
    return str(max(map(int, wf.keys())) + 1)

def _find(wf, t): return [(nid,n) for nid,n in wf.items() if n.get("class_type")==t]
def _first(wf, t):
    xs = _find(wf, t); return xs[0] if xs else (None, None)

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


    # 1) audio referensi → LoadAudio.audio
    load_id, load = _first(wf, "LoadAudio")
    if not load: raise RuntimeError("LoadAudio not found")
    audio_ref = params.get("audio_ref")
    if not audio_ref:
        raise RuntimeError("Missing audio_ref")
    load["inputs"]["audio"] = audio_ref

    # 2) TextEncodeAceStepAudio (tags/lyrics/strength) - Connect to final clip
    enc_id, enc = _first(wf, "TextEncodeAceStepAudio")
    if not enc: raise RuntimeError("TextEncodeAceStepAudio not found")
    if "tags" in params:            enc["inputs"]["tags"] = params["tags"]
    if "lyrics" in params:          enc["inputs"]["lyrics"] = params["lyrics"]
    if "lyrics_strength" in params: enc["inputs"]["lyrics_strength"] = float(params["lyrics_strength"])
    enc["inputs"]["clip"] = last_clip

    # 3) KSampler
    ks_id, ks = _first(wf, "KSampler")
    if not ks: raise RuntimeError("KSampler not found")
    if "steps" in params:     ks["inputs"]["steps"] = int(params["steps"])
    if "cfg" in params:       ks["inputs"]["cfg"] = float(params["cfg"])
    if "seed" in params:      ks["inputs"]["seed"] = int(params["seed"])
    if "sampler" in params:   ks["inputs"]["sampler_name"] = params["sampler"]
    if "scheduler" in params: ks["inputs"]["scheduler"] = params["scheduler"]
    if "denoise" in params:   ks["inputs"]["denoise"] = float(params["denoise"])

    # 4) Opsional: tonemap & model sampling shift - Connect to final model
    tm_id, tm = _first(wf, "LatentOperationTonemapReinhard")
    if tm and "tonemap_reinhard_multiplier" in params:
        tm["inputs"]["multiplier"] = float(params["tonemap_reinhard_multiplier"])

    ms_id, ms = _first(wf, "ModelSamplingSD3")
    if ms:
        ms["inputs"]["model"] = last_model
        if "model_sampling_shift" in params:
            ms["inputs"]["shift"] = float(params["model_sampling_shift"])

    # 5) Output: SaveAudioMP3 (format/quality/prefix)
    sav_mp3_id, sav_mp3 = _first(wf, "SaveAudioMP3")
    if not sav_mp3: raise RuntimeError("SaveAudioMP3 not found")
    if params.get("format","mp3").lower() == "mp3":
        if "mp3_quality" in params:     sav_mp3["inputs"]["quality"] = params["mp3_quality"]
        if "filename_prefix" in params: sav_mp3["inputs"]["filename_prefix"] = params["filename_prefix"]
    else:
        # jika ingin dukung WAV, tambahkan SaveAudioWAV di preset & pilih di sini
        pass

    return wf
