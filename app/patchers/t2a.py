# app/patchers/t2a.py
import copy

def _find_nodes(wf, class_type):
    return [(nid, n) for nid, n in wf.items() if n.get("class_type") == class_type]

def _first(wf, class_type):
    xs = _find_nodes(wf, class_type)
    return xs[0] if xs else (None, None)

def apply(workflow: dict, params: dict) -> dict:
    wf = copy.deepcopy(workflow)

    # 1) Text encoder (tags / lyrics / strength)
    enc_id, enc = _first(wf, "TextEncodeAceStepAudio")
    if not enc:
        raise RuntimeError("TextEncodeAceStepAudio not found")
    enc["inputs"]["tags"] = params.get("tags", enc["inputs"].get("tags", ""))
    enc["inputs"]["lyrics"] = params.get("lyrics", enc["inputs"].get("lyrics", ""))
    if "lyrics_strength" in params:
        enc["inputs"]["lyrics_strength"] = float(params["lyrics_strength"])

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

    # 4) Opsional: latent ops
    tonemap_id, tonemap = _first(wf, "LatentOperationTonemapReinhard")
    if tonemap and "tonemap_reinhard_multiplier" in params:
        tonemap["inputs"]["multiplier"] = float(params["tonemap_reinhard_multiplier"])

    ms_id, ms = _first(wf, "ModelSamplingSD3")
    if ms and "model_sampling_shift" in params:
        ms["inputs"]["shift"] = float(params["model_sampling_shift"])

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
        # jika ingin dukung WAV juga, kamu bisa clone/pasang SaveAudioWAV di preset dasar
        # dan pilih salah satunya di sini (atau buat node baru bila perlu).
    elif fmt == "wav":
        # jika preset punya SaveAudioWAV, set di situ & arahkan sink ke WAV
        wav_id, wav_node = _first(wf, "SaveAudioWAV")
        if not wav_node:
            # fallback: tetap mp3 jika node wav belum tersedia
            pass
        else:
            if "filename_prefix" in params:
                wav_node["inputs"]["filename_prefix"] = params["filename_prefix"]
            # pastikan jalur decode -> wav sudah benar (biasanya sama seperti mp3)

    return wf
