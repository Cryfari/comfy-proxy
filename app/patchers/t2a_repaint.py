# app/patchers/t2a_repaint.py
import copy, os

def _find(wf, t): return [(nid,n) for nid,n in wf.items() if n.get("class_type")==t]
def _first(wf, t):
    xs = _find(wf, t); return xs[0] if xs else (None, None)

def apply(workflow: dict, params: dict) -> dict:
    wf = copy.deepcopy(workflow)

    # 1) audio referensi → LoadAudio.audio
    load_id, load = _first(wf, "LoadAudio")
    if not load: raise RuntimeError("LoadAudio not found")
    audio_ref = params.get("audio_ref")
    if not audio_ref:
        raise RuntimeError("Missing audio_ref")
    load["inputs"]["audio"] = audio_ref

    # 2) TextEncodeAceStepAudio (tags/lyrics/strength)
    enc_id, enc = _first(wf, "TextEncodeAceStepAudio")
    if not enc: raise RuntimeError("TextEncodeAceStepAudio not found")
    if "tags" in params:            enc["inputs"]["tags"] = params["tags"]
    if "lyrics" in params:          enc["inputs"]["lyrics"] = params["lyrics"]
    if "lyrics_strength" in params: enc["inputs"]["lyrics_strength"] = float(params["lyrics_strength"])

    # 3) KSampler
    ks_id, ks = _first(wf, "KSampler")
    if not ks: raise RuntimeError("KSampler not found")
    if "steps" in params:     ks["inputs"]["steps"] = int(params["steps"])
    if "cfg" in params:       ks["inputs"]["cfg"] = float(params["cfg"])
    if "seed" in params:      ks["inputs"]["seed"] = int(params["seed"])
    if "sampler" in params:   ks["inputs"]["sampler_name"] = params["sampler"]
    if "scheduler" in params: ks["inputs"]["scheduler"] = params["scheduler"]
    if "denoise" in params:   ks["inputs"]["denoise"] = float(params["denoise"])

    # 4) Opsional: tonemap & model sampling shift
    tm_id, tm = _first(wf, "LatentOperationTonemapReinhard")
    if tm and "tonemap_reinhard_multiplier" in params:
        tm["inputs"]["multiplier"] = float(params["tonemap_reinhard_multiplier"])

    ms_id, ms = _first(wf, "ModelSamplingSD3")
    if ms and "model_sampling_shift" in params:
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
