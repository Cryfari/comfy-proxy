# faceswap_dyn.py
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
            "detect_gender_input": faceswap.get("detect_gender_input", "auto"),
            "detect_gender_source": faceswap.get("detect_gender_source", "auto"),
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
