def _next_id(wf):
    return str(max(map(int, wf.keys())) + 1 if wf else 1)

def _find_nodes(wf, class_type):
    return [(nid, node) for nid, node in wf.items() if node.get("class_type") == class_type]

def apply(workflow: dict, params: dict) -> dict:
    # deep copy aman (struktur datanya kecil)
    import copy
    wf = copy.deepcopy(workflow)

    # 1) Set basic t2i params (sesuaikan key sesuai preset kamu)
    prompt = params.get("prompt", "")
    negative = params.get("negative", "")
    steps = int(params.get("steps", 20))
    cfg = float(params.get("cfg", 4.5))
    seed = int(params.get("seed", -1))
    width = int(params.get("width", 512))
    height = int(params.get("height", 512))

    # CLIPTextEncode (pos/neg)
    for nid, node in _find_nodes(wf, "CLIPTextEncode"):
        title = node.get("_meta", {}).get("title", "").lower()
        if "positive" in title or node["inputs"].get("text", None) is not None:
            node["inputs"]["text"] = prompt
        if "negative" in title:
            node["inputs"]["text"] = negative

    # KSampler
    ks_nodes = _find_nodes(wf, "KSampler")
    if not ks_nodes:
        raise RuntimeError("KSampler not found in workflow")
    ks_id, ks_node = ks_nodes[0]
    ks_node["inputs"]["steps"] = steps
    ks_node["inputs"]["cfg"] = cfg
    ks_node["inputs"]["seed"] = seed

    # Resolusi (jika ada EmptySD3LatentImage / sejenis)
    for nid, node in _find_nodes(wf, "EmptySD3LatentImage"):
        node["inputs"]["width"] = width
        node["inputs"]["height"] = height

    # 2) Temukan checkpoint & clip/model asal
    ckpt_nodes = _find_nodes(wf, "CheckpointLoaderSimple")
    if not ckpt_nodes:
        raise RuntimeError("CheckpointLoaderSimple not found")
    ckpt_id, ckpt = ckpt_nodes[0]
    src_model = [ckpt_id, 0]
    src_clip  = [ckpt_id, 1]

    # 3) Sisipkan LoRA dinamis (jika ada)
    loras = params.get("loras", []) or []
    last_model, last_clip = src_model, src_clip

    for l in loras:
        new_id = _next_id(wf)
        wf[new_id] = {
            "inputs": {
                "lora_name": l.get("lora_name"),
                "strength_model": float(l.get("strength_model", 0.5)),
                "strength_clip":  float(l.get("strength_clip", 0.5)),
                "model": last_model,
                "clip":  last_clip,
            },
            "class_type": "LoraLoader",
            "_meta": {"title": "Load LoRA (dynamic)"}
        }
        # keluaran LoraLoader: 0=model, 1=clip → jadikan sumber berikutnya
        last_model = [new_id, 0]
        last_clip  = [new_id, 1]

    # 4) Rewire KSampler.model dan semua CLIPTextEncode.clip ke sumber akhir
    ks_node["inputs"]["model"] = last_model
    for nid, node in _find_nodes(wf, "CLIPTextEncode"):
        node["inputs"]["clip"] = last_clip

    return wf
