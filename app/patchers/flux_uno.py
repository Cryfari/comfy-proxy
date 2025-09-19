import copy

def _find_first(wf, class_type):
    for nid, node in wf.items():
        if node.get("class_type") == class_type:
            return nid, node
    return None, None

def _next_id(wf):
    return str(max(map(int, wf.keys())) + 1)

def apply(workflow: dict, params: dict) -> dict:
    wf = copy.deepcopy(workflow)

    # 1. Cari node-node utama
    gen_id, gen_node = _find_first(wf, "UNOGenerate")
    loader_id, loader_node = _find_first(wf, "UNOModelLoader")

    if not gen_node or not loader_node:
        raise RuntimeError("UNOGenerate or UNOModelLoader not found in the preset")

    # 2. Update parameter utama di UNOGenerate
    gen_inputs = gen_node["inputs"]
    if "prompt" in params:   gen_inputs["prompt"] = str(params["prompt"])
    if "width" in params:    gen_inputs["width"] = int(params["width"])
    if "height" in params:   gen_inputs["height"] = int(params["height"])
    if "guidance" in params: gen_inputs["guidance"] = float(params["guidance"])
    if "steps" in params:    gen_inputs["num_steps"] = int(params["steps"])
    if "seed" in params:     gen_inputs["seed"] = int(params["seed"])

    # 3. Update model di UNOModelLoader (opsional)
    loader_inputs = loader_node["inputs"]
    if "flux_model" in params: loader_inputs["flux_model"] = str(params["flux_model"])
    if "lora_model" in params: loader_inputs["lora_model"] = str(params["lora_model"])

    # 4. Handle gambar referensi secara dinamis
    reference_images = params.get("reference_images", [])
    if not isinstance(reference_images, list):
        raise TypeError("reference_images must be a list of image paths/names")

    # Hapus input referensi default dari preset jika ada gambar baru
    if reference_images:
        keys_to_remove = [k for k in gen_inputs if k.startswith("reference_image_")]
        for k in keys_to_remove:
            gen_inputs.pop(k)

    # Tambahkan node loader untuk setiap gambar referensi
    for i, image_name in enumerate(reference_images):
        # Buat node LoadImage baru untuk setiap gambar
        new_loader_id = _next_id(wf)
        wf[new_loader_id] = {
            "inputs": {"image": image_name},
            "class_type": "LoadImage",
            "_meta": {"title": f"UNO Reference Image {i+1}"}
        }

        # Sambungkan node loader baru ke node UNOGenerate
        ref_key = f"reference_image_{i+1}"
        gen_inputs[ref_key] = [new_loader_id, 0]

    return wf