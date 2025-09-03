def apply(workflow: dict, params: dict) -> dict:
    wf = workflow.copy()
    wf["6"]["inputs"]["text"] = params.get("prompt", "")
    # wf["33"]["inputs"]["text"] = params.get("negative", "")
    wf["31"]["inputs"]["steps"] = int(params.get("steps", 20))
    wf["31"]["inputs"]["cfg"] = float(params.get("cfg", 4.5))
    wf["31"]["inputs"]["sampler_name"] = params.get("sampler", "euler")
    wf["31"]["inputs"]["scheduler"] = params.get("scheduler", "simple")
    wf["31"]["inputs"]["seed"] = int(params.get("seed", -1))
    return wf