# app/integrations/framepack_runner.py
"""
Runtime FramePack untuk FastAPI:
- Memuat model-model sekali (lazy-init).
- Menjalankan job secara async, mengirim progress & file output.
- API: start_job(), stop_job(), get_status(), get_progress_queue(job_id).
"""

import os, asyncio, traceback, time, uuid
from pathlib import Path
from typing import Dict, Optional, Any

# ==== ENV & HF cache dir (sesuai demo) ====
HF_HOME = os.path.abspath(os.path.realpath(os.path.join(os.path.dirname(__file__), "../hf_download")))
os.environ.setdefault("HF_HOME", HF_HOME)

# ==== Torch & lib eksternal FramePack ====
import torch
import einops
import numpy as np
from PIL import Image

# === Komponen dari demo (di file kamu) ===
# Catatan: modul2 di bawah berasal dari paket "diffusers_helper" dan transformers/diffusers yang sama
from diffusers import AutoencoderKLHunyuanVideo
from transformers import LlamaModel, CLIPTextModel, LlamaTokenizerFast, CLIPTokenizer, SiglipVisionModel
from diffusers_helper.hunyuan import encode_prompt_conds, vae_decode, vae_encode, vae_decode_fake
from diffusers_helper.utils import (
    save_bcthw_as_mp4, crop_or_pad_yield_mask, soft_append_bcthw,
    resize_and_center_crop, generate_timestamp
)
from diffusers_helper.models.hunyuan_video_packed import HunyuanVideoTransformer3DModelPacked
from diffusers_helper.pipelines.k_diffusion_hunyuan import sample_hunyuan
from diffusers_helper.memory import (
    cpu, gpu, get_cuda_free_memory_gb,
    move_model_to_device_with_memory_preservation, offload_model_from_device_for_memory_preservation,
    unload_complete_models, load_model_as_complete, fake_diffusers_current_device, DynamicSwapInstaller
)
from diffusers_helper.clip_vision import hf_clip_vision_encode
from transformers import SiglipImageProcessor
from diffusers_helper.bucket_tools import find_nearest_bucket
from diffusers_helper.thread_utils import AsyncStream, async_run

# ====== Direktori output ======
OUTPUTS_DIR = Path("./outputs").resolve()
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

# ====== State job sederhana ======
class JobState:
    def __init__(self, job_id: str):
        self.id = job_id
        self.created_at = time.time()
        self.status = "queued"      # queued | running | done | error | stopped
        self.error: Optional[str] = None
        self.latest_preview: Optional[bytes] = None  # PNG bytes (opsional)
        self.latest_desc: str = ""
        self.output_path: Optional[str] = None

# Registry global untuk job
JOBS: Dict[str, JobState] = {}
QUEUES: Dict[str, AsyncStream] = {}  # gunakan AsyncStream dari demo untuk progress

# ====== Model global (lazy) ======
_models_loaded = False
high_vram = False

# Placeholder objek model
text_encoder = None
text_encoder_2 = None
tokenizer = None
tokenizer_2 = None
vae = None
feature_extractor = None
image_encoder = None
transformer = None

def _lazy_init_models():
    global _models_loaded, high_vram
    global text_encoder, text_encoder_2, tokenizer, tokenizer_2, vae, feature_extractor, image_encoder, transformer

    if _models_loaded:
        return

    free_mem_gb = get_cuda_free_memory_gb(gpu)
    high_vram = free_mem_gb > 60

    # === LOAD persis seperti demo ===
    # (Model repositori mengikuti demo kamu)
    text_encoder      = LlamaModel.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='text_encoder',   torch_dtype=torch.float16).cpu()
    text_encoder_2    = CLIPTextModel.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='text_encoder_2', torch_dtype=torch.float16).cpu()
    tokenizer         = LlamaTokenizerFast.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='tokenizer')
    tokenizer_2       = CLIPTokenizer.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='tokenizer_2')
    vae               = AutoencoderKLHunyuanVideo.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='vae', torch_dtype=torch.float16).cpu()

    feature_extractor = SiglipImageProcessor.from_pretrained("lllyasviel/flux_redux_bfl", subfolder='feature_extractor')
    image_encoder     = SiglipVisionModel.from_pretrained("lllyasviel/flux_redux_bfl", subfolder='image_encoder', torch_dtype=torch.float16).cpu()

    transformer       = HunyuanVideoTransformer3DModelPacked.from_pretrained('lllyasviel/FramePackI2V_HY', torch_dtype=torch.bfloat16).cpu()

    # Eval & dtype set
    for m in (vae, text_encoder, text_encoder_2, image_encoder, transformer):
        m.eval()
    if not high_vram:
        vae.enable_slicing(); vae.enable_tiling()

    transformer.high_quality_fp32_output_for_inference = True
    transformer.to(dtype=torch.bfloat16)
    vae.to(dtype=torch.float16)
    image_encoder.to(dtype=torch.float16)
    text_encoder.to(dtype=torch.float16)
    text_encoder_2.to(dtype=torch.float16)

    for m in (vae, text_encoder, text_encoder_2, image_encoder, transformer):
        m.requires_grad_(False)

    if not high_vram:
        # offload cepat (sesuai demo)
        DynamicSwapInstaller.install_model(transformer, device=gpu)
        DynamicSwapInstaller.install_model(text_encoder, device=gpu)
    else:
        text_encoder.to(gpu); text_encoder_2.to(gpu)
        image_encoder.to(gpu); vae.to(gpu); transformer.to(gpu)

    _models_loaded = True

async def _run_job(job: JobState, stream: AsyncStream, params: Dict[str, Any]):
    """
    Core FramePack worker – diadaptasi dari demo_gradio.py
    """
    _lazy_init_models()

    # Ambil parameter
    input_image_path = params["image_path"]  # path absolut image upload
    prompt           = params.get("prompt", "")
    n_prompt         = params.get("negative", "")
    seed             = int(params.get("seed", 31337))
    total_sec        = float(params.get("seconds", 5))
    latent_ws        = int(params.get("latent_window_size", 9))  # jangan diubah
    steps            = int(params.get("steps", 25))
    cfg              = float(params.get("cfg", 1.0))   # sebaiknya 1.0 (distilled)
    gs               = float(params.get("gs", 10.0))
    rs               = float(params.get("rs", 0.0))
    preserve_gb      = float(params.get("gpu_memory_preservation", 6.0))
    use_teacache     = bool(params.get("use_teacache", True))

    job.status = "running"

    # === util progress to queue (preview, desc, html_progress) ===
    def q_progress(preview_np: Optional[np.ndarray], desc: str, hint_html: str):
        stream.output_queue.push(('progress', (preview_np, desc, hint_html)))

    try:
        # === START ===
        from diffusers_helper.gradio.progress_bar import make_progress_bar_html  # hanya util HTML text

        # Read image
        if not os.path.exists(input_image_path):
            raise FileNotFoundError(f"Input image not found: {input_image_path}")
        img = Image.open(input_image_path).convert("RGB")
        H, W = img.size[1], img.size[0]

        free_mem_gb = get_cuda_free_memory_gb(gpu)
        # Text encode
        q_progress(None, "", make_progress_bar_html(0, "Text encoding ..."))

        if not high_vram:
            # strategi offload sesuai demo
            from diffusers_helper.memory import load_model_as_complete, fake_diffusers_current_device
            fake_diffusers_current_device(text_encoder, gpu)
            load_model_as_complete(text_encoder_2, target_device=gpu)

        llama_vec, clip_l_pooler = encode_prompt_conds(prompt, text_encoder, text_encoder_2, tokenizer, tokenizer_2)

        if cfg == 1:
            llama_vec_n, clip_l_pooler_n = torch.zeros_like(llama_vec), torch.zeros_like(clip_l_pooler)
        else:
            llama_vec_n, clip_l_pooler_n = encode_prompt_conds(n_prompt, text_encoder, text_encoder_2, tokenizer, tokenizer_2)

        llama_vec, llama_attention_mask = crop_or_pad_yield_mask(llama_vec, length=512)
        llama_vec_n, llama_attention_mask_n = crop_or_pad_yield_mask(llama_vec_n, length=512)

        # Image preprocessing (temukan bucket size & center crop — sesuai demo)
        q_progress(None, "", make_progress_bar_html(0, "Image processing ..."))

        height, width = find_nearest_bucket(H, W, resolution=640)
        input_image_np = resize_and_center_crop(np.array(img), target_width=width, target_height=height)
        Image.fromarray(input_image_np).save(OUTPUTS_DIR / f'{job.id}.png')

        input_image_pt = torch.from_numpy(input_image_np).float() / 127.5 - 1
        input_image_pt = input_image_pt.permute(2, 0, 1)[None, :, None]

        # VAE encode
        q_progress(None, "", make_progress_bar_html(0, "VAE encoding ..."))
        if not high_vram:
            load_model_as_complete(vae, target_device=gpu)
        start_latent = vae_encode(input_image_pt, vae)

        # CLIP Vision
        q_progress(None, "", make_progress_bar_html(0, "CLIP Vision encoding ..."))
        if not high_vram:
            load_model_as_complete(image_encoder, target_device=gpu)
        image_encoder_output = hf_clip_vision_encode(input_image_np, feature_extractor, image_encoder)
        image_encoder_last_hidden_state = image_encoder_output.last_hidden_state

        # DType match
        for t in (llama_vec, llama_vec_n, clip_l_pooler, clip_l_pooler_n, image_encoder_last_hidden_state):
            _ = t.to(transformer.dtype)

        # Sampling loop (disederhanakan minimal dari demo)
        q_progress(None, "", make_progress_bar_html(0, "Start sampling ..."))

        rnd = torch.Generator("cpu").manual_seed(seed)
        num_frames = latent_ws * 4 - 3
        total_latent_sections = int(max(round((total_sec * 30) / (latent_ws * 4)), 1))

        history_latents = torch.zeros(size=(1, 16, 1 + 2 + 16, height // 8, width // 8), dtype=torch.float32).cpu()
        history_pixels = None
        total_generated_latent_frames = 0

        latent_paddings = list(reversed(range(total_latent_sections)))
        if total_latent_sections > 4:
            latent_paddings = [3] + [2] * (total_latent_sections - 3) + [1, 0]

        def step_callback(d):
            preview = d['denoised']
            preview = vae_decode_fake(preview)
            preview = (preview * 255.0).detach().cpu().numpy().clip(0, 255).astype(np.uint8)
            preview = einops.rearrange(preview, 'b c t h w -> (b h) (t w) c')
            current_step = d['i'] + 1
            percentage = int(100.0 * current_step / steps)
            hint = f"Sampling {current_step}/{steps}"
            desc = f"Total frames: {int(max(0, total_generated_latent_frames * 4 - 3))}, Length: {max(0, (total_generated_latent_frames * 4 - 3)/30):.2f}s @30fps"
            stream.output_queue.push(('progress', (preview, desc, f'<div>{hint}</div>')))

        for latent_padding in latent_paddings:
            is_last_section = latent_padding == 0
            latent_padding_size = latent_padding * latent_ws

            if stream.input_queue.top() == 'end':
                job.status = "stopped"
                stream.output_queue.push(('end', None))
                return

            indices = torch.arange(0, sum([1, latent_padding_size, latent_ws, 1, 2, 16])).unsqueeze(0)
            clean_latent_indices_pre, blank_indices, latent_indices, clean_latent_indices_post, clean_latent_2x_indices, clean_latent_4x_indices = indices.split([1, latent_padding_size, latent_ws, 1, 2, 16], dim=1)
            clean_latent_indices = torch.cat([clean_latent_indices_pre, clean_latent_indices_post], dim=1)

            clean_latents_pre = start_latent.to(history_latents)
            clean_latents_post, clean_latents_2x, clean_latents_4x = history_latents[:, :, :1 + 2 + 16, :, :].split([1, 2, 16], dim=2)
            clean_latents = torch.cat([clean_latents_pre, clean_latents_post], dim=2)

            if not high_vram:
                unload_complete_models()
                move_model_to_device_with_memory_preservation(transformer, target_device=gpu, preserved_memory_gb=preserve_gb)

            transformer.initialize_teacache(enable_teacache=use_teacache, num_steps=steps if use_teacache else None)

            generated_latents = sample_hunyuan(
                transformer=transformer,
                sampler='unipc',
                width=width, height=height, frames=num_frames,
                real_guidance_scale=cfg, distilled_guidance_scale=gs, guidance_rescale=rs,
                num_inference_steps=steps, generator=rnd,
                prompt_embeds=llama_vec, prompt_embeds_mask=llama_attention_mask, prompt_poolers=clip_l_pooler,
                negative_prompt_embeds=llama_vec_n, negative_prompt_embeds_mask=llama_attention_mask_n, negative_prompt_poolers=clip_l_pooler_n,
                device=gpu, dtype=torch.bfloat16,
                image_embeddings=image_encoder_last_hidden_state,
                latent_indices=latent_indices,
                clean_latents=clean_latents, clean_latent_indices=clean_latent_indices,
                clean_latents_2x=clean_latents_2x, clean_latent_2x_indices=clean_latent_2x_indices,
                clean_latents_4x=clean_latents_4x, clean_latent_4x_indices=clean_latent_4x_indices,
                callback=step_callback,
            )

            if is_last_section:
                generated_latents = torch.cat([start_latent.to(generated_latents), generated_latents], dim=2)

            total_generated_latent_frames += int(generated_latents.shape[2])
            history_latents = torch.cat([generated_latents.to(history_latents), history_latents], dim=2)

            if not high_vram:
                offload_model_from_device_for_memory_preservation(transformer, target_device=gpu, preserved_memory_gb=8)
                load_model_as_complete(vae, target_device=gpu)

            real_history_latents = history_latents[:, :, :total_generated_latent_frames, :, :]

            if history_pixels is None:
                history_pixels = vae_decode(real_history_latents, vae).cpu()
            else:
                section_latent_frames = (latent_ws * 2 + 1) if is_last_section else (latent_ws * 2)
                overlapped_frames = latent_ws * 4 - 3
                current_pixels = vae_decode(real_history_latents[:, :, :section_latent_frames], vae).cpu()
                history_pixels = soft_append_bcthw(current_pixels, history_pixels, overlapped_frames)

            if not high_vram:
                unload_complete_models()

            # Simpan video parsial (dan kirim path)
            out_file = OUTPUTS_DIR / f"{job.id}_{total_generated_latent_frames}.mp4"
            save_bcthw_as_mp4(history_pixels, str(out_file), fps=30)
            stream.output_queue.push(('file', str(out_file)))

            if is_last_section:
                break

        job.output_path = str(out_file)
        job.status = "done"
    except Exception as e:
        job.status = "error"
        job.error = f"{type(e).__name__}: {e}"
        traceback.print_exc()
    finally:
        stream.output_queue.push(('end', None))

# ===== API publik =====

def start_job(image_path: str, params: Dict[str, Any]) -> str:
    """
    Mulai job FramePack. Return job_id.
    params: {prompt, negative, seed, seconds, latent_window_size, steps, cfg, gs, rs, gpu_memory_preservation, use_teacache}
    """
    job_id = generate_timestamp() + "_" + uuid.uuid4().hex[:8]
    job = JobState(job_id)
    JOBS[job_id] = job

    # siapkan stream
    stream = AsyncStream()
    QUEUES[job_id] = stream

    # pack params
    run_params = dict(params)
    run_params["image_path"] = image_path

    # jalankan async
    async_run(_run_job, job, stream, run_params)
    return job_id

def stop_job(job_id: str) -> bool:
    stream = QUEUES.get(job_id)
    job = JOBS.get(job_id)
    if not stream or not job:
        return False
    stream.input_queue.push('end')
    return True

def get_status(job_id: str) -> Dict[str, Any]:
    job = JOBS.get(job_id)
    if not job:
        return {"exists": False}
    return {
        "exists": True,
        "id": job.id,
        "status": job.status,
        "error": job.error,
        "output_path": job.output_path,
        "created_at": job.created_at,
    }

def get_progress_queue(job_id: str) -> Optional[AsyncStream]:
    return QUEUES.get(job_id)
