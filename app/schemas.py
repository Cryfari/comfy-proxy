# app/schemas/generate.py
from __future__ import annotations

from enum import Enum
from typing import List, Literal, Optional

from pydantic import BaseModel, Field, model_validator


# ===================== Enums =====================

class Sampler(str, Enum):
    euler = "euler"
    dpmpp_2m = "dpm++_2m"
    dpmpp_sde = "dpm++_sde"
    uni_pc = "uni_pc"  # umum untuk t2v/WAN


class Scheduler(str, Enum):
    simple = "simple"
    karras = "karras"


class VideoCodec(str, Enum):
    h264 = "h264"
    hevc = "hevc"
    prores = "prores"


class UpscaleMethod(str, Enum):
    lanczos = "lanczos"
    nearest = "nearest"
    bilinear = "bilinear"
    bicubic = "bicubic"


class ControlPreproc(str, Enum):
    none = "none"
    AnimeFace_SemSegPreprocessor = "AnimeFace_SemSegPreprocessor"
    AnyLineArtPreprocessor_aux = "AnyLineArtPreprocessor_aux"
    BinaryPreprocessor = "BinaryPreprocessor"
    CannyEdgePreprocessor = "CannyEdgePreprocessor"
    ColorPreprocessor = "ColorPreprocessor"
    DensePosePreprocessor = "DensePosePreprocessor"
    DepthAnythingPreprocessor = "DepthAnythingPreprocessor"
    Zoe_DepthAnythingPreprocessor = "Zoe_DepthAnythingPreprocessor"
    DepthAnythingV2Preprocessor = "DepthAnythingV2Preprocessor"
    DSINE_NormalMapPreprocessor = "DSINE-NormalMapPreprocessor"
    DWPreprocessor = "DWPreprocessor"
    AnimalPosePreprocessor = "AnimalPosePreprocessor"
    HEDPreprocessor = "HEDPreprocessor"
    FakeScribblePreprocessor = "FakeScribblePreprocessor"
    LeReS_DepthMapPreprocessor = "LeReS-DepthMapPreprocessor"
    LineArtPreprocessor = "LineArtPreprocessor"
    AnimeLineArtPreprocessor = "AnimeLineArtPreprocessor"
    LineartStandardPreprocessor = "LineartStandardPreprocessor"
    Manga2Anime_LineArt_Preprocessor = "Manga2Anime_LineArt_Preprocessor"
    MediaPipe_FaceMeshPreprocessor = "MediaPipe-FaceMeshPreprocessor"
    MeshGraphormer_DepthMapPreprocessor = "MeshGraphormer-DepthMapPreprocessor"
    Metric3D_DepthMapPreprocessor = "Metric3D-DepthMapPreprocessor"
    Metric3D_NormalMapPreprocessor = "Metric3D-NormalMapPreprocessor"
    MiDaS_NormalMapPreprocessor = "MiDaS-NormalMapPreprocessor"
    MiDaS_DepthMapPreprocessor = "MiDaS-DepthMapPreprocessor"
    M_LSDPreprocessor = "M-LSDPreprocessor"
    BAE_NormalMapPreprocessor = "BAE-NormalMapPreprocessor"
    OneFormer_COCO_SemSegPreprocessor = "OneFormer-COCO-SemSegPreprocessor"
    OneFormer_ADE20K_SemSegPreprocessor = "OneFormer-ADE20K-SemSegPreprocessor"
    OpenposePreprocessor = "OpenposePreprocessor"
    PiDiNetPreprocessor = "PiDiNetPreprocessor"
    PyraCannyPreprocessor = "PyraCannyPreprocessor"
    ImageLuminanceDetector = "ImageLuminanceDetector"
    ImageIntensityDetector = "ImageIntensityDetector"
    ScribblePreprocessor = "ScribblePreprocessor"
    Scribble_XDoG_Preprocessor = "Scribble_XDoG_Preprocessor"
    Scribble_PiDiNet_Preprocessor = "Scribble_PiDiNet_Preprocessor"
    SAMPreprocessor = "SAMPreprocessor"
    ShufflePreprocessor = "ShufflePreprocessor"
    TEEDPreprocessor = "TEEDPreprocessor"
    TilePreprocessor = "TilePreprocessor"
    TTPlanet_TileGF_Preprocessor = "TTPlanet_TileGF_Preprocessor"
    TTPlanet_TileSimple_Preprocessor = "TTPlanet_TileSimple_Preprocessor"
    UniFormer_SemSegPreprocessor = "UniFormer-SemSegPreprocessor"
    SemSegPreprocessor = "SemSegPreprocessor"
    Zoe_DepthMapPreprocessor = "Zoe_DepthMapPreprocessor"


# ===================== Sub-configs opsional =====================

class LoRAItem(BaseModel):
    name: str = Field(..., description="Nama file/alias LoRA")
    strength: float = Field(0.8, ge=0.0, le=1.0)


class LoRAConfig(BaseModel):
    items: List[LoRAItem] = Field(default_factory=list)


class ControlNetItem(BaseModel):
    preprocessor: ControlPreproc
    resolution: int = Field(512, gt=0)
    image_ref_id: Optional[str] = Field(
        None, description="ID/path gambar panduan. Wajib kecuali preprocessor='none'."
    )

    @model_validator(mode="after")
    def _need_image_when_required(self):
        if self.preprocessor != ControlPreproc.none and not self.image_ref_id:
            raise ValueError("image_ref_id wajib jika preprocessor != 'none'")
        return self


class ControlNetConfig(BaseModel):
    items: List[ControlNetItem] = Field(default_factory=list)


class UpscaleConfig(BaseModel):
    enabled: bool = False
    model_name: str = "4x-ESRGAN.safetensors"
    factor: int = Field(2, ge=2, le=4)
    method: UpscaleMethod = UpscaleMethod.lanczos
    blend: float = Field(0.5, ge=0.0, le=1.0)


class FaceSwapConfig(BaseModel):
    enabled: bool = False
    source_ref_id: Optional[str] = None
    target_ref_id: Optional[str] = None
    model: Optional[str] = "inswapper_128.onnx"
    boost: Optional[dict] = Field(default=None, description="Opsi booster/restorasi wajah")

    @model_validator(mode="after")
    def _need_both_images_if_enabled(self):
        if self.enabled and (not self.source_ref_id or not self.target_ref_id):
            raise ValueError("FaceSwap enabled: source_ref_id & target_ref_id wajib")
        return self


class I2VConfig(BaseModel):
    enabled: bool = False
    image_ref_id: Optional[str] = None
    image_b64: Optional[str] = None

    @model_validator(mode="after")
    def _at_least_one_when_enabled(self):
        if self.enabled and not (self.image_ref_id or self.image_b64):
            raise ValueError("I2V enabled: butuh image_ref_id atau image_b64")
        return self


# ===================== Opsi khusus preset =====================

class AudioOptions(BaseModel):
    tags: str = ""
    lyrics: str = ""
    lyrics_strength: float = Field(0.7, ge=0.0, le=1.0)
    format: Literal["mp3", "wav"] = "mp3"
    mp3_quality: Literal["V0", "V2", "V5", "V9"] = "V2"
    seconds: int = Field(10, gt=0)


class VideoOptions(BaseModel):
    fps: int = Field(24, gt=0, le=60)
    seconds: float = Field(5.0, gt=0.0)
    format: Literal["mp4", "mkv", "avi"] = "mp4"
    codec: VideoCodec = VideoCodec.h264


# ===================== Params umum =====================

class Params(BaseModel):
    # --- generik ---
    prompt: str = ""
    negative: str = ""
    width: Optional[int] = Field(1024, gt=0)
    height: Optional[int] = Field(576, gt=0)
    steps: int = Field(20, gt=0)
    cfg: float = Field(5.0, gt=0.0)
    seed: int = -1
    sampler: Sampler = Sampler.uni_pc
    scheduler: Scheduler = Scheduler.simple
    denoise: float = Field(1.0, gt=0.0, le=1.0)

    filename_prefix: str = "ComfyUI"
    format: Optional[str] = None  # jpg/png untuk image; mp4 untuk video (lihat VideoOptions)

    # --- fitur dinamis ---
    lora: Optional[LoRAConfig] = None
    controlnet: Optional[ControlNetConfig] = None
    upscale: Optional[UpscaleConfig] = None
    faceswap: Optional[FaceSwapConfig] = None
    i2v: Optional[I2VConfig] = None

    # --- opsi khusus preset ---
    audio: Optional[AudioOptions] = None
    video: Optional[VideoOptions] = None


# ===================== Root request =====================

class GenerateRequest(BaseModel):
    preset: str
    params: dict