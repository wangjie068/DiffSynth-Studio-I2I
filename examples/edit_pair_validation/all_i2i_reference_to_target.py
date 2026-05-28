import argparse
import csv
import gc
import glob
import inspect
import json
import math
import os
import queue
import subprocess
import sys
import threading
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image, ImageDraw, ImageFont


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

SOURCE_URL = "https://m.media-amazon.com/images/I/31u0ldoiaKL.jpg"
TARGET_URL = "https://m.media-amazon.com/images/I/41Na40-8JfL.jpg"

DEFAULT_PROMPT = (
    "Create a clean square ecommerce advertising poster from the reference product. "
    "Keep exactly the same yellow lip-care tube from the reference, including its "
    "original package text and logo; do not rewrite, replace, or redesign the label. "
    "Recompose the tube diagonally on the lower-right, suspended by a small silver "
    "cosmetic applicator clip from the top-right, with glossy honey serum dripping "
    "into a small puddle at the bottom. Use a white background. On the left add a "
    "large pink headline 'Ampule-Infused Lip Care', then gray copy 'Get honey-glow "
    "lips with ultra-dewy hydration', then a vertical list with pink plus signs: "
    "'Dead skin cells', 'Wrinkles', 'Elasticity', 'Hydration', 'Glow'. Photorealistic "
    "product advertisement."
)

NEGATIVE_PROMPT = (
    "wrong brand, changed package, unreadable label, misspelled text, extra products, "
    "deformed tube, low quality, blurry"
)

# Target-layout ROIs. They are intentionally not passed to generation.
TARGET_ROIS = {
    "full": (0.0, 0.0, 1.0, 1.0),
    "new_ad_copy_left": (0.04, 0.08, 0.63, 0.71),
    "product_right": (0.49, 0.32, 1.0, 1.0),
    "tube_label_right": (0.57, 0.45, 0.91, 0.84),
}


@dataclass(frozen=True)
class ModelSpec:
    name: str
    family: str
    description: str
    default_steps: int
    default_cfg: float
    runner: Callable
    relevant: bool = True
    notes: str = ""


@dataclass(frozen=True)
class EvalJob:
    model: str
    seed: int
    gpu: str | None = None


def download_image(url: str, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request) as response:
        output_path.write_bytes(response.read())
    with Image.open(output_path) as image:
        image.verify()


def prepare_pair(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    source_path = output_dir / "source.jpg"
    target_path = output_dir / "target.jpg"
    download_image(args.source_url, source_path)
    download_image(args.target_url, target_path)
    with Image.open(source_path) as source, Image.open(target_path) as target:
        metadata = {
            "source_url": args.source_url,
            "target_url": args.target_url,
            "source_path": str(source_path),
            "target_path": str(target_path),
            "source_size": source.size,
            "target_size": target.size,
        }
    (output_dir / "pair.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


def require_file(path: str | Path, role: str, auto_download: bool = False) -> Path:
    resolved = Path(path)
    if not resolved.is_file():
        if auto_download and role == "source image":
            print(f"Missing source image, downloading to {resolved}", flush=True)
            download_image(SOURCE_URL, resolved)
        elif auto_download and role == "target image":
            print(f"Missing target image, downloading to {resolved}", flush=True)
            download_image(TARGET_URL, resolved)
    if not resolved.is_file():
        raise SystemExit(
            f"Missing {role}: {resolved}\n"
            "Pass a valid path, or use the default Amazon source/target paths so the script can download them."
        )
    return resolved


def get_torch_dtype(dtype_name: str):
    import torch

    return getattr(torch, dtype_name)


SOURCE_REPO_ID_OVERRIDES = {
    ("huggingface", "jd-opensource/JoyAI-Image-Edit"): "jdopensource/JoyAI-Image-Edit",
}


def get_model_base_path() -> Path:
    return Path(os.environ.get("DIFFSYNTH_MODEL_BASE_PATH", "./models"))


def local_repo_exists(model_id: str, model_base_path: str | Path | None = None) -> bool:
    base_path = Path(model_base_path) if model_base_path is not None else get_model_base_path()
    return (base_path / model_id).exists()


def pattern_to_glob(pattern: str | None) -> str:
    if pattern in [None, "", "./"]:
        return "*"
    if pattern.endswith("/"):
        return pattern + "*"
    return pattern


def local_pattern_exists(
    model_id: str,
    pattern: str | None,
    model_base_path: str | Path | None = None,
) -> bool:
    base_path = Path(model_base_path) if model_base_path is not None else get_model_base_path()
    root = base_path / model_id
    if not root.exists():
        return False
    return bool(glob.glob(pattern_to_glob(pattern), root_dir=root))


def repo_aliases(model_id: str) -> list[str]:
    aliases = [model_id]
    for (_, source_model_id), target_model_id in SOURCE_REPO_ID_OVERRIDES.items():
        if source_model_id == model_id and target_model_id not in aliases:
            aliases.append(target_model_id)
        if target_model_id == model_id and source_model_id not in aliases:
            aliases.append(source_model_id)
    return aliases


def repo_id_for_source(
    model_id: str,
    download_source: str | None = None,
    model_base_path: str | Path | None = None,
    pattern: str | None = None,
    prefer_local: bool = True,
) -> str:
    if prefer_local:
        for alias in repo_aliases(model_id):
            if pattern is None and local_repo_exists(alias, model_base_path):
                return alias
            if pattern is not None and local_pattern_exists(alias, pattern, model_base_path):
                return alias
    source = download_source or os.environ.get("DIFFSYNTH_DOWNLOAD_SOURCE", "modelscope")
    return SOURCE_REPO_ID_OVERRIDES.get((source.lower(), model_id), model_id)


def model_config_for_pattern(
    ModelConfig,
    model_id: str,
    pattern: str,
    download_source: str | None = None,
    model_base_path: str | Path | None = None,
    **kwargs,
):
    resolved_model_id = repo_id_for_source(
        model_id,
        download_source=download_source,
        model_base_path=model_base_path,
        pattern=pattern,
        prefer_local=True,
    )
    has_local_files = local_pattern_exists(
        resolved_model_id,
        pattern,
        model_base_path=model_base_path,
    )
    return ModelConfig(
        model_id=resolved_model_id,
        origin_file_pattern=pattern,
        download_source=download_source,
        local_model_path=str(model_base_path) if model_base_path is not None else None,
        skip_download=has_local_files,
        **kwargs,
    )


def qwen_runner(
    *,
    model_id: str,
    source: Image.Image,
    prompt: str,
    negative_prompt: str,
    seed: int,
    height: int,
    width: int,
    steps: int,
    cfg_scale: float,
    dtype: str,
    device: str,
    use_list_input: bool,
    zero_cond_t: bool = False,
    lightning: bool = False,
    denoising_strength: float | None = None,
):
    import torch

    from diffsynth.pipelines.qwen_image import FlowMatchScheduler, ModelConfig, QwenImagePipeline

    pipe = QwenImagePipeline.from_pretrained(
        torch_dtype=get_torch_dtype(dtype),
        device=device,
        model_configs=[
            model_config_for_pattern(
                ModelConfig,
                model_id=model_id,
                pattern="transformer/diffusion_pytorch_model*.safetensors",
            ),
            model_config_for_pattern(
                ModelConfig,
                model_id="Qwen/Qwen-Image",
                pattern="text_encoder/model*.safetensors",
            ),
            model_config_for_pattern(
                ModelConfig,
                model_id="Qwen/Qwen-Image",
                pattern="vae/diffusion_pytorch_model.safetensors",
            ),
        ],
        processor_config=model_config_for_pattern(
            ModelConfig,
            model_id="Qwen/Qwen-Image-Edit",
            pattern="processor/",
        ),
    )
    if lightning:
        lora = model_config_for_pattern(
            ModelConfig,
            model_id="lightx2v/Qwen-Image-Edit-2511-Lightning",
            pattern="Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors",
        )
        pipe.load_lora(pipe.dit, lora, alpha=1)
        pipe.scheduler = FlowMatchScheduler("Qwen-Image-Lightning")
    edit_image = [source] if use_list_input else source
    kwargs = {"zero_cond_t": zero_cond_t} if zero_cond_t else {}
    return pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        edit_image=edit_image,
        edit_image_auto_resize=True,
        seed=seed,
        num_inference_steps=steps,
        height=height,
        width=width,
        cfg_scale=cfg_scale,
        **kwargs,
    )


def run_qwen_image_edit(**kwargs):
    return qwen_runner(
        model_id="Qwen/Qwen-Image-Edit",
        use_list_input=False,
        cfg_scale=kwargs.pop("cfg_scale"),
        **kwargs,
    )


def run_qwen_image_edit_2509(**kwargs):
    return qwen_runner(
        model_id="Qwen/Qwen-Image-Edit-2509",
        use_list_input=True,
        cfg_scale=kwargs.pop("cfg_scale"),
        **kwargs,
    )


def run_qwen_image_edit_2511(**kwargs):
    return qwen_runner(
        model_id="Qwen/Qwen-Image-Edit-2511",
        use_list_input=True,
        zero_cond_t=True,
        cfg_scale=kwargs.pop("cfg_scale"),
        **kwargs,
    )


def run_qwen_image_edit_2511_lightning(**kwargs):
    return qwen_runner(
        model_id="Qwen/Qwen-Image-Edit-2511",
        use_list_input=True,
        zero_cond_t=True,
        lightning=True,
        cfg_scale=kwargs.pop("cfg_scale"),
        **kwargs,
    )


def run_firered_10(**kwargs):
    return qwen_runner(
        model_id="FireRedTeam/FireRed-Image-Edit-1.0",
        use_list_input=True,
        cfg_scale=kwargs.pop("cfg_scale"),
        **kwargs,
    )


def run_firered_11(**kwargs):
    return qwen_runner(
        model_id="FireRedTeam/FireRed-Image-Edit-1.1",
        use_list_input=True,
        cfg_scale=kwargs.pop("cfg_scale"),
        **kwargs,
    )


def run_joyai_image_edit(**kwargs):
    from diffsynth.pipelines.joyai_image import JoyAIImagePipeline, ModelConfig

    model_id = "jd-opensource/JoyAI-Image-Edit"
    pipe = JoyAIImagePipeline.from_pretrained(
        torch_dtype=get_torch_dtype(kwargs["dtype"]),
        device=kwargs["device"],
        model_configs=[
            model_config_for_pattern(
                ModelConfig,
                model_id=model_id,
                pattern="transformer/transformer.pth",
            ),
            model_config_for_pattern(
                ModelConfig,
                model_id=model_id,
                pattern="JoyAI-Image-Und/model*.safetensors",
            ),
            model_config_for_pattern(
                ModelConfig,
                model_id=model_id,
                pattern="vae/Wan2.1_VAE.pth",
            ),
        ],
        processor_config=model_config_for_pattern(
            ModelConfig,
            model_id=model_id,
            pattern="JoyAI-Image-Und/",
        ),
    )
    return pipe(
        prompt=kwargs["prompt"],
        edit_image=kwargs["source"],
        height=kwargs["height"],
        width=kwargs["width"],
        seed=kwargs["seed"],
        num_inference_steps=kwargs["steps"],
        cfg_scale=kwargs["cfg_scale"],
    )


def hidream_runner(*, model_id: str, dev: bool = False, **kwargs):
    from diffsynth.core.loader.config import ModelConfig
    from diffsynth.pipelines.hidream_o1_image import HiDreamO1ImagePipeline

    pipe = HiDreamO1ImagePipeline.from_pretrained(
        torch_dtype=get_torch_dtype(kwargs["dtype"]),
        device=kwargs["device"],
        model_configs=[
            model_config_for_pattern(ModelConfig, model_id=model_id, pattern="model-*.safetensors"),
        ],
        processor_config=model_config_for_pattern(ModelConfig, model_id=model_id, pattern="./"),
    )
    if dev:
        from diffsynth.diffusion import HiDreamO1FlashScheduler

        pipe.scheduler = HiDreamO1FlashScheduler(
            noise_scale_start=7.5,
            noise_scale_end=7.5,
            noise_clip_std=2.5,
        )
    call_kwargs = {
        "prompt": kwargs["prompt"],
        "negative_prompt": kwargs["negative_prompt"],
        "cfg_scale": kwargs["cfg_scale"],
        "height": kwargs["height"],
        "width": kwargs["width"],
        "seed": kwargs["seed"],
        "num_inference_steps": kwargs["steps"],
        "edit_image": [kwargs["source"]],
    }
    if dev:
        call_kwargs.update({"model_type": "dev", "noise_scale": 7.5})
    return pipe(**call_kwargs)


def run_hidream_o1_image(**kwargs):
    return hidream_runner(model_id="HiDream-ai/HiDream-O1-Image", dev=False, **kwargs)


def run_hidream_o1_image_dev(**kwargs):
    return hidream_runner(model_id="HiDream-ai/HiDream-O1-Image-Dev", dev=True, **kwargs)


def run_z_image_omni_base(**kwargs):
    raise RuntimeError(
        "Tongyi-MAI/Z-Image-Omni-Base weights are not publicly available in the "
        "checked official model repos. This adapter is kept only as a placeholder; "
        "do not include it in default all_relevant runs."
    )
    from diffsynth.pipelines.z_image import ModelConfig, ZImagePipeline

    pipe = ZImagePipeline.from_pretrained(
        torch_dtype=get_torch_dtype(kwargs["dtype"]),
        device=kwargs["device"],
        model_configs=[
            model_config_for_pattern(
                ModelConfig,
                model_id="Tongyi-MAI/Z-Image-Omni-Base",
                pattern="transformer/*.safetensors",
            ),
            model_config_for_pattern(
                ModelConfig,
                model_id="Tongyi-MAI/Z-Image-Omni-Base",
                pattern="siglip/model.safetensors",
            ),
            model_config_for_pattern(
                ModelConfig,
                model_id="Tongyi-MAI/Z-Image-Turbo",
                pattern="text_encoder/*.safetensors",
            ),
            model_config_for_pattern(
                ModelConfig,
                model_id="Tongyi-MAI/Z-Image-Turbo",
                pattern="vae/diffusion_pytorch_model.safetensors",
            ),
        ],
        tokenizer_config=model_config_for_pattern(
            ModelConfig,
            model_id="Tongyi-MAI/Z-Image-Turbo",
            pattern="tokenizer/",
        ),
    )
    return pipe(
        prompt=kwargs["prompt"],
        edit_image=kwargs["source"],
        seed=kwargs["seed"],
        rand_device="cuda" if kwargs["device"].startswith("cuda") else kwargs["device"],
        num_inference_steps=kwargs["steps"],
        cfg_scale=kwargs["cfg_scale"],
    )


def run_flux1_kontext_dev(**kwargs):
    from diffsynth.pipelines.flux_image import FluxImagePipeline, ModelConfig

    pipe = FluxImagePipeline.from_pretrained(
        torch_dtype=get_torch_dtype(kwargs["dtype"]),
        device=kwargs["device"],
        model_configs=[
            model_config_for_pattern(
                ModelConfig,
                model_id="black-forest-labs/FLUX.1-Kontext-dev",
                pattern="flux1-kontext-dev.safetensors",
            ),
            model_config_for_pattern(
                ModelConfig,
                model_id="black-forest-labs/FLUX.1-dev",
                pattern="text_encoder/model*.safetensors",
            ),
            model_config_for_pattern(
                ModelConfig,
                model_id="black-forest-labs/FLUX.1-dev",
                pattern="text_encoder_2/model*.safetensors",
            ),
            model_config_for_pattern(
                ModelConfig,
                model_id="black-forest-labs/FLUX.1-dev",
                pattern="ae.safetensors",
            ),
        ],
    )
    return pipe(
        prompt=kwargs["prompt"],
        negative_prompt=kwargs["negative_prompt"],
        kontext_images=kwargs["source"],
        embedded_guidance=kwargs["cfg_scale"],
        seed=kwargs["seed"],
        height=kwargs["height"],
        width=kwargs["width"],
        num_inference_steps=kwargs["steps"],
    )


def run_step1x_edit(**kwargs):
    from diffsynth.pipelines.flux_image import FluxImagePipeline, ModelConfig

    pipe = FluxImagePipeline.from_pretrained(
        torch_dtype=get_torch_dtype(kwargs["dtype"]),
        device=kwargs["device"],
        model_configs=[
            model_config_for_pattern(
                ModelConfig,
                model_id="Qwen/Qwen2.5-VL-7B-Instruct",
                pattern="model-*.safetensors",
            ),
            model_config_for_pattern(
                ModelConfig,
                model_id="stepfun-ai/Step1X-Edit",
                pattern="step1x-edit-i1258.safetensors",
            ),
            model_config_for_pattern(
                ModelConfig,
                model_id="stepfun-ai/Step1X-Edit",
                pattern="vae.safetensors",
            ),
        ],
    )
    return pipe(
        prompt=kwargs["prompt"],
        negative_prompt=kwargs["negative_prompt"],
        step1x_reference_image=kwargs["source"],
        width=kwargs["width"],
        height=kwargs["height"],
        cfg_scale=kwargs["cfg_scale"],
        seed=kwargs["seed"],
        rand_device="cuda" if kwargs["device"].startswith("cuda") else kwargs["device"],
        num_inference_steps=kwargs["steps"],
    )


def run_nexus_gen_editing(**kwargs):
    import importlib

    if importlib.util.find_spec("transformers") is None:
        raise ImportError("Nexus-GenV2 requires transformers==4.49.0.")
    import transformers

    if transformers.__version__ != "4.49.0":
        raise ImportError("Nexus-GenV2 requires transformers==4.49.0.")
    from diffsynth.pipelines.flux_image import FluxImagePipeline, ModelConfig

    pipe = FluxImagePipeline.from_pretrained(
        torch_dtype=get_torch_dtype(kwargs["dtype"]),
        device=kwargs["device"],
        model_configs=[
            model_config_for_pattern(
                ModelConfig,
                model_id="DiffSynth-Studio/Nexus-GenV2",
                pattern="model*.safetensors",
            ),
            model_config_for_pattern(
                ModelConfig,
                model_id="DiffSynth-Studio/Nexus-GenV2",
                pattern="edit_decoder.bin",
            ),
            model_config_for_pattern(
                ModelConfig,
                model_id="black-forest-labs/FLUX.1-dev",
                pattern="text_encoder/model*.safetensors",
            ),
            model_config_for_pattern(
                ModelConfig,
                model_id="black-forest-labs/FLUX.1-dev",
                pattern="text_encoder_2/model*.safetensors",
            ),
            model_config_for_pattern(
                ModelConfig,
                model_id="black-forest-labs/FLUX.1-dev",
                pattern="ae.safetensors",
            ),
        ],
        nexus_gen_processor_config=model_config_for_pattern(
            ModelConfig,
            model_id="DiffSynth-Studio/Nexus-GenV2",
            pattern="processor/",
        ),
    )
    return pipe(
        prompt=kwargs["prompt"],
        negative_prompt=kwargs["negative_prompt"],
        seed=kwargs["seed"],
        cfg_scale=kwargs["cfg_scale"],
        num_inference_steps=kwargs["steps"],
        nexus_gen_reference_image=kwargs["source"],
        height=kwargs["height"],
        width=kwargs["width"],
    )


def flux2_runner(*, variant: str, base_variant: str | None = None, dev: bool = False, **kwargs):
    import torch

    from diffsynth.pipelines.flux2_image import Flux2ImagePipeline, ModelConfig

    model_configs = []
    if dev:
        vram_config = {
            "offload_dtype": torch.bfloat16,
            "offload_device": "cpu",
            "onload_dtype": torch.bfloat16,
            "onload_device": kwargs["device"],
            "preparing_dtype": torch.bfloat16,
            "preparing_device": kwargs["device"],
            "computation_dtype": torch.bfloat16,
            "computation_device": kwargs["device"],
        }
        model_configs = [
            model_config_for_pattern(
                ModelConfig,
                model_id="black-forest-labs/FLUX.2-dev",
                pattern="text_encoder/*.safetensors",
                **vram_config,
            ),
            model_config_for_pattern(
                ModelConfig,
                model_id="black-forest-labs/FLUX.2-dev",
                pattern="transformer/*.safetensors",
                **vram_config,
            ),
            model_config_for_pattern(
                ModelConfig,
                model_id="black-forest-labs/FLUX.2-dev",
                pattern="vae/diffusion_pytorch_model.safetensors",
                **vram_config,
            ),
        ]
        tokenizer_config = model_config_for_pattern(
            ModelConfig,
            model_id="black-forest-labs/FLUX.2-dev",
            pattern="tokenizer/",
        )
    else:
        transformer_model_id = base_variant or variant
        model_configs = [
            model_config_for_pattern(ModelConfig, model_id=variant, pattern="text_encoder/*.safetensors"),
            model_config_for_pattern(ModelConfig, model_id=transformer_model_id, pattern="transformer/*.safetensors"),
            model_config_for_pattern(ModelConfig, model_id=variant, pattern="vae/diffusion_pytorch_model.safetensors"),
        ]
        tokenizer_config = model_config_for_pattern(ModelConfig, model_id=variant, pattern="tokenizer/")
    pipe = Flux2ImagePipeline.from_pretrained(
        torch_dtype=get_torch_dtype(kwargs["dtype"]),
        device=kwargs["device"],
        model_configs=model_configs,
        tokenizer_config=tokenizer_config,
    )
    call_kwargs = {
        "prompt": kwargs["prompt"],
        "negative_prompt": kwargs["negative_prompt"],
        "seed": kwargs["seed"],
        "rand_device": "cuda" if kwargs["device"].startswith("cuda") else kwargs["device"],
        "edit_image": [kwargs["source"]],
        "num_inference_steps": kwargs["steps"],
        "height": kwargs["height"],
        "width": kwargs["width"],
    }
    if dev:
        call_kwargs["embedded_guidance"] = kwargs["cfg_scale"]
    else:
        call_kwargs["cfg_scale"] = kwargs["cfg_scale"]
    return pipe(**call_kwargs)


def run_flux2_dev(**kwargs):
    return flux2_runner(variant="black-forest-labs/FLUX.2-dev", dev=True, **kwargs)


def run_flux2_klein_base_4b(**kwargs):
    return flux2_runner(
        variant="black-forest-labs/FLUX.2-klein-4B",
        base_variant="black-forest-labs/FLUX.2-klein-base-4B",
        **kwargs,
    )


def run_flux2_klein_4b(**kwargs):
    return flux2_runner(variant="black-forest-labs/FLUX.2-klein-4B", **kwargs)


def run_flux2_klein_base_9b(**kwargs):
    return flux2_runner(
        variant="black-forest-labs/FLUX.2-klein-9B",
        base_variant="black-forest-labs/FLUX.2-klein-base-9B",
        **kwargs,
    )


def run_flux2_klein_9b(**kwargs):
    return flux2_runner(variant="black-forest-labs/FLUX.2-klein-9B", **kwargs)


def run_flux2_template_edit_4b(**kwargs):
    from diffsynth.diffusion.template import TemplatePipeline
    from diffsynth.pipelines.flux2_image import Flux2ImagePipeline, ModelConfig

    pipe = Flux2ImagePipeline.from_pretrained(
        torch_dtype=get_torch_dtype(kwargs["dtype"]),
        device=kwargs["device"],
        model_configs=[
            model_config_for_pattern(
                ModelConfig,
                model_id="black-forest-labs/FLUX.2-klein-base-4B",
                pattern="transformer/*.safetensors",
            ),
            model_config_for_pattern(
                ModelConfig,
                model_id="black-forest-labs/FLUX.2-klein-4B",
                pattern="text_encoder/*.safetensors",
            ),
            model_config_for_pattern(
                ModelConfig,
                model_id="black-forest-labs/FLUX.2-klein-4B",
                pattern="vae/diffusion_pytorch_model.safetensors",
            ),
        ],
        tokenizer_config=model_config_for_pattern(
            ModelConfig,
            model_id="black-forest-labs/FLUX.2-klein-4B",
            pattern="tokenizer/",
        ),
    )
    template = TemplatePipeline.from_pretrained(
        torch_dtype=get_torch_dtype(kwargs["dtype"]),
        device=kwargs["device"],
        model_configs=[
            model_config_for_pattern(
                ModelConfig,
                model_id="DiffSynth-Studio/Template-KleinBase4B-Edit",
                pattern="*",
            )
        ],
    )
    return template(
        pipe,
        prompt=kwargs["prompt"],
        seed=kwargs["seed"],
        cfg_scale=kwargs["cfg_scale"],
        num_inference_steps=kwargs["steps"],
        template_inputs=[{"image": kwargs["source"], "prompt": kwargs["prompt"]}],
        negative_template_inputs=[{"image": kwargs["source"], "prompt": ""}],
    )


def run_anima_img2img(**kwargs):
    from diffsynth.pipelines.anima_image import AnimaImagePipeline, ModelConfig

    pipe = AnimaImagePipeline.from_pretrained(
        torch_dtype=get_torch_dtype(kwargs["dtype"]),
        device=kwargs["device"],
        model_configs=[
            model_config_for_pattern(
                ModelConfig,
                model_id="circlestone-labs/Anima",
                pattern="split_files/diffusion_models/anima-preview.safetensors",
            ),
            model_config_for_pattern(
                ModelConfig,
                model_id="circlestone-labs/Anima",
                pattern="split_files/text_encoders/qwen_3_06b_base.safetensors",
            ),
            model_config_for_pattern(
                ModelConfig,
                model_id="circlestone-labs/Anima",
                pattern="split_files/vae/qwen_image_vae.safetensors",
            ),
        ],
        tokenizer_config=model_config_for_pattern(ModelConfig, model_id="Qwen/Qwen3-0.6B", pattern="./"),
        tokenizer_t5xxl_config=model_config_for_pattern(
            ModelConfig,
            model_id="stabilityai/stable-diffusion-3.5-large",
            pattern="tokenizer_3/",
        ),
    )
    return pipe(
        prompt=kwargs["prompt"],
        negative_prompt=kwargs["negative_prompt"],
        input_image=kwargs["source"].resize((kwargs["width"], kwargs["height"]), Image.Resampling.LANCZOS),
        denoising_strength=kwargs["denoising_strength"],
        cfg_scale=kwargs["cfg_scale"],
        height=kwargs["height"],
        width=kwargs["width"],
        seed=kwargs["seed"],
        rand_device="cuda" if kwargs["device"].startswith("cuda") else kwargs["device"],
        num_inference_steps=kwargs["steps"],
    )


MODEL_SPECS = [
    ModelSpec("qwen_image_edit", "qwen", "Qwen/Qwen-Image-Edit single-image edit", 40, 4.0, run_qwen_image_edit),
    ModelSpec("qwen_image_edit_2509", "qwen", "Qwen/Qwen-Image-Edit-2509 multi-image edit", 40, 4.0, run_qwen_image_edit_2509),
    ModelSpec("qwen_image_edit_2511", "qwen", "Qwen/Qwen-Image-Edit-2511 multi-image edit", 40, 4.0, run_qwen_image_edit_2511),
    ModelSpec("qwen_image_edit_2511_lightning", "qwen", "Qwen-Image-Edit-2511 + Lightning LoRA", 4, 1.0, run_qwen_image_edit_2511_lightning),
    ModelSpec("firered_image_edit_1_0", "qwen", "FireRed-Image-Edit-1.0 on Qwen pipeline", 40, 4.0, run_firered_10),
    ModelSpec("firered_image_edit_1_1", "qwen", "FireRed-Image-Edit-1.1 on Qwen pipeline", 40, 4.0, run_firered_11),
    ModelSpec("joyai_image_edit", "joyai", "JoyAI-Image-Edit", 30, 5.0, run_joyai_image_edit),
    ModelSpec("hidream_o1_image", "hidream", "HiDream-O1-Image image-to-image/edit", 50, 4.0, run_hidream_o1_image),
    ModelSpec("hidream_o1_image_dev", "hidream", "HiDream-O1-Image-Dev image-to-image/edit", 28, 1.0, run_hidream_o1_image_dev),
    ModelSpec(
        "z_image_omni_base",
        "z_image",
        "Z-Image-Omni-Base edit_image placeholder; public weights unavailable",
        40,
        4.0,
        run_z_image_omni_base,
        relevant=False,
        notes="Excluded from all_relevant because public weights are not available.",
    ),
    ModelSpec("flux1_kontext_dev", "flux", "FLUX.1-Kontext-dev", 50, 2.5, run_flux1_kontext_dev),
    ModelSpec("step1x_edit", "flux", "Step1X-Edit reference-image editing", 50, 6.0, run_step1x_edit),
    ModelSpec("nexus_gen_editing", "flux", "Nexus-GenV2 editing", 50, 2.0, run_nexus_gen_editing),
    ModelSpec("flux2_dev", "flux2", "FLUX.2-dev edit_image", 50, 2.5, run_flux2_dev),
    ModelSpec("flux2_klein_base_4b", "flux2", "FLUX.2 Klein base 4B edit_image", 50, 4.0, run_flux2_klein_base_4b),
    ModelSpec("flux2_klein_4b", "flux2", "FLUX.2 Klein 4B edit_image", 4, 4.0, run_flux2_klein_4b),
    ModelSpec("flux2_klein_base_9b", "flux2", "FLUX.2 Klein base 9B edit_image", 50, 4.0, run_flux2_klein_base_9b),
    ModelSpec("flux2_klein_9b", "flux2", "FLUX.2 Klein 9B edit_image", 4, 4.0, run_flux2_klein_9b),
    ModelSpec("flux2_template_edit_4b", "flux2", "Template-KleinBase4B-Edit", 50, 4.0, run_flux2_template_edit_4b),
    ModelSpec("anima_img2img", "baseline", "Anima input_image img2img baseline", 50, 4.0, run_anima_img2img),
]
MODEL_MAP = {spec.name: spec for spec in MODEL_SPECS}


RUNNER_COMMON_KEYS = {
    "source",
    "prompt",
    "negative_prompt",
    "seed",
    "height",
    "width",
    "steps",
    "cfg_scale",
    "dtype",
    "device",
    "denoising_strength",
}


def add_download_entry(entries: list[tuple[str, str]], model_id: str, pattern: str) -> None:
    key = (model_id, pattern)
    if key not in entries:
        entries.append(key)


def qwen_common_download_entries(entries: list[tuple[str, str]]) -> None:
    add_download_entry(entries, "Qwen/Qwen-Image", "text_encoder/model*.safetensors")
    add_download_entry(entries, "Qwen/Qwen-Image", "vae/diffusion_pytorch_model.safetensors")
    add_download_entry(entries, "Qwen/Qwen-Image", "tokenizer/")
    add_download_entry(entries, "Qwen/Qwen-Image-Edit", "processor/")


def flux1_common_download_entries(entries: list[tuple[str, str]]) -> None:
    add_download_entry(entries, "black-forest-labs/FLUX.1-dev", "tokenizer/")
    add_download_entry(entries, "black-forest-labs/FLUX.1-dev", "tokenizer_2/")
    add_download_entry(entries, "black-forest-labs/FLUX.1-dev", "text_encoder/model*.safetensors")
    add_download_entry(entries, "black-forest-labs/FLUX.1-dev", "text_encoder_2/model*.safetensors")
    add_download_entry(entries, "black-forest-labs/FLUX.1-dev", "ae.safetensors")


def download_entries_for_model(model_name: str) -> list[tuple[str, str]]:
    entries: list[tuple[str, str]] = []
    if model_name == "qwen_image_edit":
        qwen_common_download_entries(entries)
        add_download_entry(entries, "Qwen/Qwen-Image-Edit", "transformer/diffusion_pytorch_model*.safetensors")
    elif model_name == "qwen_image_edit_2509":
        qwen_common_download_entries(entries)
        add_download_entry(entries, "Qwen/Qwen-Image-Edit-2509", "transformer/diffusion_pytorch_model*.safetensors")
    elif model_name == "qwen_image_edit_2511":
        qwen_common_download_entries(entries)
        add_download_entry(entries, "Qwen/Qwen-Image-Edit-2511", "transformer/diffusion_pytorch_model*.safetensors")
    elif model_name == "qwen_image_edit_2511_lightning":
        qwen_common_download_entries(entries)
        add_download_entry(entries, "Qwen/Qwen-Image-Edit-2511", "transformer/diffusion_pytorch_model*.safetensors")
        add_download_entry(entries, "lightx2v/Qwen-Image-Edit-2511-Lightning", "Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors")
    elif model_name == "firered_image_edit_1_0":
        qwen_common_download_entries(entries)
        add_download_entry(entries, "FireRedTeam/FireRed-Image-Edit-1.0", "transformer/diffusion_pytorch_model*.safetensors")
    elif model_name == "firered_image_edit_1_1":
        qwen_common_download_entries(entries)
        add_download_entry(entries, "FireRedTeam/FireRed-Image-Edit-1.1", "transformer/diffusion_pytorch_model*.safetensors")
    elif model_name == "joyai_image_edit":
        add_download_entry(entries, "jd-opensource/JoyAI-Image-Edit", "transformer/transformer.pth")
        add_download_entry(entries, "jd-opensource/JoyAI-Image-Edit", "JoyAI-Image-Und/model*.safetensors")
        add_download_entry(entries, "jd-opensource/JoyAI-Image-Edit", "vae/Wan2.1_VAE.pth")
        add_download_entry(entries, "jd-opensource/JoyAI-Image-Edit", "JoyAI-Image-Und/")
    elif model_name == "hidream_o1_image":
        add_download_entry(entries, "HiDream-ai/HiDream-O1-Image", "model-*.safetensors")
        add_download_entry(entries, "HiDream-ai/HiDream-O1-Image", "./")
    elif model_name == "hidream_o1_image_dev":
        add_download_entry(entries, "HiDream-ai/HiDream-O1-Image-Dev", "model-*.safetensors")
        add_download_entry(entries, "HiDream-ai/HiDream-O1-Image-Dev", "./")
    elif model_name == "z_image_omni_base":
        add_download_entry(entries, "Tongyi-MAI/Z-Image-Omni-Base", "transformer/*.safetensors")
        add_download_entry(entries, "Tongyi-MAI/Z-Image-Omni-Base", "siglip/model.safetensors")
        add_download_entry(entries, "Tongyi-MAI/Z-Image-Turbo", "text_encoder/*.safetensors")
        add_download_entry(entries, "Tongyi-MAI/Z-Image-Turbo", "vae/diffusion_pytorch_model.safetensors")
        add_download_entry(entries, "Tongyi-MAI/Z-Image-Turbo", "tokenizer/")
    elif model_name == "flux1_kontext_dev":
        flux1_common_download_entries(entries)
        add_download_entry(entries, "black-forest-labs/FLUX.1-Kontext-dev", "flux1-kontext-dev.safetensors")
    elif model_name == "step1x_edit":
        flux1_common_download_entries(entries)
        add_download_entry(entries, "Qwen/Qwen2.5-VL-7B-Instruct", "model-*.safetensors")
        add_download_entry(entries, "Qwen/Qwen2.5-VL-7B-Instruct", "")
        add_download_entry(entries, "stepfun-ai/Step1X-Edit", "step1x-edit-i1258.safetensors")
        add_download_entry(entries, "stepfun-ai/Step1X-Edit", "vae.safetensors")
    elif model_name == "nexus_gen_editing":
        flux1_common_download_entries(entries)
        add_download_entry(entries, "DiffSynth-Studio/Nexus-GenV2", "model*.safetensors")
        add_download_entry(entries, "DiffSynth-Studio/Nexus-GenV2", "edit_decoder.bin")
        add_download_entry(entries, "DiffSynth-Studio/Nexus-GenV2", "processor/")
    elif model_name == "flux2_dev":
        add_download_entry(entries, "black-forest-labs/FLUX.2-dev", "text_encoder/*.safetensors")
        add_download_entry(entries, "black-forest-labs/FLUX.2-dev", "transformer/*.safetensors")
        add_download_entry(entries, "black-forest-labs/FLUX.2-dev", "vae/diffusion_pytorch_model.safetensors")
        add_download_entry(entries, "black-forest-labs/FLUX.2-dev", "tokenizer/")
    elif model_name == "flux2_klein_base_4b":
        add_download_entry(entries, "black-forest-labs/FLUX.2-klein-4B", "text_encoder/*.safetensors")
        add_download_entry(entries, "black-forest-labs/FLUX.2-klein-base-4B", "transformer/*.safetensors")
        add_download_entry(entries, "black-forest-labs/FLUX.2-klein-4B", "vae/diffusion_pytorch_model.safetensors")
        add_download_entry(entries, "black-forest-labs/FLUX.2-klein-4B", "tokenizer/")
    elif model_name == "flux2_klein_4b":
        add_download_entry(entries, "black-forest-labs/FLUX.2-klein-4B", "text_encoder/*.safetensors")
        add_download_entry(entries, "black-forest-labs/FLUX.2-klein-4B", "transformer/*.safetensors")
        add_download_entry(entries, "black-forest-labs/FLUX.2-klein-4B", "vae/diffusion_pytorch_model.safetensors")
        add_download_entry(entries, "black-forest-labs/FLUX.2-klein-4B", "tokenizer/")
    elif model_name == "flux2_klein_base_9b":
        add_download_entry(entries, "black-forest-labs/FLUX.2-klein-9B", "text_encoder/*.safetensors")
        add_download_entry(entries, "black-forest-labs/FLUX.2-klein-base-9B", "transformer/*.safetensors")
        add_download_entry(entries, "black-forest-labs/FLUX.2-klein-9B", "vae/diffusion_pytorch_model.safetensors")
        add_download_entry(entries, "black-forest-labs/FLUX.2-klein-9B", "tokenizer/")
    elif model_name == "flux2_klein_9b":
        add_download_entry(entries, "black-forest-labs/FLUX.2-klein-9B", "text_encoder/*.safetensors")
        add_download_entry(entries, "black-forest-labs/FLUX.2-klein-9B", "transformer/*.safetensors")
        add_download_entry(entries, "black-forest-labs/FLUX.2-klein-9B", "vae/diffusion_pytorch_model.safetensors")
        add_download_entry(entries, "black-forest-labs/FLUX.2-klein-9B", "tokenizer/")
    elif model_name == "flux2_template_edit_4b":
        add_download_entry(entries, "black-forest-labs/FLUX.2-klein-base-4B", "transformer/*.safetensors")
        add_download_entry(entries, "black-forest-labs/FLUX.2-klein-4B", "text_encoder/*.safetensors")
        add_download_entry(entries, "black-forest-labs/FLUX.2-klein-4B", "vae/diffusion_pytorch_model.safetensors")
        add_download_entry(entries, "black-forest-labs/FLUX.2-klein-4B", "tokenizer/")
        add_download_entry(entries, "DiffSynth-Studio/Template-KleinBase4B-Edit", "*")
    elif model_name == "anima_img2img":
        add_download_entry(entries, "circlestone-labs/Anima", "split_files/diffusion_models/anima-preview.safetensors")
        add_download_entry(entries, "circlestone-labs/Anima", "split_files/text_encoders/qwen_3_06b_base.safetensors")
        add_download_entry(entries, "circlestone-labs/Anima", "split_files/vae/qwen_image_vae.safetensors")
        add_download_entry(entries, "Qwen/Qwen3-0.6B", "./")
        add_download_entry(entries, "stabilityai/stable-diffusion-3.5-large", "tokenizer_3/")
    else:
        raise SystemExit(f"No download config for model: {model_name}")
    return entries


def download_entries_for_models(model_names: list[str]) -> list[tuple[str, str]]:
    entries: list[tuple[str, str]] = []
    seen = set()
    for model_name in model_names:
        for entry in download_entries_for_model(model_name):
            if entry not in seen:
                seen.add(entry)
                entries.append(entry)
    return entries


def list_models(args: argparse.Namespace) -> None:
    rows = []
    for spec in MODEL_SPECS:
        rows.append(
            {
                "name": spec.name,
                "family": spec.family,
                "default_steps": spec.default_steps,
                "default_cfg": spec.default_cfg,
                "description": spec.description,
                "notes": spec.notes,
            }
        )
    if args.json:
        print(json.dumps(rows, indent=2))
    else:
        for row in rows:
            print(
                f"{row['name']:<32} {row['family']:<10} "
                f"steps={row['default_steps']:<3} cfg={row['default_cfg']:<4} "
                f"{row['description']}"
            )


def self_test(args: argparse.Namespace) -> None:
    dummy_image = Image.new("RGB", (64, 64), "white")
    common_kwargs = {
        "source": dummy_image,
        "prompt": "test prompt",
        "negative_prompt": "",
        "seed": 0,
        "height": 64,
        "width": 64,
        "steps": 1,
        "cfg_scale": 1.0,
        "dtype": "bfloat16",
        "device": "cuda",
        "denoising_strength": 0.85,
    }
    for spec in MODEL_SPECS:
        unexpected = sorted(set(common_kwargs) - RUNNER_COMMON_KEYS)
        if unexpected:
            raise SystemExit(f"Unexpected common keys: {unexpected}")
        if not any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD
            for parameter in inspect.signature(spec.runner).parameters.values()
        ):
            raise SystemExit(f"{spec.name} runner must accept **kwargs")
        download_entries_for_model(spec.name)
    print(f"self-test ok: {len(MODEL_SPECS)} model adapters accept common kwargs")


def command_download_models(args: argparse.Namespace) -> None:
    model_names = resolve_model_names(args.models)
    entries = download_entries_for_models(model_names)
    print(f"Downloading {len(entries)} unique model file groups for {len(model_names)} models.")
    print(f"download_source={args.download_source}, model_base_path={args.model_base_path}")
    if not args.dry_run:
        from diffsynth.core.loader.config import ModelConfig
    for index, (model_id, pattern) in enumerate(entries, start=1):
        local_model_id = repo_id_for_source(
            model_id,
            args.download_source,
            model_base_path=args.model_base_path,
            pattern=pattern,
            prefer_local=True,
        )
        primary_model_id = repo_id_for_source(
            model_id,
            args.download_source,
            model_base_path=args.model_base_path,
            pattern=pattern,
            prefer_local=False,
        )
        has_local_files = local_pattern_exists(local_model_id, pattern, args.model_base_path)
        display_model_id = local_model_id if has_local_files else primary_model_id
        suffix = " (local files matched)" if has_local_files else ""
        print(f"[{index}/{len(entries)}] {display_model_id} :: {pattern}{suffix}", flush=True)
        if args.dry_run:
            continue
        sources = [args.download_source]
        if args.fallback_source != "none" and args.fallback_source not in sources:
            sources.append(args.fallback_source)
        last_error = None
        attempts = [("local", local_model_id)]
        for source in sources:
            source_model_id = repo_id_for_source(
                model_id,
                source,
                model_base_path=args.model_base_path,
                pattern=pattern,
                prefer_local=False,
            )
            if source_model_id not in [attempt[1] for attempt in attempts]:
                attempts.append((source, source_model_id))
        for source, source_model_id in attempts:
            try:
                config = ModelConfig(
                    model_id=source_model_id,
                    origin_file_pattern=pattern,
                    download_source=args.download_source if source == "local" else source,
                    local_model_path=args.model_base_path,
                    skip_download=local_pattern_exists(source_model_id, pattern, args.model_base_path),
                )
                config.download_if_necessary()
                last_error = None
                break
            except Exception as error:
                last_error = error
                print(
                    f"  failed from {source} as {source_model_id}: {type(error).__name__}: {error}",
                    flush=True,
                )
        if last_error is not None:
            if args.continue_on_error:
                print(f"  continue after failed download: {primary_model_id} :: {pattern}", flush=True)
                continue
            raise last_error
    if args.dry_run:
        print("Dry run complete. No files were downloaded.")
    else:
        print("All requested model files are downloaded.")


def command_verify_models(args: argparse.Namespace) -> None:
    model_names = resolve_model_names(args.models)
    entries = download_entries_for_models(model_names)
    missing = []
    print(f"Verifying {len(entries)} unique model file groups for {len(model_names)} models.")
    print(f"model_base_path={args.model_base_path}")
    for index, (model_id, pattern) in enumerate(entries, start=1):
        resolved_model_id = repo_id_for_source(
            model_id,
            args.download_source,
            model_base_path=args.model_base_path,
            pattern=pattern,
            prefer_local=True,
        )
        matched = local_pattern_exists(resolved_model_id, pattern, args.model_base_path)
        status = "OK" if matched else "MISSING"
        print(f"[{index}/{len(entries)}] {status:<7} {resolved_model_id} :: {pattern}", flush=True)
        if not matched:
            missing.append((resolved_model_id, pattern))
    if missing:
        print(f"\nMissing {len(missing)} file groups.")
        raise SystemExit(1)
    print("\nAll requested model file groups are present.")


def resolve_model_names(names: list[str]) -> list[str]:
    if not names or names == ["all"] or names == ["all_relevant"]:
        return [spec.name for spec in MODEL_SPECS if spec.relevant]
    resolved = []
    for name in names:
        if name == "all":
            resolved.extend(spec.name for spec in MODEL_SPECS)
        elif name in MODEL_MAP:
            resolved.append(name)
        else:
            valid = ", ".join(MODEL_MAP)
            raise SystemExit(f"Unknown model '{name}'. Valid models: {valid}")
    return list(dict.fromkeys(resolved))


def generate_one(args: argparse.Namespace) -> None:
    spec = MODEL_MAP[args.model]
    source_path = require_file(args.source, "source image", auto_download=True)
    source = Image.open(source_path).convert("RGB")
    output_dir = Path(args.output_dir) / spec.name
    output_dir.mkdir(parents=True, exist_ok=True)
    steps = args.num_inference_steps or spec.default_steps
    cfg_scale = args.cfg_scale if args.cfg_scale is not None else spec.default_cfg
    output_path = output_dir / f"seed{args.seed}.png"
    metadata_path = output_dir / f"seed{args.seed}.json"
    image = spec.runner(
        source=source,
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        seed=args.seed,
        height=args.height,
        width=args.width,
        steps=steps,
        cfg_scale=cfg_scale,
        dtype=args.dtype,
        device=args.device,
        denoising_strength=args.denoising_strength,
    )
    image.save(output_path)
    metadata = {
        "model": spec.name,
        "family": spec.family,
        "description": spec.description,
        "source": str(source_path),
        "generated": str(output_path),
        "target_was_not_used_for_generation": True,
        "prompt": args.prompt,
        "negative_prompt": args.negative_prompt,
        "seed": args.seed,
        "height": args.height,
        "width": args.width,
        "num_inference_steps": steps,
        "cfg_scale": cfg_scale,
        "denoising_strength": args.denoising_strength,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))
    del image
    del source
    gc.collect()


def normalized_box(box: tuple[float, float, float, float], size: tuple[int, int]):
    width, height = size
    return (
        round(box[0] * width),
        round(box[1] * height),
        round(box[2] * width),
        round(box[3] * height),
    )


def image_metrics(target: Image.Image, generated: Image.Image) -> dict[str, float]:
    target_array = np.asarray(target.convert("RGB"), dtype=np.float32)
    generated_array = np.asarray(generated.convert("RGB"), dtype=np.float32)
    error = target_array - generated_array
    mae = float(np.mean(np.abs(error)))
    rmse = float(np.sqrt(np.mean(error * error)))
    psnr = float("inf") if rmse == 0 else float(20 * math.log10(255.0 / rmse))
    return {"mae": mae, "rmse": rmse, "psnr_db": psnr}


def labelled_tile(image: Image.Image, label: str, width: int = 420) -> Image.Image:
    font = ImageFont.load_default()
    ratio = width / image.width
    resized = image.resize((width, round(image.height * ratio)), Image.Resampling.LANCZOS)
    header_height = 26
    tile = Image.new("RGB", (width, resized.height + header_height), "white")
    tile.paste(resized, (0, header_height))
    draw = ImageDraw.Draw(tile)
    draw.text((7, 8), label, fill="black", font=font)
    return tile


def horizontal_montage(tiles: list[Image.Image]) -> Image.Image:
    width = sum(tile.width for tile in tiles)
    height = max(tile.height for tile in tiles)
    canvas = Image.new("RGB", (width, height), "white")
    left = 0
    for tile in tiles:
        canvas.paste(tile, (left, 0))
        left += tile.width
    return canvas


def compare_one(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    source_path = require_file(args.source, "source image", auto_download=True)
    target_path = require_file(args.target, "target image", auto_download=True)
    generated_path = require_file(args.generated, "generated image")
    source = Image.open(source_path).convert("RGB")
    target = Image.open(target_path).convert("RGB")
    generated = Image.open(generated_path).convert("RGB")
    generated_aligned = generated.resize(target.size, Image.Resampling.LANCZOS)
    source_preview = source.resize(target.size, Image.Resampling.LANCZOS)

    horizontal_montage(
        [
            labelled_tile(source_preview, "source / reference"),
            labelled_tile(target, "held-out target"),
            labelled_tile(generated_aligned, "generated"),
        ]
    ).save(output_dir / "comparison_full.png")

    metrics = {}
    for name, box in TARGET_ROIS.items():
        pixel_box = normalized_box(box, target.size)
        target_crop = target.crop(pixel_box)
        generated_crop = generated_aligned.crop(pixel_box)
        metrics[name] = {"target_box": pixel_box, **image_metrics(target_crop, generated_crop)}
        if name != "full":
            horizontal_montage(
                [
                    labelled_tile(target_crop, f"target: {name}"),
                    labelled_tile(generated_crop, f"generated: {name}"),
                ]
            ).save(output_dir / f"comparison_{name}.png")

    result = {
        "source": str(source_path),
        "target": str(target_path),
        "generated": str(generated_path),
        "metrics": metrics,
        "note": (
            "Pixel metrics measure target-layout similarity only. For source IP/text preservation, "
            "inspect comparison_product_right.png and comparison_tube_label_right.png or run OCR."
        ),
    }
    (output_dir / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    for name, values in metrics.items():
        print(
            f"{name:<20} MAE={values['mae']:.3f} "
            f"RMSE={values['rmse']:.3f} PSNR={values['psnr_db']:.3f} dB"
        )


def build_eval_jobs(args: argparse.Namespace) -> list[EvalJob]:
    return [
        EvalJob(model=model, seed=seed)
        for model in resolve_model_names(args.models)
        for seed in args.seeds
    ]


def build_generate_command(
    args: argparse.Namespace,
    job: EvalJob,
    source_path: Path,
    output_dir: Path,
    device: str,
) -> list[str]:
    command = [
        sys.executable,
        __file__,
        "generate",
        "--model",
        job.model,
        "--source",
        str(source_path),
        "--output-dir",
        str(output_dir),
        "--seed",
        str(job.seed),
        "--height",
        str(args.height),
        "--width",
        str(args.width),
        "--dtype",
        args.dtype,
        "--device",
        device,
        "--denoising-strength",
        str(args.denoising_strength),
        "--prompt",
        args.prompt,
        "--negative-prompt",
        args.negative_prompt,
    ]
    if args.num_inference_steps is not None:
        command.extend(["--num-inference-steps", str(args.num_inference_steps)])
    if args.cfg_scale is not None:
        command.extend(["--cfg-scale", str(args.cfg_scale)])
    if hasattr(args, "download_source") and args.download_source is not None:
        command.extend(["--download-source", args.download_source])
    return command


def build_compare_command(
    job: EvalJob,
    source_path: Path,
    target_path: Path,
    generated: Path,
    eval_dir: Path,
) -> list[str]:
    return [
        sys.executable,
        __file__,
        "compare",
        "--source",
        str(source_path),
        "--target",
        str(target_path),
        "--generated",
        str(generated),
        "--output-dir",
        str(eval_dir),
    ]


def run_eval_job(
    args: argparse.Namespace,
    job: EvalJob,
    source_path: Path,
    target_path: Path,
    output_dir: Path,
    device: str,
    env: dict[str, str] | None = None,
) -> dict:
    gpu_label = f" gpu={job.gpu}" if job.gpu is not None else ""
    print(f"\n=== generate {job.model} seed={job.seed}{gpu_label} ===", flush=True)
    generated = output_dir / job.model / f"seed{job.seed}.png"
    eval_dir = output_dir / job.model / f"eval_seed{job.seed}"
    row = {"model": job.model, "seed": job.seed}
    if job.gpu is not None:
        row["gpu"] = job.gpu
    if env is None:
        env = os.environ.copy()
    if hasattr(args, "download_source") and args.download_source is not None:
        env["DIFFSYNTH_DOWNLOAD_SOURCE"] = args.download_source
    gen_proc = subprocess.run(
        build_generate_command(args, job, source_path, output_dir, device),
        cwd=args.cwd,
        env=env,
    )
    if gen_proc.returncode != 0:
        row.update({"status": "generate_failed", "returncode": gen_proc.returncode})
        return row

    print(f"=== compare {job.model} seed={job.seed}{gpu_label} ===", flush=True)
    cmp_proc = subprocess.run(
        build_compare_command(job, source_path, target_path, generated, eval_dir),
        cwd=args.cwd,
        env=env,
    )
    if cmp_proc.returncode != 0:
        row.update({"status": "compare_failed", "returncode": cmp_proc.returncode})
        return row
    metrics = json.loads((eval_dir / "metrics.json").read_text(encoding="utf-8"))["metrics"]
    row.update(
        {
            "status": "ok",
            "generated": str(generated),
            "eval_dir": str(eval_dir),
            "full_psnr_db": metrics["full"]["psnr_db"],
            "product_right_psnr_db": metrics["product_right"]["psnr_db"],
            "tube_label_right_psnr_db": metrics["tube_label_right"]["psnr_db"],
            "new_ad_copy_left_psnr_db": metrics["new_ad_copy_left"]["psnr_db"],
        }
    )
    return row


def prepare_run_paths(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    source_path = require_file(args.source, "source image", auto_download=True)
    target_path = require_file(args.target, "target image", auto_download=True)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    return source_path, target_path, output_dir


def finalize_rows(output_dir: Path, rows: list[dict]) -> None:
    results_path = output_dir / "results.jsonl"
    summary_csv_path = output_dir / "summary.csv"
    for row in rows:
        append_jsonl(results_path, row)
    write_summary_csv(summary_csv_path, rows)
    print(f"\nSaved JSONL results to {results_path}")
    print(f"Saved CSV summary to {summary_csv_path}")


def command_run_all(args: argparse.Namespace) -> None:
    source_path, target_path, output_dir = prepare_run_paths(args)
    rows = []
    for job in build_eval_jobs(args):
        rows.append(
            run_eval_job(
                args=args,
                job=job,
                source_path=source_path,
                target_path=target_path,
                output_dir=output_dir,
                device=args.device,
            )
        )
    finalize_rows(output_dir, rows)


def gpu_worker(
    *,
    gpu: str,
    jobs: queue.Queue,
    rows: list[dict],
    rows_lock: threading.Lock,
    args: argparse.Namespace,
    source_path: Path,
    target_path: Path,
    output_dir: Path,
) -> None:
    while True:
        try:
            base_job = jobs.get_nowait()
        except queue.Empty:
            return
        job = EvalJob(model=base_job.model, seed=base_job.seed, gpu=gpu)
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        if hasattr(args, "download_source") and args.download_source is not None:
            env["DIFFSYNTH_DOWNLOAD_SOURCE"] = args.download_source
        try:
            row = run_eval_job(
                args=args,
                job=job,
                source_path=source_path,
                target_path=target_path,
                output_dir=output_dir,
                device=args.worker_device,
                env=env,
            )
        except Exception as error:
            row = {
                "model": job.model,
                "seed": job.seed,
                "gpu": job.gpu,
                "status": "worker_exception",
                "error_type": type(error).__name__,
                "error": str(error),
            }
        with rows_lock:
            rows.append(row)
        jobs.task_done()


def command_run_parallel(args: argparse.Namespace) -> None:
    source_path, target_path, output_dir = prepare_run_paths(args)
    all_jobs = build_eval_jobs(args)
    job_queue: queue.Queue = queue.Queue()
    for job in all_jobs:
        job_queue.put(job)
    rows: list[dict] = []
    rows_lock = threading.Lock()
    gpus = [str(gpu) for gpu in args.gpus]
    print(f"Running {len(all_jobs)} jobs across GPUs: {', '.join(gpus)}", flush=True)
    threads = [
        threading.Thread(
            target=gpu_worker,
            kwargs={
                "gpu": gpu,
                "jobs": job_queue,
                "rows": rows,
                "rows_lock": rows_lock,
                "args": args,
                "source_path": source_path,
                "target_path": target_path,
                "output_dir": output_dir,
            },
            daemon=True,
        )
        for gpu in gpus
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    rows.sort(key=lambda row: (row.get("model", ""), int(row.get("seed", -1)), str(row.get("gpu", ""))))
    finalize_rows(output_dir, rows)


def append_jsonl(path: Path, row: dict) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_summary_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate all supported repository image-to-image/edit models on one reference-to-target product task."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser("list-models")
    list_parser.add_argument("--json", action="store_true")
    list_parser.set_defaults(func=list_models)

    self_test_parser = subparsers.add_parser("self-test")
    self_test_parser.set_defaults(func=self_test)

    download_parser = subparsers.add_parser("download-models")
    download_parser.add_argument("--models", nargs="+", default=["all_relevant"])
    download_parser.add_argument(
        "--download-source",
        choices=["modelscope", "huggingface"],
        default=os.environ.get("DIFFSYNTH_DOWNLOAD_SOURCE", "modelscope"),
    )
    download_parser.add_argument(
        "--fallback-source",
        choices=["modelscope", "huggingface", "none"],
        default="modelscope",
    )
    download_parser.add_argument("--continue-on-error", action="store_true")
    download_parser.add_argument(
        "--model-base-path",
        default=os.environ.get("DIFFSYNTH_MODEL_BASE_PATH", "./models"),
    )
    download_parser.add_argument("--dry-run", action="store_true")
    download_parser.set_defaults(func=command_download_models)

    verify_parser = subparsers.add_parser("verify-models")
    verify_parser.add_argument("--models", nargs="+", default=["all_relevant"])
    verify_parser.add_argument(
        "--download-source",
        choices=["modelscope", "huggingface"],
        default=os.environ.get("DIFFSYNTH_DOWNLOAD_SOURCE", "modelscope"),
    )
    verify_parser.add_argument(
        "--model-base-path",
        default=os.environ.get("DIFFSYNTH_MODEL_BASE_PATH", "./models"),
    )
    verify_parser.set_defaults(func=command_verify_models)

    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--source-url", default=SOURCE_URL)
    prepare.add_argument("--target-url", default=TARGET_URL)
    prepare.add_argument("--output-dir", default="data/edit_pair_validation/amazon_lipcare")
    prepare.set_defaults(func=prepare_pair)

    generate = subparsers.add_parser("generate")
    generate.add_argument("--model", required=True, choices=sorted(MODEL_MAP))
    generate.add_argument("--source", required=True)
    generate.add_argument("--output-dir", default="outputs/edit_pair_validation/all_i2i")
    add_generation_args(generate)
    generate.set_defaults(func=generate_one)

    compare = subparsers.add_parser("compare")
    compare.add_argument("--source", required=True)
    compare.add_argument("--target", required=True)
    compare.add_argument("--generated", required=True)
    compare.add_argument("--output-dir", required=True)
    compare.set_defaults(func=compare_one)

    run_all = subparsers.add_parser("run-all")
    run_all.add_argument("--models", nargs="+", default=["all_relevant"])
    run_all.add_argument("--source", required=True)
    run_all.add_argument("--target", required=True)
    run_all.add_argument("--output-dir", default="outputs/edit_pair_validation/all_i2i")
    run_all.add_argument("--seeds", nargs="+", type=int, default=[0])
    run_all.add_argument("--cwd", default=os.getcwd())
    add_generation_args(run_all, seed_required=False)
    run_all.set_defaults(func=command_run_all)

    run_parallel = subparsers.add_parser("run-parallel")
    run_parallel.add_argument("--models", nargs="+", default=["all_relevant"])
    run_parallel.add_argument("--source", required=True)
    run_parallel.add_argument("--target", required=True)
    run_parallel.add_argument("--output-dir", default="outputs/edit_pair_validation/all_i2i_parallel")
    run_parallel.add_argument("--seeds", nargs="+", type=int, default=[0])
    run_parallel.add_argument("--gpus", nargs="+", required=True)
    run_parallel.add_argument("--worker-device", default="cuda")
    run_parallel.add_argument("--cwd", default=os.getcwd())
    add_generation_args(run_parallel, seed_required=False)
    run_parallel.set_defaults(func=command_run_parallel)
    return parser


def add_generation_args(parser: argparse.ArgumentParser, seed_required: bool = True) -> None:
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--negative-prompt", default=NEGATIVE_PROMPT)
    parser.add_argument("--seed", type=int, required=seed_required)
    parser.add_argument("--num-inference-steps", type=int, default=None)
    parser.add_argument("--cfg-scale", type=float, default=None)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--dtype", choices=["bfloat16", "float16"], default="bfloat16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--denoising-strength", type=float, default=0.85)
    parser.add_argument(
        "--download-source",
        choices=["modelscope", "huggingface"],
        default=os.environ.get("DIFFSYNTH_DOWNLOAD_SOURCE", "modelscope"),
    )


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
