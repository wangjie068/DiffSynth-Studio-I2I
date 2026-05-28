import argparse
import csv
import gc
import json
import math
import os
import subprocess
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image, ImageDraw, ImageFont


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


def get_torch_dtype(dtype_name: str):
    import torch

    return getattr(torch, dtype_name)


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
):
    import torch

    from diffsynth.pipelines.qwen_image import FlowMatchScheduler, ModelConfig, QwenImagePipeline

    pipe = QwenImagePipeline.from_pretrained(
        torch_dtype=get_torch_dtype(dtype),
        device=device,
        model_configs=[
            ModelConfig(
                model_id=model_id,
                origin_file_pattern="transformer/diffusion_pytorch_model*.safetensors",
            ),
            ModelConfig(
                model_id="Qwen/Qwen-Image",
                origin_file_pattern="text_encoder/model*.safetensors",
            ),
            ModelConfig(
                model_id="Qwen/Qwen-Image",
                origin_file_pattern="vae/diffusion_pytorch_model.safetensors",
            ),
        ],
        processor_config=ModelConfig(
            model_id="Qwen/Qwen-Image-Edit",
            origin_file_pattern="processor/",
        ),
    )
    if lightning:
        lora = ModelConfig(
            model_id="lightx2v/Qwen-Image-Edit-2511-Lightning",
            origin_file_pattern="Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors",
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

    pipe = JoyAIImagePipeline.from_pretrained(
        torch_dtype=get_torch_dtype(kwargs["dtype"]),
        device=kwargs["device"],
        model_configs=[
            ModelConfig(
                model_id="jd-opensource/JoyAI-Image-Edit",
                origin_file_pattern="transformer/transformer.pth",
            ),
            ModelConfig(
                model_id="jd-opensource/JoyAI-Image-Edit",
                origin_file_pattern="JoyAI-Image-Und/model*.safetensors",
            ),
            ModelConfig(
                model_id="jd-opensource/JoyAI-Image-Edit",
                origin_file_pattern="vae/Wan2.1_VAE.pth",
            ),
        ],
        processor_config=ModelConfig(
            model_id="jd-opensource/JoyAI-Image-Edit",
            origin_file_pattern="JoyAI-Image-Und/",
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
            ModelConfig(model_id=model_id, origin_file_pattern="model-*.safetensors"),
        ],
        processor_config=ModelConfig(model_id=model_id, origin_file_pattern="./"),
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
    from diffsynth.pipelines.z_image import ModelConfig, ZImagePipeline

    pipe = ZImagePipeline.from_pretrained(
        torch_dtype=get_torch_dtype(kwargs["dtype"]),
        device=kwargs["device"],
        model_configs=[
            ModelConfig(
                model_id="Tongyi-MAI/Z-Image-Omni-Base",
                origin_file_pattern="transformer/*.safetensors",
            ),
            ModelConfig(
                model_id="Tongyi-MAI/Z-Image-Omni-Base",
                origin_file_pattern="siglip/model.safetensors",
            ),
            ModelConfig(
                model_id="Tongyi-MAI/Z-Image-Turbo",
                origin_file_pattern="text_encoder/*.safetensors",
            ),
            ModelConfig(
                model_id="Tongyi-MAI/Z-Image-Turbo",
                origin_file_pattern="vae/diffusion_pytorch_model.safetensors",
            ),
        ],
        tokenizer_config=ModelConfig(
            model_id="Tongyi-MAI/Z-Image-Turbo",
            origin_file_pattern="tokenizer/",
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
            ModelConfig(
                model_id="black-forest-labs/FLUX.1-Kontext-dev",
                origin_file_pattern="flux1-kontext-dev.safetensors",
            ),
            ModelConfig(
                model_id="black-forest-labs/FLUX.1-dev",
                origin_file_pattern="text_encoder/model.safetensors",
            ),
            ModelConfig(
                model_id="black-forest-labs/FLUX.1-dev",
                origin_file_pattern="text_encoder_2/*.safetensors",
            ),
            ModelConfig(
                model_id="black-forest-labs/FLUX.1-dev",
                origin_file_pattern="ae.safetensors",
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
            ModelConfig(
                model_id="Qwen/Qwen2.5-VL-7B-Instruct",
                origin_file_pattern="model-*.safetensors",
            ),
            ModelConfig(
                model_id="stepfun-ai/Step1X-Edit",
                origin_file_pattern="step1x-edit-i1258.safetensors",
            ),
            ModelConfig(
                model_id="stepfun-ai/Step1X-Edit",
                origin_file_pattern="vae.safetensors",
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
            ModelConfig(
                model_id="DiffSynth-Studio/Nexus-GenV2",
                origin_file_pattern="model*.safetensors",
            ),
            ModelConfig(
                model_id="DiffSynth-Studio/Nexus-GenV2",
                origin_file_pattern="edit_decoder.bin",
            ),
            ModelConfig(
                model_id="black-forest-labs/FLUX.1-dev",
                origin_file_pattern="text_encoder/model.safetensors",
            ),
            ModelConfig(
                model_id="black-forest-labs/FLUX.1-dev",
                origin_file_pattern="text_encoder_2/*.safetensors",
            ),
            ModelConfig(
                model_id="black-forest-labs/FLUX.1-dev",
                origin_file_pattern="ae.safetensors",
            ),
        ],
        nexus_gen_processor_config=ModelConfig(
            model_id="DiffSynth-Studio/Nexus-GenV2",
            origin_file_pattern="processor/",
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
            ModelConfig(
                model_id="black-forest-labs/FLUX.2-dev",
                origin_file_pattern="text_encoder/*.safetensors",
                **vram_config,
            ),
            ModelConfig(
                model_id="black-forest-labs/FLUX.2-dev",
                origin_file_pattern="transformer/*.safetensors",
                **vram_config,
            ),
            ModelConfig(
                model_id="black-forest-labs/FLUX.2-dev",
                origin_file_pattern="vae/diffusion_pytorch_model.safetensors",
                **vram_config,
            ),
        ]
        tokenizer_config = ModelConfig(
            model_id="black-forest-labs/FLUX.2-dev",
            origin_file_pattern="tokenizer/",
        )
    else:
        transformer_model_id = base_variant or variant
        model_configs = [
            ModelConfig(model_id=variant, origin_file_pattern="text_encoder/*.safetensors"),
            ModelConfig(model_id=transformer_model_id, origin_file_pattern="transformer/*.safetensors"),
            ModelConfig(model_id=variant, origin_file_pattern="vae/diffusion_pytorch_model.safetensors"),
        ]
        tokenizer_config = ModelConfig(model_id=variant, origin_file_pattern="tokenizer/")
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
            ModelConfig(
                model_id="black-forest-labs/FLUX.2-klein-base-4B",
                origin_file_pattern="transformer/*.safetensors",
            ),
            ModelConfig(
                model_id="black-forest-labs/FLUX.2-klein-4B",
                origin_file_pattern="text_encoder/*.safetensors",
            ),
            ModelConfig(
                model_id="black-forest-labs/FLUX.2-klein-4B",
                origin_file_pattern="vae/diffusion_pytorch_model.safetensors",
            ),
        ],
        tokenizer_config=ModelConfig(
            model_id="black-forest-labs/FLUX.2-klein-4B",
            origin_file_pattern="tokenizer/",
        ),
    )
    template = TemplatePipeline.from_pretrained(
        torch_dtype=get_torch_dtype(kwargs["dtype"]),
        device=kwargs["device"],
        model_configs=[ModelConfig(model_id="DiffSynth-Studio/Template-KleinBase4B-Edit")],
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
            ModelConfig(
                model_id="circlestone-labs/Anima",
                origin_file_pattern="split_files/diffusion_models/anima-preview.safetensors",
            ),
            ModelConfig(
                model_id="circlestone-labs/Anima",
                origin_file_pattern="split_files/text_encoders/qwen_3_06b_base.safetensors",
            ),
            ModelConfig(
                model_id="circlestone-labs/Anima",
                origin_file_pattern="split_files/vae/qwen_image_vae.safetensors",
            ),
        ],
        tokenizer_config=ModelConfig(model_id="Qwen/Qwen3-0.6B", origin_file_pattern="./"),
        tokenizer_t5xxl_config=ModelConfig(
            model_id="stabilityai/stable-diffusion-3.5-large",
            origin_file_pattern="tokenizer_3/",
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
    ModelSpec("z_image_omni_base", "z_image", "Z-Image-Omni-Base edit_image", 40, 4.0, run_z_image_omni_base),
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
    source = Image.open(args.source).convert("RGB")
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
        "source": str(args.source),
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
    source = Image.open(args.source).convert("RGB")
    target = Image.open(args.target).convert("RGB")
    generated = Image.open(args.generated).convert("RGB")
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
        "source": str(args.source),
        "target": str(args.target),
        "generated": str(args.generated),
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


def command_run_all(args: argparse.Namespace) -> None:
    model_names = resolve_model_names(args.models)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "results.jsonl"
    summary_csv_path = output_dir / "summary.csv"
    rows = []
    for model in model_names:
        for seed in args.seeds:
            print(f"\n=== generate {model} seed={seed} ===", flush=True)
            generated = output_dir / model / f"seed{seed}.png"
            eval_dir = output_dir / model / f"eval_seed{seed}"
            gen_cmd = [
                sys.executable,
                __file__,
                "generate",
                "--model",
                model,
                "--source",
                args.source,
                "--output-dir",
                str(output_dir),
                "--seed",
                str(seed),
                "--height",
                str(args.height),
                "--width",
                str(args.width),
                "--dtype",
                args.dtype,
                "--device",
                args.device,
                "--denoising-strength",
                str(args.denoising_strength),
                "--prompt",
                args.prompt,
                "--negative-prompt",
                args.negative_prompt,
            ]
            if args.num_inference_steps is not None:
                gen_cmd.extend(["--num-inference-steps", str(args.num_inference_steps)])
            if args.cfg_scale is not None:
                gen_cmd.extend(["--cfg-scale", str(args.cfg_scale)])
            gen_proc = subprocess.run(gen_cmd, cwd=args.cwd)
            row = {"model": model, "seed": seed}
            if gen_proc.returncode != 0:
                row.update({"status": "generate_failed", "returncode": gen_proc.returncode})
                append_jsonl(results_path, row)
                rows.append(row)
                continue

            print(f"=== compare {model} seed={seed} ===", flush=True)
            cmp_cmd = [
                sys.executable,
                __file__,
                "compare",
                "--source",
                args.source,
                "--target",
                args.target,
                "--generated",
                str(generated),
                "--output-dir",
                str(eval_dir),
            ]
            cmp_proc = subprocess.run(cmp_cmd, cwd=args.cwd)
            if cmp_proc.returncode != 0:
                row.update({"status": "compare_failed", "returncode": cmp_proc.returncode})
                append_jsonl(results_path, row)
                rows.append(row)
                continue
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
            append_jsonl(results_path, row)
            rows.append(row)
    write_summary_csv(summary_csv_path, rows)
    print(f"\nSaved JSONL results to {results_path}")
    print(f"Saved CSV summary to {summary_csv_path}")


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


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
