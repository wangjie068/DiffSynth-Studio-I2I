import argparse
import csv
import gc
import io
import json
import math
import re
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

from diffsynth.core import ModelConfig
from diffsynth.models.model_loader import ModelPool


DEFAULT_IMAGE_URL = (
    "https://p16-creative-tool-sg.tiktokcdn.com/tos-alisg-i-n2703mo9gi-sg/"
    "4de2332989354067ae9a3cff6482b7b9~tplv-n2703mo9gi-image.png"
)


@dataclass(frozen=True)
class VAESpec:
    key: str
    label: str
    model_id: str
    file_pattern: str
    model_names: tuple[str, ...]
    latent_factor: int
    adapter: str


VAE_SPECS = {
    "qwen_image": VAESpec(
        key="qwen_image",
        label="Qwen-Image / Qwen-Image-Edit VAE",
        model_id="Qwen/Qwen-Image",
        file_pattern="vae/diffusion_pytorch_model.safetensors",
        model_names=("qwen_image_vae",),
        latent_factor=16,
        adapter="direct",
    ),
    "flux1": VAESpec(
        key="flux1",
        label="FLUX.1 AE",
        model_id="black-forest-labs/FLUX.1-dev",
        file_pattern="ae.safetensors",
        model_names=("flux_vae_encoder", "flux_vae_decoder"),
        latent_factor=8,
        adapter="pair",
    ),
    "flux2": VAESpec(
        key="flux2",
        label="FLUX.2 Klein VAE",
        model_id="black-forest-labs/FLUX.2-klein-4B",
        file_pattern="vae/diffusion_pytorch_model.safetensors",
        model_names=("flux2_vae",),
        latent_factor=16,
        adapter="direct",
    ),
    "z_image": VAESpec(
        key="z_image",
        label="Z-Image VAE",
        model_id="Tongyi-MAI/Z-Image-Turbo",
        file_pattern="vae/diffusion_pytorch_model.safetensors",
        model_names=("flux_vae_encoder", "flux_vae_decoder"),
        latent_factor=8,
        adapter="pair",
    ),
    "sdxl": VAESpec(
        key="sdxl",
        label="Stable Diffusion XL VAE",
        model_id="stabilityai/stable-diffusion-xl-base-1.0",
        file_pattern="vae/diffusion_pytorch_model.safetensors",
        model_names=("stable_diffusion_xl_vae",),
        latent_factor=8,
        adapter="posterior_mode",
    ),
    "sd15": VAESpec(
        key="sd15",
        label="Stable Diffusion v1.5 VAE",
        model_id="AI-ModelScope/stable-diffusion-v1-5",
        file_pattern="vae/diffusion_pytorch_model.safetensors",
        model_names=("stable_diffusion_vae",),
        latent_factor=8,
        adapter="posterior_mode",
    ),
}
DEFAULT_VAES = ["qwen_image", "flux1", "flux2", "z_image", "sdxl", "sd15"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 0: compare pure VAE encode/decode reconstructions without diffusion denoising."
    )
    parser.add_argument("--image", default=DEFAULT_IMAGE_URL, help="Local RGB image path or HTTP(S) URL.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/phase0_vae_reconstruction"))
    parser.add_argument("--model-dir", type=Path, default=Path("models"))
    parser.add_argument(
        "--download-source",
        choices=("modelscope", "huggingface"),
        default="modelscope",
        help="Checkpoint source understood by DiffSynth ModelConfig.",
    )
    parser.add_argument(
        "--vae",
        nargs="+",
        choices=tuple(VAE_SPECS),
        default=DEFAULT_VAES,
        help="VAE families to run sequentially.",
    )
    parser.add_argument("--device", default="cuda", help="Inference device, normally cuda.")
    parser.add_argument(
        "--dtype",
        choices=("float32", "bfloat16", "float16"),
        default="float32",
        help="Use float32 for the cleanest measurement; bfloat16 reduces GPU memory.",
    )
    parser.add_argument(
        "--roi",
        action="append",
        default=[],
        metavar="NAME:X0,Y0,X1,Y1",
        help="Text-region rectangle in original input coordinates. Repeat for multiple regions.",
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        default=64,
        help="Size of non-overlapping patches used to rank where errors concentrate.",
    )
    parser.add_argument(
        "--error-scale",
        type=float,
        default=4.0,
        help="Brightness multiplier for saved grayscale absolute-error visualization only.",
    )
    parser.add_argument("--download-only", action="store_true", help="Download selected VAE files and exit.")
    parser.add_argument("--strict", action="store_true", help="Stop when any selected VAE fails.")
    return parser.parse_args()


def dtype_from_name(name: str) -> torch.dtype:
    return {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[name]


def get_spec_config(spec: VAESpec, args: argparse.Namespace) -> ModelConfig:
    return ModelConfig(
        model_id=spec.model_id,
        origin_file_pattern=spec.file_pattern,
        download_source=args.download_source,
        local_model_path=str(args.model_dir),
    )


def download_checkpoint(spec: VAESpec, args: argparse.Namespace):
    config = get_spec_config(spec, args)
    config.download_if_necessary()
    print(f"[downloaded] {spec.key}: {config.path}")
    return config.path


def open_image(source: str) -> Image.Image:
    if source.startswith(("http://", "https://")):
        request = urllib.request.Request(source, headers={"User-Agent": "vae-reconstruction-experiment/1.0"})
        with urllib.request.urlopen(request) as response:
            data = response.read()
        return Image.open(io.BytesIO(data)).convert("RGB")
    return Image.open(source).convert("RGB")


def image_to_padded_tensor(image: Image.Image, multiple: int) -> tuple[torch.Tensor, dict]:
    array = np.asarray(image, dtype=np.float32)
    tensor = torch.from_numpy(array.copy()).permute(2, 0, 1).unsqueeze(0) / 127.5 - 1.0
    height, width = array.shape[:2]
    pad_bottom = (-height) % multiple
    pad_right = (-width) % multiple
    if pad_bottom or pad_right:
        tensor = F.pad(tensor, (0, pad_right, 0, pad_bottom), mode="replicate")
    return tensor, {
        "original_height": height,
        "original_width": width,
        "pad_bottom": pad_bottom,
        "pad_right": pad_right,
        "padded_height": height + pad_bottom,
        "padded_width": width + pad_right,
        "multiple": multiple,
    }


def tensor_to_image(tensor: torch.Tensor, height: int, width: int) -> Image.Image:
    tensor = tensor[0, :, :height, :width].detach().float().cpu()
    tensor = ((tensor + 1.0) * 127.5).round().clamp(0, 255).to(torch.uint8)
    array = tensor.permute(1, 2, 0).numpy()
    return Image.fromarray(array, mode="RGB")


def model_vram_config(device: str, dtype: torch.dtype) -> dict:
    return {
        "offload_dtype": None,
        "offload_device": None,
        "onload_dtype": dtype,
        "onload_device": device,
        "preparing_dtype": dtype,
        "preparing_device": device,
        "computation_dtype": dtype,
        "computation_device": device,
    }


def reconstruct(spec: VAESpec, input_tensor: torch.Tensor, args: argparse.Namespace) -> tuple[torch.Tensor, tuple]:
    dtype = dtype_from_name(args.dtype)
    path = download_checkpoint(spec, args)
    pool = ModelPool()
    pool.auto_load_model(path, vram_config=model_vram_config(args.device, dtype))
    model_input = input_tensor.to(device=args.device, dtype=dtype)

    with torch.inference_mode():
        if spec.adapter == "pair":
            encoder = pool.fetch_model(spec.model_names[0])
            decoder = pool.fetch_model(spec.model_names[1])
            latent = encoder(model_input)
            reconstruction = decoder(latent)
        else:
            vae = pool.fetch_model(spec.model_names[0])
            if spec.adapter == "posterior_mode":
                latent = vae.encode(model_input).mode()
            else:
                latent = vae.encode(model_input)
            reconstruction = vae.decode(latent)

    latent_shape = tuple(latent.shape)
    reconstruction = reconstruction.detach().float().cpu()
    del latent, model_input, pool
    gc.collect()
    if args.device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()
    return reconstruction, latent_shape


def parse_rois(values: list[str], width: int, height: int) -> list[dict]:
    regions = [{"name": "full", "box": (0, 0, width, height)}]
    for value in values:
        try:
            name, raw_box = value.split(":", 1)
            box = tuple(int(part) for part in raw_box.split(","))
        except ValueError as exc:
            raise ValueError(f"Invalid --roi {value!r}; expected NAME:X0,Y0,X1,Y1.") from exc
        if len(box) != 4:
            raise ValueError(f"Invalid --roi {value!r}; expected four integer coordinates.")
        x0, y0, x1, y1 = box
        if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
            raise ValueError(f"ROI {value!r} is outside the original image size {width}x{height}.")
        safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", name.strip()) or "roi"
        regions.append({"name": safe_name, "box": box})
    return regions


def metrics(reference: Image.Image, reconstructed: Image.Image, box: tuple[int, int, int, int]) -> dict:
    reference_array = np.asarray(reference.crop(box), dtype=np.float32)
    reconstructed_array = np.asarray(reconstructed.crop(box), dtype=np.float32)
    difference = reference_array - reconstructed_array
    mse = float(np.mean(difference ** 2))
    rmse = math.sqrt(mse)
    return {
        "mae_255": float(np.mean(np.abs(difference))),
        "rmse_255": rmse,
        "psnr_db": float("inf") if mse == 0 else 20.0 * math.log10(255.0 / rmse),
    }


def save_error_map(
    reference: Image.Image,
    reconstructed: Image.Image,
    path: Path,
    scale: float,
    box: tuple[int, int, int, int] | None = None,
) -> None:
    if box is not None:
        reference = reference.crop(box)
        reconstructed = reconstructed.crop(box)
    difference = np.abs(
        np.asarray(reference, dtype=np.float32) - np.asarray(reconstructed, dtype=np.float32)
    ).mean(axis=2)
    visual = np.clip(difference * scale, 0, 255).astype(np.uint8)
    Image.fromarray(visual, mode="L").save(path)


def save_contact_sheet(panels: list[tuple[str, Image.Image]], path: Path) -> None:
    columns = min(3, len(panels))
    rows = math.ceil(len(panels) / columns)
    tile_width = max(panel.width for _, panel in panels)
    tile_height = max(panel.height for _, panel in panels)
    header_height = 24
    sheet = Image.new("RGB", (columns * tile_width, rows * (tile_height + header_height)), "white")
    draw = ImageDraw.Draw(sheet)
    for index, (label, panel) in enumerate(panels):
        x = index % columns * tile_width
        y = index // columns * (tile_height + header_height)
        draw.text((x + 5, y + 7), label, fill="black")
        sheet.paste(panel, (x, y + header_height))
    sheet.save(path)


def patch_metric_rows(
    reference: Image.Image, reconstructed: Image.Image, model: str, patch_size: int
) -> list[dict]:
    if patch_size <= 0:
        return []
    rows = []
    for y0 in range(0, reference.height, patch_size):
        for x0 in range(0, reference.width, patch_size):
            box = (x0, y0, min(x0 + patch_size, reference.width), min(y0 + patch_size, reference.height))
            row = {"model": model, "x0": box[0], "y0": box[1], "x1": box[2], "y1": box[3]}
            row.update(metrics(reference, reconstructed, box))
            rows.append(row)
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    selected_specs = [VAE_SPECS[key] for key in args.vae]
    if args.download_only:
        for spec in selected_specs:
            download_checkpoint(spec, args)
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)
    original = open_image(args.image)
    original.save(args.output_dir / "input_original.png")
    multiple = math.lcm(*(spec.latent_factor for spec in selected_specs))
    input_tensor, input_info = image_to_padded_tensor(original, multiple)
    regions = parse_rois(args.roi, original.width, original.height)

    result_manifest = {
        "image_source": args.image,
        "input": input_info,
        "dtype": args.dtype,
        "device": args.device,
        "download_source": args.download_source,
        "regions": regions,
        "models": [],
        "errors": [],
    }
    metric_rows = []
    patch_rows = []
    reconstructions: list[tuple[str, Image.Image]] = []
    for spec in selected_specs:
        print(f"\n[run] {spec.key}: {spec.label}")
        try:
            output_tensor, latent_shape = reconstruct(spec, input_tensor, args)
            reconstruction = tensor_to_image(
                output_tensor, input_info["original_height"], input_info["original_width"]
            )
            reconstruction.save(args.output_dir / f"recon_{spec.key}.png")
            save_error_map(
                original,
                reconstruction,
                args.output_dir / f"error_x{args.error_scale:g}_{spec.key}.png",
                args.error_scale,
            )
            reconstructions.append((spec.key, reconstruction))
            result_manifest["models"].append({**asdict(spec), "latent_shape": latent_shape})
            for region in regions:
                row = {"model": spec.key, "region": region["name"], **dict(zip(("x0", "y0", "x1", "y1"), region["box"]))}
                row.update(metrics(original, reconstruction, region["box"]))
                metric_rows.append(row)
                if region["name"] != "full":
                    reconstruction.crop(region["box"]).save(
                        args.output_dir / f"recon_{spec.key}__roi_{region['name']}.png"
                    )
                    save_error_map(
                        original,
                        reconstruction,
                        args.output_dir / f"error_x{args.error_scale:g}_{spec.key}__roi_{region['name']}.png",
                        args.error_scale,
                        region["box"],
                    )
            patch_rows.extend(patch_metric_rows(original, reconstruction, spec.key, args.patch_size))
        except Exception as exc:
            result_manifest["errors"].append({"model": spec.key, "error": repr(exc)})
            print(f"[failed] {spec.key}: {exc!r}")
            if args.strict:
                raise
        finally:
            gc.collect()
            if args.device.startswith("cuda") and torch.cuda.is_available():
                torch.cuda.empty_cache()

    if not reconstructions:
        raise RuntimeError("Every selected VAE failed; see errors printed above.")

    original_panels = [("input", original), *reconstructions]
    save_contact_sheet(original_panels, args.output_dir / "comparison_full.png")
    for region in regions:
        if region["name"] != "full":
            panels = [("input", original.crop(region["box"]))]
            panels.extend((key, image.crop(region["box"])) for key, image in reconstructions)
            save_contact_sheet(panels, args.output_dir / f"comparison_roi_{region['name']}.png")

    write_csv(args.output_dir / "metrics.csv", metric_rows)
    write_csv(args.output_dir / "patch_metrics.csv", patch_rows)
    with (args.output_dir / "manifest.json").open("w", encoding="utf-8") as file:
        json.dump(result_manifest, file, ensure_ascii=False, indent=2)

    print("\n[metrics]")
    for row in metric_rows:
        print(
            f"{row['model']:12s} {row['region']:16s} "
            f"MAE={row['mae_255']:.3f} RMSE={row['rmse_255']:.3f} PSNR={row['psnr_db']:.3f} dB"
        )
    if patch_rows:
        print("\n[top-error patches, inspect these alongside text regions]")
        for key, _ in reconstructions:
            worst = sorted(
                (row for row in patch_rows if row["model"] == key),
                key=lambda row: row["mae_255"],
                reverse=True,
            )[:5]
            boxes = ", ".join(
                f"({row['x0']},{row['y0']},{row['x1']},{row['y1']}):{row['mae_255']:.2f}"
                for row in worst
            )
            print(f"{key:12s} {boxes}")
    print(f"\nSaved results to {args.output_dir}")


if __name__ == "__main__":
    main()
