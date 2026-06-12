import argparse
import csv
import gc
import json
import os
import queue
import subprocess
import sys
import threading
from dataclasses import dataclass
from pathlib import Path

from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import all_i2i_reference_to_target as base


DEFAULT_DESIGN_SOURCE = "outputs/edit_pair_validation/amazon_lipcare_design/design_preview.png"
DEFAULT_TARGET = "data/edit_pair_validation/amazon_lipcare/target.jpg"
DEFAULT_OUTPUT_DIR = "outputs/edit_pair_validation/design_i2i_to_target"

DESIGN_TO_TARGET_PROMPT = (
    "Create the final photorealistic square ecommerce advertising poster in the same composition as Picture 1. "
    "Picture 1 is the publication design draft and layout blueprint. If Picture 2 is provided, it is only a "
    "high-resolution package-label reference for the yellow tube; preserve its logo geometry, brand text, and small "
    "package text in the tube-label area. Do not place Picture 2 as a separate object. The result should match the "
    "Amazon target ad style: a clean white-background lip-care product poster with a yellow CNP Laboratory lip-care "
    "tube diagonally placed on the lower right, held by a small polished silver cosmetic applicator clip from the upper "
    "right. Keep the left-side typography and copy from the design draft: the large pink headline 'Ampule-Infused Lip "
    "Care', the gray copy 'Get honey-glow lips with ultra-dewy hydration', and the vertical benefit list with pink plus "
    "signs: 'Dead skin cells', 'Wrinkles', 'Elasticity', 'Hydration', 'Glow'. The flat orange puddle and drip in the "
    "design draft are only rough placeholders. Replace them completely with photorealistic glossy amber honey serum: "
    "transparent golden highlights, natural viscosity, surface tension, a smooth continuous drip from the tube nozzle, "
    "a small realistic pooled mound at the bottom, subtle refraction, wet specular reflections, and soft contact "
    "shadows. Make the clip, tube plastic, nozzle, honey, reflections, and background look like a polished studio "
    "product photograph. Preserve the existing yellow product tube identity: do not change the tube silhouette, logo "
    "placement, label layout, brand text, or visible small package text. Refine the design draft into a polished "
    "target-like product advertisement rather than creating a new layout."
)

DESIGN_TO_TARGET_NEGATIVE_PROMPT = (
    "changed brand, changed package text, rewritten label, unreadable small text, misspelled text, extra product, "
    "duplicated tube, wrong layout, missing headline, missing benefit list, deformed tube, cropped product, blurry, "
    "low quality, messy background"
)

# Target-layout ROIs are inherited from the original target task. These measure
# target-like layout/style, not source text preservation.
TARGET_ROIS = base.TARGET_ROIS

# ROIs on the 1024 design draft. These measure whether the design-draft product
# label survives when the design itself is used as reference input.
DESIGN_PRESERVE_ROIS = {
    "design_product_right": (0.46, 0.18, 0.89, 0.84),
    "design_tube_label": (0.58, 0.37, 0.78, 0.60),
    "design_vertical_brand": (0.50, 0.49, 0.66, 0.72),
}

QWEN_MULTI_IMAGE_MODELS = {
    "qwen_image_edit_2509",
    "qwen_image_edit_2511",
    "qwen_image_edit_2511_lightning",
    "firered_image_edit_1_0",
    "firered_image_edit_1_1",
}


@dataclass(frozen=True)
class EvalJob:
    model: str
    seed: int
    gpu: str | None = None


def resolve_model_names(names: list[str]) -> list[str]:
    return base.resolve_model_names(names)


def require_file(path: str | Path, role: str) -> Path:
    resolved = Path(path)
    if not resolved.is_file():
        raise SystemExit(f"Missing {role}: {resolved}")
    return resolved


def normalized_box(box: tuple[float, float, float, float], size: tuple[int, int]):
    return base.normalized_box(box, size)


def parse_box(value: str | None) -> tuple[float, float, float, float] | None:
    if value is None:
        return None
    parts = [float(part.strip()) for part in value.split(",")]
    if len(parts) != 4:
        raise SystemExit(f"Expected box as left,top,right,bottom; got: {value}")
    return tuple(parts)


def crop_box(image: Image.Image, box: tuple[float, float, float, float] | None) -> Image.Image:
    if box is None:
        return image.copy()
    pixel_box = normalized_box(box, image.size) if all(0.0 <= value <= 1.0 for value in box) else tuple(round(v) for v in box)
    left, top, right, bottom = pixel_box
    left = max(0, min(left, image.width - 1))
    top = max(0, min(top, image.height - 1))
    right = max(left + 1, min(right, image.width))
    bottom = max(top + 1, min(bottom, image.height))
    return image.crop((left, top, right, bottom))


def prepare_roi_reference_images(
    args: argparse.Namespace,
    source: Image.Image,
    output_dir: Path,
    model_name: str,
) -> tuple[list[Image.Image], list[dict]]:
    if not args.use_roi_reference and not args.use_roi_attention_steering:
        return [], []
    if model_name not in QWEN_MULTI_IMAGE_MODELS:
        return [], []

    references: list[Image.Image] = []
    metadata: list[dict] = []
    reference_dir = output_dir / model_name / "reference_inputs"
    reference_dir.mkdir(parents=True, exist_ok=True)

    roi_source_path = Path(args.roi_reference_image) if args.roi_reference_image else None
    roi_source = Image.open(roi_source_path).convert("RGB") if roi_source_path else source
    roi_box = parse_box(args.roi_reference_box)
    roi_crop = crop_box(roi_source, roi_box)
    roi_crop = roi_crop.resize((args.roi_reference_size, args.roi_reference_size), Image.Resampling.LANCZOS)
    roi_output_path = reference_dir / f"seed{args.seed}_label_roi.png"
    roi_crop.save(roi_output_path)
    references.append(roi_crop)
    metadata.append(
        {
            "role": "high_resolution_label_reference",
            "source": str(roi_source_path or args.source),
            "box": args.roi_reference_box,
            "image": str(roi_output_path),
        }
    )

    if args.product_reference:
        product_path = require_file(args.product_reference, "product reference image")
        product_reference = Image.open(product_path).convert("RGB")
        product_box = parse_box(args.product_reference_box)
        product_reference = crop_box(product_reference, product_box)
        if args.product_reference_size:
            product_reference.thumbnail((args.product_reference_size, args.product_reference_size), Image.Resampling.LANCZOS)
            canvas = Image.new("RGB", (args.product_reference_size, args.product_reference_size), "white")
            canvas.paste(
                product_reference,
                ((canvas.width - product_reference.width) // 2, (canvas.height - product_reference.height) // 2),
            )
            product_reference = canvas
        product_output_path = reference_dir / f"seed{args.seed}_product_reference.png"
        product_reference.save(product_output_path)
        references.append(product_reference)
        metadata.append(
            {
                "role": "product_identity_reference",
                "source": str(product_path),
                "box": args.product_reference_box,
                "image": str(product_output_path),
            }
        )

    return references, metadata


def generate_one(args: argparse.Namespace) -> None:
    spec = base.MODEL_MAP[args.model]
    source_path = require_file(args.source, "design source image")
    source = Image.open(source_path).convert("RGB")
    output_dir = Path(args.output_dir) / spec.name
    output_dir.mkdir(parents=True, exist_ok=True)
    steps = args.num_inference_steps or spec.default_steps
    cfg_scale = args.cfg_scale if args.cfg_scale is not None else spec.default_cfg
    output_path = output_dir / f"seed{args.seed}.png"
    metadata_path = output_dir / f"seed{args.seed}.json"
    extra_edit_images, extra_edit_image_metadata = prepare_roi_reference_images(args, source, Path(args.output_dir), spec.name)
    runner_kwargs = {
        "source": source,
        "prompt": args.prompt,
        "negative_prompt": args.negative_prompt,
        "seed": args.seed,
        "height": args.height,
        "width": args.width,
        "steps": steps,
        "cfg_scale": cfg_scale,
        "dtype": args.dtype,
        "device": args.device,
        "denoising_strength": args.denoising_strength,
    }
    if extra_edit_images:
        runner_kwargs["extra_edit_images"] = extra_edit_images
    if args.use_roi_attention_steering and extra_edit_images and spec.name in QWEN_MULTI_IMAGE_MODELS:
        runner_kwargs["edit_latent_attention_repeat_indices"] = [1]
        runner_kwargs["edit_latent_attention_repeat"] = args.roi_attention_repeat
    image = spec.runner(**runner_kwargs)
    image.save(output_path)
    metadata = {
        "model": spec.name,
        "family": spec.family,
        "description": spec.description,
        "design_source": str(source_path),
        "generated": str(output_path),
        "target_was_not_used_for_generation": True,
        "prompt": args.prompt,
        "negative_prompt": args.negative_prompt,
        "extra_edit_images": extra_edit_image_metadata,
        "roi_attention_steering": {
            "enabled": bool(args.use_roi_attention_steering and extra_edit_images and spec.name in QWEN_MULTI_IMAGE_MODELS),
            "method": "repeat_roi_reference_edit_latents",
            "repeat_indices": [1] if args.use_roi_attention_steering and extra_edit_images and spec.name in QWEN_MULTI_IMAGE_MODELS else [],
            "repeat": args.roi_attention_repeat,
        },
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


def compare_one(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    design_path = require_file(args.source, "design source image")
    target_path = require_file(args.target, "target image")
    generated_path = require_file(args.generated, "generated image")
    design = Image.open(design_path).convert("RGB")
    target = Image.open(target_path).convert("RGB")
    generated = Image.open(generated_path).convert("RGB")

    generated_for_target = generated.resize(target.size, Image.Resampling.LANCZOS)
    generated_for_design = generated.resize(design.size, Image.Resampling.LANCZOS)
    target_preview = target.resize(design.size, Image.Resampling.LANCZOS)

    base.horizontal_montage(
        [
            base.labelled_tile(design, "design source / reference"),
            base.labelled_tile(target_preview, "held-out target"),
            base.labelled_tile(generated_for_design, "generated"),
        ]
    ).save(output_dir / "comparison_full.png")

    target_metrics = {}
    for name, box in TARGET_ROIS.items():
        pixel_box = normalized_box(box, target.size)
        target_crop = target.crop(pixel_box)
        generated_crop = generated_for_target.crop(pixel_box)
        target_metrics[name] = {"target_box": pixel_box, **base.image_metrics(target_crop, generated_crop)}
        if name != "full":
            base.horizontal_montage(
                [
                    base.labelled_tile(target_crop, f"target: {name}"),
                    base.labelled_tile(generated_crop, f"generated: {name}"),
                ]
            ).save(output_dir / f"comparison_target_{name}.png")

    preserve_metrics = {}
    for name, box in DESIGN_PRESERVE_ROIS.items():
        pixel_box = normalized_box(box, design.size)
        design_crop = design.crop(pixel_box)
        generated_crop = generated_for_design.crop(pixel_box)
        preserve_metrics[name] = {"design_box": pixel_box, **base.image_metrics(design_crop, generated_crop)}
        base.horizontal_montage(
            [
                base.labelled_tile(design_crop, f"design: {name}"),
                base.labelled_tile(generated_crop, f"generated: {name}"),
            ]
        ).save(output_dir / f"comparison_preserve_{name}.png")

    result = {
        "design_source": str(design_path),
        "target": str(target_path),
        "generated": str(generated_path),
        "target_layout_metrics": target_metrics,
        "design_preservation_metrics": preserve_metrics,
        "note": (
            "Target metrics measure whether the design-driven output moves toward the held-out target. "
            "Design preservation metrics measure whether product/label pixels from design_preview survive; "
            "inspect comparison_preserve_design_tube_label.png for small package text."
        ),
    }
    (output_dir / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    for name, values in target_metrics.items():
        print(
            f"target::{name:<22} MAE={values['mae']:.3f} "
            f"RMSE={values['rmse']:.3f} PSNR={values['psnr_db']:.3f} dB"
        )
    for name, values in preserve_metrics.items():
        print(
            f"preserve::{name:<20} MAE={values['mae']:.3f} "
            f"RMSE={values['rmse']:.3f} PSNR={values['psnr_db']:.3f} dB"
        )


def build_eval_jobs(args: argparse.Namespace) -> list[EvalJob]:
    return [
        EvalJob(model=model, seed=seed)
        for model in resolve_model_names(args.models)
        for seed in args.seeds
    ]


def build_generate_command(args: argparse.Namespace, job: EvalJob, source_path: Path, output_dir: Path, device: str) -> list[str]:
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
    if args.use_roi_reference:
        command.append("--use-roi-reference")
    if args.use_roi_attention_steering:
        command.append("--use-roi-attention-steering")
        command.extend(["--roi-attention-repeat", str(args.roi_attention_repeat)])
    command.extend(["--roi-reference-box", args.roi_reference_box])
    command.extend(["--roi-reference-size", str(args.roi_reference_size)])
    if args.roi_reference_image:
        command.extend(["--roi-reference-image", args.roi_reference_image])
    if args.product_reference:
        command.extend(["--product-reference", args.product_reference])
    if args.product_reference_box:
        command.extend(["--product-reference-box", args.product_reference_box])
    if args.product_reference_size:
        command.extend(["--product-reference-size", str(args.product_reference_size)])
    if args.num_inference_steps is not None:
        command.extend(["--num-inference-steps", str(args.num_inference_steps)])
    if args.cfg_scale is not None:
        command.extend(["--cfg-scale", str(args.cfg_scale)])
    if args.download_source is not None:
        command.extend(["--download-source", args.download_source])
    return command


def build_compare_command(job: EvalJob, source_path: Path, target_path: Path, generated: Path, eval_dir: Path) -> list[str]:
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
    print(f"\n=== generate {job.model} seed={job.seed}{gpu_label} from design draft ===", flush=True)
    generated = output_dir / job.model / f"seed{job.seed}.png"
    eval_dir = output_dir / job.model / f"eval_seed{job.seed}"
    row = {"model": job.model, "seed": job.seed}
    if job.gpu is not None:
        row["gpu"] = job.gpu
    if env is None:
        env = os.environ.copy()
    if args.download_source is not None:
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
    metrics = json.loads((eval_dir / "metrics.json").read_text(encoding="utf-8"))
    target_metrics = metrics["target_layout_metrics"]
    preserve_metrics = metrics["design_preservation_metrics"]
    row.update(
        {
            "status": "ok",
            "generated": str(generated),
            "eval_dir": str(eval_dir),
            "target_full_psnr_db": target_metrics["full"]["psnr_db"],
            "target_product_right_psnr_db": target_metrics["product_right"]["psnr_db"],
            "target_tube_label_right_psnr_db": target_metrics["tube_label_right"]["psnr_db"],
            "preserve_design_tube_label_psnr_db": preserve_metrics["design_tube_label"]["psnr_db"],
            "preserve_design_vertical_brand_psnr_db": preserve_metrics["design_vertical_brand"]["psnr_db"],
        }
    )
    return row


def prepare_run_paths(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    source_path = require_file(args.source, "design source image")
    target_path = require_file(args.target, "target image")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    return source_path, target_path, output_dir


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
    rows = [
        run_eval_job(
            args=args,
            job=job,
            source_path=source_path,
            target_path=target_path,
            output_dir=output_dir,
            device=args.device,
        )
        for job in build_eval_jobs(args)
    ]
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
        if args.download_source is not None:
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
    print(f"Running {len(all_jobs)} design-driven jobs across GPUs: {', '.join(gpus)}", flush=True)
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


def add_generation_args(parser: argparse.ArgumentParser, seed_required: bool = True) -> None:
    parser.add_argument("--prompt", default=DESIGN_TO_TARGET_PROMPT)
    parser.add_argument("--negative-prompt", default=DESIGN_TO_TARGET_NEGATIVE_PROMPT)
    parser.add_argument("--seed", type=int, required=seed_required)
    parser.add_argument("--num-inference-steps", type=int, default=None)
    parser.add_argument("--cfg-scale", type=float, default=None)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--dtype", choices=["bfloat16", "float16"], default="bfloat16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--denoising-strength", type=float, default=0.85)
    parser.add_argument(
        "--use-roi-reference",
        action="store_true",
        help="For Qwen multi-image edit models, add a high-resolution label crop as an extra native image input.",
    )
    parser.add_argument(
        "--use-roi-attention-steering",
        action="store_true",
        help=(
            "For Qwen multi-image edit models, add the label ROI reference and repeat its edit latents in the DiT "
            "image-token sequence to increase attention mass without spatial masks or training."
        ),
    )
    parser.add_argument(
        "--roi-attention-repeat",
        type=int,
        default=3,
        help="How many total copies of the ROI reference edit latents to expose to DiT attention.",
    )
    parser.add_argument(
        "--roi-reference-image",
        default=None,
        help="Image to crop the ROI reference from. Defaults to --source.",
    )
    parser.add_argument(
        "--roi-reference-box",
        default="0.58,0.37,0.78,0.60",
        help="ROI crop box as normalized or pixel left,top,right,bottom. Defaults to design_tube_label.",
    )
    parser.add_argument("--roi-reference-size", type=int, default=1024)
    parser.add_argument(
        "--product-reference",
        default=None,
        help="Optional extra product identity reference image, passed after the label ROI for Qwen models.",
    )
    parser.add_argument(
        "--product-reference-box",
        default=None,
        help="Optional crop box for --product-reference.",
    )
    parser.add_argument("--product-reference-size", type=int, default=1024)
    parser.add_argument(
        "--download-source",
        choices=["modelscope", "huggingface"],
        default=os.environ.get("DIFFSYNTH_DOWNLOAD_SOURCE", "modelscope"),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate whether a detailed target-like design draft improves small package-text preservation "
            "for repository image-to-image/edit models."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser("list-models")
    list_parser.add_argument("--json", action="store_true")
    list_parser.set_defaults(func=base.list_models)

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
    verify_parser.set_defaults(func=base.command_verify_models)

    generate = subparsers.add_parser("generate")
    generate.add_argument("--model", required=True, choices=sorted(base.MODEL_MAP))
    generate.add_argument("--source", default=DEFAULT_DESIGN_SOURCE)
    generate.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    add_generation_args(generate)
    generate.set_defaults(func=generate_one)

    compare = subparsers.add_parser("compare")
    compare.add_argument("--source", default=DEFAULT_DESIGN_SOURCE)
    compare.add_argument("--target", default=DEFAULT_TARGET)
    compare.add_argument("--generated", required=True)
    compare.add_argument("--output-dir", required=True)
    compare.set_defaults(func=compare_one)

    run_all = subparsers.add_parser("run-all")
    run_all.add_argument("--models", nargs="+", default=["all_relevant"])
    run_all.add_argument("--source", default=DEFAULT_DESIGN_SOURCE)
    run_all.add_argument("--target", default=DEFAULT_TARGET)
    run_all.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    run_all.add_argument("--seeds", nargs="+", type=int, default=[0])
    run_all.add_argument("--cwd", default=os.getcwd())
    add_generation_args(run_all, seed_required=False)
    run_all.set_defaults(func=command_run_all)

    run_parallel = subparsers.add_parser("run-parallel")
    run_parallel.add_argument("--models", nargs="+", default=["all_relevant"])
    run_parallel.add_argument("--source", default=DEFAULT_DESIGN_SOURCE)
    run_parallel.add_argument("--target", default=DEFAULT_TARGET)
    run_parallel.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    run_parallel.add_argument("--seeds", nargs="+", type=int, default=[0])
    run_parallel.add_argument("--gpus", nargs="+", required=True)
    run_parallel.add_argument("--worker-device", default="cuda")
    run_parallel.add_argument("--cwd", default=os.getcwd())
    add_generation_args(run_parallel, seed_required=False)
    run_parallel.set_defaults(func=command_run_parallel)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
