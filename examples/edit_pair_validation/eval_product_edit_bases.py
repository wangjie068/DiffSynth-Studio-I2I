#!/usr/bin/env python3
import argparse
import gc
import json
import random
import urllib.request
from io import BytesIO
from pathlib import Path

import torch
from PIL import Image, ImageDraw


USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)


def read_records(path):
    if path.suffix == ".jsonl":
        with path.open(encoding="utf-8", errors="ignore") as in_file:
            return [json.loads(line) for line in in_file if line.strip()]
    return json.load(path.open(encoding="utf-8"))


def prompt_of(record):
    return record.get("prompt") or record.get("edit_instruction_detailed") or record.get("edit_instruction") or ""


def pair_id_of(record, index):
    return str(record.get("pair_id") or f"case_{index:05d}")


def safe_name(value):
    return "".join(ch if ch.isalnum() or ch in "-_" else "-" for ch in value)[:120]


def fit_size(width, height, max_pixels):
    scale = min(1.0, (max_pixels / max(1, width * height)) ** 0.5)
    width = max(256, int(width * scale) // 16 * 16)
    height = max(256, int(height * scale) // 16 * 16)
    return width, height


def download_image(url, path, timeout, retries=2):
    if path.exists():
        return
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    last_error = None
    for _ in range(retries + 1):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                raw = response.read()
            break
        except Exception as error:
            last_error = error
    else:
        raise last_error
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.open(BytesIO(raw)).convert("RGB").save(path, quality=95)


def materialize_case(record, index, image_root, output_dir, timeout):
    pair_id = safe_name(pair_id_of(record, index))
    case_dir = output_dir / "cases" / f"{index:03d}-{pair_id}"
    source_path = case_dir / "source.jpg"
    target_path = case_dir / "target.jpg"

    if record.get("edit_image") and record.get("image"):
        src = image_root / record["edit_image"]
        tgt = image_root / record["image"]
        source_path.parent.mkdir(parents=True, exist_ok=True)
        if not source_path.exists():
            Image.open(src).convert("RGB").save(source_path, quality=95)
        if not target_path.exists():
            Image.open(tgt).convert("RGB").save(target_path, quality=95)
    else:
        download_image(record["source_url"], source_path, timeout)
        download_image(record["target_url"], target_path, timeout)

    return {
        "index": index,
        "pair_id": pair_id,
        "prompt": prompt_of(record),
        "source": str(source_path),
        "target": str(target_path),
        "raw": record,
    }


def load_pipe(model):
    if model in {"qwen2511", "firered11"}:
        from diffsynth.pipelines.qwen_image import ModelConfig, QwenImagePipeline

        model_id = {
            "qwen2511": "Qwen/Qwen-Image-Edit-2511",
            "firered11": "FireRedTeam/FireRed-Image-Edit-1.1",
        }[model]
        return QwenImagePipeline.from_pretrained(
            torch_dtype=torch.bfloat16,
            device="cuda",
            model_configs=[
                ModelConfig(model_id=model_id, origin_file_pattern="transformer/diffusion_pytorch_model*.safetensors"),
                ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="text_encoder/model*.safetensors"),
                ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="vae/diffusion_pytorch_model.safetensors"),
            ],
            processor_config=ModelConfig(model_id="Qwen/Qwen-Image-Edit", origin_file_pattern="processor/"),
        )
    if model == "boogu":
        from diffsynth.pipelines.boogu_image import BooguImagePipeline, ModelConfig

        return BooguImagePipeline.from_pretrained(
            torch_dtype=torch.bfloat16,
            device="cuda",
            model_configs=[
                ModelConfig(model_id="Boogu/Boogu-Image-0.1-Edit", origin_file_pattern="transformer/*.safetensors"),
                ModelConfig(model_id="Boogu/Boogu-Image-0.1-Edit", origin_file_pattern="mllm/*.safetensors"),
                ModelConfig(model_id="Boogu/Boogu-Image-0.1-Edit", origin_file_pattern="vae/*.safetensors"),
            ],
            processor_config=ModelConfig(model_id="Boogu/Boogu-Image-0.1-Edit", origin_file_pattern="mllm/"),
        )
    raise ValueError(f"Unsupported model: {model}")


def infer(pipe, model, case, steps, seed, max_pixels):
    source = Image.open(case["source"]).convert("RGB")
    target = Image.open(case["target"]).convert("RGB")
    width, height = fit_size(*target.size, max_pixels=max_pixels)
    prompt = case["prompt"]
    if model == "qwen2511":
        return pipe(
            prompt,
            edit_image=[source],
            seed=seed,
            num_inference_steps=steps,
            height=height,
            width=width,
            edit_image_auto_resize=True,
            zero_cond_t=True,
        )
    if model == "firered11":
        return pipe(
            prompt,
            edit_image=[source],
            seed=seed,
            num_inference_steps=steps,
            height=height,
            width=width,
            edit_image_auto_resize=True,
        )
    return pipe(
        prompt=prompt,
        negative_prompt="",
        edit_image=source,
        height=height,
        width=width,
        seed=seed,
        rand_device="cuda",
        num_inference_steps=steps,
        cfg_scale=1.0,
    )


def tile(image, label, size=256):
    image = image.copy()
    image.thumbnail((size, size))
    canvas = Image.new("RGB", (size, size + 24), "white")
    canvas.paste(image, ((size - image.width) // 2, (size - image.height) // 2))
    ImageDraw.Draw(canvas).text((6, size + 6), label[:34], fill=(0, 0, 0))
    return canvas


def write_contact_sheet(cases, models, output_dir):
    rows = []
    for case in cases:
        cells = [
            tile(Image.open(case["source"]).convert("RGB"), "source"),
            tile(Image.open(case["target"]).convert("RGB"), "target"),
        ]
        for model in models:
            path = output_dir / "outputs" / model / f"{case['index']:03d}.jpg"
            if path.exists():
                cells.append(tile(Image.open(path).convert("RGB"), model))
        row = Image.new("RGB", (sum(cell.width for cell in cells), cells[0].height), "white")
        x = 0
        for cell in cells:
            row.paste(cell, (x, 0))
            x += cell.width
        rows.append(row)
    sheet = Image.new("RGB", (max(row.width for row in rows), sum(row.height for row in rows)), "white")
    y = 0
    for row in rows:
        sheet.paste(row, (0, y))
        y += row.height
    sheet.save(output_dir / "contact_sheet.jpg", quality=95)


def main():
    parser = argparse.ArgumentParser(description="Run a small product edit base-model bakeoff.")
    parser.add_argument("--metadata", type=Path, default=Path("data/amazon_reviews_2023/hf_product_edit_annotations_smalltext_full_v3/pair_view/validation-00000-of-00001.jsonl"))
    parser.add_argument("--image-root", type=Path, default=Path("data/amazon_reviews_2023/qwen_image_edit_smalltext_full_v3"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/amazon_reviews_2023/base_model_eval"))
    parser.add_argument("--models", default="qwen2511,firered11,boogu")
    parser.add_argument("--limit", type=int, default=16)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--sample-seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--max-pixels", type=int, default=1048576)
    parser.add_argument("--timeout", type=float, default=30)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = read_records(args.metadata)
    random.Random(args.sample_seed).shuffle(records)

    cases = []
    skipped_cases = []
    for record in records:
        if len(cases) >= args.limit:
            break
        try:
            cases.append(materialize_case(record, len(cases), args.image_root, args.output_dir, args.timeout))
        except Exception as error:
            skipped_cases.append({"pair_id": pair_id_of(record, len(cases)), "error": str(error)})
            print(f"skip case {pair_id_of(record, len(cases))}: {error}")
    (args.output_dir / "cases.json").write_text(json.dumps(cases, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.output_dir / "skipped_cases.json").write_text(json.dumps(skipped_cases, ensure_ascii=False, indent=2), encoding="utf-8")
    if not cases:
        raise RuntimeError("No cases were materialized. Use downloaded metadata.json or increase --timeout.")

    models = [item.strip() for item in args.models.split(",") if item.strip()]
    failures = []
    for model in models:
        out_dir = args.output_dir / "outputs" / model
        out_dir.mkdir(parents=True, exist_ok=True)
        pipe = load_pipe(model)
        for case in cases:
            out_path = out_dir / f"{case['index']:03d}.jpg"
            if out_path.exists():
                print(f"skip {model} {case['index']:03d}")
                continue
            try:
                image = infer(pipe, model, case, args.steps, args.seed, args.max_pixels)
                image.save(out_path, quality=95)
                print(f"wrote {model} {case['index']:03d} {case['pair_id']}")
            except Exception as error:
                failures.append({"model": model, "index": case["index"], "pair_id": case["pair_id"], "error": str(error)})
                print(f"failed {model} {case['index']:03d}: {error}")
        del pipe
        gc.collect()
        torch.cuda.empty_cache()

    (args.output_dir / "failures.json").write_text(json.dumps(failures, ensure_ascii=False, indent=2), encoding="utf-8")
    write_contact_sheet(cases, models, args.output_dir)
    print(f"output_dir={args.output_dir}")
    print(f"contact_sheet={args.output_dir / 'contact_sheet.jpg'}")
    print(f"failures={len(failures)}")


if __name__ == "__main__":
    main()
