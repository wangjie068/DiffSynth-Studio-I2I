#!/usr/bin/env python3
import argparse
import base64
import json
import os
import secrets
import ssl
import time
import urllib.error
import urllib.request
from io import BytesIO
from pathlib import Path

from PIL import Image, ImageDraw


DEFAULT_BASE_URL = "https://gpt-i18n.byteintl.net/gpt/openapi/online/v2/crawl/openai"
DEFAULT_MODELS_FOR_SHEET = "qwen2511,qwen2511_lora,hidream_o1,flux2kleinbase9b,firered11,gpt_image"


def load_cases(path, limit):
    cases = json.load(path.open(encoding="utf-8"))
    if limit:
        return cases[:limit]
    return cases


def encode_multipart(fields, files):
    boundary = f"----codex-{secrets.token_hex(16)}"
    chunks = []
    for name, value in fields:
        chunks.append(f"--{boundary}\r\n".encode())
        chunks.append(f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode())
        chunks.append(str(value).encode())
        chunks.append(b"\r\n")
    for name, filename, content_type, data in files:
        chunks.append(f"--{boundary}\r\n".encode())
        chunks.append(
            f'Content-Disposition: form-data; name="{name}"; filename="{filename}"\r\n'.encode()
        )
        chunks.append(f"Content-Type: {content_type}\r\n\r\n".encode())
        chunks.append(data)
        chunks.append(b"\r\n")
    chunks.append(f"--{boundary}--\r\n".encode())
    return b"".join(chunks), f"multipart/form-data; boundary={boundary}"


def image_content_type(path):
    suffix = path.suffix.lower()
    if suffix in {".jpg", ".jpeg"}:
        return "image/jpeg"
    if suffix == ".png":
        return "image/png"
    if suffix == ".webp":
        return "image/webp"
    return "application/octet-stream"


def post_edit(base_url, api_key, case, model, quality, size, timeout, ssl_context):
    source_path = Path(case["source"])
    body, content_type = encode_multipart(
        fields=[
            ("prompt", case["prompt"]),
            ("model", model),
            ("quality", quality),
            ("size", size),
            ("n", "1"),
        ],
        files=[
            ("image[]", source_path.name, image_content_type(source_path), source_path.read_bytes()),
        ],
    )
    url = f"{base_url.rstrip('/')}/images/edits"
    request = urllib.request.Request(
        url,
        data=body,
        method="POST",
        headers={
            "Content-Type": content_type,
            "api-key": api_key,
            "X-TT-LOGID": f"product-gpt-image-{case['index']:03d}-{int(time.time())}",
        },
    )
    with urllib.request.urlopen(request, timeout=timeout, context=ssl_context) as response:
        payload = json.loads(response.read().decode("utf-8"))
    return base64.b64decode(payload["data"][0]["b64_json"])


def save_image(raw, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.open(BytesIO(raw)).convert("RGB").save(path, quality=95)


def tile(image, label, size=256):
    image = image.copy()
    image.thumbnail((size, size))
    canvas = Image.new("RGB", (size, size + 24), "white")
    canvas.paste(image, ((size - image.width) // 2, (size - image.height) // 2))
    ImageDraw.Draw(canvas).text((6, size + 6), label[:34], fill=(0, 0, 0))
    return canvas


def write_contact_sheet(cases, models, output_dir, sheet_name):
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
    sheet.save(output_dir / sheet_name, quality=95)


def retry_wait_seconds(error_code, attempt, retry_delay):
    if error_code in {429, 500, 502, 503, 504}:
        return retry_delay * (2 ** attempt)
    return 2 * (attempt + 1)


def main():
    parser = argparse.ArgumentParser(description="Add GPT Image edit outputs to product edit eval cases.")
    parser.add_argument("--cases", type=Path, default=Path("data/amazon_reviews_2023/base_model_eval/cases.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/amazon_reviews_2023/base_model_eval"))
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--api-key-env", default="AIDP_GPT_IMAGE_AK")
    parser.add_argument("--model", default="gpt-image-2")
    parser.add_argument("--quality", default="low", choices=["low", "medium", "high", "auto"])
    parser.add_argument("--size", default="1024x1024", choices=["1024x1024", "1536x1024", "1024x1536", "auto"])
    parser.add_argument("--limit", type=int, default=16)
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--retries", type=int, default=5)
    parser.add_argument("--retry-delay", type=float, default=30)
    parser.add_argument("--insecure-skip-tls-verify", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--models-for-sheet", default=DEFAULT_MODELS_FOR_SHEET)
    parser.add_argument("--sheet-name", default="contact_sheet_with_gpt_image.jpg")
    args = parser.parse_args()

    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        raise SystemExit(f"Set {args.api_key_env} before running this script.")

    cases = load_cases(args.cases, args.limit)
    out_dir = args.output_dir / "outputs" / "gpt_image"
    failures = []
    ssl_context = ssl._create_unverified_context() if args.insecure_skip_tls_verify else None

    for case in cases:
        out_path = out_dir / f"{case['index']:03d}.jpg"
        if out_path.exists() and not args.overwrite:
            print(f"skip gpt_image {case['index']:03d}")
            continue
        last_error = None
        for attempt in range(args.retries + 1):
            error_code = None
            try:
                raw = post_edit(
                    args.base_url,
                    api_key,
                    case,
                    args.model,
                    args.quality,
                    args.size,
                    args.timeout,
                    ssl_context,
                )
                save_image(raw, out_path)
                print(f"wrote gpt_image {case['index']:03d} {case['pair_id']}")
                last_error = None
                break
            except urllib.error.HTTPError as error:
                error_code = error.code
                detail = error.read().decode("utf-8", errors="replace")
                last_error = f"HTTP {error.code}: {detail[:1000]}"
            except Exception as error:
                last_error = str(error)
            if attempt < args.retries:
                wait = retry_wait_seconds(error_code, attempt, args.retry_delay)
                print(f"retry gpt_image {case['index']:03d} in {wait:.0f}s: {last_error}")
                time.sleep(wait)
        if last_error:
            failures.append({"index": case["index"], "pair_id": case["pair_id"], "error": last_error})
            print(f"failed gpt_image {case['index']:03d}: {last_error}")

    (args.output_dir / "gpt_image_failures.json").write_text(
        json.dumps(failures, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    models = [item.strip() for item in args.models_for_sheet.split(",") if item.strip()]
    write_contact_sheet(cases, models, args.output_dir, args.sheet_name)
    print(f"outputs={out_dir}")
    print(f"contact_sheet={args.output_dir / args.sheet_name}")
    print(f"failures={len(failures)}")


if __name__ == "__main__":
    main()
