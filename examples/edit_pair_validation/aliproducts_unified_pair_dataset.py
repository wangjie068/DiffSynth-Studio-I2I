#!/usr/bin/env python3
import argparse
import base64
import importlib.util
import io
import json
import os
import random
import re
import time
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps
from tqdm import tqdm


DEFAULT_BASE_URL = "https://gpt-i18n.byteintl.net/gpt/openapi/online/multimodal/crawl"
DEFAULT_API_VERSION = "2024-03-01-preview"
DEFAULT_MODEL = "gpt-5.4-mini-2026-03-17"
DEFAULT_IMAGEX_SERVICE_PY = Path(__file__).with_name("imagex_service.py")


UNIFIED_PROMPT = """# Role & Objective
You are an expert e-commerce visual analysis AI creating a high-quality image-editing validation dataset.

You will receive numbered product images from the same SKU/category bucket, either as a numbered grid or as an ordered list of images.

Your goals:
1. Identify the best "Main Image".
2. Select high-quality "Related Images" that depict the exact same product.
3. For each valid [Main Image -> Related Image] pair, produce image-editing instructions and text-preservation metadata.
4. The related image index MUST be different from the main image index. Never pair an image with itself.

# Strict Related Image Criteria
A valid related image MUST satisfy ALL:
- Completely Identical: same product appearance, model, color, material, packaging, label/logo, and structural details.
- Complete Composition: fully contains the entire main product. Exclude partial details, ingredient lists, nutrition labels, zoomed-in textures, or pure packaging text.
- High Quality: clear, sharp, well-lit, visually useful.
- Useful Edit Target: differs from the main image in perspective, pose, background, scene, physical state, lighting, or composition enough to define a meaningful edit.

# Required Text Preservation Analysis
For the main image and each valid target image:
- Decide whether small text exists.
- Extract visible small text contents with best-effort OCR.
- Include brand/logo/IP text when visible.
- If text exists but is unreadable, include "unreadable" and mark legibility.

# Return STRICT JSON only, no markdown:
{
  "main_image_index": 0,
  "category_id_visible_or_inferred": null,
  "product_category": "coarse product category",
  "product_subcategory": "fine product type",
  "main_image_description": "faithful description of original/source image",
  "main_has_small_text": true,
  "main_small_text_contents": [
    {
      "text": "best-effort OCR or unreadable",
      "language": "en/zh/other/unknown",
      "location": "front label / package / tag / etc",
      "legibility": "clear/partial/tiny/blurry",
      "estimated_text_height_px": number_or_null
    }
  ],
  "main_brand_or_ip_text": ["brand/logo/IP text"],
  "pairs": [
    {
      "related_image_index": 1,
      "is_valid_pair": true,
      "identity_confidence": 0.0-1.0,
      "quality_score": 0.0-1.0,
      "has_small_text": true,
      "small_text_contents": [
        {
          "text": "best-effort OCR or unreadable",
          "language": "en/zh/other/unknown",
          "location": "front label / package / tag / etc",
          "legibility": "clear/partial/tiny/blurry",
          "estimated_text_height_px": number_or_null
        }
      ],
      "brand_or_ip_text": ["brand/logo/IP text"],
      "same_product_evidence": ["same label", "same color", "same package shape"],
      "reject_reason": null,
      "edit_instruction": "specific instruction for an image-to-image model to generate the target from the source while preserving identity and text",
      "target_image_description": "faithful description of target/related image",
      "preservation_requirements": ["product identity", "brand/logo", "small text", "package geometry"],
      "usable_for_text_preservation_benchmark": true,
      "suitability_score": 0.0-1.0
    }
  ]
}

Scoring guidance:
- identity_confidence >= 0.85 only when the product is truly the same.
- suitability_score should be high when the pair is same-product, high-quality, has meaningful edit differences, and contains text/brand/IP worth preserving.
- If no valid related image exists, return pairs=[].
"""


def parse_json_response(text: str) -> dict:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.S)
        if not match:
            raise
        return json.loads(match.group(0))


def iter_annotations(json_dir: Path):
    for json_path in sorted(json_dir.rglob("*.json")):
        try:
            data = json.loads(json_path.read_text())
        except Exception:
            continue
        yield from iter_annotation_records(data, json_path)


def iter_annotation_records(data, json_path: Path):
    if isinstance(data, list):
        for item in data:
            record = normalize_annotation_item(item, None, json_path)
            if record is not None:
                yield record
        return

    if not isinstance(data, dict):
        return

    images = data.get("images")
    annotations = data.get("annotations")
    if isinstance(images, list):
        image_by_id = {
            image.get("id", image.get("image_id")): image
            for image in images
            if isinstance(image, dict)
        }
        if isinstance(annotations, list):
            for item in annotations:
                if not isinstance(item, dict):
                    continue
                image = image_by_id.get(item.get("image_id"))
                record = normalize_annotation_item(item, image, json_path)
                if record is not None:
                    yield record
        else:
            for image in images:
                record = normalize_annotation_item(image, None, json_path)
                if record is not None:
                    yield record
        return

    if isinstance(annotations, list):
        for item in annotations:
            record = normalize_annotation_item(item, None, json_path)
            if record is not None:
                yield record


def normalize_annotation_item(item: dict | None, image: dict | None, json_path: Path) -> dict | None:
    if not isinstance(item, dict):
        return None
    merged = {}
    if isinstance(image, dict):
        merged.update(image)
    merged.update(item)

    fpath = first_present(
        merged,
        ["fpath", "file_name", "filename", "path", "image_path", "relative_path", "url"],
    )
    image_id = merged.get("image_id", merged.get("id"))
    category_id = merged.get("category_id", merged.get("label", merged.get("class_id")))

    if fpath is None and image_id is not None:
        fpath = str(image_id)
    if fpath is None:
        return None
    if category_id is None:
        category_id = infer_category_id(str(fpath))
    if category_id is None:
        return None
    return {
        "annotation_json": str(json_path),
        "fpath": str(fpath),
        "image_id": None if image_id is None else str(image_id),
        "category_id": str(category_id),
    }


def first_present(item: dict, keys: list[str]):
    for key in keys:
        value = item.get(key)
        if value not in [None, ""]:
            return value
    return None


def infer_category_id(fpath: str) -> str | None:
    parts = Path(fpath.lstrip("/")).parts
    if len(parts) >= 2:
        return parts[-2]
    return None


def resolve_image_path(image_root: Path, fpath: str) -> Path | None:
    rel = str(fpath).lstrip("/")
    rel_path = Path(rel)
    candidates = [
        image_root / rel_path,
        image_root / "train" / rel_path,
        image_root / "val" / rel_path,
        image_root / "train_val" / rel_path,
        image_root / "images" / rel_path,
        image_root / rel_path.name,
    ]
    if rel_path.suffix == "":
        for ext in [".jpg", ".jpeg", ".png", ".webp"]:
            candidates.extend([
                image_root / f"{rel}{ext}",
                image_root / "train" / f"{rel}{ext}",
                image_root / "val" / f"{rel}{ext}",
                image_root / "train_val" / f"{rel}{ext}",
                image_root / "images" / f"{rel}{ext}",
            ])
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    search_names = [rel_path.name]
    if rel_path.suffix == "":
        search_names.extend(f"{rel_path.name}{ext}" for ext in [".jpg", ".jpeg", ".png", ".webp"])
    for name in search_names:
        matches = list(image_root.rglob(name))
        if matches:
            return matches[0]
    return None


def collect_groups(json_dir: Path, image_root: Path, max_categories: int, max_per_category: int, shuffle: bool):
    grouped = defaultdict(list)
    for item in tqdm(iter_annotations(json_dir), desc="reading annotations"):
        image_path = resolve_image_path(image_root, item["fpath"])
        if image_path is None:
            continue
        item["image_path"] = str(image_path)
        grouped[item["category_id"]].append(item)
    groups = []
    for category_id, items in grouped.items():
        if len(items) < 2:
            continue
        if shuffle:
            random.shuffle(items)
        groups.append((category_id, items[:max_per_category]))
    groups.sort(key=lambda x: len(x[1]), reverse=True)
    if max_categories:
        groups = groups[:max_categories]
    return groups


def make_grid(records: list[dict], thumb: int, cols: int) -> tuple[str, list[dict]]:
    tiles = []
    for index, record in enumerate(records):
        image = Image.open(record["image_path"]).convert("RGB")
        image = ImageOps.contain(image, (thumb, thumb))
        tile = Image.new("RGB", (thumb, thumb + 40), "white")
        tile.paste(image, ((thumb - image.width) // 2, 40 + (thumb - image.height) // 2))
        draw = ImageDraw.Draw(tile)
        draw.text((8, 10), f"#{index}", fill=(255, 0, 0))
        tiles.append(tile)
        record["grid_index"] = index
    rows = (len(tiles) + cols - 1) // cols
    grid = Image.new("RGB", (cols * thumb, rows * (thumb + 40)), "white")
    for index, tile in enumerate(tiles):
        grid.paste(tile, ((index % cols) * thumb, (index // cols) * (thumb + 40)))
    buffer = io.BytesIO()
    grid.save(buffer, format="JPEG", quality=92)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("utf-8"), records


def load_imgx_service(imagex_service_py: Path):
    spec = importlib.util.spec_from_file_location("imagex_service_test", imagex_service_py)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load module from {imagex_service_py}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "ImgXService"):
        raise RuntimeError(f"{imagex_service_py} must define ImgXService")
    return module.ImgXService()


def load_url_cache(path: Path | None) -> dict[str, str]:
    cache = {}
    if path is None or not path.exists():
        return cache
    with path.open("r", encoding="utf-8") as fin:
        for line in fin:
            try:
                item = json.loads(line)
            except Exception:
                continue
            image_path = item.get("image_path")
            imagex_url = item.get("imagex_url")
            if image_path and imagex_url:
                cache[str(image_path)] = str(imagex_url)
    return cache


def append_url_cache(path: Path, items: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fout:
        for item in items:
            fout.write(json.dumps(item, ensure_ascii=False) + "\n")
        fout.flush()


def ensure_imagex_urls(
    records: list[dict],
    imagex_service,
    url_cache: dict[str, str],
    url_cache_path: Path,
    upload_batch_size: int,
):
    missing = [record for record in records if record["image_path"] not in url_cache]
    for start in range(0, len(missing), upload_batch_size):
        batch = missing[start:start + upload_batch_size]
        paths = [record["image_path"] for record in batch]
        urls = imagex_service.imagex_upload(paths)
        cache_items = []
        for record, url in zip(batch, urls):
            url_cache[record["image_path"]] = url
            cache_items.append({
                "image_path": record["image_path"],
                "imagex_url": url,
                "fpath": record.get("fpath"),
                "image_id": record.get("image_id"),
                "category_id": record.get("category_id"),
            })
        append_url_cache(url_cache_path, cache_items)
    for index, record in enumerate(records):
        record["grid_index"] = index
        record["imagex_url"] = url_cache[record["image_path"]]
    return records


def build_image_content(records: list[dict] | None = None, grid_url: str | None = None):
    if records is not None and all(record.get("imagex_url") for record in records):
        content = [{
            "type": "text",
            "text": (
                UNIFIED_PROMPT
                + "\n\nThe following images are provided in order. "
                  "Use these zero-based indices exactly: #0, #1, #2, ..."
            ),
        }]
        for index, record in enumerate(records):
            content.append({"type": "text", "text": f"Image #{index}:"})
            content.append({"type": "image_url", "image_url": {"url": record["imagex_url"]}})
        return content
    if grid_url is None:
        raise RuntimeError("Either uploaded image records or grid_url is required.")
    return [
        {"type": "text", "text": UNIFIED_PROMPT},
        {"type": "image_url", "image_url": {"url": grid_url}},
    ]


def call_model(client, model: str, retries: int, records: list[dict] | None = None, grid_url: str | None = None):
    last_error = None
    for attempt in range(retries):
        try:
            completion = client.chat.completions.create(
                model=model,
                messages=[{
                    "role": "user",
                    "content": build_image_content(records=records, grid_url=grid_url),
                }],
            )
            return parse_json_response(completion.choices[0].message.content)
        except Exception as error:
            last_error = error
            time.sleep(2 ** attempt)
    raise last_error


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json-dir", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-categories", type=int, default=1000)
    parser.add_argument("--max-per-category", type=int, default=8)
    parser.add_argument("--thumb", type=int, default=384)
    parser.add_argument("--cols", type=int, default=4)
    parser.add_argument("--min-identity-confidence", type=float, default=0.85)
    parser.add_argument("--min-quality-score", type=float, default=0.65)
    parser.add_argument("--min-suitability-score", type=float, default=0.6)
    parser.add_argument("--require-small-text", action="store_true")
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--imagex-service-py", type=Path, default=DEFAULT_IMAGEX_SERVICE_PY)
    parser.add_argument("--no-imagex-upload", action="store_true")
    parser.add_argument("--url-cache-jsonl", type=Path, default=None)
    parser.add_argument("--upload-batch-size", type=int, default=8)
    parser.add_argument("--sleep", type=float, default=0.2)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--base-url", default=os.environ.get("GPT_BASE_URL", DEFAULT_BASE_URL))
    parser.add_argument("--api-version", default=os.environ.get("GPT_API_VERSION", DEFAULT_API_VERSION))
    parser.add_argument("--model", default=os.environ.get("GPT_MODEL", DEFAULT_MODEL))
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY") or os.environ.get("GPT_AK"))
    args = parser.parse_args()

    if not args.api_key:
        raise SystemExit("Missing API key. Set OPENAI_API_KEY or GPT_AK.")
    if not args.json_dir.exists():
        raise SystemExit(f"Missing json dir: {args.json_dir}")
    if not args.image_root.exists():
        raise SystemExit(f"Missing image root: {args.image_root}")

    groups = collect_groups(args.json_dir, args.image_root, args.max_categories, args.max_per_category, args.shuffle)
    if not groups:
        raise SystemExit("No category groups with at least two resolved local images.")

    from openai import AzureOpenAI

    client = AzureOpenAI(azure_endpoint=args.base_url, api_version=args.api_version, api_key=args.api_key)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    valid_pairs_path = args.out.with_suffix(".pairs.jsonl")
    url_cache_path = args.url_cache_jsonl or args.out.with_suffix(".urls.jsonl")
    if args.overwrite:
        for path in [args.out, valid_pairs_path, url_cache_path]:
            if path.exists():
                path.unlink()
    use_imagex = not args.no_imagex_upload and args.imagex_service_py is not None and args.imagex_service_py.exists()
    imagex_service = load_imgx_service(args.imagex_service_py) if use_imagex else None
    if not args.no_imagex_upload and imagex_service is None:
        print(f"ImageX service file not found, using local grid data URL mode: {args.imagex_service_py}", flush=True)
    url_cache = load_url_cache(url_cache_path)

    with args.out.open("a", encoding="utf-8") as fout, valid_pairs_path.open("a", encoding="utf-8") as pfout:
        for category_id, records in tqdm(groups, desc="mining unified pairs"):
            if imagex_service is not None:
                indexed_records = ensure_imagex_urls(
                    records,
                    imagex_service=imagex_service,
                    url_cache=url_cache,
                    url_cache_path=url_cache_path,
                    upload_batch_size=args.upload_batch_size,
                )
                grid_url = None
            else:
                grid_url, indexed_records = make_grid(records, args.thumb, args.cols)
            group_output = {
                "category_id": category_id,
                "images": indexed_records,
            }
            try:
                result = call_model(
                    client,
                    args.model,
                    args.retries,
                    records=indexed_records if imagex_service is not None else None,
                    grid_url=grid_url,
                )
                group_output.update(result)
            except Exception as error:
                group_output["error"] = f"{type(error).__name__}: {error}"
            fout.write(json.dumps(group_output, ensure_ascii=False) + "\n")
            fout.flush()

            main_index = group_output.get("main_image_index")
            if isinstance(main_index, int) and 0 <= main_index < len(indexed_records):
                main_image = indexed_records[main_index]
                for pair in group_output.get("pairs", []):
                    target_index = pair.get("related_image_index")
                    if not pair.get("is_valid_pair"):
                        continue
                    if target_index == main_index:
                        continue
                    if float(pair.get("identity_confidence", 0.0)) < args.min_identity_confidence:
                        continue
                    if float(pair.get("quality_score", 0.0)) < args.min_quality_score:
                        continue
                    if float(pair.get("suitability_score", 0.0)) < args.min_suitability_score:
                        continue
                    if args.require_small_text and not (group_output.get("main_has_small_text") or pair.get("has_small_text")):
                        continue
                    if not isinstance(target_index, int) or not 0 <= target_index < len(indexed_records):
                        continue
                    if indexed_records[target_index].get("image_path") == main_image.get("image_path"):
                        continue
                    unified_pair = {
                        "category_id": category_id,
                        "product_category": group_output.get("product_category"),
                        "product_subcategory": group_output.get("product_subcategory"),
                        "source_image": main_image,
                        "target_image": indexed_records[target_index],
                        "source_image_description": group_output.get("main_image_description"),
                        "source_has_small_text": group_output.get("main_has_small_text"),
                        "source_small_text_contents": group_output.get("main_small_text_contents"),
                        "source_brand_or_ip_text": group_output.get("main_brand_or_ip_text"),
                    }
                    unified_pair.update(pair)
                    pfout.write(json.dumps(unified_pair, ensure_ascii=False) + "\n")
                    pfout.flush()
            if args.sleep:
                time.sleep(args.sleep)

    print(f"Wrote group-level decisions to {args.out}")
    print(f"Wrote unified valid pairs to {valid_pairs_path}")


if __name__ == "__main__":
    main()
