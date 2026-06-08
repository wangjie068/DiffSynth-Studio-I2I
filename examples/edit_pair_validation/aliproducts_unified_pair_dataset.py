#!/usr/bin/env python3
import argparse
import base64
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


UNIFIED_PROMPT = """# Role & Objective
You are an expert e-commerce visual analysis AI creating a high-quality image-editing validation dataset.

You will receive a grid image containing numbered product images from the same SKU/category bucket.

Your goals:
1. Identify the best "Main Image".
2. Select high-quality "Related Images" that depict the exact same product.
3. For each valid [Main Image -> Related Image] pair, produce image-editing instructions and text-preservation metadata.

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
        annotations = data.get("annotations") if isinstance(data, dict) else None
        if not isinstance(annotations, list):
            continue
        for item in annotations:
            if isinstance(item, dict) and "fpath" in item and "category_id" in item:
                yield {
                    "annotation_json": str(json_path),
                    "fpath": item["fpath"],
                    "category_id": str(item["category_id"]),
                }


def resolve_image_path(image_root: Path, fpath: str) -> Path | None:
    rel = str(fpath).lstrip("/")
    candidates = [image_root / rel, image_root / Path(rel).name]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    matches = list(image_root.rglob(Path(rel).name))
    return matches[0] if matches else None


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


def call_model(client, model: str, grid_url: str, retries: int):
    last_error = None
    for attempt in range(retries):
        try:
            completion = client.chat.completions.create(
                model=model,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": UNIFIED_PROMPT},
                        {"type": "image_url", "image_url": {"url": grid_url}},
                    ],
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

    with args.out.open("a", encoding="utf-8") as fout, valid_pairs_path.open("a", encoding="utf-8") as pfout:
        for category_id, records in tqdm(groups, desc="mining unified pairs"):
            grid_url, indexed_records = make_grid(records, args.thumb, args.cols)
            group_output = {
                "category_id": category_id,
                "images": indexed_records,
            }
            try:
                result = call_model(client, args.model, grid_url, args.retries)
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
