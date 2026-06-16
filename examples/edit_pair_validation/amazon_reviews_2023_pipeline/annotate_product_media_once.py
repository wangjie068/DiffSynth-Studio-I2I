#!/usr/bin/env python3
import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Optional
from urllib.parse import quote


DEFAULT_BASE_URL = "https://aidp-i18ntt-sg.tiktok-row.net/api/modelhub/online/v2/crawl"
DEFAULT_API_VERSION = "2024-03-01-preview"
DEFAULT_MODEL = "gpt-5.4-mini-2026-03-17"


ANNOTATION_PROMPT = """# Role
You are an expert e-commerce visual dataset annotator for image-to-image edit training.

# Task
You will receive metadata and numbered images from ONE Amazon product/ASIN.
Do all annotation in ONE pass:
1. Understand the product.
2. Label each image role and quality.
3. Select all useful image-edit pair candidates without forcing a fixed count.
4. Write edit instructions for the selected pairs.

# Important Constraints
- The input images are supposed to belong to the same product, but some may show bundles, details, charts, or unrelated accessories. Do not assume all are equally useful.
- Do NOT discard data. Use labels, scores, and reject reasons.
- Do NOT output all pair combinations. Select all genuinely useful pair candidates, and reject uncertain ones with reasons.
- Prefer a clean main product/source image to a moderate target image.
- A valid target image MUST visibly contain the same physical product/package/model. Do not create a valid pair when the product is absent and only implied by eyelashes, ingredients, effects, claims, charts, or before/after examples.
- Avoid pairs where the target is a pure manual, ingredient panel, size chart, zoomed texture, claim graphic, before/after panel without the product, or a scene where the product becomes incidental.
- Keep transformations controlled: low or medium transformation is preferred over high.
- Videos are not provided to you. Ignore video URLs.

# Image Roles
Use one of:
main_product, alternate_angle, advertising_layout, lifestyle, infographic, detail_closeup,
packaging_text, bundle_or_set, instruction_manual, size_chart, before_after, swatch_or_texture,
bad_or_unclear

# Pair Types
Use one of:
main_to_ad, main_to_angle, main_to_lifestyle, main_to_infographic, main_to_detail,
angle_to_ad, bad_or_uncertain

# Return STRICT JSON only, no markdown
{
  "product": {
    "asin": "string",
    "domain": "beauty/electronics/crafts/toys/music/industrial/other",
    "product_type": "short noun phrase",
    "form_factor": "bottle/jar/tube/box/pouch/device/accessory/kit/container/card/unknown",
    "is_bottle_like": true,
    "brand": "string or unknown",
    "usable_for_i2i": true,
    "product_consistency_risk": "low/medium/high",
    "notes": "short reason"
  },
  "images": [
    {
      "image_index": 0,
      "image_id": "string",
      "role": "main_product",
      "source_candidate_score": 0.0,
      "target_candidate_score": 0.0,
      "quality_score": 0.0,
      "aesthetic_score": 0.0,
      "full_product_visible": true,
      "main_product_visibility": 0.0,
      "background_type": "white/transparent/studio/lifestyle/graphic/cluttered/unknown",
      "layout_type": "single_product/product_with_props/multi_panel/collage/text_heavy_ad/unknown",
      "text_density": "none/low/medium/high",
      "has_marketing_text": true,
      "has_logo_or_brand": true,
      "has_human": false,
      "has_face": false,
      "has_hand": false,
      "has_multiple_products": false,
      "is_bundle_or_set": false,
      "is_closeup": false,
      "is_instructional": false,
      "is_low_resolution_or_blurry": false,
      "same_product_confidence": 0.0,
      "visible_text": ["best-effort visible brand/package text"],
      "recommended_use": "train/validation/review_only/exclude",
      "exclude_reasons": []
    }
  ],
  "pairs": [
    {
      "source_image_index": 0,
      "target_image_index": 1,
      "pair_type": "main_to_ad",
      "identity_confidence": 0.0,
      "transformation_magnitude": "low/medium/high",
      "edit_difficulty": "easy/medium/hard",
      "edit_usefulness_score": 0.0,
      "training_value_score": 0.0,
      "text_preservation_risk": "low/medium/high",
      "object_change_risk": "low/medium/high",
      "camera_change": "none/angle/zoom/crop/orientation/mixed",
      "background_change": "none/white_to_graphic/white_to_lifestyle/studio_to_lifestyle/other",
      "reject": false,
      "reject_reasons": [],
      "edit_instruction": "specific image editing instruction from source to target",
      "preservation_requirements": ["product identity", "brand/logo", "package geometry"]
    }
  ],
  "recommended_pairs_summary": "short summary"
}

# Scoring Guide
- source_candidate_score: high for a clean complete product image suitable as source.
- target_candidate_score: high for useful edit target: ad layout, angle change, moderate lifestyle, or informative but not overwhelming infographic.
- identity_confidence: only high if source and target clearly show the same product/package/model.
- If the target does not clearly show the product, set identity_confidence <= 0.65 and reject=true.
- Set reject=true when identity is uncertain, transformation is too high, target is unusable, or source is not a good source.
- recommended_use=train only for images that are clean enough and product-relevant.
"""


def load_env(path: Path):
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export "):].strip()
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def parse_json_response(text: str) -> dict:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.S)
        if not match:
            raise
        return json.loads(match.group(0))


class CurlAzureChatClient:
    def __init__(self, base_url: str, api_version: str, api_key: str, timeout: int = 600):
        self.base_url = base_url.rstrip("/")
        self.api_version = api_version
        self.api_key = api_key
        self.timeout = timeout

    def create(self, model: str, content: list[dict]) -> str:
        url = (
            f"{self.base_url}/openai/deployments/{quote(model, safe='')}"
            f"/chat/completions?api-version={self.api_version}"
        )
        payload = {"model": model, "messages": [{"role": "user", "content": content}]}
        headers = [
            "Content-Type: application/json",
            f"api-key: {self.api_key}",
            f"Authorization: Bearer {self.api_key}",
        ]
        log_id = os.environ.get("X_TT_LOGID") or os.environ.get("X-TT-LOGID")
        if log_id:
            headers.append(f"X-TT-LOGID: {log_id}")

        with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False) as payload_file, \
                tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False) as config_file:
            json.dump(payload, payload_file, ensure_ascii=False)
            payload_file.flush()
            config_file.write(f'url = "{url}"\n')
            config_file.write("request = POST\n")
            config_file.write("silent\nshow-error\nfail\ncompressed\n")
            config_file.write(f"max-time = {self.timeout}\n")
            for header in headers:
                config_file.write(f'header = "{header}"\n')
            config_file.write(f'data-binary = "@{payload_file.name}"\n')
            config_file.flush()
            payload_path = payload_file.name
            config_path = config_file.name
        try:
            result = subprocess.run(["curl", "--config", config_path], check=True, capture_output=True, text=True)
        finally:
            for path in [payload_path, config_path]:
                try:
                    os.unlink(path)
                except OSError:
                    pass
        data = json.loads(result.stdout)
        return data["choices"][0]["message"]["content"]


def build_client(base_url: str, api_version: str, api_key: str):
    return CurlAzureChatClient(base_url, api_version, api_key)


def is_url_accessible(url: str, timeout: int = 15) -> bool:
    result = subprocess.run(
        [
            "curl",
            "-L",
            "--silent",
            "--show-error",
            "--max-time",
            str(timeout),
            "--range",
            "0-1023",
            "--write-out",
            "%{http_code}\t%{content_type}",
            "--output",
            "/dev/null",
            url,
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return False
    parts = result.stdout.strip().split("\t")
    return len(parts) >= 2 and parts[0].startswith("2") and parts[1].startswith("image/")


def iter_product_groups(image_jsonl: Path, max_scan_lines: Optional[int] = None):
    current_id = None
    current_items = []
    with image_jsonl.open(encoding="utf-8", errors="ignore") as in_file:
        for line_index, line in enumerate(in_file):
            if max_scan_lines is not None and line_index >= max_scan_lines:
                break
            if not line.strip():
                continue
            item = json.loads(line)
            product_id = item.get("product_id")
            if not product_id:
                continue
            if current_id is not None and product_id != current_id:
                yield current_id, current_items
                current_items = []
            current_id = product_id
            current_items.append(item)
    if current_id is not None and current_items:
        yield current_id, current_items


def select_images(items: list[dict], max_images: int):
    seen_urls = set()
    unique = []
    for item in items:
        url = item.get("fpath")
        if not url or url in seen_urls:
            continue
        seen_urls.add(url)
        unique.append(item)
    unique.sort(key=lambda item: (
        0 if item.get("variant") == "MAIN" else 1,
        str(item.get("variant") or ""),
        str(item.get("image_id") or ""),
    ))
    selected = unique[:max_images]
    for index, item in enumerate(selected):
        item["image_index"] = index
    return selected


def build_content(product_id: str, items: list[dict], raw_product: Optional[dict] = None):
    first = items[0]
    metadata = {
        "asin": product_id,
        "title": first.get("product_title", ""),
        "category": first.get("category", ""),
        "product_page_url": first.get("product_page_url", ""),
        "image_count_sent": len(items),
        "raw_product_for_prompt": raw_for_prompt(raw_product),
        "images": [
            {
                "image_index": item["image_index"],
                "image_id": item.get("image_id", ""),
                "variant": item.get("variant", ""),
                "url": item.get("fpath", ""),
                "source_annotation": item,
            }
            for item in items
        ],
    }
    content = [
        {"type": "text", "text": ANNOTATION_PROMPT + "\n\nProduct metadata:\n" + json.dumps(metadata, ensure_ascii=False, indent=2)}
    ]
    for item in items:
        content.append({"type": "text", "text": f"Image #{item['image_index']} | image_id={item.get('image_id')} | variant={item.get('variant', '')}"})
        content.append({"type": "image_url", "image_url": {"url": item["fpath"]}})
    return content, metadata


def as_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def postprocess_annotation(annotation: dict, args) -> tuple[dict, list[dict]]:
    images = {
        image.get("image_index"): image
        for image in annotation.get("images", [])
        if isinstance(image, dict)
    }
    valid_pairs = []
    processed_pairs = []
    blocked_roles = {
        "instruction_manual",
        "size_chart",
        "swatch_or_texture",
        "detail_closeup",
        "packaging_text",
        "bad_or_unclear",
    }
    for pair in annotation.get("pairs", []):
        if not isinstance(pair, dict):
            continue
        pair = dict(pair)
        source = images.get(pair.get("source_image_index"), {})
        target = images.get(pair.get("target_image_index"), {})
        reasons = list(pair.get("reject_reasons") or [])

        source_score = as_float(source.get("source_candidate_score"))
        target_score = as_float(target.get("target_candidate_score"))
        target_visibility = as_float(target.get("main_product_visibility"))
        target_same = as_float(target.get("same_product_confidence"))
        identity = as_float(pair.get("identity_confidence"))
        target_role = target.get("role")

        if source_score < args.min_pair_source_score:
            reasons.append(f"source_candidate_score<{args.min_pair_source_score}")
        if target_score < args.min_pair_target_score:
            reasons.append(f"target_candidate_score<{args.min_pair_target_score}")
        if target_visibility < args.min_pair_target_visibility:
            reasons.append(f"target_product_visibility<{args.min_pair_target_visibility}")
        if target_same < args.min_pair_target_same_confidence:
            reasons.append(f"target_same_product_confidence<{args.min_pair_target_same_confidence}")
        if identity < args.min_pair_identity_confidence:
            reasons.append(f"pair_identity_confidence<{args.min_pair_identity_confidence}")
        if target_role in blocked_roles:
            reasons.append(f"blocked_target_role:{target_role}")
        if pair.get("transformation_magnitude") == "high":
            reasons.append("high_transformation_magnitude")

        warnings = []
        high_confidence_override = (
            pair.get("pair_type") in {"main_to_ad", "main_to_angle", "main_to_lifestyle", "angle_to_ad"}
            and identity >= args.high_confidence_override_identity
            and as_float(pair.get("training_value_score")) >= args.high_confidence_override_training_value
            and as_float(pair.get("edit_usefulness_score")) >= args.high_confidence_override_usefulness
            and target_same >= args.min_pair_target_same_confidence
            and target_role not in blocked_roles
            and pair.get("transformation_magnitude") != "high"
        )
        if high_confidence_override:
            overridable = {
                f"target_candidate_score<{args.min_pair_target_score}",
                f"target_product_visibility<{args.min_pair_target_visibility}",
            }
            kept_reasons = []
            for reason in reasons:
                if reason in overridable:
                    warnings.append(f"overrode_{reason}")
                else:
                    kept_reasons.append(reason)
            reasons = kept_reasons

        if reasons:
            pair["reject"] = True
            pair["reject_reasons"] = sorted(set(str(reason) for reason in reasons))
            pair["post_filter_rejected"] = True
        else:
            pair["reject"] = False
            pair["reject_reasons"] = []
            pair["post_filter_rejected"] = False
            if warnings:
                pair["post_filter_warnings"] = sorted(set(warnings))
            valid_pairs.append(pair)
        processed_pairs.append(pair)

    valid_pairs.sort(
        key=lambda pair: (
            as_float(pair.get("training_value_score")),
            as_float(pair.get("edit_usefulness_score")),
            as_float(pair.get("identity_confidence")),
        ),
        reverse=True,
    )
    allowed = None
    if args.max_valid_pairs_per_product and args.max_valid_pairs_per_product > 0:
        allowed = {
            (pair.get("source_image_index"), pair.get("target_image_index"))
            for pair in valid_pairs[:args.max_valid_pairs_per_product]
        }
    final_valid_pairs = []
    for pair in processed_pairs:
        key = (pair.get("source_image_index"), pair.get("target_image_index"))
        if allowed is not None and not pair.get("reject") and key not in allowed:
            pair["reject"] = True
            pair["post_filter_rejected"] = True
            pair["reject_reasons"] = ["exceeds_max_valid_pairs_per_product"]
        if not pair.get("reject"):
            final_valid_pairs.append(pair)

    annotation = dict(annotation)
    annotation["pairs"] = processed_pairs
    annotation["valid_pairs"] = final_valid_pairs
    annotation["postprocess_policy"] = {
        "min_pair_source_score": args.min_pair_source_score,
        "min_pair_target_score": args.min_pair_target_score,
        "min_pair_target_visibility": args.min_pair_target_visibility,
        "min_pair_target_same_confidence": args.min_pair_target_same_confidence,
        "min_pair_identity_confidence": args.min_pair_identity_confidence,
        "max_valid_pairs_per_product": args.max_valid_pairs_per_product,
    }
    return annotation, final_valid_pairs


def already_done(path: Path) -> set[str]:
    done = set()
    if not path.exists():
        return done
    with path.open(encoding="utf-8", errors="ignore") as in_file:
        for line in in_file:
            try:
                item = json.loads(line)
            except Exception:
                continue
            product_id = item.get("product_id") or item.get("product", {}).get("asin")
            if product_id:
                done.add(product_id)
    return done


class ProductRawReader:
    def __init__(self, path: Optional[Path]):
        self.path = path
        self.file = None
        self.current = None
        if path is not None and path.exists():
            self.file = path.open(encoding="utf-8", errors="ignore")

    def close(self):
        if self.file is not None:
            self.file.close()

    def _read_next(self):
        if self.file is None:
            self.current = None
            return
        for line in self.file:
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except Exception:
                continue
            product_id = item.get("parent_asin") or item.get("product_id")
            if product_id:
                self.current = (str(product_id), item)
                return
        self.current = None

    def get(self, product_id: str):
        if self.file is None:
            return None
        if self.current is None:
            self._read_next()
        while self.current is not None and self.current[0] != product_id:
            self._read_next()
        return self.current[1] if self.current is not None else None


def raw_for_prompt(raw_product: Optional[dict]) -> dict:
    if not raw_product:
        return {}
    return {
        "main_category": raw_product.get("main_category"),
        "title": raw_product.get("title"),
        "average_rating": raw_product.get("average_rating"),
        "rating_number": raw_product.get("rating_number"),
        "features": raw_product.get("features"),
        "description": raw_product.get("description"),
        "price": raw_product.get("price"),
        "store": raw_product.get("store"),
        "categories": raw_product.get("categories"),
        "details": raw_product.get("details"),
        "parent_asin": raw_product.get("parent_asin"),
        "subtitle": raw_product.get("subtitle"),
        "author": raw_product.get("author"),
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Annotate one Amazon product per multimodal model call.")
    parser.add_argument("--image-jsonl", type=Path, default=Path("data/amazon_reviews_2023/media_all/annotations/amazon_reviews_2023_media_urls.jsonl"))
    parser.add_argument("--product-raw-jsonl", type=Path)
    parser.add_argument("--out", type=Path, default=Path("outputs/edit_pair_validation/amazon_reviews_2023_product_annotations.jsonl"))
    parser.add_argument("--max-products", type=int, default=10)
    parser.add_argument("--max-images-per-product", type=int, default=10)
    parser.add_argument("--min-images", type=int, default=4)
    parser.add_argument("--max-scan-lines", type=int, default=500000)
    parser.add_argument("--category-regex", default="All_Beauty")
    parser.add_argument("--title-regex", default=r"serum|cream|moisturizer|lotion|shampoo|conditioner|sunscreen|cleanser|face wash|toner|oil|balm|body wash|spray|gel|essence|mask|scrub")
    parser.add_argument("--validate-urls", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--sleep", type=float, default=0.2)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--min-pair-source-score", type=float, default=0.75)
    parser.add_argument("--min-pair-target-score", type=float, default=0.5)
    parser.add_argument("--min-pair-target-visibility", type=float, default=0.25)
    parser.add_argument("--min-pair-target-same-confidence", type=float, default=0.85)
    parser.add_argument("--min-pair-identity-confidence", type=float, default=0.85)
    parser.add_argument("--max-valid-pairs-per-product", type=int, default=0, help="0 means no cap.")
    parser.add_argument("--high-confidence-override-identity", type=float, default=0.9)
    parser.add_argument("--high-confidence-override-training-value", type=float, default=0.7)
    parser.add_argument("--high-confidence-override-usefulness", type=float, default=0.7)
    parser.add_argument("--base-url", default=os.environ.get("GPT_BASE_URL") or os.environ.get("LLM_BASE_URL") or DEFAULT_BASE_URL)
    parser.add_argument("--api-version", default=os.environ.get("GPT_API_VERSION", DEFAULT_API_VERSION))
    parser.add_argument("--model", default=os.environ.get("GPT_MODEL", DEFAULT_MODEL))
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY") or os.environ.get("GPT_AK") or os.environ.get("GPT5_API_KEY"))
    return parser.parse_args()


def main():
    load_env(Path(".env"))
    args = parse_args()
    category_re = re.compile(args.category_regex, re.I) if args.category_regex else None
    title_re = re.compile(args.title_regex, re.I) if args.title_regex else None
    if args.overwrite and args.out.exists():
        args.out.unlink()
    if not args.dry_run and not args.api_key:
        raise SystemExit("Missing API key. Set OPENAI_API_KEY, GPT_AK, or GPT5_API_KEY in .env.")

    done = already_done(args.out)
    product_raw_reader = ProductRawReader(args.product_raw_jsonl)
    client = None if args.dry_run else build_client(args.base_url, args.api_version, args.api_key)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    try:
        with args.out.open("a", encoding="utf-8") as out_file:
            for product_id, items in iter_product_groups(args.image_jsonl, args.max_scan_lines):
                if product_id in done:
                    continue
                if len(items) < args.min_images:
                    continue
                first = items[0]
                title = first.get("product_title", "")
                category = first.get("category", "")
                if category_re and not category_re.search(category):
                    continue
                if title_re and not title_re.search(title):
                    continue
                selected = select_images(items, args.max_images_per_product)
                if len(selected) < args.min_images:
                    continue
                if args.validate_urls:
                    selected = [item for item in selected if is_url_accessible(item["fpath"])]
                    if len(selected) < args.min_images:
                        continue
                    for index, item in enumerate(selected):
                        item["image_index"] = index

                raw_product = product_raw_reader.get(product_id)
                content, metadata = build_content(product_id, selected, raw_product)
                if args.dry_run:
                    result = {
                        "product_id": product_id,
                        "source_dataset": "McAuley-Lab/Amazon-Reviews-2023",
                        "raw_product": raw_product,
                        "metadata": metadata,
                        "source_images": selected,
                    }
                else:
                    last_error = None
                    for attempt in range(args.retries):
                        try:
                            response_text = client.create(args.model, content)
                            raw_annotation = parse_json_response(response_text)
                            annotation, valid_pairs = postprocess_annotation(raw_annotation, args)
                            result = {
                                "product_id": product_id,
                                "source_dataset": "McAuley-Lab/Amazon-Reviews-2023",
                                "raw_product": raw_product,
                                "metadata": metadata,
                                "source_images": selected,
                                "raw_annotation": raw_annotation,
                                "annotation": annotation,
                                "valid_pairs": valid_pairs,
                            }
                            break
                        except Exception as error:
                            last_error = error
                            time.sleep(2 ** attempt)
                    else:
                        result = {
                            "product_id": product_id,
                            "source_dataset": "McAuley-Lab/Amazon-Reviews-2023",
                            "raw_product": raw_product,
                            "metadata": metadata,
                            "source_images": selected,
                            "error": str(last_error),
                        }
                out_file.write(json.dumps(result, ensure_ascii=False) + "\n")
                out_file.flush()
                count += 1
                print(f"wrote {count}: {product_id} | {title[:100]}", flush=True)
                if count >= args.max_products:
                    break
                if args.sleep:
                    time.sleep(args.sleep)
    finally:
        product_raw_reader.close()
    print(f"done wrote={count} out={args.out}")


if __name__ == "__main__":
    main()
