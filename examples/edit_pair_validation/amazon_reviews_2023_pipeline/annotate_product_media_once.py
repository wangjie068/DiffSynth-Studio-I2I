#!/usr/bin/env python3
import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Optional
from urllib.parse import quote


DEFAULT_BASE_URL = "https://aidp-i18ntt-sg.tiktok-row.net/api/modelhub/online/v2/crawl"
DEFAULT_API_VERSION = "2024-03-01-preview"
DEFAULT_MODEL = "gpt-5.4-mini-2026-03-17"


ANNOTATION_PROMPT = """You are an e-commerce visual annotator for image-to-image edit training.

Return STRICT JSON only. Annotate one Amazon product/ASIN in one pass.

Goal: select high-quality source-target product image pairs for training. Keep output compact, but do not weaken pair quality judgment.

Hard rules:
- You receive N numbered images. Return exactly one "images" item for every input image_index. Do not omit bad images; mark them exclude.
- Select the best clean source image yourself; Amazon MAIN is only a hint. Prefer a simple product-only catalog image on a white/transparent/studio background with readable product logo/text/small text. If both a white-background catalog source and a styled/lifestyle/prop source exist, choose the white-background catalog source. If both a single product and a multi-pack/bundle/group shot exist, choose the source whose complete product set is most likely to reappear in targets. Do not choose a source whose product identity text is mostly unreadable unless no usable text-bearing source exists. Images with fruit, flowers, ingredient props, lifestyle background, marketing layout, or richer styling are usually targets, not sources.
- If several clean source candidates exist in the same product family, choose the one whose visible color, package form, size, and SKU best match the strongest target candidates. Do not create a color/package-change pair when a cleaner same-color/same-package source exists.
- If two images show the same product where one is clean/product-only and the other is styled/prop/ad/lifestyle, output the pair in clean-to-styled direction, not the reverse.
- Select every high-value training pair from the chosen source, usually 1-6 if available. Do not chase coverage; skip borderline, redundant, or merely acceptable pairs.
- The "pairs" array should contain only plausible training candidates. Do not add rejected pairs just to explain failures; describe bad targets in their image fields instead.
- A valid target must contain the same complete source product object or source product set as real visible objects. Category does not matter: bottles, jars, boxes, bags, tools, devices, accessories, kits, and other products are all valid if the source product has useful logo/text/small-text identity to preserve.
- It is OK if the source product is smaller, partly occluded, held by a person, surrounded by props, or accompanied by related products in the target. It is not OK if the source product object/set is absent, replaced by a different SKU, reduced to only a brand/logo/text panel, or only appears as printed packaging artwork.
- For every pair, explicitly verify the source product object/set inventory against the target. Same brand, same logo, same text, related products, or related contents do not count unless the actual source product object/set is visible in the target.
- Example: if the source is a green retail soap box, a target that only shows matching soap bars, fragrance props, or brand text without that same green box must set source_product_object_set_present_in_target=false. If the target still shows the same green box plus soap bars/props, it can be true.
- Example: if the source is a bottle/jar/tube and the target shows that same object held by a person, in a lifestyle scene, or inside an ad layout, it can be true even with hands, faces, props, or partial occlusion.
- If the source shows multiple major saleable objects, for example box + bottle, kit + applicator, two bottles, bag + compartment, or device + accessory, the target must preserve those same major objects unless the model explicitly marks the source extra object as a non-saleable prop. Do not accept targets that keep only one source component.
- Do not accept package-form substitutions or additions: box-to-bottle, bottle-to-box, box-to-soap-bar-only, box-to-flat-lay-contents, jar-to-label, product-to-front-panel, adding an outer retail box that was not in the source, replacing the source SKU with only related collection/bundle/different SKU products, or targets where the product only appears as a printed picture on retail packaging.
- Reject/down-score if source and target readable product-type words conflict, such as spray vs shampoo, bottle vs retail box, lotion vs soap, or serum vs shampoo.
- Reject/down-score target images where the source product object/set is missing, replaced, cropped out, back/rear view when source is front, dense side/back label view, a pure size chart/manual/ingredient panel, feature table, text-only infographic, swatch board, before-after-only panel, or complex multi-panel/product-grid comparison collage.
- Reject/down-score clean sources that include external color swatches, shade strips, smear samples, or variant comparison samples outside the product itself.
- Do not reject a polished one-panel ad/lifestyle/infographic target only because it has headlines, benefit bullets, icon labels, ingredient props, rich styling, or a repeated copy of the same exact product, as long as the exact same source product object/set remains identifiable in the target.
- Reject/down-score side/back/rear packaging views with dense side-panel text when the source is a front package view.
- Reject/down-score if the target appears to be a different product type, brand, formula, SKU, package, or visible product text, even if colors or category look similar.
- Humans/faces/hands and partial occlusion are OK when the same source product object/set is still recognizable as present in the target.
- Preserve brand/logo, front label, package shape, color, cap/pump/tube geometry, and useful small text.
- Prefer controlled transformations that improve aesthetics: clean ad layout, polished lifestyle scene, better lighting/background, or controlled angle change. A high transformation is valid only when the same complete source product object/set remains identifiable and identity-safe.
- Small text means fine label/capacity/ingredient/warning text, dense package copy, small ad callouts, and icon labels. OCR best-effort; use "[unreadable]" instead of inventing text.

Text and description rules:
- Image descriptions should be specific but compact: product count, view, placement, background, visible text, logo/small-text evidence, and risks.
- For valid pairs, edit_instruction_detailed must be directly usable as a training prompt. Include exact source identity to preserve, target composition, product position/scale/orientation, props/effects, exact readable target text, typography/color/style, lighting, and forbidden identity changes.
- Name visible words when readable, for example headline/body/icon labels; do not write only "add text" or "benefit icons".

Enums:
- image role: main_product, alternate_angle, advertising_layout, lifestyle, infographic, detail_closeup, packaging_text, bundle_or_set, instruction_manual, size_chart, before_after, swatch_or_texture, bad_or_unclear
- pair_type: main_to_ad, main_to_angle, main_to_lifestyle, main_to_infographic, main_to_detail, angle_to_ad, bad_or_uncertain
- scores are 0.0-1.0. pair_quality_score >=0.8 excellent, 0.7-0.8 good, 0.5-0.7 borderline, <0.5 bad.

Return this compact JSON shape:
{
  "product": {
    "asin": "string",
    "domain": "beauty/electronics/crafts/toys/music/industrial/other",
    "product_type": "short noun phrase",
    "form_factor": "bottle/jar/tube/box/pouch/device/accessory/kit/container/card/unknown",
    "brand": "string or unknown",
    "has_logo_or_brand": true,
    "logo_or_brand_notes": "brief evidence",
    "has_small_text": true,
    "small_text_training_potential": "none/low/medium/high",
    "small_text_notes": "brief evidence",
    "is_suitable_logo_text_product": true,
    "logo_text_product_suitability_score": 0.0,
    "product_quality_for_training_score": 0.0,
    "product_quality_judgement": "brief strict judgement",
    "is_flat_decorative_sheet_or_sticker": false,
    "unsuitable_product_reasons": [],
    "usable_for_i2i": true,
    "product_consistency_risk": "low/medium/high",
    "notes": "short"
  },
  "source_selection": {
    "selected_main_source_image_index": 0,
    "selected_main_source_image_id": "string",
    "selected_source_is_amazon_main_variant": true,
    "selection_confidence": 0.0,
    "selection_reason": "brief reason",
    "rejected_source_candidates": [{"image_index": 1, "reason": "brief reason"}]
  },
  "images": [
    {
      "image_index": 0,
      "image_id": "string",
      "role": "main_product",
      "recommended_use": "train/validation/review_only/exclude",
      "source_candidate_score": 0.0,
      "target_candidate_score": 0.0,
      "quality_score": 0.0,
      "aesthetic_score": 0.0,
      "selected_as_main_source": false,
      "main_source_rank": 1,
      "main_source_reason": "brief reason",
      "detailed_description": "specific 1-3 sentences, usually 180-320 chars",
      "product_identity_description": "package/model/logo/color/shape evidence, usually 120-220 chars",
      "target_edit_description": "brief target-use description and risks",
      "product_view": "front/back/side/angled_front/top/multi_view/unknown",
      "full_product_visible": true,
      "main_product_visibility": 0.0,
      "same_product_confidence": 0.0,
      "background_type": "white/transparent/studio/lifestyle/graphic/cluttered/unknown",
      "layout_type": "single_product/product_with_props/multi_panel/collage/text_heavy_ad/unknown",
      "layout_complexity": "simple/moderate/complex",
      "has_multiple_product_views": false,
      "has_color_or_variant_swatches": false,
      "has_multiple_products": false,
      "product_instance_count": 1,
      "is_bundle_or_set": false,
      "is_closeup": false,
      "is_instructional": false,
      "is_low_resolution_or_blurry": false,
      "has_human": false,
      "has_face": false,
      "has_hand": false,
      "text_density": "none/low/medium/high",
      "has_marketing_text": true,
      "has_logo_or_brand": true,
      "logo_or_brand_regions": ["package front"],
      "logo_or_brand_legibility": "none/poor/partial/readable",
      "logo_or_brand_transcription": ["visible brand/logo words"],
      "has_small_text": true,
      "small_text_density": "none/low/medium/high",
      "small_text_regions": ["package front label"],
      "small_text_legibility": "none/poor/partial/readable",
      "small_text_transcription": ["short readable snippets"],
      "small_text_ocr_text": "best-effort OCR string",
      "small_text_ocr_spans": [{"region":"package front label","text":"best-effort or [unreadable]","legibility":"poor/partial/readable","confidence":0.0,"is_exact":false}],
      "visible_text_inventory": {
        "product_package_text": ["exact readable package words"],
        "ad_headlines": ["exact visible headlines"],
        "ad_body_copy": ["exact body copy"],
        "benefit_list_items": ["exact benefits"],
        "icon_labels": ["exact icon labels"],
        "badges_or_callouts": ["exact callouts"],
        "disclaimers_or_other_text": ["other text"]
      },
      "visible_text": ["best-effort visible text"],
      "small_text_importance": "none/low/medium/high",
      "small_text_risk": "none/low/medium/high",
      "exclude_reasons": []
    }
  ],
  "pairs": [
    {
      "source_image_index": 0,
      "target_image_index": 1,
      "pair_type": "main_to_ad",
      "is_high_quality_pair": true,
      "pair_quality_score": 0.0,
      "pair_quality_tier": "excellent/good/borderline/bad",
      "pair_quality_judgement": "strict concise judgement",
      "pair_failure_modes": [],
      "source_product_object_set_present_in_target": true,
      "source_product_presence_confidence": 0.0,
      "source_product_missing_components": [],
      "identity_confidence": 0.0,
      "transformation_magnitude": "low/medium/high",
      "edit_scope_complexity": "simple/moderate/complex",
      "edit_difficulty": "easy/medium/hard",
      "edit_usefulness_score": 0.0,
      "training_value_score": 0.0,
      "aesthetic_improvement_score": 0.0,
      "logo_preservation_score": 0.0,
      "text_preservation_risk": "low/medium/high",
      "object_change_risk": "low/medium/high",
      "camera_change": "none/angle/zoom/crop/orientation/mixed",
      "view_change": "same_front/front_to_angle/front_to_side/front_to_back/back_to_front/multi_view/uncertain",
      "background_change": "none/white_to_graphic/white_to_lifestyle/studio_to_lifestyle/other",
      "small_text_change": "preserve_same_text/reposition_text/resize_text/add_marketing_small_text/remove_or_obscure_text/no_small_text/uncertain",
      "small_text_training_value_score": 0.0,
      "product_text_match_score": 0.0,
      "product_text_match_notes": "brief comparison of source and target brand/product/formula/package text",
      "logo_preservation": "exact logo/brand requirements",
      "small_text_preservation": "exact label/text-zone requirements",
      "small_text_generation": "new text/callouts if any",
      "source_visible_text_to_preserve": ["exact source words"],
      "target_visible_text_to_generate": ["exact target words"],
      "text_rendering_requirements": "legibility/placement/style requirements",
      "reject": false,
      "reject_reasons": [],
      "edit_instruction": "concise instruction",
      "edit_instruction_detailed": "complete training prompt, usually 120-220 words, with source identity, target layout, visible target text, style, lighting, and forbidden changes",
      "preservation_requirements": ["product identity","logo/brand","small label text","package geometry"]
    }
  ],
  "recommended_pairs_summary": "short summary"
}
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
            result = subprocess.run(["curl", "--config", config_path], capture_output=True, text=True)
            if result.returncode != 0:
                detail = (result.stderr or result.stdout or "").strip()
                raise RuntimeError(f"curl failed with exit {result.returncode}: {detail}")
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


def split_api_keys(value: Optional[str]) -> list[str]:
    if not value:
        return []
    keys = []
    for part in re.split(r"[,;\n]", value):
        part = part.strip()
        if part:
            keys.append(part)
    return keys


def get_api_keys(args) -> list[str]:
    keys = split_api_keys(args.api_keys)
    if keys:
        return keys
    keys = split_api_keys(args.api_key)
    if keys:
        return keys
    return []


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
        "images": [
            {
                "image_index": item["image_index"],
                "image_id": item.get("image_id", ""),
                "variant": item.get("variant", ""),
                "url": item.get("fpath", ""),
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


def annotate_one_product(task: dict, args, api_keys: list[str]):
    product_id = task["product_id"]
    selected = task["selected"]
    raw_product = task.get("raw_product")
    start_key_index = task.get("start_key_index", 0)
    content, metadata = build_content(product_id, selected, raw_product)
    if args.dry_run:
        return {
            "product_id": product_id,
            "source_dataset": "McAuley-Lab/Amazon-Reviews-2023",
            "raw_product": raw_product,
            "metadata": metadata,
            "source_images": selected,
        }

    last_error = None
    attempt_errors = []
    key_count = max(1, len(api_keys))
    for attempt in range(args.retries):
        for key_offset in range(key_count):
            key_index = (start_key_index + key_offset) % key_count
            api_key = api_keys[key_index] if api_keys else None
            client = build_client(args.base_url, args.api_version, api_key)
            try:
                response_text = client.create(args.model, content)
                raw_annotation = parse_json_response(response_text)
                annotation, valid_pairs = postprocess_annotation(raw_annotation, args)
                return {
                    "product_id": product_id,
                    "source_dataset": "McAuley-Lab/Amazon-Reviews-2023",
                    "raw_product": raw_product,
                    "metadata": metadata,
                    "source_images": selected,
                    "raw_annotation": raw_annotation,
                    "annotation": annotation,
                    "valid_pairs": valid_pairs,
                    "api_key_index": key_index,
                    "retry_attempt": attempt,
                }
            except Exception as error:
                last_error = error
                attempt_errors.append({
                    "attempt": attempt,
                    "api_key_index": key_index,
                    "error": str(error),
                })
        time.sleep(2 ** attempt)
    return {
        "product_id": product_id,
        "source_dataset": "McAuley-Lab/Amazon-Reviews-2023",
        "raw_product": raw_product,
        "metadata": metadata,
        "source_images": selected,
        "error": str(last_error),
        "attempt_errors": attempt_errors,
    }


def as_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def as_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


def text_len(value) -> int:
    if value is None:
        return 0
    if isinstance(value, list):
        value = " ".join(str(item) for item in value)
    return len(str(value).strip())


def small_text_ocr_len(image: dict) -> int:
    parts = []
    if image.get("small_text_ocr_text"):
        parts.append(str(image.get("small_text_ocr_text")))
    for span in image.get("small_text_ocr_spans") or []:
        if isinstance(span, dict) and span.get("text"):
            parts.append(str(span.get("text")))
    return text_len(" ".join(parts))


def flatten_text_values(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        parts = []
        for child in value.values():
            parts.extend(flatten_text_values(child))
        return parts
    if isinstance(value, list):
        parts = []
        for child in value:
            parts.extend(flatten_text_values(child))
        return parts
    return [str(value)]


def selected_main_source_index(annotation: dict):
    selection = annotation.get("source_selection") or {}
    selected = selection.get("selected_main_source_image_index")
    if selected is not None:
        try:
            return int(selected)
        except (TypeError, ValueError):
            return selected
    for image in annotation.get("images", []):
        if isinstance(image, dict) and as_bool(image.get("selected_as_main_source")):
            return image.get("image_index")
    return None


def product_text(product: dict) -> str:
    fields = [
        product.get("product_type"),
        product.get("form_factor"),
        product.get("brand"),
        product.get("logo_or_brand_notes"),
        product.get("small_text_notes"),
        product.get("notes"),
    ]
    reasons = product.get("unsuitable_product_reasons") or []
    fields.extend(str(reason) for reason in reasons)
    return " ".join(str(field or "") for field in fields)


def product_reject_reasons(annotation: dict, args) -> list[str]:
    product = annotation.get("product") or {}
    reasons = []
    if as_bool(product.get("is_flat_decorative_sheet_or_sticker")):
        reasons.append("blocked_product_type:flat_decorative_sheet_or_sticker")
    if product.get("is_suitable_logo_text_product") is False:
        reasons.append("unsuitable_logo_text_product")
    suitability = product.get("logo_text_product_suitability_score")
    if suitability is not None and as_float(suitability, 1.0) < args.min_product_logo_text_suitability_score:
        reasons.append(f"logo_text_product_suitability_score<{args.min_product_logo_text_suitability_score}")
    if args.blocked_product_regex:
        match = re.search(args.blocked_product_regex, product_text(product), re.I)
        if match:
            reasons.append(f"blocked_product_type:{match.group(0)}")
    return reasons


def target_drops_source_product_count(source: dict, target: dict) -> bool:
    source_count = as_float(source.get("product_instance_count"), 1.0)
    target_count = as_float(target.get("product_instance_count"), 1.0)
    if source_count <= 1:
        return False
    return target_count + 0.25 < source_count


def target_is_back_view(target: dict, pair: dict) -> bool:
    target_view = str(target.get("product_view") or target.get("view_orientation") or "").lower()
    pair_view = str(pair.get("view_change") or "").lower()
    if target_view in {"back", "rear", "reverse", "back_view", "rear_view"}:
        return True
    if pair_view in {"front_to_back", "front_to_rear", "front_to_reverse"}:
        return True
    descriptive_text = " ".join(
        str(value or "")
        for value in [
            target.get("detailed_description"),
            target.get("product_identity_description"),
            target.get("target_edit_description"),
        ]
    )
    return bool(re.search(
        r"\b(side/back|back[- ]?(view|label|panel|side|facing|copy)|rear[- ]?(view|label|panel)|"
        r"reverse[- ]?(view|side)|dense (back|rear) (label|copy|text)|full back label)\b",
        descriptive_text,
        re.I,
    ))


def target_is_side_text_view(source: dict, target: dict, pair: dict) -> bool:
    source_view = str(source.get("product_view") or "").lower()
    target_view = str(target.get("product_view") or "").lower()
    pair_view = str(pair.get("view_change") or "").lower()
    target_text_density = str(target.get("text_density") or "").lower()
    source_is_front = source_view in {"front", "angled_front", "unknown", ""}
    target_is_side = target_view == "side" or pair_view == "front_to_side"
    descriptive_text = " ".join(
        str(value or "")
        for value in [
            target.get("detailed_description"),
            target.get("product_identity_description"),
            target.get("target_edit_description"),
            pair.get("pair_quality_judgement"),
            pair.get("pair_failure_modes"),
        ]
    )
    if not source_is_front or not target_is_side:
        return False
    return target_text_density in {"medium", "high"} or bool(re.search(
        r"\b(side text|side copy|side label|dense side|side becomes more visible|"
        r"dense copy increases text[- ]preservation difficulty|dense package copy)\b",
        descriptive_text,
        re.I,
    ))


def pair_declares_product_mismatch(pair: dict) -> bool:
    text = " ".join(flatten_text_values([
        pair.get("pair_quality_judgement"),
        pair.get("pair_failure_modes"),
        pair.get("product_text_match_notes"),
    ]))
    if re.search(r"\b(no|not|without)\s+(a\s+)?different\s+(sku|product|formula|item|package|variant)\b", text, re.I):
        return False
    return bool(re.search(
        r"\b(different sku|different product|different formula|different item|different package|"
        r"different variant|within the same bundle|same[- ]brand same[- ]line|same brand same line|"
        r"same[- ]brand pair|same line pair|same brand and compatible|compatible beauty line|"
        r"not an exact product match|different visible product text|product text mismatch)\b",
        text,
        re.I,
    ))


def pair_declares_missing_source_component(pair: dict) -> bool:
    text = " ".join(flatten_text_values([
        pair.get("pair_quality_judgement"),
        pair.get("pair_failure_modes"),
        pair.get("product_text_match_notes"),
    ]))
    instruction_text = " ".join(flatten_text_values([
        pair.get("edit_instruction"),
        pair.get("edit_instruction_detailed"),
    ]))
    if re.search(r"\b(keep|preserv(e|ing)|contains?|includes?|remains?)\b.{0,40}\b(product|box|package|packaging|bottle|jar|tube)\b", text, re.I):
        return False
    return bool(re.search(
        r"\b((product object|physical product|package|box|bottle|jar|tube) (is )?absent|"
        r"missing (the )?(product object|physical product|box|package|packaging|bottle|jar|tube)|"
        r"drops? (the )?(product object|physical product|box|package|packaging|bottle|jar|tube)|"
        r"target drops? (the )?(product object|physical product|box|package|packaging|bottle|jar|tube)|"
        r"bar[- ]only|bottle[- ]only|jar[- ]only|label[- ]only|front[- ]panel|"
        r"less direct identity match|package form|packaging transformation|package is absent|"
        r"box[- ]to[- ]bottle|bottle[- ]to[- ]box|jar[- ]to[- ]label|"
        r"retail[- ]box (ad|target|front pack)|front pack target|"
        r"printed (picture|image|artwork) on (the )?(box|package)|"
        r"brand[- ]led rather than pack[- ]led|usage scene without (the )?product)\b",
        text,
        re.I,
    ) or re.search(
        r"\b(box .{0,40} into .{0,40}(soap bars?|bars?|flat[- ]?lay)|"
        r"box shot .{0,40} into .{0,40}(soap bars?|bars?|flat[- ]?lay)|"
        r"(bottle|jar|tube) .{0,30} into (a )?(box|boxed|retail pack)|"
        r"source .{0,30} into .{0,30}(related collection|collection image|bundle|different sku))\b",
        instruction_text,
        re.I,
    ))


def image_identity_text(image: dict) -> str:
    parts = []
    for key in [
        "visible_text",
        "visible_text_inventory",
        "logo_or_brand_transcription",
        "small_text_transcription",
        "small_text_ocr_text",
        "product_identity_description",
        "detailed_description",
    ]:
        parts.extend(flatten_text_values(image.get(key)))
    for span in image.get("small_text_ocr_spans") or []:
        if isinstance(span, dict):
            parts.extend(flatten_text_values(span.get("text")))
    return " ".join(part for part in parts if part and part != "[unreadable]")


def image_visible_identity_text(image: dict) -> str:
    parts = []
    for key in [
        "visible_text",
        "visible_text_inventory",
        "logo_or_brand_transcription",
        "small_text_transcription",
        "small_text_ocr_text",
    ]:
        parts.extend(flatten_text_values(image.get(key)))
    for span in image.get("small_text_ocr_spans") or []:
        if isinstance(span, dict):
            parts.extend(flatten_text_values(span.get("text")))
    return " ".join(part for part in parts if part and part != "[unreadable]")


def product_kind_terms(text: str) -> set[str]:
    lowered = f" {str(text).lower()} "
    patterns = {
        "shampoo": r"\bshampoo\b",
        "conditioner": r"\bconditioner\b",
        "spray": r"\b(spray|mist)\b",
        "serum": r"\bserum\b",
        "cream": r"\bcream\b",
        "lotion": r"\blotion\b",
        "soap": r"\b(soap|beauty bar)\b",
        "scrub": r"\bscrub\b",
        "mask": r"\bmask\b",
        "balm": r"\bbalm\b",
        "deodorant": r"\bdeodorant\b",
        "perfume": r"\b(perfume|cologne|fragrance)\b",
        "oil": r"\boil\b",
    }
    return {kind for kind, pattern in patterns.items() if re.search(pattern, lowered)}


def source_identity_reasons(source: dict) -> list[str]:
    text = image_visible_identity_text(source)
    logo_legibility = str(source.get("logo_or_brand_legibility") or "").lower()
    small_text_legibility = str(source.get("small_text_legibility") or "").lower()
    has_readable_logo = logo_legibility in {"partial", "readable"}
    has_readable_small_text = small_text_legibility in {"partial", "readable"}
    if len(re.sub(r"\W+", "", text)) < 8 and not has_readable_logo and not has_readable_small_text:
        return ["source_identity_text_too_weak"]
    return []


def meaningful_missing_components(value) -> list[str]:
    items = []
    for item in flatten_text_values(value):
        item = str(item).strip()
        if not item:
            continue
        if re.fullmatch(r"(none|no|n/a|na|null|\[\]|not applicable)", item, re.I):
            continue
        items.append(item)
    return items


def source_product_presence_reasons(pair: dict, args) -> list[str]:
    reasons = []
    present = pair.get("source_product_object_set_present_in_target")
    if present is not None and not as_bool(present):
        reasons.append("source_product_object_set_not_present_in_target")
    confidence = pair.get("source_product_presence_confidence")
    if confidence is not None and as_float(confidence, 1.0) < args.min_source_product_presence_confidence:
        reasons.append(f"source_product_presence_confidence<{args.min_source_product_presence_confidence}")
    if meaningful_missing_components(pair.get("source_product_missing_components")):
        reasons.append("source_product_missing_components")
    return reasons


def pair_confirms_source_product_present(pair: dict, args) -> bool:
    present = pair.get("source_product_object_set_present_in_target")
    confidence = pair.get("source_product_presence_confidence")
    if present is None:
        return False
    if not as_bool(present):
        return False
    if confidence is not None and as_float(confidence, 1.0) < args.min_source_product_presence_confidence:
        return False
    return not meaningful_missing_components(pair.get("source_product_missing_components"))


def source_target_kind_conflict(source: dict, target: dict) -> bool:
    source_terms = product_kind_terms(image_identity_text(source))
    target_terms = product_kind_terms(image_identity_text(target))
    if not source_terms or not target_terms:
        return False
    generic_terms = {"oil"}
    source_primary = source_terms - generic_terms
    target_primary = target_terms - generic_terms
    if source_primary and target_primary:
        return not bool(source_primary & target_primary)
    if source_terms & target_terms:
        return False
    return False


def target_introduces_outer_packaging(source: dict, target: dict, pair: dict) -> bool:
    source_text = image_identity_text(source)
    target_text = " ".join([
        image_identity_text(target),
        " ".join(flatten_text_values(pair.get("pair_quality_judgement"))),
        " ".join(flatten_text_values(pair.get("edit_instruction"))),
        " ".join(flatten_text_values(pair.get("edit_instruction_detailed"))),
    ])
    source_has_box = re.search(r"\b(box|boxed|retail pack|outer package|outer box|carton)\b", source_text, re.I)
    target_has_added_box = re.search(
        r"\b(matching box|outer box|retail box|boxed retail|retail pack|"
        r"plus (a )?(matching )?(box|outer box|retail box)|adds? (a )?(box|outer box|retail box)|"
        r"with (a )?(matching )?(box|outer box|retail box))\b",
        target_text,
        re.I,
    )
    return bool(target_has_added_box and not source_has_box)


def target_drops_source_package_or_container(source: dict, target: dict, pair: dict) -> bool:
    source_text = image_identity_text(source)
    target_text = " ".join([
        image_identity_text(target),
        " ".join(flatten_text_values(target.get("detailed_description"))),
        " ".join(flatten_text_values(target.get("target_edit_description"))),
        " ".join(flatten_text_values(pair.get("pair_quality_judgement"))),
        " ".join(flatten_text_values(pair.get("pair_failure_modes"))),
        " ".join(flatten_text_values(pair.get("edit_instruction"))),
        " ".join(flatten_text_values(pair.get("edit_instruction_detailed"))),
    ])
    source_has_box = re.search(r"\b(box|boxed|retail pack|outer package|outer box|carton)\b", source_text, re.I)
    target_preserves_box = re.search(r"\b(same|full|complete|preserve[sd]?|keeps?) .{0,25}\b(box|package|packaging|carton)\b", target_text, re.I)
    target_drops_to_contents = re.search(
        r"\b(box[- ]to[- ]((soap[- ])?bar|bottle|jar|tube|container|product[- ]only)|"
        r"box .{0,40} into .{0,40}(soap bars?|bars?|flat[- ]?lay|contents|bottle[- ]only|jar[- ]only|tube[- ]only|single bottle|single jar)|"
        r"target drops? .{0,20}(box|package|packaging)|package is absent|"
        r"drops? the box|bar[- ]only|soap[- ]bar[- ]only|shifts? to bar[- ]only)\b",
        target_text,
        re.I,
    )
    target_has_box = re.search(r"\b(box|boxed|package|packaging|retail pack|outer box|carton)\b", target_text, re.I)
    target_shows_loose_contents = re.search(
        r"\b(soap bars?|beauty bars?|loose contents?|individual items?|flat[- ]?lay|unboxed contents?)\b",
        target_text,
        re.I,
    )
    return bool(source_has_box and not target_preserves_box and (
        target_drops_to_contents or (target_shows_loose_contents and not target_has_box)
    ))


def source_complexity_reasons(source: dict, args) -> list[str]:
    reasons = []
    layout_type = str(source.get("layout_type") or "").lower()
    layout_complexity = str(source.get("layout_complexity") or "").lower()
    descriptive_text = " ".join(
        str(value or "")
        for value in [
            source.get("detailed_description"),
            source.get("target_edit_description"),
            source.get("product_identity_description"),
        ]
    )
    if layout_type in {"multi_panel", "collage"}:
        reasons.append(f"blocked_source_layout:{layout_type}")
    if layout_complexity == "complex":
        reasons.append("source_layout_complexity:complex")
    if as_bool(source.get("has_color_or_variant_swatches")):
        reasons.append("source_has_color_or_variant_swatches")
    if re.search(r"\b(color|shade|variant)?\s*(swatches?|smears?|streaks?|sample strips?)\b", descriptive_text, re.I):
        reasons.append("source_has_external_swatch_samples")
    return reasons


def target_complexity_reasons(source: dict, target: dict, pair: dict, args) -> list[str]:
    reasons = []
    layout_type = str(target.get("layout_type") or "").lower()
    layout_complexity = str(target.get("layout_complexity") or "").lower()
    pair_type = str(pair.get("pair_type") or "").lower()
    descriptive_text = " ".join(
        str(value or "")
        for value in [
            target.get("detailed_description"),
            target.get("target_edit_description"),
            target.get("product_identity_description"),
            pair.get("edit_instruction"),
            pair.get("edit_instruction_detailed"),
            pair.get("pair_quality_judgement"),
        ]
    )
    if not args.allow_infographic_pairs and pair_type == "main_to_infographic":
        reasons.append("blocked_pair_type:main_to_infographic")
    if layout_type == "collage":
        reasons.append(f"blocked_target_layout:{layout_type}")
    if as_bool(target.get("has_multiple_product_views")):
        reasons.append("target_has_multiple_product_views")
    if as_bool(target.get("has_color_or_variant_swatches")):
        reasons.append("target_has_color_or_variant_swatches")
    if re.search(
        r"\b(collage|multiple views?|multi[- ]?view|variant comparison|color comparison|"
        r"swatches?|swatch board|shade chart|color chart|product grid|several product views|"
        r"comparison chart)\b",
        descriptive_text,
        re.I,
    ):
        reasons.append("target_complex_collage_or_variant_layout")
    if not pair_confirms_source_product_present(pair, args) and re.search(
        r"\b(text[- ]only infographic|feature table|specification table|ingredient panel|size chart|"
        r"instruction manual|label[- ]only|no physical product|"
        r"product only appears as (a )?printed (picture|image))\b",
        descriptive_text,
        re.I,
    ):
        reasons.append("target_text_or_label_only_layout")
    return reasons


def postprocess_annotation(annotation: dict, args) -> tuple[dict, list[dict]]:
    images = {
        image.get("image_index"): image
        for image in annotation.get("images", [])
        if isinstance(image, dict)
    }
    selected_source_index = selected_main_source_index(annotation)
    valid_pairs = []
    processed_pairs = []
    blocked_roles = {
        "instruction_manual",
        "size_chart",
        "swatch_or_texture",
        "packaging_text",
        "bad_or_unclear",
    }
    allowed_pair_types = {
        "main_to_ad",
        "main_to_angle",
        "main_to_lifestyle",
        "main_to_detail",
        "angle_to_ad",
        "bad_or_uncertain",
    }
    if args.allow_infographic_pairs:
        allowed_pair_types.add("main_to_infographic")
    product_reasons = product_reject_reasons(annotation, args)
    for pair in annotation.get("pairs", []):
        if not isinstance(pair, dict):
            continue
        pair = dict(pair)
        source = images.get(pair.get("source_image_index"), {})
        target = images.get(pair.get("target_image_index"), {})
        reasons = list(pair.get("reject_reasons") or [])

        source_score = as_float(source.get("source_candidate_score"))
        target_score = as_float(target.get("target_candidate_score"))
        source_quality = as_float(source.get("quality_score"))
        target_quality = as_float(target.get("quality_score"))
        target_visibility = as_float(target.get("main_product_visibility"))
        target_same = as_float(target.get("same_product_confidence"))
        target_aesthetic = as_float(target.get("aesthetic_score"))
        identity = as_float(pair.get("identity_confidence"))
        pair_quality = as_float(pair.get("pair_quality_score"))
        logo_score = as_float(pair.get("logo_preservation_score"))
        small_text_score = as_float(pair.get("small_text_training_value_score"))
        target_role = target.get("role")
        pair_type = pair.get("pair_type")
        target_full_product_visible = as_bool(target.get("full_product_visible"))
        source_description_chars = text_len(source.get("detailed_description"))
        target_description_chars = text_len(target.get("detailed_description"))
        source_identity_chars = text_len(source.get("product_identity_description"))
        target_identity_chars = text_len(target.get("product_identity_description"))
        detailed_instruction_chars = text_len(pair.get("edit_instruction_detailed"))
        logo_preservation_chars = text_len(pair.get("logo_preservation"))
        small_text_preservation_chars = text_len(pair.get("small_text_preservation"))
        source_small_text_ocr_chars = small_text_ocr_len(source)
        target_small_text_ocr_chars = small_text_ocr_len(target)
        presence_reasons = source_product_presence_reasons(pair, args)
        strong_model_pair = (
            pair_type in {"main_to_ad", "main_to_angle", "main_to_lifestyle", "angle_to_ad", "main_to_infographic"}
            and pair.get("is_high_quality_pair") is True
            and pair_quality >= args.min_pair_quality_score
            and identity >= args.min_pair_identity_confidence
            and logo_score >= args.min_pair_logo_preservation_score
            and small_text_score >= args.min_pair_small_text_training_score
            and source_quality >= args.min_pair_source_quality_score
            and not product_reasons
            and not pair_declares_product_mismatch(pair)
            and not pair_declares_missing_source_component(pair)
            and not source_target_kind_conflict(source, target)
            and not target_introduces_outer_packaging(source, target, pair)
            and not target_drops_source_package_or_container(source, target, pair)
            and not target_drops_source_product_count(source, target)
            and not presence_reasons
        )

        if pair_type not in allowed_pair_types:
            reasons.append(f"unknown_pair_type:{pair_type}")
        reasons.extend(product_reasons)
        if args.require_model_high_quality_pair and pair.get("is_high_quality_pair") is not True:
            reasons.append("model_judged_not_high_quality_pair")
        if pair_quality < args.min_pair_quality_score:
            reasons.append(f"pair_quality_score<{args.min_pair_quality_score}")
        if args.require_selected_main_source:
            if selected_source_index is None:
                reasons.append("missing_selected_main_source")
            elif pair.get("source_image_index") != selected_source_index:
                reasons.append(f"source_image_index_not_selected_main_source:{selected_source_index}")
        if source_score < args.min_pair_source_score:
            reasons.append(f"source_candidate_score<{args.min_pair_source_score}")
        if target_score < args.min_pair_target_score and not strong_model_pair:
            reasons.append(f"target_candidate_score<{args.min_pair_target_score}")
        if source_quality < args.min_pair_source_quality_score:
            reasons.append(f"source_quality_score<{args.min_pair_source_quality_score}")
        if target_quality < args.min_pair_target_quality_score and not strong_model_pair:
            reasons.append(f"target_quality_score<{args.min_pair_target_quality_score}")
        if target_visibility < args.min_pair_target_visibility and not strong_model_pair:
            reasons.append(f"target_product_visibility<{args.min_pair_target_visibility}")
        if not target_full_product_visible and not pair_confirms_source_product_present(pair, args):
            reasons.append("target_full_product_not_visible")
        if target_same < args.min_pair_target_same_confidence:
            reasons.append(f"target_same_product_confidence<{args.min_pair_target_same_confidence}")
        if identity < args.min_pair_identity_confidence:
            reasons.append(f"pair_identity_confidence<{args.min_pair_identity_confidence}")
        if target_aesthetic < args.min_pair_target_aesthetic_score and not strong_model_pair:
            reasons.append(f"target_aesthetic_score<{args.min_pair_target_aesthetic_score}")
        if logo_score < args.min_pair_logo_preservation_score:
            reasons.append(f"logo_preservation_score<{args.min_pair_logo_preservation_score}")
        if small_text_score < args.min_pair_small_text_training_score:
            reasons.append(f"small_text_training_value_score<{args.min_pair_small_text_training_score}")
        if source_description_chars < args.min_image_detailed_description_chars:
            reasons.append(f"source_detailed_description_chars<{args.min_image_detailed_description_chars}")
        if target_description_chars < args.min_image_detailed_description_chars:
            reasons.append(f"target_detailed_description_chars<{args.min_image_detailed_description_chars}")
        if source_identity_chars < args.min_product_identity_description_chars:
            reasons.append(f"source_product_identity_description_chars<{args.min_product_identity_description_chars}")
        if target_identity_chars < args.min_product_identity_description_chars:
            reasons.append(f"target_product_identity_description_chars<{args.min_product_identity_description_chars}")
        if detailed_instruction_chars < args.min_pair_detailed_instruction_chars:
            reasons.append(f"edit_instruction_detailed_chars<{args.min_pair_detailed_instruction_chars}")
        if logo_preservation_chars < args.min_pair_logo_preservation_chars:
            reasons.append(f"logo_preservation_chars<{args.min_pair_logo_preservation_chars}")
        if small_text_preservation_chars < args.min_pair_small_text_preservation_chars:
            reasons.append(f"small_text_preservation_chars<{args.min_pair_small_text_preservation_chars}")
        if target_small_text_ocr_chars < args.min_target_small_text_ocr_chars:
            reasons.append(f"target_small_text_ocr_chars<{args.min_target_small_text_ocr_chars}")
        reasons.extend(source_complexity_reasons(source, args))
        reasons.extend(source_identity_reasons(source))
        if args.reject_target_back_view and target_is_back_view(target, pair):
            reasons.append("target_back_or_rear_view_not_allowed")
        if target_is_side_text_view(source, target, pair):
            reasons.append("target_side_text_panel_not_allowed")
        if pair_declares_product_mismatch(pair):
            reasons.append("model_declared_different_sku_or_product")
        if pair_declares_missing_source_component(pair):
            reasons.append("model_declared_missing_source_component")
        if source_target_kind_conflict(source, target):
            reasons.append("source_target_product_kind_conflict")
        if target_introduces_outer_packaging(source, target, pair):
            reasons.append("target_introduces_outer_packaging_not_in_source")
        if target_drops_source_package_or_container(source, target, pair):
            reasons.append("target_drops_source_package_or_container")
        if target_drops_source_product_count(source, target):
            reasons.append("target_drops_source_product_count")
        reasons.extend(presence_reasons)
        reasons.extend(target_complexity_reasons(source, target, pair, args))
        if target_role in blocked_roles:
            reasons.append(f"blocked_target_role:{target_role}")
        if pair.get("transformation_magnitude") == "high" and not strong_model_pair:
            reasons.append("high_transformation_magnitude")

        if reasons:
            pair["reject"] = True
            pair["reject_reasons"] = sorted(set(str(reason) for reason in reasons))
            pair["post_filter_rejected"] = True
        else:
            pair["reject"] = False
            pair["reject_reasons"] = []
            pair["post_filter_rejected"] = False
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
        "require_model_high_quality_pair": args.require_model_high_quality_pair,
        "min_pair_quality_score": args.min_pair_quality_score,
        "min_pair_source_quality_score": args.min_pair_source_quality_score,
        "min_pair_target_quality_score": args.min_pair_target_quality_score,
        "min_pair_target_visibility": args.min_pair_target_visibility,
        "min_pair_target_same_confidence": args.min_pair_target_same_confidence,
        "min_source_product_presence_confidence": args.min_source_product_presence_confidence,
        "min_pair_identity_confidence": args.min_pair_identity_confidence,
        "min_pair_target_aesthetic_score": args.min_pair_target_aesthetic_score,
        "min_pair_logo_preservation_score": args.min_pair_logo_preservation_score,
        "min_pair_small_text_training_score": args.min_pair_small_text_training_score,
        "min_image_detailed_description_chars": args.min_image_detailed_description_chars,
        "min_product_identity_description_chars": args.min_product_identity_description_chars,
        "min_pair_detailed_instruction_chars": args.min_pair_detailed_instruction_chars,
        "min_pair_logo_preservation_chars": args.min_pair_logo_preservation_chars,
        "min_pair_small_text_preservation_chars": args.min_pair_small_text_preservation_chars,
        "min_target_small_text_ocr_chars": args.min_target_small_text_ocr_chars,
        "require_target_visible_text_in_instruction": args.require_target_visible_text_in_instruction,
        "min_target_visible_text_inventory_chars": args.min_target_visible_text_inventory_chars,
        "min_target_text_instruction_overlap": args.min_target_text_instruction_overlap,
        "min_product_logo_text_suitability_score": args.min_product_logo_text_suitability_score,
        "blocked_product_regex": args.blocked_product_regex,
        "reject_target_back_view": args.reject_target_back_view,
        "allow_infographic_pairs": args.allow_infographic_pairs,
        "allow_multi_product_targets": args.allow_multi_product_targets,
        "max_source_product_instance_count": args.max_source_product_instance_count,
        "max_target_product_instance_count": args.max_target_product_instance_count,
        "require_selected_main_source": args.require_selected_main_source,
        "selected_main_source_image_index": selected_source_index,
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


def parse_args():
    parser = argparse.ArgumentParser(description="Annotate one Amazon product per multimodal model call.")
    parser.add_argument("--image-jsonl", type=Path, default=Path("data/amazon_reviews_2023/media_all/annotations/amazon_reviews_2023_media_urls.jsonl"))
    parser.add_argument("--product-raw-jsonl", type=Path)
    parser.add_argument("--out", type=Path, default=Path("data/amazon_reviews_2023/annotations/product_annotations.jsonl"))
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
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--max-pending", type=int, default=0, help="0 means workers * 2.")
    parser.add_argument("--min-pair-source-score", type=float, default=0.75)
    parser.add_argument("--min-pair-target-score", type=float, default=0.5)
    parser.add_argument("--require-model-high-quality-pair", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--min-pair-quality-score", type=float, default=0.0)
    parser.add_argument("--min-pair-source-quality-score", type=float, default=0.0)
    parser.add_argument("--min-pair-target-quality-score", type=float, default=0.0)
    parser.add_argument("--min-pair-target-visibility", type=float, default=0.25)
    parser.add_argument("--min-pair-human-target-visibility", type=float, default=0.75, help=argparse.SUPPRESS)
    parser.add_argument("--min-pair-target-same-confidence", type=float, default=0.85)
    parser.add_argument("--min-source-product-presence-confidence", type=float, default=0.75)
    parser.add_argument("--min-pair-identity-confidence", type=float, default=0.85)
    parser.add_argument("--min-pair-target-aesthetic-score", type=float, default=0.0)
    parser.add_argument("--min-pair-logo-preservation-score", type=float, default=0.0)
    parser.add_argument("--min-pair-small-text-training-score", type=float, default=0.0)
    parser.add_argument("--min-product-logo-text-suitability-score", type=float, default=0.0)
    parser.add_argument(
        "--blocked-product-regex",
        default=(
            r"nail art|nail sticker|sticker|decal|temporary tattoo|water transfer|"
            r"pattern sheet|flat decorative sheet|swatch-like design"
        ),
    )
    parser.add_argument("--reject-target-back-view", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--allow-infographic-pairs", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--allow-multi-product-targets", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--max-source-product-instance-count", type=float, default=2)
    parser.add_argument("--max-target-product-instance-count", type=float, default=1)
    parser.add_argument("--min-image-detailed-description-chars", type=int, default=0)
    parser.add_argument("--min-product-identity-description-chars", type=int, default=0)
    parser.add_argument("--min-pair-detailed-instruction-chars", type=int, default=0)
    parser.add_argument("--min-pair-logo-preservation-chars", type=int, default=0)
    parser.add_argument("--min-pair-small-text-preservation-chars", type=int, default=0)
    parser.add_argument("--require-target-visible-text-in-instruction", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--min-target-visible-text-inventory-chars", type=int, default=20)
    parser.add_argument("--min-target-text-instruction-overlap", type=float, default=0.65)
    parser.add_argument(
        "--min-target-small-text-ocr-chars",
        "--min-image-small-text-ocr-chars",
        dest="min_target_small_text_ocr_chars",
        type=int,
        default=0,
    )
    parser.add_argument("--require-selected-main-source", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-valid-pairs-per-product", type=int, default=0, help="0 means no cap.")
    parser.add_argument("--allow-human-dominant-targets", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--high-confidence-override-identity", type=float, default=0.9)
    parser.add_argument("--high-confidence-override-training-value", type=float, default=0.7)
    parser.add_argument("--high-confidence-override-usefulness", type=float, default=0.7)
    parser.add_argument("--base-url", default=os.environ.get("GPT_BASE_URL") or os.environ.get("LLM_BASE_URL") or DEFAULT_BASE_URL)
    parser.add_argument("--api-version", default=os.environ.get("GPT_API_VERSION", DEFAULT_API_VERSION))
    parser.add_argument("--model", default=os.environ.get("GPT_MODEL", DEFAULT_MODEL))
    parser.add_argument("--api-keys", default=os.environ.get("GPT_API_KEYS") or os.environ.get("OPENAI_API_KEYS") or os.environ.get("GPT5_API_KEYS"))
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY") or os.environ.get("GPT_AK") or os.environ.get("GPT5_API_KEY"))
    return parser.parse_args()


def main():
    load_env(Path(".env"))
    args = parse_args()
    category_re = re.compile(args.category_regex, re.I) if args.category_regex else None
    title_re = re.compile(args.title_regex, re.I) if args.title_regex else None
    if args.overwrite and args.out.exists():
        args.out.unlink()
    api_keys = get_api_keys(args)
    if not args.dry_run and not api_keys:
        raise SystemExit("Missing API key. Set GPT_API_KEYS, OPENAI_API_KEYS, OPENAI_API_KEY, GPT_AK, or GPT5_API_KEY in .env.")
    if args.workers < 1:
        raise SystemExit("--workers must be >= 1")

    done = already_done(args.out)
    product_raw_reader = ProductRawReader(args.product_raw_jsonl)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    submitted = 0
    written = 0
    max_pending = args.max_pending if args.max_pending > 0 else args.workers * 2

    def write_completed(futures, out_file, wait_mode):
        nonlocal written
        if not futures:
            return futures
        done_futures, pending_futures = wait(futures, return_when=wait_mode)
        for future in done_futures:
            result = future.result()
            out_file.write(json.dumps(result, ensure_ascii=False) + "\n")
            out_file.flush()
            written += 1
            title = result.get("metadata", {}).get("title", "")
            print(f"wrote {written}: {result.get('product_id')} | {title[:100]}", flush=True)
        return set(pending_futures)

    try:
        with args.out.open("a", encoding="utf-8") as out_file, ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = set()
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
                task = {
                    "product_id": product_id,
                    "selected": selected,
                    "raw_product": raw_product,
                    "start_key_index": submitted % len(api_keys) if api_keys else 0,
                }
                futures.add(executor.submit(annotate_one_product, task, args, api_keys))
                submitted += 1
                if submitted >= args.max_products:
                    break
                if len(futures) >= max_pending:
                    futures = write_completed(futures, out_file, FIRST_COMPLETED)
                if args.sleep:
                    time.sleep(args.sleep)
            while futures:
                futures = write_completed(futures, out_file, FIRST_COMPLETED)
    finally:
        product_raw_reader.close()
    print(f"done submitted={submitted} wrote={written} out={args.out}")


if __name__ == "__main__":
    main()
