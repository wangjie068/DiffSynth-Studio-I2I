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


ANNOTATION_PROMPT = """# Role
You are an expert e-commerce visual dataset annotator for image-to-image edit training.

# Task
You will receive metadata and numbered images from ONE Amazon product/ASIN.
Do all annotation in ONE pass:
1. Understand the product.
2. Label each image role, quality, logo/brand evidence, and small-text OCR properties.
3. Explicitly choose the best main source image. Do not assume Amazon's MAIN variant is automatically the source.
4. Judge whether each candidate is truly high-quality training data.
5. Select all useful image-edit pair candidates without forcing a fixed count.
6. Write detailed edit instructions for the selected pairs.

# Important Constraints
- The input images are supposed to belong to the same product, but some may show bundles, details, charts, or unrelated accessories. Do not assume all are equally useful.
- Do NOT discard data. Use labels, scores, and reject reasons.
- Do NOT output all pair combinations. Select all genuinely useful pair candidates, and reject uncertain ones with reasons.
- Prefer a clean complete product/source image to a moderate target image, but explicitly select it by image_index. Amazon's variant=MAIN is only a hint, not a decision.
- A valid target image MUST contain the complete same physical product/package/model from the source image. The product can be recomposed into an ad/lifestyle/graphic layout, but the full saleable product must remain visible as a clear primary subject.
- Do not create a valid pair when the target product is cropped, partially hidden, absent, tiny, incidental, or only implied by eyelashes, eyes, skin, ingredients, effects, claims, charts, or before/after examples.
- Avoid pairs where the target is a pure manual, ingredient panel, size chart, zoomed texture, claim graphic, before/after panel without a dominant product, or a scene where the product becomes incidental.
- Humans, faces, or hands are allowed when the complete product remains clearly visible and product-dominant. Do not reject a pair only because a person appears.
- Keep previous consistency requirements: source and target must be the same product/variant/package, with no brand swap, no product identity drift, no missing full product, no high transformation, and no target where the product is incidental.
- Do not select front-to-back or front-to-rear-view edits. A valid target should preserve the same usable front/logo/small-text surface, not turn a front package shot into a back-label or rear-panel product view.
- The main training goal is beautiful product-image editing while preserving the product LOGO, brand marks, package identity, and small text. Prefer pairs that improve layout/aesthetics without changing or corrupting the product identity.
- Prefer pairs that preserve, move, resize, or add readable small text while keeping the complete product. Do not count bottle shape alone as the goal.
- Small text means fine product-label text, ingredient/capacity text, warning text, dense package copy, small brand/subtitle text, or small ad callouts. Do not count only large headlines as small text.
- OCR small text explicitly. Use best-effort transcription only for visible text. If a region is unreadable, write "[unreadable]" and mark low confidence instead of hallucinating.
- For every image with visible text, transcribe the text inventory explicitly. Separate product/package text from ad headline, body copy, benefit list, icon labels, badges, callouts, and disclaimer text.
- Description and instruction fields must name the important visible words, not only say "headline", "benefit icons", "small text", or "branding". For example, write the headline text and icon labels such as "For Longer, Stronger Hair", "Natural", "Vegan", "Cruelty Free", "Gluten Free", "Non-Toxic", and "Palm Free" when visible.
- edit_instruction_detailed must be long-form and directly usable as a training prompt. It should include: exact product identity to preserve, exact source package/logo/small text to preserve when readable, target composition, target visible marketing text to generate, target icon/benefit/callout labels, typography/color/style, lighting/background, and explicit forbidden changes.
- Exclude products that are themselves flat decorative sheets, nail-art sticker/decal sheets, temporary tattoo sheets, pattern sheets, or swatch-like design collections. These do not provide stable product LOGO/package text preservation training.
- Reject or down-score pairs where the logo, brand name, key package text, or fine label text is missing, heavily distorted, replaced by a different brand, or visually implausible.
- A good target should be aesthetically useful: cleaner composition, nicer lighting/background, better ad layout, or more polished product presentation.
- Description fields must be specific and sufficient. Avoid generic one-line descriptions; include exact product identity, object count, product placement, camera/view, background/layout, props, material/texture/finish, lighting/shadow, text/graphics, logo/brand evidence, small-text regions, aesthetic qualities, and any visible risks.
- detailed_description should let a human imagine the image without seeing it. Do not write only "polished ad-style image"; describe what is in the image, where it is, what text is present, what props/effects appear, and how the product looks.
- Be strict when judging high-quality pairs. A high-quality pair must have a good source, a visually polished target, complete same product, no front-to-back view change, preserved logo/brand identity, useful small-text supervision, and a detailed actionable instruction. If any of these are weak, set is_high_quality_pair=false and explain why.
- Do not select complex reconstruction targets: no multi-panel collage, no multi-view product grid, no color/variant comparison chart, no swatch board, and no target containing many duplicated product instances. The target should be one coherent polished product image/ad scene, not a product-detail infographic page.
- Keep transformations controlled: low or medium transformation is preferred over high.
- Videos are not provided to you. Ignore video URLs.

# Detailed Instruction Requirements
For each valid pair, write edit_instruction_detailed in the style of a complete training prompt, not a short caption.
It must cover:
- Source identity reference: exact product type, color, material/finish, package shape, cap/pump/nozzle, label design, brand/logo words, key package text, and product proportions to preserve.
- Target composition: square/portrait/landscape if apparent, product position, scale, orientation, crop, negative space, background, props, graphics, icons, ingredient elements, human/body parts if any, and whether shadows/reflections are needed.
- Target text design: exact visible headline/body/list/icon/badge text, typography style, color, hierarchy, and placement.
- Aesthetic direction: commercial style, lighting, color palette, realism level, premium/clean/lifestyle/editorial feeling, and Amazon listing suitability.
- Forbidden changes: no product identity drift, no back/rear view switch, no missing product, no logo/text corruption, no extra unrelated product, no unreadable key text.
If the target contains little or no ad text, still describe the full visual transformation in detail: camera, pose, lighting, background, props, texture, product surface, and exact preservation requirements.

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
    "brand": "string or unknown",
    "has_logo_or_brand": true,
    "logo_or_brand_notes": "visible logo/brand mark evidence and whether it is useful for preservation training",
    "has_small_text": true,
    "small_text_training_potential": "none/low/medium/high",
    "small_text_notes": "where useful small text appears across the product images",
    "is_suitable_logo_text_product": true,
    "logo_text_product_suitability_score": 0.0,
    "product_quality_for_training_score": 0.0,
    "product_quality_judgement": "why this product is or is not suitable for high-quality logo/small-text/aesthetic product editing training",
    "is_flat_decorative_sheet_or_sticker": false,
    "unsuitable_product_reasons": [],
    "usable_for_i2i": true,
    "product_consistency_risk": "low/medium/high",
    "notes": "short reason"
  },
  "source_selection": {
    "selected_main_source_image_index": 0,
    "selected_main_source_image_id": "string",
    "selected_source_is_amazon_main_variant": true,
    "selection_confidence": 0.0,
    "selection_reason": "why this image is the best clean complete source, considering full product visibility, logo/text readability, quality, and identity consistency",
    "rejected_source_candidates": [
      {
        "image_index": 1,
        "reason": "why this image is not the best source"
      }
    ]
  },
  "images": [
    {
      "image_index": 0,
      "image_id": "string",
      "detailed_description": "2-4 sentences describing the full visible content, product placement, background, text/graphics, props, and any visible issues",
      "product_identity_description": "detailed description of the visible product/package/model/color/shape/logo evidence",
      "target_edit_description": "detailed description of what an image-editing model would need to create if this image is used as target, including layout, product position, background, visible text, and small-text requirements",
      "visual_inventory": {
        "product_objects": ["what product objects are visible and how many"],
        "product_pose_and_camera": "front/angled/side view, orientation, crop, scale, camera distance",
        "composition": "where the product, props, graphics, and text sit in the frame",
        "background_and_surface": "background color/type, tabletop/floor/reflection/shadow if visible",
        "props_and_effects": ["mint leaves, flowers, ingredient cutouts, liquid splash, glow, icons, human/hand/face, etc."],
        "materials_and_textures": ["plastic, glass, metallic cap, glossy serum, matte label, transparent liquid, etc."],
        "lighting_and_style": "studio/lifestyle/editorial/ad style, softness, shadows, color palette"
      },
      "role": "main_product",
      "source_candidate_score": 0.0,
      "selected_as_main_source": false,
      "main_source_rank": 1,
      "main_source_reason": "why this image is or is not suitable as the selected source",
      "target_candidate_score": 0.0,
      "quality_score": 0.0,
      "aesthetic_score": 0.0,
      "product_view": "front/back/side/angled_front/top/multi_view/unknown",
      "full_product_visible": true,
      "main_product_visibility": 0.0,
      "background_type": "white/transparent/studio/lifestyle/graphic/cluttered/unknown",
      "layout_type": "single_product/product_with_props/multi_panel/collage/text_heavy_ad/unknown",
      "layout_complexity": "simple/moderate/complex",
      "has_multiple_product_views": false,
      "has_color_or_variant_swatches": false,
      "text_density": "none/low/medium/high",
      "has_marketing_text": true,
      "has_logo_or_brand": true,
      "logo_or_brand_regions": ["package front", "cap", "box front"],
      "logo_or_brand_legibility": "none/poor/partial/readable",
      "logo_or_brand_transcription": ["best-effort visible logo or brand text"],
      "has_small_text": true,
      "small_text_density": "none/low/medium/high",
      "small_text_regions": ["package front", "package side", "bottom label", "ad callout"],
      "small_text_legibility": "none/poor/partial/readable",
      "small_text_transcription": ["best-effort short snippets of readable small text, not full OCR"],
      "small_text_ocr_text": "best-effort OCR string for visible small text only; use [unreadable] for unreadable regions and do not invent text",
      "small_text_ocr_spans": [
        {
          "region": "package front lower label",
          "text": "best-effort OCR text or [unreadable]",
          "legibility": "poor/partial/readable",
          "confidence": 0.0,
          "is_exact": false
        }
      ],
      "visible_text_inventory": {
        "product_package_text": ["exact readable product/package words, brand, subtitle, size, claims"],
        "ad_headlines": ["exact headline text visible in the image"],
        "ad_body_copy": ["exact body or paragraph copy visible in the image"],
        "benefit_list_items": ["exact benefit list items visible in the image"],
        "icon_labels": ["exact icon labels visible in the image"],
        "badges_or_callouts": ["exact badge, seal, sticker, or small callout text"],
        "disclaimers_or_other_text": ["exact other visible text, or [unreadable] when present but illegible"]
      },
      "small_text_importance": "none/low/medium/high",
      "small_text_risk": "none/low/medium/high",
      "has_human": false,
      "has_face": false,
      "has_hand": false,
      "has_multiple_products": false,
      "product_instance_count": 1,
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
      "is_high_quality_pair": true,
      "pair_quality_score": 0.0,
      "pair_quality_tier": "excellent/good/borderline/bad",
      "pair_quality_judgement": "strict judgement of whether this pair should be used for training, covering source quality, target aesthetics, product consistency, logo/small-text preservation, view consistency, and instruction quality",
      "pair_failure_modes": ["front_to_back_view_change", "weak_logo_preservation", "low_aesthetic_value"],
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
      "logo_preservation": "what logo, brand wordmark, icon, or package mark should be preserved from source to target",
      "small_text_preservation": "what exact small text, label zones, or typography should be preserved from source to target",
      "small_text_generation": "what new small text or tiny callouts should be generated in the target, if any",
      "source_visible_text_to_preserve": ["exact visible source words that should be preserved, including brand/logo/package text"],
      "target_visible_text_to_generate": ["exact visible target words that should be generated, including headline/body/icon/benefit labels"],
      "text_rendering_requirements": "specific typography, placement, size, color, legibility, and no-hallucination requirements for all visible target text",
      "reject": false,
      "reject_reasons": [],
      "edit_instruction": "concise image editing instruction from source to target",
      "edit_instruction_detailed": "complete detailed training prompt: preserve exact source product identity; describe target composition, object placement, camera/view, product scale/orientation, props/effects, background/surface, lighting/shadows/reflections, exact visible text and typography, aesthetic style, and forbidden changes",
      "preservation_requirements": ["product identity", "logo/brand", "small label text", "package geometry"]
    }
  ],
  "recommended_pairs_summary": "short summary"
}

# Scoring Guide
- source_candidate_score: high for a clean complete product image suitable as source.
- target_candidate_score: high for useful edit target: ad layout, angle change, moderate lifestyle, or informative but not overwhelming infographic.
- pair_quality_score: strict overall training quality. Use >=0.8 only for excellent pairs, 0.7-0.8 for good pairs, 0.5-0.7 for borderline review-only pairs, and <0.5 for bad pairs.
- identity_confidence: only high if source and target clearly show the same product/package/model.
- If the target does not clearly show the product, set identity_confidence <= 0.65 and reject=true.
- logo_preservation_score: high only when the target keeps the same readable logo/brand mark or clearly preserves the visible package mark.
- small_text_training_value_score: high when the source/target pair gives useful supervision for preserving or generating readable fine text on the same complete product.
- aesthetic_improvement_score: high when the target is a more polished, attractive product/ad image while still preserving product identity.
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


def annotate_one_product(task: dict, args, api_key: Optional[str]):
    product_id = task["product_id"]
    selected = task["selected"]
    raw_product = task.get("raw_product")
    content, metadata = build_content(product_id, selected, raw_product)
    if args.dry_run:
        return {
            "product_id": product_id,
            "source_dataset": "McAuley-Lab/Amazon-Reviews-2023",
            "raw_product": raw_product,
            "metadata": metadata,
            "source_images": selected,
        }

    client = build_client(args.base_url, args.api_version, api_key)
    last_error = None
    for attempt in range(args.retries):
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
            }
        except Exception as error:
            last_error = error
            time.sleep(2 ** attempt)
    return {
        "product_id": product_id,
        "source_dataset": "McAuley-Lab/Amazon-Reviews-2023",
        "raw_product": raw_product,
        "metadata": metadata,
        "source_images": selected,
        "error": str(last_error),
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


def image_visible_text_inventory(image: dict) -> str:
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


def text_tokens(value: str) -> set[str]:
    stopwords = {
        "and", "the", "with", "for", "from", "into", "same", "keep", "text",
        "product", "bottle", "image", "target", "source", "visible", "label",
    }
    tokens = set()
    for token in re.findall(r"[a-z0-9][a-z0-9'-]{2,}", str(value).lower()):
        token = token.strip("'-")
        if token and token not in stopwords:
            tokens.add(token)
    return tokens


def text_inventory_overlap(inventory: str, instruction: str) -> float:
    inventory_tokens = text_tokens(inventory)
    if not inventory_tokens:
        return 1.0
    instruction_tokens = text_tokens(instruction)
    if not instruction_tokens:
        return 0.0
    required = min(len(inventory_tokens), 16)
    matched = len(inventory_tokens & instruction_tokens)
    return min(matched, required) / required


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
        r"\b(side/back|back[- ]?(view|label|panel|side|facing)|rear[- ]?(view|label|panel)|reverse[- ]?(view|side))\b",
        descriptive_text,
        re.I,
    ))


def source_complexity_reasons(source: dict, args) -> list[str]:
    reasons = []
    layout_type = str(source.get("layout_type") or "").lower()
    layout_complexity = str(source.get("layout_complexity") or "").lower()
    product_count = as_float(source.get("product_instance_count"), 1.0)
    if layout_type in {"multi_panel", "collage"}:
        reasons.append(f"blocked_source_layout:{layout_type}")
    if layout_complexity == "complex":
        reasons.append("source_layout_complexity:complex")
    if as_bool(source.get("has_multiple_product_views")):
        reasons.append("source_has_multiple_product_views")
    if as_bool(source.get("has_color_or_variant_swatches")):
        reasons.append("source_has_color_or_variant_swatches")
    if product_count > args.max_source_product_instance_count:
        reasons.append(f"source_product_instance_count>{args.max_source_product_instance_count}")
    return reasons


def target_complexity_reasons(source: dict, target: dict, pair: dict, args) -> list[str]:
    reasons = []
    layout_type = str(target.get("layout_type") or "").lower()
    layout_complexity = str(target.get("layout_complexity") or "").lower()
    edit_scope = str(pair.get("edit_scope_complexity") or "").lower()
    pair_type = str(pair.get("pair_type") or "").lower()
    source_count = as_float(source.get("product_instance_count"), 1.0)
    target_count = as_float(target.get("product_instance_count"), 1.0)
    allowed_target_count = max(args.max_target_product_instance_count, source_count)
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
    if layout_type in {"multi_panel", "collage"}:
        reasons.append(f"blocked_target_layout:{layout_type}")
    if layout_complexity == "complex":
        reasons.append("target_layout_complexity:complex")
    if edit_scope == "complex":
        reasons.append("edit_scope_complexity:complex")
    if (
        not args.allow_multi_product_targets
        and as_bool(target.get("has_multiple_products"))
        and target_count > allowed_target_count
    ):
        reasons.append("target_has_extra_products")
    if not args.allow_multi_product_targets and target_count > allowed_target_count:
        reasons.append(f"target_product_instance_count>{allowed_target_count:g}")
    if as_bool(target.get("has_multiple_product_views")):
        reasons.append("target_has_multiple_product_views")
    if as_bool(target.get("has_color_or_variant_swatches")):
        reasons.append("target_has_color_or_variant_swatches")
    if re.search(
        r"\b(multi[- ]?panel|collage|multiple views?|multi[- ]?view|variant comparison|color comparison|"
        r"swatches?|swatch board|shade chart|color chart|product grid|grid of|several product views|"
        r"multiple copies|duplicated bottles|comparison chart)\b",
        descriptive_text,
        re.I,
    ):
        reasons.append("target_complex_collage_or_variant_layout")
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
        "detail_closeup",
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
        target_text_inventory = image_visible_text_inventory(target)
        target_text_inventory_chars = text_len(target_text_inventory)
        instruction_text = " ".join(flatten_text_values(pair.get("edit_instruction_detailed")))
        target_text_instruction_overlap = text_inventory_overlap(target_text_inventory, instruction_text)

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
        if target_score < args.min_pair_target_score:
            reasons.append(f"target_candidate_score<{args.min_pair_target_score}")
        if source_quality < args.min_pair_source_quality_score:
            reasons.append(f"source_quality_score<{args.min_pair_source_quality_score}")
        if target_quality < args.min_pair_target_quality_score:
            reasons.append(f"target_quality_score<{args.min_pair_target_quality_score}")
        if target_visibility < args.min_pair_target_visibility:
            reasons.append(f"target_product_visibility<{args.min_pair_target_visibility}")
        if not target_full_product_visible:
            reasons.append("target_full_product_not_visible")
        if target_same < args.min_pair_target_same_confidence:
            reasons.append(f"target_same_product_confidence<{args.min_pair_target_same_confidence}")
        if identity < args.min_pair_identity_confidence:
            reasons.append(f"pair_identity_confidence<{args.min_pair_identity_confidence}")
        if target_aesthetic < args.min_pair_target_aesthetic_score:
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
        if (
            args.require_target_visible_text_in_instruction
            and as_bool(target.get("has_marketing_text"))
            and target_text_inventory_chars >= args.min_target_visible_text_inventory_chars
            and target_text_instruction_overlap < args.min_target_text_instruction_overlap
        ):
            reasons.append(f"target_visible_text_instruction_overlap<{args.min_target_text_instruction_overlap}")
        reasons.extend(source_complexity_reasons(source, args))
        if args.reject_target_back_view and target_is_back_view(target, pair):
            reasons.append("target_back_or_rear_view_not_allowed")
        reasons.extend(target_complexity_reasons(source, target, pair, args))
        if target_role in blocked_roles:
            reasons.append(f"blocked_target_role:{target_role}")
        if pair.get("transformation_magnitude") == "high":
            reasons.append("high_transformation_magnitude")

        warnings = []
        high_confidence_override = (
            pair.get("pair_type") in {"main_to_ad", "main_to_angle", "main_to_lifestyle", "angle_to_ad"}
            and not product_reasons
            and (not args.require_model_high_quality_pair or pair.get("is_high_quality_pair") is True)
            and pair_quality >= args.min_pair_quality_score
            and (not args.require_selected_main_source or pair.get("source_image_index") == selected_source_index)
            and identity >= args.high_confidence_override_identity
            and as_float(pair.get("training_value_score")) >= args.high_confidence_override_training_value
            and as_float(pair.get("edit_usefulness_score")) >= args.high_confidence_override_usefulness
            and source_quality >= args.min_pair_source_quality_score
            and target_quality >= args.min_pair_target_quality_score
            and target_aesthetic >= args.min_pair_target_aesthetic_score
            and logo_score >= args.min_pair_logo_preservation_score
            and small_text_score >= args.min_pair_small_text_training_score
            and source_description_chars >= args.min_image_detailed_description_chars
            and target_description_chars >= args.min_image_detailed_description_chars
            and source_identity_chars >= args.min_product_identity_description_chars
            and target_identity_chars >= args.min_product_identity_description_chars
            and detailed_instruction_chars >= args.min_pair_detailed_instruction_chars
            and logo_preservation_chars >= args.min_pair_logo_preservation_chars
            and small_text_preservation_chars >= args.min_pair_small_text_preservation_chars
            and target_small_text_ocr_chars >= args.min_target_small_text_ocr_chars
            and (
                not args.require_target_visible_text_in_instruction
                or not as_bool(target.get("has_marketing_text"))
                or target_text_inventory_chars < args.min_target_visible_text_inventory_chars
                or target_text_instruction_overlap >= args.min_target_text_instruction_overlap
            )
            and (not args.reject_target_back_view or not target_is_back_view(target, pair))
            and not source_complexity_reasons(source, args)
            and not target_complexity_reasons(source, target, pair, args)
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
        "require_model_high_quality_pair": args.require_model_high_quality_pair,
        "min_pair_quality_score": args.min_pair_quality_score,
        "min_pair_source_quality_score": args.min_pair_source_quality_score,
        "min_pair_target_quality_score": args.min_pair_target_quality_score,
        "min_pair_target_visibility": args.min_pair_target_visibility,
        "min_pair_target_same_confidence": args.min_pair_target_same_confidence,
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
    parser.add_argument("--min-pair-identity-confidence", type=float, default=0.85)
    parser.add_argument("--min-pair-target-aesthetic-score", type=float, default=0.0)
    parser.add_argument("--min-pair-logo-preservation-score", type=float, default=0.0)
    parser.add_argument("--min-pair-small-text-training-score", type=float, default=0.0)
    parser.add_argument("--min-product-logo-text-suitability-score", type=float, default=0.0)
    parser.add_argument(
        "--blocked-product-regex",
        default=r"nail art|nail sticker|sticker|decal|temporary tattoo|water transfer|pattern sheet|flat decorative sheet|swatch-like design",
    )
    parser.add_argument("--reject-target-back-view", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--allow-infographic-pairs", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--allow-multi-product-targets", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--max-source-product-instance-count", type=float, default=2)
    parser.add_argument("--max-target-product-instance-count", type=float, default=1)
    parser.add_argument("--min-image-detailed-description-chars", type=int, default=0)
    parser.add_argument("--min-product-identity-description-chars", type=int, default=0)
    parser.add_argument("--min-pair-detailed-instruction-chars", type=int, default=0)
    parser.add_argument("--min-pair-logo-preservation-chars", type=int, default=0)
    parser.add_argument("--min-pair-small-text-preservation-chars", type=int, default=0)
    parser.add_argument("--require-target-visible-text-in-instruction", action=argparse.BooleanOptionalAction, default=True)
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
                api_key = api_keys[submitted % len(api_keys)] if api_keys else None
                task = {
                    "product_id": product_id,
                    "selected": selected,
                    "raw_product": raw_product,
                }
                futures.add(executor.submit(annotate_one_product, task, args, api_key))
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
