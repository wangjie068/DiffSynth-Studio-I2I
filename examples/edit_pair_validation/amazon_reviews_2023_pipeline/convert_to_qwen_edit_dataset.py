#!/usr/bin/env python3
import argparse
import json
import re
import subprocess
from io import BytesIO
from pathlib import Path


USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)


def safe_name(value, fallback):
    value = re.sub(r"\s+", " ", (value or "").strip()).lower()
    value = re.sub(r"[^a-z0-9]+", "-", value).strip("-")
    return value[:80] or fallback


def iter_jsonl(path: Path):
    with path.open(encoding="utf-8", errors="ignore") as in_file:
        for line in in_file:
            if line.strip():
                yield json.loads(line)


def iter_pair_records(path: Path):
    for record in iter_jsonl(path):
        if "source_image" in record and "target_image" in record and "pair" in record:
            yield record
            continue

        source_images = {
            item.get("image_index"): item
            for item in record.get("source_images", [])
        }
        image_labels = {
            item.get("image_index"): item
            for item in record.get("annotation", {}).get("images", [])
            if isinstance(item, dict)
        }
        product = record.get("annotation", {}).get("product", {})
        for index, pair in enumerate(record.get("valid_pairs") or []):
            source = source_images.get(pair.get("source_image_index"), {})
            target = source_images.get(pair.get("target_image_index"), {})
            source_label = image_labels.get(pair.get("source_image_index"), {})
            target_label = image_labels.get(pair.get("target_image_index"), {})
            yield {
                "pair_id": f"{record.get('product_id')}_{index:03d}",
                "product_id": record.get("product_id"),
                "product": product,
                "raw_product": record.get("raw_product"),
                "source_image": source,
                "target_image": target,
                "source_image_label": source_label,
                "target_image_label": target_label,
                "pair": pair,
                "edit_instruction": pair.get("edit_instruction", ""),
                "edit_instruction_detailed": pair.get("edit_instruction_detailed", ""),
                "source_url": source.get("fpath"),
                "target_url": target.get("fpath"),
                "pair_type": pair.get("pair_type"),
                "identity_confidence": pair.get("identity_confidence"),
                "training_value_score": pair.get("training_value_score"),
                "edit_usefulness_score": pair.get("edit_usefulness_score"),
                "target_aesthetic_score": target_label.get("aesthetic_score"),
                "aesthetic_improvement_score": pair.get("aesthetic_improvement_score"),
                "logo_preservation_score": pair.get("logo_preservation_score"),
                "logo_preservation": pair.get("logo_preservation"),
                "small_text_change": pair.get("small_text_change"),
                "small_text_training_value_score": pair.get("small_text_training_value_score"),
                "small_text_preservation": pair.get("small_text_preservation"),
                "small_text_generation": pair.get("small_text_generation"),
                "transformation_magnitude": pair.get("transformation_magnitude"),
                "post_filter_warnings": pair.get("post_filter_warnings", []),
            }


def as_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def keep_pair(record, args):
    product = record.get("product") or {}
    pair = record.get("pair") or {}
    if pair.get("reject"):
        return False
    if record.get("post_filter_warnings") and not args.include_warning_pairs:
        return False
    if args.bottle_like_only and not product.get("is_bottle_like"):
        return False
    if args.small_text_only and as_float(record.get("small_text_training_value_score")) < args.min_small_text_training_score:
        return False
    if args.pair_type_regex and not re.search(args.pair_type_regex, str(record.get("pair_type") or ""), re.I):
        return False
    if args.product_domain_regex and not re.search(args.product_domain_regex, str(product.get("domain") or ""), re.I):
        return False
    if args.max_transformation != "high" and record.get("transformation_magnitude") == "high":
        return False
    if args.max_transformation == "low" and record.get("transformation_magnitude") != "low":
        return False
    if as_float(record.get("identity_confidence")) < args.min_identity_confidence:
        return False
    if as_float(record.get("training_value_score")) < args.min_training_value_score:
        return False
    if as_float(record.get("edit_usefulness_score")) < args.min_edit_usefulness_score:
        return False
    if as_float(record.get("target_aesthetic_score")) < args.min_target_aesthetic_score:
        return False
    if as_float(record.get("aesthetic_improvement_score")) < args.min_aesthetic_improvement_score:
        return False
    if as_float(record.get("logo_preservation_score")) < args.min_logo_preservation_score:
        return False
    return bool(
        record.get("source_url")
        and record.get("target_url")
        and (record.get("edit_instruction_detailed") or record.get("edit_instruction"))
    )


def run_curl(url, timeout):
    result = subprocess.run(
        [
            "curl",
            "-L",
            "--compressed",
            "--fail",
            "--silent",
            "--show-error",
            "--max-time",
            str(timeout),
            "--user-agent",
            USER_AGENT,
            url,
        ],
        check=True,
        capture_output=True,
    )
    return result.stdout


def download_image(url, path, timeout, overwrite):
    from PIL import Image

    if path.exists() and not overwrite:
        return
    raw = run_curl(url, timeout)
    path.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(BytesIO(raw)) as image:
        image.convert("RGB").save(path, quality=95)


def parse_args():
    parser = argparse.ArgumentParser(description="Convert annotated Amazon product edit pairs to local Qwen-Image-Edit training data.")
    parser.add_argument("--annotations", type=Path, required=True, help="Product annotations JSONL or pair-level JSONL.")
    parser.add_argument("--output-dir", type=Path, default=Path("data/amazon_reviews_2023/qwen_image_edit"))
    parser.add_argument("--max-pairs", type=int)
    parser.add_argument("--timeout", type=float, default=30)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--include-warning-pairs", action="store_true")
    parser.add_argument("--bottle-like-only", action="store_true")
    parser.add_argument("--small-text-only", action="store_true")
    parser.add_argument("--min-small-text-training-score", type=float, default=0.5)
    parser.add_argument("--pair-type-regex", default="main_to_ad|main_to_angle|main_to_lifestyle|angle_to_ad")
    parser.add_argument("--product-domain-regex", default="")
    parser.add_argument("--max-transformation", choices=["low", "medium", "high"], default="medium")
    parser.add_argument("--min-identity-confidence", type=float, default=0.85)
    parser.add_argument("--min-training-value-score", type=float, default=0.0)
    parser.add_argument("--min-edit-usefulness-score", type=float, default=0.0)
    parser.add_argument("--min-target-aesthetic-score", type=float, default=0.0)
    parser.add_argument("--min-aesthetic-improvement-score", type=float, default=0.0)
    parser.add_argument("--min-logo-preservation-score", type=float, default=0.0)
    return parser.parse_args()


def main():
    args = parse_args()
    metadata = []
    pairs_sidecar = []
    skipped = []

    for record in iter_pair_records(args.annotations):
        if not keep_pair(record, args):
            continue
        if args.max_pairs is not None and len(metadata) >= args.max_pairs:
            break

        pair_slug = f"{len(metadata):06d}-{record.get('product_id', 'unknown')}-{safe_name(record.get('pair_id'), 'pair')}"
        source_path = args.output_dir / "images" / pair_slug / "source.jpg"
        target_path = args.output_dir / "images" / pair_slug / "target.jpg"
        if not args.dry_run:
            try:
                download_image(record["source_url"], source_path, args.timeout, args.overwrite)
                download_image(record["target_url"], target_path, args.timeout, args.overwrite)
            except Exception as error:
                skipped.append({
                    "pair_id": record.get("pair_id"),
                    "reason": str(error),
                    "source_url": record.get("source_url"),
                    "target_url": record.get("target_url"),
                })
                continue

        prompt = record.get("edit_instruction_detailed") or record.get("edit_instruction")
        metadata.append({
            "image": str(target_path.relative_to(args.output_dir)),
            "edit_image": str(source_path.relative_to(args.output_dir)),
            "prompt": prompt,
            "product_id": record.get("product_id"),
            "pair_id": record.get("pair_id"),
            "pair_type": record.get("pair_type"),
            "identity_confidence": record.get("identity_confidence"),
            "training_value_score": record.get("training_value_score"),
            "edit_usefulness_score": record.get("edit_usefulness_score"),
            "target_aesthetic_score": record.get("target_aesthetic_score"),
            "aesthetic_improvement_score": record.get("aesthetic_improvement_score"),
            "logo_preservation_score": record.get("logo_preservation_score"),
            "logo_preservation": record.get("logo_preservation"),
            "small_text_change": record.get("small_text_change"),
            "small_text_training_value_score": record.get("small_text_training_value_score"),
            "small_text_preservation": record.get("small_text_preservation"),
            "small_text_generation": record.get("small_text_generation"),
            "transformation_magnitude": record.get("transformation_magnitude"),
            "source_url": record.get("source_url"),
            "target_url": record.get("target_url"),
        })
        pairs_sidecar.append(record)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.output_dir / "pairs.json").write_text(json.dumps(pairs_sidecar, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.output_dir / "skipped.json").write_text(json.dumps(skipped, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote metadata={len(metadata)} skipped={len(skipped)} output_dir={args.output_dir}")
    if args.dry_run:
        print("dry-run: images were not downloaded")


if __name__ == "__main__":
    main()
