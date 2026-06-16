#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

from annotate_product_media_once import postprocess_annotation


def iter_jsonl(path: Path):
    with path.open(encoding="utf-8", errors="ignore") as in_file:
        for line in in_file:
            if line.strip():
                yield json.loads(line)


def parse_args():
    parser = argparse.ArgumentParser(description="Re-run postprocessing on existing product annotations without model calls.")
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--min-pair-source-score", type=float, default=0.75)
    parser.add_argument("--min-pair-target-score", type=float, default=0.5)
    parser.add_argument("--require-model-high-quality-pair", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--min-pair-quality-score", type=float, default=0.0)
    parser.add_argument("--min-pair-source-quality-score", type=float, default=0.0)
    parser.add_argument("--min-pair-target-quality-score", type=float, default=0.0)
    parser.add_argument("--min-pair-target-visibility", type=float, default=0.25)
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
    parser.add_argument("--min-image-detailed-description-chars", type=int, default=0)
    parser.add_argument("--min-product-identity-description-chars", type=int, default=0)
    parser.add_argument("--min-pair-detailed-instruction-chars", type=int, default=0)
    parser.add_argument("--min-pair-logo-preservation-chars", type=int, default=0)
    parser.add_argument("--min-pair-small-text-preservation-chars", type=int, default=0)
    parser.add_argument(
        "--min-target-small-text-ocr-chars",
        "--min-image-small-text-ocr-chars",
        dest="min_target_small_text_ocr_chars",
        type=int,
        default=0,
    )
    parser.add_argument("--require-selected-main-source", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-valid-pairs-per-product", type=int, default=0)
    parser.add_argument("--high-confidence-override-identity", type=float, default=0.9)
    parser.add_argument("--high-confidence-override-training-value", type=float, default=0.7)
    parser.add_argument("--high-confidence-override-usefulness", type=float, default=0.7)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.out.exists() and not args.overwrite:
        raise SystemExit(f"{args.out} exists; pass --overwrite")
    args.out.parent.mkdir(parents=True, exist_ok=True)

    policy = SimpleNamespace(**vars(args))
    count = 0
    valid_pairs = 0
    with args.out.open("w", encoding="utf-8") as out_file:
        for record in iter_jsonl(args.annotations):
            raw_annotation = record.get("raw_annotation") or record.get("annotation")
            if raw_annotation:
                annotation, pairs = postprocess_annotation(raw_annotation, policy)
                record["annotation"] = annotation
                record["valid_pairs"] = pairs
                valid_pairs += len(pairs)
            out_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    print(f"products={count} valid_pairs={valid_pairs} out={args.out}")


if __name__ == "__main__":
    main()
