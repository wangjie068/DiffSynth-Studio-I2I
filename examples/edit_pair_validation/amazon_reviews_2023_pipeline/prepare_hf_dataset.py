#!/usr/bin/env python3
import argparse
import json
import random
import shutil
from pathlib import Path


def iter_jsonl(path: Path):
    with path.open(encoding="utf-8", errors="ignore") as in_file:
        for line in in_file:
            if not line.strip():
                continue
            yield json.loads(line)


def is_valid_pair(pair: dict, include_warnings: bool):
    if pair.get("reject"):
        return False
    if pair.get("post_filter_warnings") and not include_warnings:
        return False
    return True


def pair_record(product_record: dict, pair: dict, pair_index: int):
    product_id = product_record.get("product_id")
    source_images = {
        item.get("image_index"): item
        for item in product_record.get("source_images", [])
    }
    source = source_images.get(pair.get("source_image_index"), {})
    target = source_images.get(pair.get("target_image_index"), {})
    annotation = product_record.get("annotation", {})
    product_labels = annotation.get("product", {})
    return {
        "pair_id": f"{product_id}_{pair_index:03d}",
        "product_id": product_id,
        "source_dataset": product_record.get("source_dataset"),
        "product": product_labels,
        "raw_product": product_record.get("raw_product"),
        "source_image": source,
        "target_image": target,
        "pair": pair,
        "edit_instruction": pair.get("edit_instruction", ""),
        "edit_instruction_detailed": pair.get("edit_instruction_detailed", ""),
        "source_url": source.get("fpath"),
        "target_url": target.get("fpath"),
        "pair_type": pair.get("pair_type"),
        "identity_confidence": pair.get("identity_confidence"),
        "training_value_score": pair.get("training_value_score"),
        "edit_usefulness_score": pair.get("edit_usefulness_score"),
        "small_text_change": pair.get("small_text_change"),
        "small_text_training_value_score": pair.get("small_text_training_value_score"),
        "small_text_preservation": pair.get("small_text_preservation"),
        "small_text_generation": pair.get("small_text_generation"),
        "transformation_magnitude": pair.get("transformation_magnitude"),
        "post_filter_warnings": pair.get("post_filter_warnings", []),
    }


def write_jsonl(path: Path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as out_file:
        for record in records:
            out_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count


def pairs_from_products(products, include_warnings: bool):
    pairs = []
    for product in products:
        pair_source = product.get("valid_pairs")
        if pair_source is None:
            pair_source = [
                pair
                for pair in product.get("annotation", {}).get("pairs", [])
                if is_valid_pair(pair, include_warnings)
            ]
        for index, pair in enumerate(pair_source):
            if not is_valid_pair(pair, include_warnings):
                continue
            pairs.append(pair_record(product, pair, index))
    return pairs


def dataset_card(pair_count: int, product_count: int):
    return f"""---
task_categories:
- image-to-image
language:
- en
tags:
- amazon
- ecommerce
- image-editing
- product-images
pretty_name: Amazon Reviews 2023 Product Edit Pair Annotations
configs:
- config_name: product_annotations
  data_files:
  - split: full
    path: product_annotations/full-*.jsonl
  - split: train
    path: product_annotations/train-*.jsonl
  - split: validation
    path: product_annotations/validation-*.jsonl
- config_name: pair_view
  data_files:
  - split: full
    path: pair_view/full-*.jsonl
  - split: train
    path: pair_view/train-*.jsonl
  - split: validation
    path: pair_view/validation-*.jsonl
---

# Amazon Reviews 2023 Product Edit Pair Annotations

This dataset contains product-level multimodal annotations derived from
`McAuley-Lab/Amazon-Reviews-2023` raw metadata. Each product annotation keeps
the original product metadata, source image URL records, model-produced image
labels, small-text annotations, and post-processed candidate edit pairs.

## Files

- `product_annotations/full-*.jsonl`: one complete record per Amazon product/ASIN.
- `product_annotations/train-*.jsonl`, `product_annotations/validation-*.jsonl`: product-level splits.
- `pair_view/*.jsonl`: derived pair-level view for convenience only.

## Current Export

- Product records: {product_count}
- Pair records: {pair_count}

Each product record keeps original product metadata, original image URL records,
model annotation, small-text labels, post-processed valid pairs, and any
warnings/reject reasons.
Images are referenced by URL and are not redistributed in this annotation dataset.
Use the conversion script in the source repository to download only selected
training pairs.

## Example Load

```python
from datasets import load_dataset

products = load_dataset("namespace/dataset_name", "product_annotations")
pairs = load_dataset("namespace/dataset_name", "pair_view")
```
"""


def parse_args():
    parser = argparse.ArgumentParser(description="Package product annotations as a Hugging Face uploadable JSONL dataset.")
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("data/amazon_reviews_2023/hf_product_edit_annotations"))
    parser.add_argument("--validation-ratio", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--include-warning-pairs", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.output_dir.exists():
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    products = []
    for product in iter_jsonl(args.annotations):
        products.append(product)

    products_for_split = list(products)
    random.Random(args.seed).shuffle(products_for_split)
    val_count = int(round(len(products_for_split) * args.validation_ratio))
    validation = products_for_split[:val_count]
    train = products_for_split[val_count:]

    pairs = pairs_from_products(products, args.include_warning_pairs)
    train_pairs = pairs_from_products(train, args.include_warning_pairs)
    validation_pairs = pairs_from_products(validation, args.include_warning_pairs)

    product_count = write_jsonl(args.output_dir / "product_annotations" / "full-00000-of-00001.jsonl", products)
    write_jsonl(args.output_dir / "product_annotations" / "train-00000-of-00001.jsonl", train)
    write_jsonl(args.output_dir / "product_annotations" / "validation-00000-of-00001.jsonl", validation)
    pair_count = write_jsonl(args.output_dir / "pair_view" / "full-00000-of-00001.jsonl", pairs)
    write_jsonl(args.output_dir / "pair_view" / "train-00000-of-00001.jsonl", train_pairs)
    write_jsonl(args.output_dir / "pair_view" / "validation-00000-of-00001.jsonl", validation_pairs)
    (args.output_dir / "README.md").write_text(dataset_card(pair_count, product_count), encoding="utf-8")

    print(f"products={product_count}")
    print(f"pairs={pair_count}")
    print(f"train={len(train)} validation={len(validation)}")
    print(f"output_dir={args.output_dir}")


if __name__ == "__main__":
    main()
