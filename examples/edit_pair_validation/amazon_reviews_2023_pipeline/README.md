# Amazon Reviews 2023 Product Edit Dataset Pipeline

This pipeline builds reusable product-level annotations for image-to-image edit training from
`McAuley-Lab/Amazon-Reviews-2023`.

The design goal is to avoid wasting data or model calls:

- Keep original product metadata.
- Keep all image/video URL metadata.
- Call the multimodal model once per product/ASIN.
- Label product, image roles, and candidate pairs in one response.
- Do not delete uncertain data; keep labels, scores, reject reasons, and warnings.
- Download images only after pair selection.

## Files

Official scripts:

- `examples/edit_pair_validation/amazon_reviews_2023_pipeline/export_hf_amazon_reviews_media.py`
  Export product CSV, full raw product JSONL, image URL JSONL, and video URL JSONL from Hugging Face parquet shards.
- `examples/edit_pair_validation/amazon_reviews_2023_pipeline/annotate_product_media_once.py`
  One product per multimodal model call. Produces product/image/pair annotations.
- `examples/edit_pair_validation/amazon_reviews_2023_pipeline/prepare_hf_dataset.py`
  Package annotations into a Hugging Face uploadable JSONL dataset directory.
- `examples/edit_pair_validation/amazon_reviews_2023_pipeline/convert_to_qwen_edit_dataset.py`
  Download selected valid pairs and produce Qwen-Image-Edit style `metadata.json`.
- `data/amazon_reviews_2023/notebooks/preview_amazon_reviews_2023.ipynb`
  Local preview notebook for image quality and URL accessibility checks.

## Data Layout

Full export output:

```text
data/amazon_reviews_2023/media_all/
  products.csv
  products_raw.jsonl
  stats.json
  annotations/
    amazon_reviews_2023_media_urls.jsonl
    amazon_reviews_2023_video_urls.jsonl
```

Annotation output:

```text
data/amazon_reviews_2023/annotations/product_annotations.jsonl
```

Each annotation line keeps:

- `raw_product`: original product row from Amazon Reviews 2023 parquet.
- `source_images`: original image URL annotation records.
- `raw_annotation`: model response before post-processing.
- `annotation`: post-processed annotation with `pairs` and `valid_pairs`.
- `valid_pairs`: pair candidates accepted by rule post-processing.

## 1. Local Smoke Test

Use local data only for flow validation:

```bash
python3 examples/edit_pair_validation/amazon_reviews_2023_pipeline/annotate_product_media_once.py \
  --image-jsonl data/amazon_reviews_2023/media_all/annotations/amazon_reviews_2023_media_urls.jsonl \
  --product-raw-jsonl data/amazon_reviews_2023/media_all/products_raw_sample.jsonl \
  --out data/amazon_reviews_2023/annotations/product_annotations_test.jsonl \
  --max-products 2 \
  --max-images-per-product 6 \
  --min-images 4 \
  --max-scan-lines 1000000 \
  --overwrite
```

Preview:

```text
data/amazon_reviews_2023/previews/product_annotation_test_preview.html
```

## 2. Server Full Metadata Export

Install parquet dependency on the server:

```bash
python3 -m pip install pyarrow
```

Export all available `raw_meta_*` configs:

```bash
python3 examples/edit_pair_validation/amazon_reviews_2023_pipeline/export_hf_amazon_reviews_media.py \
  --output-dir data/amazon_reviews_2023/media_all \
  --cache-dir data/amazon_reviews_2023/cache \
  --min-images 2
```

If you only want some categories:

```bash
python3 examples/edit_pair_validation/amazon_reviews_2023_pipeline/export_hf_amazon_reviews_media.py \
  --include-config raw_meta_All_Beauty \
  --include-config raw_meta_Electronics \
  --output-dir data/amazon_reviews_2023/media_all \
  --cache-dir data/amazon_reviews_2023/cache \
  --min-images 2
```

## 3. Server Product-Level Annotation

Beauty bottles/skincare first:

```bash
python3 examples/edit_pair_validation/amazon_reviews_2023_pipeline/annotate_product_media_once.py \
  --image-jsonl data/amazon_reviews_2023/media_all/annotations/amazon_reviews_2023_media_urls.jsonl \
  --product-raw-jsonl data/amazon_reviews_2023/media_all/products_raw.jsonl \
  --out data/amazon_reviews_2023/annotations/product_annotations_beauty.jsonl \
  --category-regex 'All_Beauty' \
  --title-regex 'serum|cream|moisturizer|lotion|shampoo|conditioner|sunscreen|cleanser|face wash|toner|oil|balm|body wash|spray|gel|essence|mask|scrub' \
  --max-products 100000000 \
  --max-images-per-product 10 \
  --min-images 4 \
  --workers 4 \
  --sleep 0.1
```

All categories:

```bash
python3 examples/edit_pair_validation/amazon_reviews_2023_pipeline/annotate_product_media_once.py \
  --image-jsonl data/amazon_reviews_2023/media_all/annotations/amazon_reviews_2023_media_urls.jsonl \
  --product-raw-jsonl data/amazon_reviews_2023/media_all/products_raw.jsonl \
  --out data/amazon_reviews_2023/annotations/product_annotations_all.jsonl \
  --category-regex '' \
  --title-regex '' \
  --max-products 100000000 \
  --max-images-per-product 10 \
  --min-images 4 \
  --workers 4 \
  --sleep 0.1
```

Notes:

- `--max-valid-pairs-per-product 0` is the default and means no cap.
- The model is still instructed to select useful pairs only, not all combinations.
- A valid pair requires the target image to contain the complete same product/package as the source image. Partial eye/skin/detail/claim graphics are retained with reject reasons, not used as valid pairs.
- Existing output is resumable; without `--overwrite`, already annotated ASINs are skipped.
- For multiple API keys, set `GPT_API_KEYS=key1,key2` in `.env`, or pass `--api-keys 'key1,key2'`.
- `--workers` controls concurrent product-level model calls. Keep it no larger than your key/rate-limit capacity.

## 4. Package Annotation Dataset for Hugging Face

```bash
python3 examples/edit_pair_validation/amazon_reviews_2023_pipeline/prepare_hf_dataset.py \
  --annotations data/amazon_reviews_2023/annotations/product_annotations_beauty.jsonl \
  --output-dir data/amazon_reviews_2023/hf_product_edit_annotations \
  --validation-ratio 0.02
```

This creates:

```text
data/amazon_reviews_2023/hf_product_edit_annotations/
  README.md
  product_annotations/
    full-00000-of-00001.jsonl
    train-00000-of-00001.jsonl
    validation-00000-of-00001.jsonl
  pair_view/
    full-00000-of-00001.jsonl
    train-00000-of-00001.jsonl
    validation-00000-of-00001.jsonl
```

The primary config is `product_annotations`: one complete product/ASIN record
per line, keeping original raw metadata plus model annotations. `pair_view` is
only a derived convenience view.

Upload example:

```bash
huggingface-cli login
huggingface-cli upload <namespace>/<dataset_name> data/amazon_reviews_2023/hf_product_edit_annotations . --repo-type dataset
```

## 5. Convert Selected Pairs to Training Format

Download only selected pair images and create Qwen-Image-Edit style metadata:

```bash
python3 examples/edit_pair_validation/amazon_reviews_2023_pipeline/convert_to_qwen_edit_dataset.py \
  --annotations data/amazon_reviews_2023/hf_product_edit_annotations/product_annotations/train-00000-of-00001.jsonl \
  --output-dir data/amazon_reviews_2023/qwen_image_edit_train \
  --bottle-like-only \
  --max-transformation medium \
  --min-identity-confidence 0.85 \
  --min-training-value-score 0.0 \
  --min-edit-usefulness-score 0.0
```

Output:

```text
data/amazon_reviews_2023/qwen_image_edit_train/
  metadata.json
  pairs.json
  skipped.json
  images/
    000000-ASIN-pair/
      source.jpg
      target.jpg
```

`metadata.json` uses:

- `edit_image`: source image path.
- `image`: target image path.
- `prompt`: edit instruction.

## Recommended Selection Policy

For high precision training data:

- `reject == false`
- `pair_type in main_to_ad, main_to_angle, main_to_lifestyle, angle_to_ad`
- `transformation_magnitude in low, medium`
- `identity_confidence >= 0.85`
- `product.is_bottle_like == true` for bottle/jar/tube focused training.

Pairs with `post_filter_warnings` are retained in the annotation dataset. Exclude them by default for training, or include them with `--include-warning-pairs` after review.
