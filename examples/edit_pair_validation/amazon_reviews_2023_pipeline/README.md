# Amazon Reviews 2023 Product Edit Dataset Pipeline

This pipeline builds reusable product-level annotations for image-to-image edit training from
`McAuley-Lab/Amazon-Reviews-2023`.

The design goal is to avoid wasting data or model calls:

- Keep original product metadata.
- Keep all image/video URL metadata.
- Call the multimodal model once per product/ASIN.
- Label product, image roles, small-text properties, and candidate pairs in one response.
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

Logo, small text, and aesthetics are first-class labels. Product records include
logo/brand notes, `has_small_text`, and `small_text_training_potential`; image
records include aesthetic score, logo regions/legibility, small-text density,
regions, legibility, OCR text/spans, transcription snippets, and risk. The model
must also write `source_selection.selected_main_source_image_index`; Amazon's
`variant=MAIN` is only an input hint, not a default source decision. Pair records
include the model's high-quality-pair judgement, pair quality score/tier, logo
preservation score, aesthetic improvement score, small-text change type,
preservation/generation notes, detailed edit instructions, and
`small_text_training_value_score`.

Quality gates can also require sufficient text descriptions, detailed edit
instructions, product identity descriptions, and preservation notes before a pair
is accepted for training.

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

Beauty/package small-text products first:

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
  --require-model-high-quality-pair \
  --min-pair-quality-score 0.7 \
  --min-pair-source-quality-score 0.65 \
  --min-pair-target-quality-score 0.65 \
  --min-pair-logo-preservation-score 0.7 \
  --min-pair-small-text-training-score 0.5 \
  --min-pair-target-aesthetic-score 0.6 \
  --min-image-detailed-description-chars 120 \
  --min-product-identity-description-chars 60 \
  --min-pair-detailed-instruction-chars 180 \
  --min-pair-logo-preservation-chars 30 \
  --min-pair-small-text-preservation-chars 30 \
  --min-target-small-text-ocr-chars 20 \
  --require-selected-main-source \
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
- `--require-model-high-quality-pair --min-pair-quality-score 0.7` uses the model's own strict high-quality-pair judgement before rule gates.
- `--require-selected-main-source` requires the model to explicitly choose the source image; pairs using another source image are rejected. This prevents blindly trusting Amazon's `MAIN` variant.
- `--min-pair-source-quality-score` and `--min-pair-target-quality-score` keep blurry/low-quality images out of `valid_pairs`.
- `--min-pair-logo-preservation-score 0.7` makes `valid_pairs` focus on pairs where the product logo/brand mark remains usable.
- `--min-pair-small-text-training-score 0.5` makes `valid_pairs` focus on useful small-text supervision. Leave it at `0.0` if you want to keep non-small-text valid pairs and filter later.
- `--min-pair-target-aesthetic-score 0.6` keeps targets visually polished enough for product/ad training.
- Description length gates prevent underspecified image descriptions, identity descriptions, edit instructions, and preservation notes from entering `valid_pairs`.
- `--min-target-small-text-ocr-chars 20` requires target small-text OCR content to be written out. Source OCR is useful metadata but is not a hard reject reason.
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
  --small-text-only \
  --min-small-text-training-score 0.5 \
  --require-model-high-quality-pair \
  --min-pair-quality-score 0.7 \
  --min-logo-preservation-score 0.7 \
  --min-target-aesthetic-score 0.6 \
  --min-aesthetic-improvement-score 0.4 \
  --min-source-quality-score 0.65 \
  --min-target-quality-score 0.65 \
  --min-image-detailed-description-chars 120 \
  --min-product-identity-description-chars 60 \
  --min-prompt-chars 180 \
  --min-logo-preservation-chars 30 \
  --min-small-text-preservation-chars 30 \
  --min-target-small-text-ocr-chars 20 \
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
- `prompt`: detailed edit instruction when available, otherwise the concise edit instruction.

## Recommended Selection Policy

For high precision training data:

- `reject == false`
- `pair_type in main_to_ad, main_to_angle, main_to_lifestyle, angle_to_ad`
- `transformation_magnitude in low, medium`
- `identity_confidence >= 0.85`
- model judged `is_high_quality_pair == true`
- `pair_quality_score >= 0.7`
- `logo_preservation_score >= 0.7`
- `small_text_training_value_score >= 0.5`
- `target_aesthetic_score >= 0.6`
- `aesthetic_improvement_score >= 0.4`
- `source/target quality_score >= 0.65`
- detailed image descriptions, product identity descriptions, prompts, and preservation notes are long enough to be useful.
- target includes explicit small-text OCR text/spans.
- `target.full_product_visible == true`
- `annotation.images[*].small_text_legibility in partial/readable` for manual inspection subsets.

Pairs with `post_filter_warnings` are retained in the annotation dataset. Exclude them by default for training, or include them with `--include-warning-pairs` after review.
