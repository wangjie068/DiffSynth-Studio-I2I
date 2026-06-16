#!/usr/bin/env python3
import argparse
import csv
import json
import re
import subprocess
import sys
import urllib.request
from pathlib import Path


HF_REPO = "McAuley-Lab/Amazon-Reviews-2023"
HF_BASE = f"https://huggingface.co/datasets/{HF_REPO}"
ASIN_RE = re.compile(r"^[A-Z0-9]{10}$")


def require_pyarrow():
    try:
        import pyarrow.parquet as pq
    except ImportError:
        print(
            "Missing dependency: pyarrow. Install it first, for example:\n"
            "  python3 -m pip install --target data/amazon_beauty_products/.python_deps pyarrow\n"
            "  PYTHONPATH=data/amazon_beauty_products/.python_deps python3 ...\n",
            file=sys.stderr,
        )
        raise
    return pq


def read_hf_tree(path=""):
    url = f"https://huggingface.co/api/datasets/{HF_REPO}/tree/main"
    if path:
        url += f"/{path}"
    with urllib.request.urlopen(url, timeout=60) as response:
        return json.loads(response.read().decode("utf-8"))


def discover_configs(include_configs, exclude_configs):
    if include_configs:
        configs = include_configs
    else:
        configs = [
            item["path"]
            for item in read_hf_tree()
            if item.get("path", "").startswith("raw_meta_")
        ]
    excluded = set(exclude_configs or [])
    return [config for config in sorted(configs) if config not in excluded]


def discover_parquet_files(config):
    files = []
    for item in read_hf_tree(config):
        path = item.get("path", "")
        if path.endswith(".parquet"):
            files.append((path, item.get("size", 0)))
    return sorted(files)


def download_if_needed(remote_path, cache_dir):
    local_path = cache_dir / remote_path
    if local_path.exists() and local_path.stat().st_size > 0:
        return local_path
    local_path.parent.mkdir(parents=True, exist_ok=True)
    url = f"{HF_BASE}/resolve/main/{remote_path}"
    subprocess.run(
        ["curl", "-L", "--fail", "--retry", "3", "--output", str(local_path), url],
        check=True,
    )
    return local_path


def first_non_empty(values):
    for value in values or []:
        if value:
            return value
    return ""


def compact_images(images):
    if not images:
        return []
    hi_res = images.get("hi_res") or []
    large = images.get("large") or []
    thumb = images.get("thumb") or []
    variant = images.get("variant") or []
    rows = []
    for index in range(max(len(hi_res), len(large), len(thumb), len(variant))):
        url = first_non_empty([
            hi_res[index] if index < len(hi_res) else None,
            large[index] if index < len(large) else None,
            thumb[index] if index < len(thumb) else None,
        ])
        if url:
            rows.append({
                "url": url,
                "variant": variant[index] if index < len(variant) else "",
            })
    return rows


def compact_videos(videos):
    if not videos:
        return []
    titles = videos.get("title") or []
    urls = videos.get("url") or []
    rows = []
    for index in range(max(len(titles), len(urls))):
        url = urls[index] if index < len(urls) else ""
        if url:
            rows.append({
                "url": url,
                "title": titles[index] if index < len(titles) else "",
            })
    return rows


def scalar(value):
    return "" if value is None else value


def iter_rows(parquet_path, batch_size):
    pq = require_pyarrow()
    parquet_file = pq.ParquetFile(parquet_path)
    for batch in parquet_file.iter_batches(batch_size=batch_size):
        yield from batch.to_pylist()


def append_jsonl(path, item):
    path.write(json.dumps(item, ensure_ascii=False) + "\n")


def parse_args():
    parser = argparse.ArgumentParser(description="Export media URLs from Amazon Reviews 2023 raw_meta categories.")
    parser.add_argument("--include-config", action="append", default=[], help="Specific raw_meta_* config to export. Defaults to all.")
    parser.add_argument("--exclude-config", action="append", default=[], help="raw_meta_* config to skip.")
    parser.add_argument("--cache-dir", type=Path, default=Path("data/amazon_reviews_2023/cache"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/amazon_reviews_2023/media_all"))
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--min-images", type=int, default=2)
    parser.add_argument("--keep-parquet", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    configs = discover_configs(args.include_config, args.exclude_config)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    annotations_dir = args.output_dir / "annotations"
    annotations_dir.mkdir(parents=True, exist_ok=True)

    product_csv = args.output_dir / "products.csv"
    product_raw_jsonl = args.output_dir / "products_raw.jsonl"
    image_jsonl = annotations_dir / "amazon_reviews_2023_media_urls.jsonl"
    video_jsonl = annotations_dir / "amazon_reviews_2023_video_urls.jsonl"
    stats_json = args.output_dir / "stats.json"

    seen_products = set()
    stats = {
        "configs": {},
        "scanned_rows": 0,
        "exported_products": 0,
        "exported_images": 0,
        "exported_videos": 0,
    }

    fieldnames = [
        "product_id",
        "product_title",
        "product_category",
        "product_page_url",
        "review_count",
        "avg_rating",
        "source",
        "image_count",
        "video_count",
    ]
    with product_csv.open("w", newline="", encoding="utf-8") as csv_file, \
            product_raw_jsonl.open("w", encoding="utf-8") as product_raw_file, \
            image_jsonl.open("w", encoding="utf-8") as image_file, \
            video_jsonl.open("w", encoding="utf-8") as video_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        for config in configs:
            config_stats = {"scanned_rows": 0, "exported_products": 0, "exported_images": 0, "exported_videos": 0}
            stats["configs"][config] = config_stats
            for remote_path, size in discover_parquet_files(config):
                print(f"downloading/reading {remote_path} ({size} bytes)", flush=True)
                parquet_path = download_if_needed(remote_path, args.cache_dir)
                for row in iter_rows(parquet_path, args.batch_size):
                    stats["scanned_rows"] += 1
                    config_stats["scanned_rows"] += 1
                    asin = scalar(row.get("parent_asin"))
                    if not ASIN_RE.fullmatch(asin) or asin in seen_products:
                        continue
                    images = compact_images(row.get("images"))
                    if len(images) < args.min_images:
                        continue
                    videos = compact_videos(row.get("videos"))
                    seen_products.add(asin)

                    category = config.removeprefix("raw_meta_")
                    title = scalar(row.get("title"))
                    product_url = f"https://www.amazon.com/dp/{asin}"
                    product_raw = dict(row)
                    product_raw["source_config"] = config
                    product_raw["product_page_url"] = product_url
                    append_jsonl(product_raw_file, product_raw)
                    writer.writerow({
                        "product_id": asin,
                        "product_title": title,
                        "product_category": category,
                        "product_page_url": product_url,
                        "review_count": scalar(row.get("rating_number")),
                        "avg_rating": scalar(row.get("average_rating")),
                        "source": "hf_mcauley_amazon_reviews_2023",
                        "image_count": len(images),
                        "video_count": len(videos),
                    })
                    stats["exported_products"] += 1
                    config_stats["exported_products"] += 1

                    for index, image in enumerate(images):
                        append_jsonl(image_file, {
                            "category": category,
                            "fpath": image["url"],
                            "image_id": f"{asin}_img_{index:02d}",
                            "product_id": asin,
                            "product_title": title,
                            "product_page_url": product_url,
                            "media_type": "image",
                            "variant": image.get("variant", ""),
                        })
                        stats["exported_images"] += 1
                        config_stats["exported_images"] += 1

                    for index, video in enumerate(videos):
                        append_jsonl(video_file, {
                            "category": category,
                            "url": video["url"],
                            "video_id": f"{asin}_video_{index:02d}",
                            "product_id": asin,
                            "product_title": title,
                            "product_page_url": product_url,
                            "media_type": "video",
                            "title": video.get("title", ""),
                        })
                        stats["exported_videos"] += 1
                        config_stats["exported_videos"] += 1

                stats_json.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
                if not args.keep_parquet and parquet_path.exists():
                    parquet_path.unlink()
            print(f"{config}: {config_stats}", flush=True)

    stats_json.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    print(f"products={product_csv}")
    print(f"products_raw={product_raw_jsonl}")
    print(f"image_annotations={image_jsonl}")
    print(f"video_urls={video_jsonl}")


if __name__ == "__main__":
    main()
