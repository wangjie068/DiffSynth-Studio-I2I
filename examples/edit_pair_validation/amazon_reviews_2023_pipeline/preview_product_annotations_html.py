#!/usr/bin/env python3
import argparse
import collections
import html
import json
from pathlib import Path


def esc(value):
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        value = json.dumps(value, ensure_ascii=False, indent=2)
    return html.escape(str(value))


def iter_jsonl(path: Path):
    with path.open(encoding="utf-8", errors="ignore") as in_file:
        for line in in_file:
            if line.strip():
                yield json.loads(line)


def image_url(item: dict):
    return item.get("fpath") or item.get("url") or ""


def short(value, limit=220):
    text = str(value or "").strip()
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "..."


def span_list(items):
    if not items:
        return '<span class="muted">none</span>'
    if isinstance(items, str):
        return f"<span>{esc(items)}</span>"
    parts = []
    for item in items:
        if isinstance(item, dict):
            region = item.get("region") or "region"
            text = item.get("text") or ""
            legibility = item.get("legibility") or ""
            confidence = item.get("confidence")
            meta = f"{legibility}"
            if confidence is not None:
                meta = f"{meta} {confidence}".strip()
            parts.append(
                f'<span class="chip" title="{esc(meta)}">'
                f"<b>{esc(region)}:</b> {esc(text)}</span>"
            )
        else:
            parts.append(f'<span class="chip">{esc(item)}</span>')
    return "".join(parts)


def metric(label, value, cls=""):
    if value is None:
        value = ""
    return f'<span class="metric {cls}"><b>{esc(label)}</b>{esc(value)}</span>'


def image_card(source_item: dict, label: dict, selected_index):
    index = label.get("image_index", source_item.get("image_index"))
    selected = index == selected_index
    url = image_url(source_item)
    variant = source_item.get("variant") or ""
    role = label.get("role") or ""
    classes = "image-card selected" if selected else "image-card"
    return f"""
    <article class="{classes}" id="img-{esc(index)}">
      <div class="image-head">
        <b>#{esc(index)} {esc(variant)}</b>
        <span>{esc(role)}</span>
        {'<em>SELECTED SOURCE</em>' if selected else ''}
      </div>
      <a href="{esc(url)}" target="_blank"><img src="{esc(url)}" loading="lazy"></a>
      <div class="metrics">
        {metric("S", label.get("source_candidate_score"))}
        {metric("T", label.get("target_candidate_score"))}
        {metric("Q", label.get("quality_score"))}
        {metric("A", label.get("aesthetic_score"))}
        {metric("vis", label.get("main_product_visibility"))}
        {metric("same", label.get("same_product_confidence"))}
      </div>
      <p><b>description</b><br>{esc(label.get("detailed_description"))}</p>
      <p><b>identity</b><br>{esc(label.get("product_identity_description"))}</p>
      <p><b>visual inventory</b><br>{esc(label.get("visual_inventory"))}</p>
      <p><b>visible text inventory</b><br>{esc(label.get("visible_text_inventory"))}</p>
      <p><b>logo</b><br>{span_list(label.get("logo_or_brand_transcription"))}</p>
      <p><b>small text OCR</b><br>{esc(label.get("small_text_ocr_text"))}</p>
      <div class="chips">{span_list(label.get("small_text_ocr_spans"))}</div>
      <p class="muted"><b>target edit</b><br>{esc(label.get("target_edit_description"))}</p>
    </article>
    """


def pair_card(pair: dict, source_item: dict, target_item: dict, source_label: dict, target_label: dict, index: int):
    final_status = pair.get("final_status") or ("rejected" if pair.get("reject") else "selected")
    status = str(final_status).upper()
    status_class = "good" if final_status == "selected" else "bad" if final_status == "rejected" else "review"
    source_url = image_url(source_item)
    target_url = image_url(target_item)
    reasons = pair.get("post_filter_reasons") or pair.get("reject_reasons") or []
    return f"""
    <article class="pair-card {status_class}">
      <header>
        <h4>Pair #{index} <span class="{status_class}">{status}</span></h4>
        <div class="metrics">
          {metric("type", pair.get("pair_type"))}
          {metric("model", pair.get("model_status"))}
          {metric("post", pair.get("post_filter_status"))}
          {metric("final", final_status)}
          {metric("identity", pair.get("identity_confidence"))}
          {metric("quality", pair.get("pair_quality_score"))}
          {metric("tier", pair.get("pair_quality_tier"))}
          {metric("logo", pair.get("logo_preservation_score"))}
          {metric("small", pair.get("small_text_training_value_score"))}
          {metric("target A", target_label.get("aesthetic_score"))}
          {metric("aesthetic +", pair.get("aesthetic_improvement_score"))}
          {metric("transform", pair.get("transformation_magnitude"))}
        </div>
      </header>
      <div class="pair-images">
        <div>
          <b>source #{esc(pair.get("source_image_index"))}</b>
          <a href="{esc(source_url)}" target="_blank"><img src="{esc(source_url)}" loading="lazy"></a>
        </div>
        <div>
          <b>target #{esc(pair.get("target_image_index"))}</b>
          <a href="{esc(target_url)}" target="_blank"><img src="{esc(target_url)}" loading="lazy"></a>
        </div>
      </div>
      {'<p class="reject"><b>post filter reasons</b><br>' + esc(", ".join(reasons)) + '</p>' if reasons else ''}
      <p><b>review reasons</b><br>{esc(pair.get("review_reasons"))}</p>
      <p><b>model reject diagnostics</b><br>{esc(pair.get("model_reject_reasons"))}</p>
      <p><b>model quality judgement</b><br>{esc(pair.get("pair_quality_judgement"))}</p>
      <p><b>failure modes</b><br>{esc(pair.get("pair_failure_modes"))}</p>
      <p><b>instruction</b><br>{esc(pair.get("edit_instruction"))}</p>
      <p><b>detailed instruction</b><br>{esc(pair.get("edit_instruction_detailed"))}</p>
      <p><b>source text to preserve</b><br>{esc(pair.get("source_visible_text_to_preserve"))}</p>
      <p><b>target text to generate</b><br>{esc(pair.get("target_visible_text_to_generate"))}</p>
      <p><b>text rendering requirements</b><br>{esc(pair.get("text_rendering_requirements"))}</p>
      <p><b>logo preservation</b><br>{esc(pair.get("logo_preservation"))}</p>
      <p><b>small text preservation</b><br>{esc(pair.get("small_text_preservation"))}</p>
      <p><b>small text generation</b><br>{esc(pair.get("small_text_generation"))}</p>
    </article>
    """


def product_section(record: dict, show_all_pairs: bool):
    product_id = record.get("product_id")
    ann = record.get("annotation") or {}
    product = ann.get("product") or {}
    metadata = record.get("metadata") or {}
    raw = record.get("raw_product") or {}
    source_selection = ann.get("source_selection") or {}
    selected_index = source_selection.get("selected_main_source_image_index")
    source_images = {item.get("image_index"): item for item in record.get("source_images", [])}
    image_labels = {
        item.get("image_index"): item
        for item in ann.get("images", [])
        if isinstance(item, dict)
    }
    pairs = ann.get("pairs") or []
    valid_pairs = record.get("valid_pairs") or ann.get("valid_pairs") or []
    visible_pairs = pairs if show_all_pairs else valid_pairs
    product_url = metadata.get("product_page_url") or raw.get("product_page_url") or ""
    title = metadata.get("title") or raw.get("title") or ""
    images_html = "\n".join(
        image_card(source_images.get(idx, {}), label, selected_index)
        for idx, label in sorted(image_labels.items(), key=lambda kv: kv[0] if kv[0] is not None else 999)
    )
    pair_html = "\n".join(
        pair_card(
            pair,
            source_images.get(pair.get("source_image_index"), {}),
            source_images.get(pair.get("target_image_index"), {}),
            image_labels.get(pair.get("source_image_index"), {}),
            image_labels.get(pair.get("target_image_index"), {}),
            index,
        )
        for index, pair in enumerate(visible_pairs)
    )
    return f"""
    <section class="product">
      <header class="product-head">
        <div>
          <h2>{esc(product_id)} <span>{esc(product.get("domain"))}</span></h2>
          <p><a href="{esc(product_url)}" target="_blank">{esc(short(title, 180))}</a></p>
          <p class="muted">{esc(product.get("notes"))}</p>
        </div>
        <div class="summary-box">
          {metric("images", len(image_labels))}
          {metric("pairs", len(pairs))}
          {metric("valid", len(valid_pairs), "good")}
          {metric("source", selected_index)}
          {metric("source conf", source_selection.get("selection_confidence"))}
          {metric("small text", product.get("small_text_training_potential"))}
        </div>
      </header>
      <details open>
        <summary>source selection</summary>
        <p>{esc(source_selection.get("selection_reason"))}</p>
        <pre>{esc(source_selection.get("rejected_source_candidates"))}</pre>
      </details>
      <details>
        <summary>product labels</summary>
        <pre>{esc(product)}</pre>
      </details>
      <h3>Images</h3>
      <div class="image-grid">{images_html}</div>
      <h3>Pairs {'(all)' if show_all_pairs else '(selected only)'}</h3>
      <div class="pair-grid">{pair_html or '<p class="muted">No pairs to show.</p>'}</div>
    </section>
    """


def build_html(records: list[dict], args, stats: dict, rejects: collections.Counter):
    show_all_pairs = not args.selected_only
    sections = "\n".join(product_section(record, show_all_pairs) for record in records)
    reject_rows = "\n".join(
        f"<tr><td>{esc(reason)}</td><td>{count}</td></tr>"
        for reason, count in rejects.most_common(30)
    )
    return f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Amazon Product Annotation Preview</title>
<style>
body {{ margin:0; font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Arial,sans-serif; color:#222; background:#f5f5f3; }}
a {{ color:#164f9f; }}
.top {{ position:sticky; top:0; z-index:5; background:#fff; border-bottom:1px solid #ddd; padding:14px 22px; }}
.top h1 {{ margin:0 0 8px; font-size:20px; }}
.stats, .metrics {{ display:flex; gap:8px; flex-wrap:wrap; align-items:center; }}
.metric {{ display:inline-flex; gap:6px; border:1px solid #d6d6d1; background:#fff; border-radius:6px; padding:4px 7px; font-size:12px; }}
.metric.good, .good {{ color:#087443; border-color:#97c9b0; }}
.bad {{ color:#b42318; border-color:#f0a8a1; }}
.review {{ color:#9a5b00; border-color:#e7c27f; }}
.muted {{ color:#666; }}
.product {{ margin:18px 22px 28px; background:#fff; border:1px solid #ddd; border-radius:8px; padding:18px; }}
.product-head {{ display:flex; justify-content:space-between; gap:16px; border-bottom:1px solid #eee; padding-bottom:12px; }}
.product-head h2 {{ margin:0; font-size:18px; }}
.product-head h2 span {{ color:#666; font-size:13px; font-weight:400; }}
.summary-box {{ min-width:270px; }}
details {{ margin:12px 0; background:#fafafa; border:1px solid #e5e5e5; border-radius:6px; padding:8px 10px; }}
summary {{ cursor:pointer; font-weight:700; }}
pre {{ white-space:pre-wrap; word-break:break-word; font-size:12px; background:#f6f6f6; padding:8px; border-radius:5px; }}
.image-grid {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(260px,1fr)); gap:14px; }}
.image-card {{ border:1px solid #ddd; border-radius:8px; padding:10px; background:#fff; }}
.image-card.selected {{ border:3px solid #168a52; }}
.image-head {{ display:flex; gap:8px; justify-content:space-between; align-items:center; font-size:13px; min-height:24px; }}
.image-head em {{ font-size:11px; color:#087443; font-style:normal; font-weight:700; }}
img {{ max-width:100%; height:220px; object-fit:contain; background:#fff; display:block; margin:8px auto; }}
.image-card p, .pair-card p {{ font-size:12px; line-height:1.35; }}
.chips {{ display:flex; flex-wrap:wrap; gap:4px; }}
.chip {{ display:inline-block; border:1px solid #d7d7d7; border-radius:5px; padding:3px 5px; margin:2px; font-size:11px; background:#fbfbfb; }}
.pair-grid {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(430px,1fr)); gap:14px; }}
.pair-card {{ border:1px solid #ddd; border-radius:8px; padding:12px; background:#fff; }}
.pair-card.good {{ border-left:5px solid #168a52; }}
.pair-card.bad {{ border-left:5px solid #c3372b; }}
.pair-card.review {{ border-left:5px solid #c98213; }}
.pair-card h4 {{ margin:0 0 8px; }}
.pair-images {{ display:grid; grid-template-columns:1fr 1fr; gap:10px; }}
.pair-images img {{ height:210px; }}
.reject {{ background:#fff1ef; border:1px solid #f2b8b2; border-radius:6px; padding:8px; }}
table {{ border-collapse:collapse; font-size:12px; }}
td, th {{ border:1px solid #ddd; padding:4px 7px; }}
</style>
</head>
<body>
<div class="top">
  <h1>Amazon Product Annotation Preview</h1>
  <div class="stats">
    {metric("products", stats["products"])}
    {metric("images", stats["images"])}
    {metric("pairs", stats["pairs"])}
    {metric("valid pairs", stats["valid_pairs"], "good")}
    {metric("show", "all pairs" if show_all_pairs else "selected only")}
  </div>
  <details>
    <summary>Top reject reasons</summary>
    <table><tr><th>reason</th><th>count</th></tr>{reject_rows}</table>
  </details>
</div>
{sections}
</body>
</html>
"""


def parse_args():
    parser = argparse.ArgumentParser(description="Create a browsable HTML preview for product-level Amazon annotation JSONL.")
    parser.add_argument("--annotations", "--input", dest="annotations", type=Path, required=True)
    parser.add_argument("--out", "--output", dest="out", type=Path, required=True)
    parser.add_argument("--max-products", type=int, default=0, help="0 means all products.")
    parser.add_argument("--selected-only", action="store_true", help="Only show final_status=selected pairs.")
    parser.add_argument("--show-rejected", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args()


def main():
    args = parse_args()
    records = []
    stats = collections.Counter()
    rejects = collections.Counter()
    for record in iter_jsonl(args.annotations):
        ann = record.get("annotation") or {}
        stats["products"] += 1
        stats["images"] += len(ann.get("images") or [])
        stats["pairs"] += len(ann.get("pairs") or [])
        stats["valid_pairs"] += len(record.get("valid_pairs") or ann.get("valid_pairs") or [])
        for pair in ann.get("pairs") or []:
            if pair.get("reject"):
                rejects.update(pair.get("reject_reasons") or [])
        if args.max_products <= 0 or len(records) < args.max_products:
            records.append(record)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(build_html(records, args, stats, rejects), encoding="utf-8")
    print(f"products={stats['products']} images={stats['images']} pairs={stats['pairs']} valid_pairs={stats['valid_pairs']}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
