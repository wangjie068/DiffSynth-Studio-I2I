import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


DEFAULT_SOURCE = "outputs/edit_pair_validation/amazon_lipcare_design/design_preview.png"
DEFAULT_SOURCE_BOX = "0.58,0.37,0.78,0.60"
DEFAULT_TARGET_BOX = "0.50,0.22,0.80,0.66"


def parse_box(value: str) -> tuple[float, float, float, float]:
    parts = [float(part.strip()) for part in value.split(",")]
    if len(parts) != 4:
        raise SystemExit(f"Expected box as left,top,right,bottom; got: {value}")
    return tuple(parts)


def parse_quad(value: str | None) -> np.ndarray | None:
    if value is None:
        return None
    parts = [float(part.strip()) for part in value.split(",")]
    if len(parts) != 8:
        raise SystemExit(f"Expected quad as x1,y1,x2,y2,x3,y3,x4,y4; got: {value}")
    return np.asarray(parts, dtype=np.float32).reshape(4, 2)


def box_to_pixels(box: tuple[float, float, float, float], size: tuple[int, int]) -> tuple[int, int, int, int]:
    width, height = size
    if all(0.0 <= value <= 1.0 for value in box):
        left, top, right, bottom = box
        box = (left * width, top * height, right * width, bottom * height)
    left, top, right, bottom = [int(round(value)) for value in box]
    left = max(0, min(left, width - 1))
    top = max(0, min(top, height - 1))
    right = max(left + 1, min(right, width))
    bottom = max(top + 1, min(bottom, height))
    return left, top, right, bottom


def quad_to_pixels(quad: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    width, height = size
    quad = quad.astype(np.float32).copy()
    if np.all((quad >= 0.0) & (quad <= 1.0)):
        quad[:, 0] *= width
        quad[:, 1] *= height
    return quad.astype(np.float32)


def crop_np(image: Image.Image, box: tuple[float, float, float, float]) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    pixel_box = box_to_pixels(box, image.size)
    return np.asarray(image.crop(pixel_box).convert("RGB")), pixel_box


def gaussian_mask(mask: np.ndarray, blur: int) -> np.ndarray:
    blur = max(1, int(blur))
    if blur % 2 == 0:
        blur += 1
    return cv2.GaussianBlur(mask.astype(np.float32), (blur, blur), 0).clip(0.0, 1.0)


def make_text_detail_mask(source_rgb: np.ndarray, dilate: int, blur: int) -> np.ndarray:
    gray = cv2.cvtColor(source_rgb, cv2.COLOR_RGB2GRAY)
    lab = cv2.cvtColor(source_rgb, cv2.COLOR_RGB2LAB)
    luma = lab[:, :, 0]

    adaptive = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31,
        7,
    )
    local_mean = cv2.GaussianBlur(luma, (31, 31), 0)
    dark = (luma.astype(np.int16) < local_mean.astype(np.int16) - 10).astype(np.uint8) * 255
    edges = cv2.Canny(gray, 60, 160)
    mask = cv2.bitwise_or(adaptive, dark)
    mask = cv2.bitwise_or(mask, edges)

    # Suppress broad yellow-plastic gradients; keep thin strokes and label rules.
    kernel_small = np.ones((2, 2), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_small)
    if dilate > 0:
        kernel = np.ones((dilate, dilate), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=1)
    return gaussian_mask(mask / 255.0, blur)


def match_local_color(reference_rgb: np.ndarray, target_rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    ref = reference_rgb.astype(np.float32)
    target = target_rgb.astype(np.float32)
    bg = (1.0 - mask[..., None]).clip(0.0, 1.0)
    denom = bg.sum(axis=(0, 1)).clip(min=1.0)
    ref_mean = (ref * bg).sum(axis=(0, 1)) / denom
    target_mean = (target * bg).sum(axis=(0, 1)) / denom
    ref_std = np.sqrt((((ref - ref_mean) ** 2) * bg).sum(axis=(0, 1)) / denom).clip(min=1.0)
    target_std = np.sqrt((((target - target_mean) ** 2) * bg).sum(axis=(0, 1)) / denom).clip(min=1.0)
    matched = (ref - ref_mean) * (target_std / ref_std) + target_mean
    return matched.clip(0, 255).astype(np.uint8)


def restore_roi(
    generated: Image.Image,
    source: Image.Image,
    source_box: tuple[float, float, float, float],
    target_box: tuple[float, float, float, float],
    alpha: float,
    detail_alpha: float,
    dilate: int,
    mask_blur: int,
    detail_blur: int,
    match_color: bool,
) -> tuple[Image.Image, Image.Image, Image.Image]:
    source_crop, _ = crop_np(source, source_box)
    target_crop, target_pixels = crop_np(generated, target_box)
    left, top, right, bottom = target_pixels
    target_h, target_w = target_crop.shape[:2]

    source_resized = cv2.resize(source_crop, (target_w, target_h), interpolation=cv2.INTER_CUBIC)
    mask = make_text_detail_mask(source_resized, dilate=dilate, blur=mask_blur)
    source_matched = match_local_color(source_resized, target_crop, mask) if match_color else source_resized

    blur = max(1, int(detail_blur))
    if blur % 2 == 0:
        blur += 1
    source_detail = source_matched.astype(np.float32) - cv2.GaussianBlur(source_matched.astype(np.float32), (blur, blur), 0)
    target_base = target_crop.astype(np.float32)

    direct = (1.0 - alpha * mask[..., None]) * target_base + (alpha * mask[..., None]) * source_matched.astype(np.float32)
    restored = direct + detail_alpha * mask[..., None] * source_detail
    restored = restored.clip(0, 255).astype(np.uint8)

    output = np.asarray(generated.convert("RGB")).copy()
    output[top:bottom, left:right] = restored
    mask_vis = np.zeros_like(output)
    mask_uint8 = (mask * 255).clip(0, 255).astype(np.uint8)
    mask_vis[top:bottom, left:right] = np.stack([mask_uint8, mask_uint8, mask_uint8], axis=-1)
    reference_vis = np.asarray(generated.convert("RGB")).copy()
    reference_vis[top:bottom, left:right] = source_resized
    return Image.fromarray(output), Image.fromarray(mask_vis), Image.fromarray(reference_vis)


def restore_quad(
    generated: Image.Image,
    source: Image.Image,
    source_quad: np.ndarray,
    target_quad: np.ndarray,
    alpha: float,
    detail_alpha: float,
    dilate: int,
    mask_blur: int,
    detail_blur: int,
    match_color: bool,
) -> tuple[Image.Image, Image.Image, Image.Image]:
    source_rgb = np.asarray(source.convert("RGB"))
    target_rgb = np.asarray(generated.convert("RGB"))
    height, width = target_rgb.shape[:2]
    source_points = quad_to_pixels(source_quad, source.size)
    target_points = quad_to_pixels(target_quad, generated.size)

    source_mask = make_text_detail_mask(source_rgb, dilate=dilate, blur=mask_blur)
    polygon_mask = np.zeros(source_mask.shape, dtype=np.float32)
    cv2.fillConvexPoly(polygon_mask, source_points.astype(np.int32), 1.0)
    source_mask = source_mask * polygon_mask

    homography = cv2.getPerspectiveTransform(source_points, target_points)
    warped_source = cv2.warpPerspective(
        source_rgb,
        homography,
        (width, height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
    warped_mask = cv2.warpPerspective(
        source_mask,
        homography,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    ).clip(0.0, 1.0)

    source_matched = match_local_color(warped_source, target_rgb, warped_mask) if match_color else warped_source
    blur = max(1, int(detail_blur))
    if blur % 2 == 0:
        blur += 1
    source_detail = source_matched.astype(np.float32) - cv2.GaussianBlur(source_matched.astype(np.float32), (blur, blur), 0)
    mask = warped_mask[..., None]
    restored = (1.0 - alpha * mask) * target_rgb.astype(np.float32) + (alpha * mask) * source_matched.astype(np.float32)
    restored = restored + detail_alpha * mask * source_detail
    restored = restored.clip(0, 255).astype(np.uint8)

    mask_uint8 = (warped_mask * 255).clip(0, 255).astype(np.uint8)
    mask_vis = np.stack([mask_uint8, mask_uint8, mask_uint8], axis=-1)
    reference_vis = target_rgb.copy()
    reference_alpha = gaussian_mask(warped_mask, 7)[..., None]
    reference_vis = ((1.0 - reference_alpha) * reference_vis + reference_alpha * warped_source).clip(0, 255).astype(np.uint8)
    return Image.fromarray(restored), Image.fromarray(mask_vis), Image.fromarray(reference_vis)


def iter_input_images(input_path: Path):
    if input_path.is_file():
        yield input_path
        return
    for path in sorted(input_path.glob("*/seed*.png")):
        yield path


def output_path_for(input_image: Path, input_root: Path, output_dir: Path) -> Path:
    if input_root.is_file():
        return output_dir / input_image.name
    return output_dir / input_image.relative_to(input_root)


def main():
    parser = argparse.ArgumentParser(description="Restore small text/logo high-frequency details inside a local ROI.")
    parser.add_argument("--input", required=True, help="Generated image path or a run directory containing model/seed*.png.")
    parser.add_argument("--source", default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--source-box", default=DEFAULT_SOURCE_BOX)
    parser.add_argument("--target-box", default=DEFAULT_TARGET_BOX)
    parser.add_argument("--source-quad", default=None, help="Optional full-image source quad: x1,y1,x2,y2,x3,y3,x4,y4.")
    parser.add_argument("--target-quad", default=None, help="Optional full-image target quad: x1,y1,x2,y2,x3,y3,x4,y4.")
    parser.add_argument("--alpha", type=float, default=0.70, help="Direct masked reference blend strength.")
    parser.add_argument("--detail-alpha", type=float, default=0.85, help="High-frequency detail boost strength.")
    parser.add_argument("--mask-dilate", type=int, default=2)
    parser.add_argument("--mask-blur", type=int, default=5)
    parser.add_argument("--detail-blur", type=int, default=7)
    parser.add_argument("--match-color", action="store_true", help="Match reference ROI color statistics before blending. Off by default to keep dark text strokes dark.")
    parser.add_argument("--write-debug", action="store_true")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    source = Image.open(args.source).convert("RGB")
    source_box = parse_box(args.source_box)
    target_box = parse_box(args.target_box)
    source_quad = parse_quad(args.source_quad)
    target_quad = parse_quad(args.target_quad)
    if (source_quad is None) != (target_quad is None):
        raise SystemExit("--source-quad and --target-quad must be provided together.")

    rows = []
    for image_path in iter_input_images(input_path):
        generated = Image.open(image_path).convert("RGB")
        if source_quad is not None and target_quad is not None:
            restored, mask_vis, reference_vis = restore_quad(
                generated=generated,
                source=source,
                source_quad=source_quad,
                target_quad=target_quad,
                alpha=args.alpha,
                detail_alpha=args.detail_alpha,
                dilate=args.mask_dilate,
                mask_blur=args.mask_blur,
                detail_blur=args.detail_blur,
                match_color=args.match_color,
            )
        else:
            restored, mask_vis, reference_vis = restore_roi(
                generated=generated,
                source=source,
                source_box=source_box,
                target_box=target_box,
                alpha=args.alpha,
                detail_alpha=args.detail_alpha,
                dilate=args.mask_dilate,
                mask_blur=args.mask_blur,
                detail_blur=args.detail_blur,
                match_color=args.match_color,
            )
        out_path = output_path_for(image_path, input_path, output_dir)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        restored.save(out_path)
        debug_paths = {}
        if args.write_debug:
            debug_dir = out_path.parent / "roi_restore_debug"
            debug_dir.mkdir(parents=True, exist_ok=True)
            mask_path = debug_dir / f"{out_path.stem}_mask.png"
            reference_path = debug_dir / f"{out_path.stem}_warped_reference.png"
            mask_vis.save(mask_path)
            reference_vis.save(reference_path)
            debug_paths = {"mask": str(mask_path), "warped_reference": str(reference_path)}
        sidecar = out_path.with_suffix(".roi_restore.json")
        metadata = {
            "input": str(image_path),
            "output": str(out_path),
            "source": args.source,
            "source_box": args.source_box,
            "target_box": args.target_box,
            "source_quad": args.source_quad,
            "target_quad": args.target_quad,
            "alpha": args.alpha,
            "detail_alpha": args.detail_alpha,
            "mask_dilate": args.mask_dilate,
            "mask_blur": args.mask_blur,
            "detail_blur": args.detail_blur,
            "match_color": args.match_color,
            "debug": debug_paths,
        }
        sidecar.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        rows.append(metadata)

    summary_path = output_dir / "roi_restore_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "input",
                "output",
                "source",
                "source_box",
                "target_box",
                "alpha",
                "detail_alpha",
                "mask_dilate",
                "mask_blur",
                "detail_blur",
                "match_color",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in writer.fieldnames})
    print(f"Wrote {len(rows)} restored image(s) to {output_dir}")


if __name__ == "__main__":
    main()
