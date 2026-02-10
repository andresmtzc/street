#!/usr/bin/env python3
"""
Batch inpainting for street view images using LaMa ONNX model.

Usage:
  1. Draw a mask in the webapp, click "Export Mask PNG"
  2. Download lama_fp32.onnx from https://huggingface.co/Carve/LaMa-ONNX/resolve/main/lama_fp32.onnx
  3. Run:  python3 inpaint_batch.py --images barcelona/ --mask mask.png --model lama_fp32.onnx --output output/

  Options:
    --images   Folder with input images (or a glob pattern)
    --mask     Path to mask PNG (white = remove, black = keep)
    --model    Path to ONNX model file
    --output   Output folder (created if needed)
    --size     Model input resolution (default: auto-detect or 512)
    --overlap  Tile overlap fraction (default: 0.25)
    --padding  Context padding around mask as fraction (default: 0.5)
"""

import argparse
import os
import sys
import glob
import time
import numpy as np
from PIL import Image

import onnxruntime as ort


def load_model(model_path):
    print(f"Loading model: {model_path}")
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = os.cpu_count()
    session = ort.InferenceSession(model_path, opts, providers=['CPUExecutionProvider'])
    input_names = [inp.name for inp in session.get_inputs()]
    output_names = [out.name for out in session.get_outputs()]
    # Detect model input size from shape metadata
    shape = session.get_inputs()[0].shape  # e.g. [1, 3, 512, 512]
    model_size = shape[2] if isinstance(shape[2], int) else 512
    print(f"  Inputs: {input_names}, Outputs: {output_names}, Size: {model_size}x{model_size}")
    return session, input_names, output_names, model_size


def run_tile(session, input_names, output_names, img_tile, mask_tile, ts):
    """Run model on a single ts x ts tile."""
    # img_tile: numpy (ts, ts, 3) uint8
    # mask_tile: numpy (ts, ts) uint8 (255=inpaint, 0=keep)
    img_f = img_tile.astype(np.float32) / 255.0  # [0, 1]
    img_f = img_f.transpose(2, 0, 1)[np.newaxis]  # (1, 3, H, W)

    mask_f = (mask_tile.astype(np.float32) / 255.0)
    mask_f = (mask_f > 0.5).astype(np.float32)
    mask_f = mask_f[np.newaxis, np.newaxis]  # (1, 1, H, W)

    feeds = {input_names[0]: img_f, input_names[1]: mask_f}
    result = session.run(output_names, feeds)[0]  # (1, 3, H, W) in [0, 255]

    out = result[0].transpose(1, 2, 0)  # (H, W, 3)
    out = np.clip(out, 0, 255).astype(np.uint8)
    return out


def build_tile_weights(size, overlap):
    """Linear taper for blending overlapping tiles."""
    w = np.ones((size, size), dtype=np.float32)
    for i in range(overlap):
        t = (i + 1) / overlap
        w[i, :] *= t
        w[-(i+1), :] *= t
        w[:, i] *= t
        w[:, -(i+1)] *= t
    return w


def get_mask_bbox(mask):
    """Get bounding box of non-zero region in mask."""
    rows = np.any(mask > 128, axis=1)
    cols = np.any(mask > 128, axis=0)
    if not rows.any():
        return None
    y1, y2 = np.where(rows)[0][[0, -1]]
    x1, x2 = np.where(cols)[0][[0, -1]]
    return int(x1), int(y1), int(x2 + 1), int(y2 + 1)


def inpaint_image(session, input_names, output_names, img, mask, model_size, overlap_frac=0.25, padding_frac=0.5):
    """Inpaint a single image using tiled inference."""
    h, w = img.shape[:2]
    ts = model_size
    overlap = int(ts * overlap_frac)
    stride = ts - overlap

    # Find mask bbox and add padding
    bbox = get_mask_bbox(mask)
    if bbox is None:
        return img  # no mask, return original

    bx1, by1, bx2, by2 = bbox
    bw, bh = bx2 - bx1, by2 - by1
    px = max(64, int(bw * padding_frac))
    py = max(64, int(bh * padding_frac))
    cx1 = max(0, bx1 - px)
    cy1 = max(0, by1 - py)
    cx2 = min(w, bx2 + px)
    cy2 = min(h, by2 + py)
    cw, ch = cx2 - cx1, cy2 - cy1

    crop_img = img[cy1:cy2, cx1:cx2].copy()
    crop_mask = mask[cy1:cy2, cx1:cx2].copy()

    # Single tile case
    if cw <= ts and ch <= ts:
        padded_img = np.zeros((ts, ts, 3), dtype=np.uint8)
        padded_mask = np.zeros((ts, ts), dtype=np.uint8)
        padded_img[:ch, :cw] = crop_img
        padded_mask[:ch, :cw] = crop_mask
        result = run_tile(session, input_names, output_names, padded_img, padded_mask, ts)
        crop_result = result[:ch, :cw]
    else:
        # Multi-tile with overlap blending
        tiles_x = max(1, int(np.ceil((cw - overlap) / stride)))
        tiles_y = max(1, int(np.ceil((ch - overlap) / stride)))
        weights = build_tile_weights(ts, overlap)

        accum = np.zeros((ch, cw, 3), dtype=np.float64)
        accum_w = np.zeros((ch, cw), dtype=np.float64)

        total = tiles_x * tiles_y
        done = 0
        for ty in range(tiles_y):
            for tx in range(tiles_x):
                x0 = min(tx * stride, max(0, cw - ts))
                y0 = min(ty * stride, max(0, ch - ts))
                tw = min(ts, cw - x0)
                th = min(ts, ch - y0)

                # Extract and pad tile
                tile_img = np.zeros((ts, ts, 3), dtype=np.uint8)
                tile_mask = np.zeros((ts, ts), dtype=np.uint8)
                tile_img[:th, :tw] = crop_img[y0:y0+th, x0:x0+tw]
                tile_mask[:th, :tw] = crop_mask[y0:y0+th, x0:x0+tw]

                # Skip tiles without mask
                if tile_mask.max() < 128:
                    tile_result = tile_img
                else:
                    tile_result = run_tile(session, input_names, output_names, tile_img, tile_mask, ts)

                # Accumulate with weights
                for py in range(th):
                    for px in range(tw):
                        wt = weights[py, px]
                        accum[y0+py, x0+px] += tile_result[py, px].astype(np.float64) * wt
                        accum_w[y0+py, x0+px] += wt

                done += 1
                print(f"    Tile {done}/{total}", end='\r')

        print()
        denom = np.maximum(accum_w, 1e-8)
        crop_result = (accum / denom[:, :, np.newaxis]).clip(0, 255).astype(np.uint8)

    # Composite: use inpainted result only where mask is white, feathered blend
    result = img.copy()
    # Build feathered alpha from crop_mask
    from PIL import ImageFilter
    mask_pil = Image.fromarray(crop_mask)
    mask_blurred = mask_pil.filter(ImageFilter.GaussianBlur(radius=3))
    alpha = np.array(mask_blurred).astype(np.float32) / 255.0

    for c in range(3):
        orig = result[cy1:cy2, cx1:cx2, c].astype(np.float32)
        inpainted = crop_result[:, :, c].astype(np.float32)
        result[cy1:cy2, cx1:cx2, c] = (orig * (1 - alpha) + inpainted * alpha).astype(np.uint8)

    return result


def main():
    parser = argparse.ArgumentParser(description='Batch inpaint street view images')
    parser.add_argument('--images', required=True, help='Input image folder')
    parser.add_argument('--mask', required=True, help='Mask PNG (white=remove, black=keep)')
    parser.add_argument('--model', required=True, help='ONNX model path')
    parser.add_argument('--output', required=True, help='Output folder')
    parser.add_argument('--size', type=int, default=0, help='Model input size (0=auto-detect)')
    parser.add_argument('--overlap', type=float, default=0.25, help='Tile overlap fraction')
    parser.add_argument('--padding', type=float, default=0.5, help='Context padding fraction')
    args = parser.parse_args()

    # Load model
    session, input_names, output_names, detected_size = load_model(args.model)
    model_size = args.size if args.size > 0 else detected_size

    # Load mask
    mask_img = Image.open(args.mask).convert('L')  # grayscale
    mask_np = np.array(mask_img)
    print(f"Mask: {mask_np.shape[1]}x{mask_np.shape[0]}, {(mask_np > 128).sum()} pixels to inpaint")

    # Find images
    img_dir = args.images.rstrip('/')
    extensions = ('*.jpg', '*.jpeg', '*.png', '*.webp', '*.JPG', '*.JPEG')
    files = []
    for ext in extensions:
        files.extend(glob.glob(os.path.join(img_dir, ext)))
    files.sort()
    print(f"Found {len(files)} images in {img_dir}")

    if not files:
        print("No images found!")
        sys.exit(1)

    # Create output dir
    os.makedirs(args.output, exist_ok=True)

    # Process
    total = len(files)
    for i, fpath in enumerate(files):
        name = os.path.basename(fpath)
        out_path = os.path.join(args.output, name)

        if os.path.exists(out_path):
            print(f"[{i+1}/{total}] {name} — already exists, skipping")
            continue

        t0 = time.time()
        print(f"[{i+1}/{total}] {name}...", end=' ')

        img = np.array(Image.open(fpath).convert('RGB'))

        # Resize mask to match image if needed
        if mask_np.shape[:2] != img.shape[:2]:
            mask_resized = np.array(mask_img.resize((img.shape[1], img.shape[0]), Image.NEAREST))
        else:
            mask_resized = mask_np

        result = inpaint_image(
            session, input_names, output_names,
            img, mask_resized, model_size,
            overlap_frac=args.overlap, padding_frac=args.padding
        )

        Image.fromarray(result).save(out_path, quality=92)
        dt = time.time() - t0
        print(f"done in {dt:.1f}s")

    print(f"\nAll done! Results in {args.output}/")


if __name__ == '__main__':
    main()
