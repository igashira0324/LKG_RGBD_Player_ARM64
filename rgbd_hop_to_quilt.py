#!/usr/bin/env python3
import argparse
import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

# --- Phase 3-B: Quality Presets ---
# PRESETS: Recommended settings for different use cases.
# natural: Balanced, fast (backward warp). Recommended for general video.
# studio: Highest quality, very slow (forward splat + inpaint). Recommended for stills or short clips.
PRESETS = {
    "fast": {
        "synthesis": "backward",
        "auto_depth_range": True,
        "depth_contrast": 1.1,
        "depth_gamma": 1.0,
        "depth_filter": "gaussian",
        "depth_blur": 3,
        "max_shift_ratio": 0.035,
        "foreground_dilate": 0,
        "hole_fill": "none",
        "splat_radius": 1,
    },
    "natural": {
        "synthesis": "backward",
        "auto_depth_range": True,
        "depth_contrast": 1.25,
        "depth_gamma": 0.95,
        "depth_filter": "bilateral",
        "depth_blur": 3,
        "max_shift_ratio": 0.045,
        "foreground_dilate": 1,
        "hole_fill": "dilate",
        "splat_radius": 1,
    },
    "strong": {
        "synthesis": "forward",
        "auto_depth_range": True,
        "depth_contrast": 1.45,
        "depth_gamma": 0.9,
        "depth_filter": "bilateral",
        "depth_blur": 3,
        "max_shift_ratio": 0.055,
        "foreground_dilate": 1,
        "hole_fill": "inpaint",
        "splat_radius": 1,
    },
    "studio": {
        "synthesis": "forward",
        "auto_depth_range": True,
        "depth_contrast": 1.35,
        "depth_gamma": 0.9,
        "depth_filter": "bilateral",
        "depth_blur": 3,
        "max_shift_ratio": 0.05,
        "foreground_dilate": 2,
        "hole_fill": "inpaint",
        "splat_radius": 1,
    }
}

def ffprobe_value(path, entries):
    cmd = ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", entries, "-of", "default=nokey=1:noprint_wrappers=1", str(path)]
    return subprocess.check_output(cmd, text=True).strip()

def get_fps(path):
    val = ffprobe_value(path, "stream=r_frame_rate")
    if "/" in val:
        a, b = val.split("/")
        return float(a) / float(b)
    return float(val)

def read_hop_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def depth_position_to_loc(pos):
    pos = str(pos).lower()
    mapping = {"top": 0, "bottom": 1, "left": 2, "right": 3}
    if pos in mapping: return mapping[pos]
    raise ValueError(f"Unknown depthPosition: {pos}")

def split_rgbd(frame_bgr, depth_loc):
    h, w = frame_bgr.shape[:2]
    if depth_loc == 2:  # left
        mid = w // 2
        dep, rgb = frame_bgr[:, :mid], frame_bgr[:, mid:]
    elif depth_loc == 3:  # right
        mid = w // 2
        rgb, dep = frame_bgr[:, :mid], frame_bgr[:, mid:]
    elif depth_loc == 0:  # top
        mid = h // 2
        dep, rgb = frame_bgr[:mid, :], frame_bgr[mid:, :]
    else: # bottom
        mid = h // 2
        rgb, dep = frame_bgr[:mid, :], frame_bgr[mid:, :]
    
    if dep.shape[:2] != rgb.shape[:2]:
        dep = cv2.resize(dep, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_LINEAR)
    return rgb, dep

# --- Depth Preprocessing (Phase 3-B) ---

def normalize_depth_range(depth_gray, auto_range=False, low_p=2.0, high_p=98.0, near=None, far=None, focus=0.5, contrast=1.0, gamma=1.0):
    d = depth_gray.astype(np.float32)
    # Ensure 0..1 scale (fix uint8 0..255 case)
    if d.max() > 1.5:
        d /= 255.0

    if auto_range:
        lo, hi = np.percentile(d, low_p), np.percentile(d, high_p)
    else:
        lo = 0.0 if near is None else float(near)
        hi = 1.0 if far is None else float(far)
    
    if hi <= lo + 1e-6: hi = lo + 1e-6
    d = np.clip((d - lo) / (hi - lo), 0.0, 1.0)
    d = np.clip(focus + (d - focus) * contrast, 0.0, 1.0)
    if gamma != 1.0: d = np.power(d, max(gamma, 1e-6))
    return np.clip(d, 0.0, 1.0)

def filter_depth(depth, mode="bilateral", blur=3, bilateral_d=5, sigma_color=0.05, sigma_space=5.0):
    if mode == "none": return depth
    if mode == "gaussian":
        k = int(blur); k = k + 1 if k % 2 == 0 else k
        return cv2.GaussianBlur(depth, (k, k), 0)
    if mode == "bilateral":
        return cv2.bilateralFilter(depth, int(bilateral_d), float(sigma_color), float(sigma_space))
    return depth

def foreground_dilate_depth(depth, amount=1):
    if amount <= 0: return depth
    kernel = np.ones((amount * 2 + 1, amount * 2 + 1), np.uint8)
    return cv2.dilate(depth, kernel, iterations=1)

# --- View Synthesis (Phase 3-B) ---

def synthesize_view_backward(rgb_bgr, depth, view_offset, zero_depth, max_shift_px, reverse_parallax=False):
    h, w = rgb_bgr.shape[:2]
    xs, ys = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    sign = -1.0 if reverse_parallax else 1.0
    disp = sign * view_offset * max_shift_px * (depth - zero_depth)
    map_x, map_y = xs + disp, ys
    return cv2.remap(rgb_bgr, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT101)

def synthesize_view_forward(rgb_bgr, depth, view_offset, zero_depth, max_shift_px, reverse_parallax=False, splat_radius=1, hole_fill="none"):
    h, w = rgb_bgr.shape[:2]
    sign = -1.0 if reverse_parallax else 1.0
    xs, ys = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    disp = sign * view_offset * max_shift_px * (depth - zero_depth)
    x2 = xs + disp
    
    out = np.zeros_like(rgb_bgr)
    # Mask to keep track of painted pixels
    mask = np.zeros((h, w), dtype=np.uint8)
    
    xi = np.rint(x2).astype(np.int32)
    yi = ys.astype(np.int32)
    valid = (xi >= 0) & (xi < w)
    
    src_y, src_x = np.where(valid)
    dst_x = xi[src_y, src_x]
    dst_y = yi[src_y, src_x]
    z = depth[src_y, src_x]
    
    # Simple Z-buffer: Near pixel wins (higher depth value in 0..1 range)
    sorted_idx = np.argsort(z) # Far pixels first, so near pixels overwrite
    for i in sorted_idx:
        sy, sx, dx, dy = src_y[i], src_x[i], dst_x[i], dst_y[i]
        out[dy, dx] = rgb_bgr[sy, sx]
        mask[dy, dx] = 255
        
    if splat_radius > 0:
        kernel = np.ones((splat_radius * 2 + 1, splat_radius * 2 + 1), np.uint8)
        dilated = cv2.dilate(out, kernel, iterations=1)
        dilated_mask = cv2.dilate(mask, kernel, iterations=1)
        holes = (mask == 0) & (dilated_mask > 0)
        out[holes] = dilated[holes]
        mask[holes] = 255
        
    if hole_fill == "inpaint":
        hole_mask = (mask == 0).astype(np.uint8) * 255
        if np.any(hole_mask): out = cv2.inpaint(out, hole_mask, 3, cv2.INPAINT_TELEA)
    elif hole_fill == "dilate":
        hole_mask = (mask == 0); kernel = np.ones((3, 3), np.uint8)
        filled = cv2.dilate(out, kernel, iterations=2)
        out[hole_mask] = filled[hole_mask]
        
    return out

def fit_to_tile(img, tile_w, tile_h, mode="contain"):
    h, w = img.shape[:2]
    if mode == "stretch": return cv2.resize(img, (tile_w, tile_h), interpolation=cv2.INTER_AREA)
    src_aspect, dst_aspect = w / h, tile_w / tile_h
    if mode == "contain":
        if src_aspect > dst_aspect:
            new_w, new_h = tile_w, int(round(tile_w / src_aspect))
        else:
            new_h, new_w = tile_h, int(round(tile_h * src_aspect))
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        canvas = np.zeros((tile_h, tile_w, 3), dtype=np.uint8)
        x, y = (tile_w - new_w) // 2, (tile_h - new_h) // 2
        canvas[y:y+new_h, x:x+new_w] = resized
        return canvas
    if mode == "cover":
        if src_aspect > dst_aspect:
            new_h, new_w = tile_h, int(round(tile_h * src_aspect))
        else:
            new_w, new_h = tile_w, int(round(tile_w / src_aspect))
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        x, y = max((new_w - tile_w) // 2, 0), max((new_h - tile_h) // 2, 0)
        return resized[y:y+tile_h, x:x+tile_w]
    raise ValueError(f"Unknown fit mode: {mode}")

def assemble_quilt(views, cols, rows, tile_w, tile_h):
    quilt = np.zeros((rows * tile_h, cols * tile_w, 3), dtype=np.uint8)
    for i, tile in enumerate(views):
        col, row_from_bottom = i % cols, i // cols
        row = (rows - 1) - row_from_bottom
        x0, y0 = col * tile_w, row * tile_h
        quilt[y0:y0+tile_h, x0:x0+tile_w] = tile
    return quilt

def build_ffmpeg_writer(output, input_video, width, height, fps, crf=18, preset="medium", audio=True, audio_start=0.0, pix_fmt="yuv420p"):
    cmd = ["ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{width}x{height}", "-r", str(fps), "-i", "pipe:0"]
    if audio:
        if audio_start and audio_start > 0:
            cmd += ["-ss", str(audio_start)]
        cmd += ["-i", str(input_video), "-map", "0:v:0", "-map", "1:a?"]
    else:
        cmd += ["-map", "0:v:0"]
    cmd += ["-c:v", "libx264", "-preset", preset, "-crf", str(crf), "-pix_fmt", pix_fmt]
    if audio: cmd += ["-c:a", "copy", "-shortest"]
    cmd += ["-movflags", "+faststart", str(output)]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE)

def main():
    ap = argparse.ArgumentParser(description="Phase 3-B Final: High-Quality RGBD to Quilt Converter")
    ap.add_argument("--json", required=True); ap.add_argument("--video", required=True); ap.add_argument("--output", required=True)
    ap.add_argument("--cols", type=int, default=11); ap.add_argument("--rows", type=int, default=6)
    ap.add_argument("--quilt-size", type=int, default=4092)
    ap.add_argument("--quality-preset", choices=list(PRESETS.keys()), default="natural")
    
    # Advanced / Manual Overrides
    ap.add_argument("--synthesis", choices=["backward", "forward"], default=None)
    ap.add_argument("--auto-depth-range", dest="auto_depth_range", action="store_true", default=None)
    ap.add_argument("--no-auto-depth-range", dest="auto_depth_range", action="store_false")
    ap.add_argument("--depth-low-percentile", type=float, default=2.0)
    ap.add_argument("--depth-high-percentile", type=float, default=98.0)
    ap.add_argument("--depth-near", type=float, default=None); ap.add_argument("--depth-far", type=float, default=None)
    ap.add_argument("--depth-contrast", type=float, default=None); ap.add_argument("--depth-gamma", type=float, default=None)
    ap.add_argument("--depth-filter", choices=["none", "gaussian", "bilateral"], default=None)
    ap.add_argument("--depth-blur", type=int, default=None)
    ap.add_argument("--foreground-dilate", type=int, default=None)
    ap.add_argument("--hole-fill", choices=["none", "inpaint", "dilate"], default=None)
    ap.add_argument("--splat-radius", type=int, default=None)
    ap.add_argument("--max-shift-ratio", type=float, default=None); ap.add_argument("--zero-depth", type=float, default=None)
    
    # Basic Config
    ap.add_argument("--depthiness", type=float, default=None); ap.add_argument("--focus", type=float, default=None)
    ap.add_argument("--depth-position", default=None); ap.add_argument("--depth-inversion", type=int, default=None)
    ap.add_argument("--fit", choices=["contain", "cover", "stretch"], default="contain")
    ap.add_argument("--reverse-views", action="store_true"); ap.add_argument("--reverse-parallax", action="store_true")
    ap.add_argument("--seconds", type=float, default=0.0); ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--crf", type=int, default=18); ap.add_argument("--preset", default="medium"); ap.add_argument("--no-audio", action="store_true")
    ap.add_argument("--pix-fmt", default="yuv420p", choices=["yuv420p", "yuv444p"])
    ap.add_argument("--debug-dir", default=None); ap.add_argument("--debug-frame", type=int, default=0)
    args = ap.parse_args()

    # Apply Presets
    P = PRESETS[args.quality_preset]
    def get_arg(key, preset_key): return getattr(args, key) if getattr(args, key) is not None else P[preset_key]
    synthesis = get_arg("synthesis", "synthesis")
    auto_depth_range = get_arg("auto_depth_range", "auto_depth_range")
    depth_contrast = get_arg("depth_contrast", "depth_contrast")
    depth_gamma = get_arg("depth_gamma", "depth_gamma")
    depth_filter_mode = get_arg("depth_filter", "depth_filter")
    depth_blur = get_arg("depth_blur", "depth_blur")
    foreground_dilate = get_arg("foreground_dilate", "foreground_dilate")
    hole_fill = get_arg("hole_fill", "hole_fill")
    splat_radius = get_arg("splat_radius", "splat_radius")
    max_shift_ratio = get_arg("max_shift_ratio", "max_shift_ratio")

    # Load Source
    json_path, video_path, output_path = Path(args.json).resolve(), Path(args.video).resolve(), Path(args.output).resolve()
    cfg = read_hop_json(json_path)
    depthiness = float(args.depthiness if args.depthiness is not None else cfg.get("depthiness", 1.0))
    focus = float(args.focus if args.focus is not None else cfg.get("focus", 0.0))
    depth_pos = args.depth_position if args.depth_position is not None else cfg.get("depthPosition", "right")
    depth_loc = depth_position_to_loc(depth_pos)
    depth_inversion = bool(args.depth_inversion) if args.depth_inversion is not None else bool(cfg.get("depthInversion", False))
    
    fps = get_fps(video_path); cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if args.start > 0: cap.set(cv2.CAP_PROP_POS_MSEC, args.start * 1000.0)
    max_frames = int(round(args.seconds * fps)) if args.seconds > 0 else None
    
    ok, frame = cap.read()
    if not ok or frame is None:
        raise RuntimeError(f"Failed to read first frame from video: {video_path}")
    
    rgb0, dep0 = split_rgbd(frame, depth_loc)
    rgb_h, rgb_w = rgb0.shape[:2]
    max_shift_px = rgb_w * max_shift_ratio * depthiness
    zero_depth = float(np.clip(args.zero_depth if args.zero_depth is not None else 0.5 + focus * 0.5, 0.0, 1.0))
    
    tile_w, tile_h = int(args.quilt_size // args.cols), int(args.quilt_size // args.rows)
    quilt_w, quilt_h = tile_w * args.cols, tile_h * args.rows
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = build_ffmpeg_writer(output_path, video_path, quilt_w, quilt_h, fps, crf=args.crf, preset=args.preset, audio=not args.no_audio, audio_start=args.start, pix_fmt=args.pix_fmt)
    
    offsets = np.linspace(-1.0, 1.0, args.cols * args.rows, dtype=np.float32)
    if args.reverse_views: offsets = offsets[::-1]
    
    processed = 0; pbar = tqdm(total=(max_frames if max_frames else total_frames), desc=f"Phase 3-B [{args.quality_preset}]")
    try:
        while True:
            current = frame if processed == 0 else cap.read()[1]
            if current is None or (max_frames and processed >= max_frames): break
            rgb, dep = split_rgbd(current, depth_loc)
            
            # 1. Depth Preprocessing
            dep_gray = cv2.cvtColor(dep, cv2.COLOR_BGR2GRAY)
            if depth_inversion: dep_gray = 255 - dep_gray
            depth = normalize_depth_range(dep_gray, auto_range=auto_depth_range, low_p=args.depth_low_percentile, high_p=args.depth_high_percentile, near=args.depth_near, far=args.depth_far, focus=focus, contrast=depth_contrast, gamma=depth_gamma)
            depth = filter_depth(depth, mode=depth_filter_mode, blur=depth_blur)
            depth_for_warp = foreground_dilate_depth(depth, amount=foreground_dilate)
            
            # 2. View Synthesis
            views = []
            for off in offsets:
                if synthesis == "forward":
                    v = synthesize_view_forward(rgb, depth_for_warp, float(off), zero_depth, max_shift_px, args.reverse_parallax, splat_radius=splat_radius, hole_fill=hole_fill)
                else:
                    v = synthesize_view_backward(rgb, depth_for_warp, float(off), zero_depth, max_shift_px, args.reverse_parallax)
                views.append(fit_to_tile(v, tile_w, tile_h, args.fit))
            
            # 3. Assemble & Write
            quilt = assemble_quilt(views, args.cols, args.rows, tile_w, tile_h)
            try:
                writer.stdin.write(quilt.tobytes())
            except BrokenPipeError:
                raise RuntimeError("ffmpeg writer pipe closed unexpectedly")
            
            # 4. Debug Save
            if args.debug_dir and processed == args.debug_frame:
                d_dir = Path(args.debug_dir); d_dir.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(d_dir/"rgb.png"), rgb)
                cv2.imwrite(str(d_dir/"depth_raw.png"), dep_gray)
                cv2.imwrite(str(d_dir/"depth_final.png"), (depth * 255).astype(np.uint8))
                cv2.imwrite(str(d_dir/"view_00.png"), views[0])
                cv2.imwrite(str(d_dir/"view_mid.png"), views[len(views)//2])
                cv2.imwrite(str(d_dir/"view_last.png"), views[-1])
                cv2.imwrite(str(d_dir/"quilt_frame.jpg"), quilt, [cv2.IMWRITE_JPEG_QUALITY, 90])
            
            processed += 1; pbar.update(1)
    finally:
        pbar.close(); cap.release()
        if writer.stdin:
            writer.stdin.close()
        ret = writer.wait()
        if ret != 0:
            print(f"\n[ERROR] ffmpeg writer failed with return code {ret}")
    
    suggested = f"{output_path.stem}_qs{args.cols}x{args.rows}a0.5625{output_path.suffix}"
    print(f"\nPhase 3-B DONE")
    print(f"Output: {output_path}")
    print(f"Suggested filename for Player: {suggested}\n")

if __name__ == "__main__":
    main()
