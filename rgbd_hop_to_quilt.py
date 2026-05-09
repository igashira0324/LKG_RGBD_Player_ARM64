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


def ffprobe_value(path, entries):
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", entries,
        "-of", "default=nokey=1:noprint_wrappers=1",
        str(path)
    ]
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
    """
    Internal (Aligned with Player):
      0 = top
      1 = bottom
      2 = left
      3 = right
    """
    pos = str(pos).lower()
    if pos == "top":
        return 0
    if pos == "bottom":
        return 1
    if pos == "left":
        return 2
    if pos == "right":
        return 3
    raise ValueError(f"Unknown depthPosition: {pos}")


def split_rgbd(frame_bgr, depth_loc):
    h, w = frame_bgr.shape[:2]

    if depth_loc == 2:  # left
        mid = w // 2
        dep = frame_bgr[:, :mid]
        rgb = frame_bgr[:, mid:]
    elif depth_loc == 3:  # right
        mid = w // 2
        rgb = frame_bgr[:, :mid]
        dep = frame_bgr[:, mid:]
    elif depth_loc == 0:  # top
        mid = h // 2
        dep = frame_bgr[:mid, :]
        rgb = frame_bgr[mid:, :]
    elif depth_loc == 1:  # bottom
        mid = h // 2
        rgb = frame_bgr[:mid, :]
        dep = frame_bgr[mid:, :]
    else:
        raise ValueError(f"depth_loc must be 0,1,2,3. got {depth_loc}")

    if dep.shape[:2] != rgb.shape[:2]:
        dep = cv2.resize(dep, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_LINEAR)

    return rgb, dep

# ... (rest of the file remains same) ...
def depth_to_float(dep_bgr, invert=False, gamma=1.0, blur=3):
    gray = cv2.cvtColor(dep_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    if invert: gray = 1.0 - gray
    gamma = max(float(gamma), 1e-6)
    if gamma != 1.0: gray = np.power(np.clip(gray, 0, 1), gamma)
    if blur and blur > 0:
        k = int(blur)
        if k % 2 == 0: k += 1
        gray = cv2.GaussianBlur(gray, (k, k), 0)
    return np.clip(gray, 0.0, 1.0)

def fit_to_tile(img, tile_w, tile_h, mode="contain"):
    h, w = img.shape[:2]
    if mode == "stretch": return cv2.resize(img, (tile_w, tile_h), interpolation=cv2.INTER_AREA)
    src_aspect, dst_aspect = w / h, tile_w / tile_h
    if mode == "contain":
        if src_aspect > dst_aspect:
            new_w = tile_w
            new_h = int(round(tile_w / src_aspect))
        else:
            new_h = tile_h
            new_w = int(round(tile_h * src_aspect))
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

def synthesize_view(rgb_bgr, depth, view_offset, zero_depth, max_shift_px, reverse_parallax=False):
    h, w = rgb_bgr.shape[:2]
    xs, ys = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    sign = -1.0 if reverse_parallax else 1.0
    disp = sign * view_offset * max_shift_px * (depth - zero_depth)
    map_x, map_y = xs + disp.astype(np.float32), ys
    view = cv2.remap(rgb_bgr, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT101)
    return view

def assemble_quilt(views, cols, rows, tile_w, tile_h):
    quilt = np.zeros((rows * tile_h, cols * tile_w, 3), dtype=np.uint8)
    for i, tile in enumerate(views):
        col, row_from_bottom = i % cols, i // cols
        row = rows - 1 - row_from_bottom
        x0, y0 = col * tile_w, row * tile_h
        quilt[y0:y0+tile_h, x0:x0+tile_w] = tile
    return quilt

def build_ffmpeg_writer(output, input_video, width, height, fps, crf=18, preset="medium", audio=True):
    cmd = ["ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{width}x{height}", "-r", str(fps), "-i", "pipe:0"]
    if audio: cmd += ["-i", str(input_video), "-map", "0:v:0", "-map", "1:a?"]
    else: cmd += ["-map", "0:v:0"]
    cmd += ["-c:v", "libx264", "-preset", preset, "-crf", str(crf), "-pix_fmt", "yuv420p"]
    if audio: cmd += ["-c:a", "copy", "-shortest"]
    cmd += ["-movflags", "+faststart", str(output)]
    print("[ffmpeg]", " ".join(cmd))
    return subprocess.Popen(cmd, stdin=subprocess.PIPE)

def main():
    ap = argparse.ArgumentParser(description="Convert Looking Glass HOP RGBD JSON + RGBD video to Quilt video on ARM64/Linux.")
    ap.add_argument("--json", required=True); ap.add_argument("--video", required=True); ap.add_argument("--output", required=True)
    ap.add_argument("--cols", type=int, default=11); ap.add_argument("--rows", type=int, default=6)
    ap.add_argument("--quilt-size", type=int, default=4092)
    ap.add_argument("--depthiness", type=float, default=None); ap.add_argument("--focus", type=float, default=None)
    ap.add_argument("--depth-position", default=None); ap.add_argument("--depth-inversion", type=int, default=None)
    ap.add_argument("--chroma-depth", type=int, default=None)
    ap.add_argument("--max-shift-ratio", type=float, default=0.035); ap.add_argument("--zero-depth", type=float, default=None)
    ap.add_argument("--depth-gamma", type=float, default=1.0); ap.add_argument("--depth-blur", type=int, default=3)
    ap.add_argument("--fit", choices=["contain", "cover", "stretch"], default="contain")
    ap.add_argument("--reverse-views", action="store_true"); ap.add_argument("--reverse-parallax", action="store_true")
    ap.add_argument("--seconds", type=float, default=0.0); ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--crf", type=int, default=18); ap.add_argument("--preset", default="medium"); ap.add_argument("--no-audio", action="store_true")
    args = ap.parse_args()

    json_path, video_path, output_path = Path(args.json).resolve(), Path(args.video).resolve(), Path(args.output).resolve()
    cfg = read_hop_json(json_path)
    depthiness = float(args.depthiness if args.depthiness is not None else cfg.get("depthiness", 1.0))
    focus = float(args.focus if args.focus is not None else cfg.get("focus", 0.0))
    depth_pos = args.depth_position if args.depth_position is not None else cfg.get("depthPosition", "right")
    depth_loc = depth_position_to_loc(depth_pos)
    depth_inversion = bool(args.depth_inversion) if args.depth_inversion is not None else bool(cfg.get("depthInversion", False))
    chroma_depth = bool(args.chroma_depth) if args.chroma_depth is not None else bool(cfg.get("chromaDepth", False))
    cols, rows = int(args.cols), int(args.rows)
    view_count = cols * rows
    quilt_size = int(args.quilt_size)
    tile_w, tile_h = quilt_size // cols, quilt_size // rows
    quilt_w, quilt_h = tile_w * cols, tile_h * rows
    fps = get_fps(video_path)
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if args.start > 0: cap.set(cv2.CAP_PROP_POS_MSEC, args.start * 1000.0)
    max_frames = int(round(args.seconds * fps)) if args.seconds > 0 else None
    ok, frame = cap.read()
    rgb0, dep0 = split_rgbd(frame, depth_loc)
    rgb_h, rgb_w = rgb0.shape[:2]
    max_shift_px = rgb_w * args.max_shift_ratio * depthiness
    zero_depth = float(np.clip(args.zero_depth if args.zero_depth is not None else 0.5 + focus * 0.5, 0.0, 1.0))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = build_ffmpeg_writer(output_path, video_path, quilt_w, quilt_h, fps, crf=args.crf, preset=args.preset, audio=not args.no_audio)
    processed, pbar_total = 0, (max_frames if max_frames else total_frames)
    pbar = tqdm(total=pbar_total if pbar_total > 0 else None, desc="RGBD -> Quilt")
    offsets = np.linspace(-1.0, 1.0, view_count, dtype=np.float32)
    if args.reverse_views: offsets = offsets[::-1]
    try:
        while True:
            current = frame if processed == 0 else cap.read()[1]
            if current is None or (max_frames and processed >= max_frames): break
            rgb, dep = split_rgbd(current, depth_loc)
            depth = depth_to_float(dep, invert=depth_inversion, gamma=args.depth_gamma, blur=args.depth_blur)
            views = [fit_to_tile(synthesize_view(rgb, depth, float(off), zero_depth, max_shift_px, args.reverse_parallax), tile_w, tile_h, args.fit) for off in offsets]
            writer.stdin.write(assemble_quilt(views, cols, rows, tile_w, tile_h).tobytes())
            processed += 1; pbar.update(1)
    finally:
        pbar.close(); cap.release(); writer.stdin.close(); writer.wait()
    print(f"\nDONE\nOutput: {output_path}\nRecommended filename pattern: *_qs{cols}x{rows}a0.5625.mp4\n")

if __name__ == "__main__":
    main()
