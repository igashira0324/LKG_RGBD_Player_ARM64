#!/usr/bin/env python3
import cv2
import numpy as np
import argparse

def generate_test_quilt(cols=11, rows=6, res=4092, output=None):
    # Calculate tile size
    tile_w = res // cols
    tile_h = res // rows
    views = cols * rows
    
    # Create blank quilt
    out = np.zeros((tile_h * rows, tile_w * cols, 3), dtype=np.uint8)
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    for i in range(views):
        col = i % cols
        row = i // cols
        
        # LKG Quilt standard: view 0 is bottom-left.
        img_row = rows - 1 - row
        
        tile = np.zeros((tile_h, tile_w, 3), dtype=np.uint8)
        
        # Background color gradient (Red to Blue)
        t = i / max(1, views - 1)
        tile[:, :, 0] = int(255 * (1.0 - t)) # Blue
        tile[:, :, 2] = int(255 * t)         # Red
        
        # Parallax Square: Moves horizontally across views
        sq_size = tile_w // 8
        x_center = int(tile_w * (0.2 + 0.6 * t))
        y_center = tile_h // 2
        cv2.rectangle(tile, (x_center - sq_size, y_center - sq_size), 
                      (x_center + sq_size, y_center + sq_size), (255, 255, 255), -1)
        
        # View Number
        text = f"{i:02d}"
        cv2.putText(tile, text, (tile_w//4, tile_h//2 + tile_h//8), font, 3.0, (0, 255, 0), 8, cv2.LINE_AA)
        
        # Border
        cv2.rectangle(tile, (0, 0), (tile_w-1, tile_h-1), (255, 255, 255), 2)
        
        y0 = img_row * tile_h
        x0 = col * tile_w
        out[y0:y0+tile_h, x0:x0+tile_w] = tile
        
    aspect = 0.5625 # LKG Go Portrait
    filename = output if output else f"test_quilt_qs{cols}x{rows}a{aspect}.png"
    cv2.imwrite(filename, out)
    print(f"Generated {filename} ({out.shape[1]}x{out.shape[0]})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cols", type=int, default=11)
    parser.add_argument("--rows", type=int, default=6)
    parser.add_argument("--res", type=int, default=4092)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    generate_test_quilt(cols=args.cols, rows=args.rows, res=args.res, output=args.output)
