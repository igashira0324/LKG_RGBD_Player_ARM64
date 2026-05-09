import cv2
import numpy as np
import os

def generate_test_quilt(cols=8, rows=6, tile_w=512, tile_h=512):
    quilt_w = cols * tile_w
    quilt_h = rows * tile_h
    quilt = np.zeros((quilt_h, quilt_w, 3), dtype=np.uint8)
    
    total_views = cols * rows
    
    for i in range(total_views):
        # OpenGL/LKG convention: 0 is bottom-left
        col = i % cols
        row = i // cols
        
        # Tile coordinates (image space: 0,0 is top-left)
        x = col * tile_w
        y = (rows - 1 - row) * tile_h
        
        # Draw gradient background (Red to Blue across views)
        t = i / (total_views - 1)
        color = (
            int((1.0 - t) * 255), # B
            int(t * 128),         # G
            int(t * 255)          # R
        )
        
        tile = np.zeros((tile_h, tile_w, 3), dtype=np.uint8)
        tile[:] = color
        
        # Add horizontal gradient to check left/right orientation
        for gx in range(tile_w):
            tile[:, gx] = tile[:, gx] * (0.5 + 0.5 * gx / tile_w)
            
        # Add view number
        text = f"V:{i:02d}"
        cv2.putText(tile, text, (tile_w//4, tile_h//2), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 5)
        
        # Add grid lines
        cv2.rectangle(tile, (0, 0), (tile_w-1, tile_h-1), (255, 255, 255), 2)
        
        quilt[y:y+tile_h, x:x+tile_w] = tile
        
    output_path = "test_quilt_qs8x6.png"
    cv2.imwrite(output_path, quilt)
    print(f"Generated test quilt: {output_path} ({quilt_w}x{quilt_h})")

if __name__ == "__main__":
    generate_test_quilt()
