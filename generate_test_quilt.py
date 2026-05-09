import cv2
import numpy as np

def generate_test_quilt(cols=11, rows=6, res=4092):
    # Calculate tile size
    tw = res // cols
    th = res // rows
    
    # Create blank quilt
    quilt = np.zeros((th * rows, tw * cols, 3), dtype=np.uint8)
    
    for row in range(rows):
        for col in range(cols):
            view_index = row * cols + col
            
            # Create tile
            tile = np.zeros((th, tw, 3), dtype=np.uint8)
            
            # Background color gradient based on view index
            hue = int(180 * view_index / (cols * rows))
            tile_hsv = np.zeros((th, tw, 3), dtype=np.uint8)
            tile_hsv[..., 0] = hue
            tile_hsv[..., 1] = 200
            tile_hsv[..., 2] = 50
            tile_bgr = cv2.cvtColor(tile_hsv, cv2.COLOR_HSV2BGR)
            tile[:] = tile_bgr
            
            # Draw view number
            text = f"V:{view_index:02d}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 2.0
            thickness = 5
            (text_w, text_h), _ = cv2.getTextSize(text, font, font_scale, thickness)
            
            # Center text
            text_x = (tw - text_w) // 2
            text_y = (th + text_h) // 2
            
            # Draw shadow
            cv2.putText(tile, text, (text_x+3, text_y+3), font, font_scale, (0,0,0), thickness+2)
            # Draw white text
            cv2.putText(tile, text, (text_x, text_y), font, font_scale, (255,255,255), thickness)
            
            # Draw borders
            cv2.rectangle(tile, (0, 0), (tw-1, th-1), (255, 255, 255), 2)
            
            # Place in quilt (View 0 is bottom-left in LKG Quilt standard)
            # But OpenCV Y is top-down, so row 0 in quilt is TOP.
            # LKG View 0 is bottom-left, View (rows-1)*cols is top-left.
            # So LKG row 'r' is quilt row 'rows - 1 - r'
            q_row = rows - 1 - row
            quilt[q_row*th:(q_row+1)*th, col*tw:(col+1)*tw] = tile
            
    # Save with naming convention
    aspect = 0.5625 # Go portrait
    filename = f"test_quilt_qs{cols}x{rows}a{aspect}.png"
    cv2.imwrite(filename, quilt)
    print(f"Generated {filename} ({quilt.shape[1]}x{quilt.shape[0]})")

if __name__ == "__main__":
    generate_test_quilt()
