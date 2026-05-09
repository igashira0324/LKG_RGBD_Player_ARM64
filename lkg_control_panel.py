#!/usr/bin/env python3
import sys
import json
import socket
import argparse
import os
import math
from PyQt6.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout, 
                             QSlider, QLabel, QComboBox, QPushButton, QDoubleSpinBox, QSpinBox)
from PyQt6.QtCore import Qt

class LKGControlPanel(QWidget):
    def __init__(self, args, unknown):
        super().__init__()
        self.initializing = True
        self.monitor_index = args.monitor
        self.pipeline = args.pipeline
        self.udp_port = 5000 + args.monitor
        self.udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        
        # Initial states from args
        self.pitch = 143.6; self.tilt = -0.324; self.center = 0.0; self.invView = args.inv_view
        self.screen_w = 1440.0; self.screen_h = 2560.0
        self.focus = args.focus; self.depthiness = args.depthiness; self.maxParallaxPx = args.max_parallax_px
        self.depthContrast = 1.2; self.depthGamma = 1.2; self.depthSmooth = 0.5; self.edgeFade = 0.8; self.depthLoc = args.depth_loc
        self.quiltViews = args.quilt_views or (args.quilt_cols * args.quilt_rows)
        self.quiltFit = {"stretch": 0, "contain": 1, "cover": 2}.get(args.quilt_fit, 0)
        self.flipRows = 1 if args.quilt_flip_rows else 0
        self.quiltZoom = args.quilt_zoom; self.overscan = args.overscan; self.debugFixedView = args.debug_fixed_view

        self.load_calibration(args.calib_file)
        self.init_ui()
        self.initializing = False

    def get_calib_value(self, config, key, default):
        v = config.get(key, default)
        if isinstance(v, dict): return float(v.get("value", default))
        return float(v)

    def load_calibration(self, calib_file):
        if not calib_file:
            for p in ["/media", "/mnt"]:
                if os.path.exists(p):
                    for root, dirs, files in os.walk(p):
                        if "visual.json" in files: calib_file = os.path.join(root, "visual.json"); break
        if calib_file and os.path.exists(calib_file):
            try:
                with open(calib_file, 'r', encoding='utf-8') as f: config = json.load(f)
                raw_pitch = self.get_calib_value(config, "pitch", 49.818); raw_slope = self.get_calib_value(config, "slope", -5.48)
                raw_center = self.get_calib_value(config, "center", 0.157); dpi = self.get_calib_value(config, "DPI", 491.0)
                self.screen_w = self.get_calib_value(config, "screenW", 1440.0); self.screen_h = self.get_calib_value(config, "screenH", 2560.0)
                screen_inches = math.sqrt(self.screen_w**2 + self.screen_h**2) / dpi
                self.pitch = raw_pitch * screen_inches * math.cos(math.atan(1.0 / raw_slope))
                self.tilt = self.screen_h / (self.screen_w * raw_slope); self.center = raw_center
            except: pass

    def init_ui(self):
        self.setWindowTitle(f"LKG Go Control - Monitor {self.monitor_index}"); layout = QVBoxLayout()
        layout.addWidget(QLabel(f"Active Pipeline: {self.pipeline.upper()}"))
        self.inv_btn = QPushButton(f"INVERT VIEW: {'ON' if self.invView else 'OFF'}")
        self.inv_btn.setCheckable(True); self.inv_btn.setChecked(bool(self.invView))
        self.inv_btn.clicked.connect(self.toggle_inv); layout.addWidget(self.inv_btn)

        if self.pipeline == "rgbd": self.setup_rgbd_ui(layout)
        else: self.setup_quilt_ui(layout)
            
        layout.addWidget(QLabel("--- Calibration ---"))
        self.pitch_spin = self.add_spin(layout, "Pitch:", 50, 500, self.pitch, 0.001)
        self.tilt_spin = self.add_spin(layout, "Tilt:", -5.0, 5.0, self.tilt, 0.0001)
        self.center_spin = self.add_spin(layout, "Center:", -2.0, 2.0, self.center, 0.001)
        self.apply_btn = QPushButton("APPLY CALIBRATION (Overwrites Device Offsets)")
        self.apply_btn.setStyleSheet("background-color: #444; color: #fff; font-weight: bold; padding: 8px;")
        self.apply_btn.clicked.connect(self.send_calibration); layout.addWidget(self.apply_btn)
        self.setLayout(layout)

    def setup_rgbd_ui(self, layout):
        layout.addWidget(QLabel("--- RGBD Controls ---"))
        self.focus_slider = self.add_slider(layout, "Focus:", 0, 100, int(self.focus*100))
        self.depth_slider = self.add_slider(layout, "Depthiness:", 0, 500, int(self.depthiness*100))
        self.parallax_slider = self.add_slider(layout, "Max Parallax:", 0, 100, int(self.maxParallaxPx*10))
        self.contrast_slider = self.add_slider(layout, "Depth Contrast:", 50, 200, int(self.depthContrast*100))
        self.gamma_slider = self.add_slider(layout, "Depth Gamma:", 50, 200, int(self.depthGamma*100))
        self.smooth_slider = self.add_slider(layout, "Depth Smooth:", 0, 100, int(self.depthSmooth*100))
        self.edge_slider = self.add_slider(layout, "Edge Fade:", 0, 100, int(self.edgeFade*100))
        
        self.depth_loc_combo = QComboBox()
        self.depth_loc_combo.addItems(["Top", "Bottom", "Left", "Right"])
        self.depth_loc_combo.setCurrentIndex(self.depthLoc)
        self.depth_loc_combo.currentIndexChanged.connect(self.update_params)
        layout.addWidget(QLabel("Depth Location:")); layout.addWidget(self.depth_loc_combo)
        
        self.invert_depth_btn = QPushButton("INVERT DEPTH: OFF"); self.invert_depth_btn.setCheckable(True)
        self.invert_depth_btn.clicked.connect(self.update_params); layout.addWidget(self.invert_depth_btn)
        self.debug_combo = QComboBox(); self.debug_combo.addItems(["Standard", "RGB Only", "Smooth Depth", "Raw Depth", "Parallax Mask", "Edge Mask"])
        self.debug_combo.currentIndexChanged.connect(self.update_params); layout.addWidget(QLabel("Debug Mode:")); layout.addWidget(self.debug_combo)

    def setup_quilt_ui(self, layout):
        layout.addWidget(QLabel("--- Quilt Controls ---"))
        self.fixed_view_label = QLabel(f"Fixed View Index: {self.debugFixedView if self.debugFixedView >=0 else 'OFF'}")
        self.fixed_view_slider = QSlider(Qt.Orientation.Horizontal); self.fixed_view_slider.setRange(-1, self.quiltViews - 1)
        self.fixed_view_slider.setValue(self.debugFixedView); self.fixed_view_slider.valueChanged.connect(self.update_fixed_view)
        layout.addWidget(self.fixed_view_label); layout.addWidget(self.fixed_view_slider)
        self.fit_combo = QComboBox(); self.fit_combo.addItems(["Stretch", "Contain", "Cover"]); self.fit_combo.setCurrentIndex(self.quiltFit)
        self.fit_combo.currentIndexChanged.connect(self.update_params); layout.addWidget(QLabel("Fit Mode:")); layout.addWidget(self.fit_combo)
        self.zoom_slider = self.add_slider(layout, "Quilt Zoom:", 50, 200, int(self.quiltZoom*100))
        self.overscan_slider = self.add_slider(layout, "Overscan:", 0, 100, int(self.overscan*100))
        self.flip_rows_btn = QPushButton(f"FLIP ROWS: {'ON' if self.flipRows else 'OFF'}")
        self.flip_rows_btn.setCheckable(True); self.flip_rows_btn.setChecked(bool(self.flipRows))
        self.flip_rows_btn.clicked.connect(self.update_params); layout.addWidget(self.flip_rows_btn)

    def add_slider(self, layout, label, min_v, max_v, init_v):
        l = QLabel(label); s = QSlider(Qt.Orientation.Horizontal); s.setRange(min_v, max_v); s.setValue(init_v)
        s.valueChanged.connect(self.update_params); layout.addWidget(l); layout.addWidget(s); return s

    def add_spin(self, layout, label, min_v, max_v, init_v, step):
        row = QHBoxLayout(); row.addWidget(QLabel(label)); s = QDoubleSpinBox(); s.setRange(min_v, max_v); s.setValue(init_v); s.setSingleStep(step); s.setDecimals(4)
        row.addWidget(s); layout.addLayout(row); return s

    def toggle_inv(self): self.invView = 1 if self.inv_btn.isChecked() else 0; self.inv_btn.setText(f"INVERT VIEW: {'ON' if self.invView else 'OFF'}"); self.update_params()
    def update_fixed_view(self): v = self.fixed_view_slider.value(); self.fixed_view_label.setText(f"Fixed View Index: {v}" if v >= 0 else "Fixed View Index: OFF"); self.update_params()
    def send_calibration(self):
        msg = {"pitch": self.pitch_spin.value(), "tilt": self.tilt_spin.value(), "center": self.center_spin.value()}
        self.udp_socket.sendto(json.dumps(msg).encode('utf-8'), ('127.0.0.1', self.udp_port))

    def update_params(self):
        if getattr(self, "initializing", False): return
        msg = {"pipeline": self.pipeline, "invView": self.invView}
        if self.pipeline == "quilt":
            self.flipRows = 1 if self.flip_rows_btn.isChecked() else 0; self.flip_rows_btn.setText(f"FLIP ROWS: {'ON' if self.flipRows else 'OFF'}")
            msg.update({"quiltFit": self.fit_combo.currentIndex(), "flipRows": self.flipRows, "debugFixedView": self.fixed_view_slider.value(), "quiltZoom": self.zoom_slider.value() / 100.0, "overscan": self.overscan_slider.value() / 100.0})
        else:
            self.invert_depth_btn.setText(f"INVERT DEPTH: {'ON' if self.invert_depth_btn.isChecked() else 'OFF'}")
            msg.update({"focus": self.focus_slider.value() / 100.0, "depthiness": self.depth_slider.value() / 100.0, "maxParallaxPx": self.parallax_slider.value() / 10.0, "depthContrast": self.contrast_slider.value() / 100.0, "depthGamma": self.gamma_slider.value() / 100.0, "depthSmooth": self.smooth_slider.value() / 100.0, "edgeFade": self.edge_slider.value() / 100.0, "depthLoc": self.depth_loc_combo.currentIndex(), "invertDepth": 1 if self.invert_depth_btn.isChecked() else 0, "debugMode": self.debug_combo.currentIndex()})
        self.udp_socket.sendto(json.dumps(msg).encode('utf-8'), ('127.0.0.1', self.udp_port))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--monitor", type=int, default=1); parser.add_argument("--pipeline", default="rgbd"); parser.add_argument("--calib-file", default=None); parser.add_argument("--inv-view", type=int, default=1)
    parser.add_argument("--quilt-cols", type=int, default=11); parser.add_argument("--quilt-rows", type=int, default=6); parser.add_argument("--quilt-views", type=int, default=None); parser.add_argument("--quilt-aspect", type=float, default=0.5625); parser.add_argument("--debug-fixed-view", type=int, default=-1); parser.add_argument("--quilt-fit", default="stretch"); parser.add_argument("--quilt-zoom", type=float, default=1.0); parser.add_argument("--overscan", type=float, default=0.0); parser.add_argument("--quilt-flip-rows", action="store_true", default=True); parser.add_argument("--no-quilt-flip-rows", dest="quilt_flip_rows", action="store_false")
    parser.add_argument("--focus", type=float, default=0.5); parser.add_argument("--depthiness", type=float, default=1.0); parser.add_argument("--max-parallax-px", type=float, default=3.0); parser.add_argument("--depth-loc", type=int, default=3)
    args, unknown = parser.parse_known_args()
    app = QApplication(sys.argv); win = LKGControlPanel(args, unknown); win.show(); sys.exit(app.exec())
