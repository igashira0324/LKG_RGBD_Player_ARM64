#!/usr/bin/env python3
import sys
import json
import socket
import argparse
import os
from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout, 
                             QSlider, QLabel, QComboBox, QPushButton, QDoubleSpinBox, QSpinBox)
from PyQt5.QtCore import Qt

class LKGControlPanel(QWidget):
    def __init__(self, monitor_index=1, calib_file=None, pipeline="rgbd"):
        super().__init__()
        self.monitor_index = monitor_index
        self.pipeline = pipeline
        self.udp_port = 5000 + monitor_index
        self.udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        
        # Calibration / Common
        self.pitch = 143.6
        self.tilt = -0.324
        self.center = 0.0
        self.flipSubp = 0
        self.invView = 1
        
        # RGBD Params
        self.focus = 0.5
        self.depthiness = 1.0
        self.maxParallaxPx = 3.0
        self.depthGamma = 1.2
        self.depthContrast = 1.2
        self.depthSmooth = 0.5
        self.edgeFade = 0.8
        self.depthLoc = 3 # Right
        self.invertDepth = 0
        self.debugMode = 0
        
        # Quilt Params
        self.quiltCols = 11
        self.quiltRows = 6
        self.quiltViews = 66
        self.quiltAspect = 0.5625
        self.quiltFit = 0
        self.flipRows = 0
        self.debugFixedView = -1
        self.quiltZoom = 1.0
        self.overscan = 0.0
        
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle(f"LKG Go Control - Monitor {self.monitor_index}")
        layout = QVBoxLayout()
        
        # Pipeline Info
        layout.addWidget(QLabel(f"Active Pipeline: {self.pipeline.upper()}"))
        
        # Common: Invert View
        self.inv_btn = QPushButton("INVERT VIEW: ON")
        self.inv_btn.setCheckable(True)
        self.inv_btn.setChecked(True)
        self.inv_btn.clicked.connect(self.toggle_inv)
        layout.addWidget(self.inv_btn)

        if self.pipeline == "rgbd":
            self.setup_rgbd_ui(layout)
        else:
            self.setup_quilt_ui(layout)
            
        # Calibration (Common)
        layout.addWidget(QLabel("--- Calibration ---"))
        self.pitch_spin = self.add_spin(layout, "Pitch:", 100, 200, self.pitch, 0.1)
        self.tilt_spin = self.add_spin(layout, "Tilt:", -1.0, 1.0, self.tilt, 0.001)
        self.center_spin = self.add_spin(layout, "Center:", -1.0, 1.0, self.center, 0.001)
        
        self.setLayout(layout)
        self.update_params()

    def setup_rgbd_ui(self, layout):
        layout.addWidget(QLabel("--- RGBD Controls ---"))
        self.focus_slider = self.add_slider(layout, "Focus:", 0, 100, int(self.focus*100))
        self.depth_slider = self.add_slider(layout, "Depthiness:", 0, 500, int(self.depthiness*100))
        self.parallax_slider = self.add_slider(layout, "Max Parallax:", 0, 100, int(self.maxParallaxPx*10))
        
        self.debug_combo = QComboBox()
        self.debug_combo.addItems(["Standard", "RGB Only", "Smooth Depth", "Raw Depth", "Parallax Mask", "Edge Mask"])
        self.debug_combo.currentIndexChanged.connect(self.update_params)
        layout.addWidget(QLabel("Debug Mode:"))
        layout.addWidget(self.debug_combo)

    def setup_quilt_ui(self, layout):
        layout.addWidget(QLabel("--- Quilt Controls ---"))
        self.fixed_view_label = QLabel("Fixed View Index: OFF")
        self.fixed_view_slider = QSlider(Qt.Horizontal)
        self.fixed_view_slider.setRange(-1, 65)
        self.fixed_view_slider.setValue(-1)
        self.fixed_view_slider.valueChanged.connect(self.update_fixed_view)
        layout.addWidget(self.fixed_view_label)
        layout.addWidget(self.fixed_view_slider)
        
        self.fit_combo = QComboBox()
        self.fit_combo.addItems(["Stretch", "Contain", "Cover"])
        self.fit_combo.currentIndexChanged.connect(self.update_params)
        layout.addWidget(QLabel("Fit Mode:"))
        layout.addWidget(self.fit_combo)
        
        self.flip_rows_btn = QPushButton("FLIP ROWS: OFF")
        self.flip_rows_btn.setCheckable(True)
        self.flip_rows_btn.clicked.connect(self.update_params)
        layout.addWidget(self.flip_rows_btn)

    def add_slider(self, layout, label, min_v, max_v, init_v):
        l = QLabel(label)
        s = QSlider(Qt.Horizontal)
        s.setRange(min_v, max_v)
        s.setValue(init_v)
        s.valueChanged.connect(self.update_params)
        layout.addWidget(l)
        layout.addWidget(s)
        return s

    def add_spin(self, layout, label, min_v, max_v, init_v, step):
        row = QHBoxLayout()
        row.addWidget(QLabel(label))
        s = QDoubleSpinBox()
        s.setRange(min_v, max_v)
        s.setValue(init_v)
        s.setSingleStep(step)
        s.setDecimals(4)
        s.valueChanged.connect(self.update_params)
        row.addWidget(s)
        layout.addLayout(row)
        return s

    def toggle_inv(self):
        self.invView = 1 if self.inv_btn.isChecked() else 0
        self.inv_btn.setText(f"INVERT VIEW: {'ON' if self.invView else 'OFF'}")
        self.update_params()

    def update_fixed_view(self):
        v = self.fixed_view_slider.value()
        self.fixed_view_label.setText(f"Fixed View Index: {v}" if v >= 0 else "Fixed View Index: OFF")
        self.update_params()

    def update_params(self):
        msg = {
            "pipeline": self.pipeline,
            "invView": self.invView,
            "pitch": self.pitch_spin.value(),
            "tilt": self.tilt_spin.value(),
            "center": self.center_spin.value(),
            "flipSubp": self.flipSubp
        }
        
        if self.pipeline == "quilt":
            msg.update({
                "quiltCols": self.quiltCols,
                "quiltRows": self.quiltRows,
                "quiltAspect": self.quiltAspect,
                "quiltFit": self.fit_combo.currentIndex(),
                "flipRows": 1 if self.flip_rows_btn.isChecked() else 0,
                "debugFixedView": self.fixed_view_slider.value(),
                "quiltZoom": self.quiltZoom,
                "overscan": self.overscan
            })
        else:
            msg.update({
                "focus": self.focus_slider.value() / 100.0,
                "depthiness": self.depth_slider.value() / 100.0,
                "maxParallaxPx": self.parallax_slider.value() / 10.0,
                "depthContrast": self.depthContrast,
                "depthGamma": self.depthGamma,
                "depthSmooth": self.depthSmooth,
                "edgeFade": self.edgeFade,
                "depthLoc": self.depthLoc,
                "invertDepth": self.invertDepth,
                "debugMode": self.debug_combo.currentIndex()
            })
            
        data = json.dumps(msg).encode('utf-8')
        self.udp_socket.sendto(data, ('127.0.0.1', self.udp_port))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--monitor", type=int, default=1)
    parser.add_argument("--pipeline", default="rgbd")
    args = parser.parse_args()
    app = QApplication(sys.argv)
    win = LKGControlPanel(monitor_index=args.monitor, pipeline=args.pipeline)
    win.show()
    sys.exit(app.exec())
