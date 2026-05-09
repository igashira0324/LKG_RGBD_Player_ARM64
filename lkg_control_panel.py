import sys
import json
import socket
import argparse
import math
import os
import getpass
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QSlider, QLabel, QPushButton, QGroupBox,
                             QDoubleSpinBox, QComboBox)
from PySide6.QtCore import Qt

class LKGControlPanel(QMainWindow):
    def __init__(self, monitor_index=1):
        super().__init__()
        self.setWindowTitle(f"Looking Glass Go - Control Panel (Monitor {monitor_index})")
        self.setFixedWidth(520)
        self.setFixedHeight(650)
        
        # UDP Setup
        self.udp_ip = "127.0.0.1"
        self.udp_port = 5000 + monitor_index
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        
        # Parameters
        self.focus = 0.5
        self.depthiness = 1.0
        self.invView = 1
        self.depthLoc = 3  # 3=Depth on Right (ComfyUI default)
        self.invertDepth = 0
        self.testPattern = 0
        self.flipSubp = 0
        self.debugMode = 0
        
        # High Quality Parameters
        self.maxParallaxPx = 8.0
        self.depthNear = 0.05
        self.depthFar = 0.95
        self.depthGamma = 1.2
        self.depthSmooth = 0.5
        self.edgeFade = 0.8
        
        # Base values from factory config
        self.base_pitch = 143.6
        self.base_tilt = -0.324
        self.base_center = 0.0
        
        # Final values = base + offset
        self.pitch = self.base_pitch
        self.tilt = self.base_tilt
        self.center = self.base_center
        
        # Load calibration defaults
        self.load_calibration()
        
        self.init_ui()
        self.apply_styles()

    def discover_factory_calibration(self):
        """Search for visual.json on mounted Looking Glass drives."""
        search_paths = ["/media", "/mnt"]
        try:
            search_paths.append(f"/run/media/{getpass.getuser()}")
        except Exception:
            pass

        for base in search_paths:
            if not os.path.exists(base): continue
            try:
                for drive in os.listdir(base):
                    drive_path = os.path.join(base, drive)
                    full_path = os.path.join(drive_path, "LKG_calibration", "visual.json")
                    if os.path.exists(full_path): return full_path
                    full_path = os.path.join(drive_path, "visual.json")
                    if os.path.exists(full_path): return full_path
                    
                    if os.path.isdir(drive_path):
                        for sub in os.listdir(drive_path):
                            full_path = os.path.join(drive_path, sub, "LKG_calibration", "visual.json")
                            if os.path.exists(full_path): return full_path
            except:
                continue
        return None

    def load_calibration(self):
        """Load calibration defaults and runtime overrides."""
        def load_json_if_exists(path):
            if path and os.path.exists(path):
                with open(path, "r") as f:
                    return json.load(f)
            return {}

        def get_calib_value(config_dict, key, default_val):
            v = config_dict.get(key, default_val)
            if isinstance(v, dict):
                return v.get("value", default_val)
            return v

        script_dir = os.path.dirname(os.path.abspath(__file__))
        override_path = os.path.join(script_dir, "lkg_calibration.json")
        factory_path = self.discover_factory_calibration()

        if factory_path:
            calib_file = factory_path
            print(f"GUI: Using factory calibration: {factory_path}")
        else:
            calib_file = override_path
            print(f"GUI: Using fallback calibration: {calib_file}")

        calib_data = load_json_if_exists(calib_file)
        override_data = load_json_if_exists(override_path)

        # Base config
        config = calib_data.get('configValue', calib_data)
        raw_pitch = float(get_calib_value(config, "pitch", 49.818))
        raw_slope = float(get_calib_value(config, "slope", -5.48))
        raw_center = float(get_calib_value(config, "center", 0.157))
        screen_w = float(get_calib_value(config, "screenW", 1440.0))
        screen_h = float(get_calib_value(config, "screenH", 2560.0))
        dpi = float(get_calib_value(config, "DPI", 491.0))
        
        self.invView = int(get_calib_value(config, "invView", self.invView))
        self.flipSubp = int(get_calib_value(config, "flipSubp", self.flipSubp))
        
        screen_inches = screen_w / dpi
        self.base_pitch = raw_pitch * screen_inches * math.cos(math.atan(1.0 / raw_slope))
        self.base_tilt = screen_h / (screen_w * raw_slope)
        self.base_center = raw_center
        
        # Overrides
        overrides = override_data.get('runtimeOverride', {})
        pitchOffset = float(overrides.get("pitchOffset", 0.0))
        tiltOffset = float(overrides.get("tiltOffset", 0.0))
        centerOffset = float(overrides.get("centerOffset", 0.0))
        
        self.pitch = self.base_pitch + pitchOffset
        self.tilt = self.base_tilt + tiltOffset
        self.center = self.base_center + centerOffset
        
        self.maxParallaxPx = float(overrides.get("maxParallaxPx", self.maxParallaxPx))
        self.depthNear = float(overrides.get("depthNear", self.depthNear))
        self.depthFar = float(overrides.get("depthFar", self.depthFar))
        self.depthGamma = float(overrides.get("depthGamma", self.depthGamma))
        self.depthSmooth = float(overrides.get("depthSmooth", self.depthSmooth))
        self.edgeFade = float(overrides.get("edgeFade", self.edgeFade))
                
    def save_calibration(self):
        calib_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lkg_calibration.json")
        data = {}
        if os.path.exists(calib_path):
            with open(calib_path, 'r') as f:
                data = json.load(f)
                
        overrides = data.setdefault("runtimeOverride", {})
        overrides["pitchOffset"] = self.pitch - self.base_pitch
        overrides["tiltOffset"] = self.tilt - self.base_tilt
        overrides["centerOffset"] = self.center - self.base_center
        overrides["maxParallaxPx"] = self.maxParallaxPx
        overrides["depthNear"] = self.depthNear
        overrides["depthFar"] = self.depthFar
        overrides["depthGamma"] = self.depthGamma
        overrides["depthSmooth"] = self.depthSmooth
        overrides["edgeFade"] = self.edgeFade
        
        with open(calib_path, 'w') as f:
            json.dump(data, f, indent=2)
            
        print(f"Saved calibration offsets to {calib_path}")
        
    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        layout.setContentsMargins(15, 10, 15, 10)
        layout.setSpacing(5)
        
        # === Depth Controls ===
        depth_group = QGroupBox("Depth Settings")
        depth_layout = QVBoxLayout(depth_group)
        
        # Focus & Depthiness in one row
        depth_row = QHBoxLayout()
        self.focus_label = QLabel(f"Focus: {self.focus:.2f}")
        depth_row.addWidget(self.focus_label)
        self.focus_slider = QSlider(Qt.Horizontal)
        self.focus_slider.setRange(0, 100)
        self.focus_slider.setValue(int(self.focus * 100))
        self.focus_slider.valueChanged.connect(self.update_params)
        depth_row.addWidget(self.focus_slider)
        depth_layout.addLayout(depth_row)
        
        depthiness_row = QHBoxLayout()
        self.depth_label = QLabel(f"Depthiness: {self.depthiness:.2f}")
        depthiness_row.addWidget(self.depth_label)
        self.depth_slider = QSlider(Qt.Horizontal)
        self.depth_slider.setRange(0, 500)
        self.depth_slider.setValue(int(self.depthiness * 100))
        self.depth_slider.valueChanged.connect(self.update_params)
        depthiness_row.addWidget(self.depth_slider)
        depth_layout.addLayout(depthiness_row)
        
        # Toggle Buttons
        btn_layout = QHBoxLayout()
        self.swap_btn = QPushButton("SWAP RGB/DEPTH")
        self.swap_btn.setCheckable(True)
        self.swap_btn.setChecked(self.depthLoc == 3)
        self.swap_btn.clicked.connect(self.toggle_swap)
        btn_layout.addWidget(self.swap_btn)
        
        self.inv_btn = QPushButton("INVERT VIEW")
        self.inv_btn.setCheckable(True)
        self.inv_btn.setChecked(self.invView == 1)
        self.inv_btn.clicked.connect(self.toggle_inv)
        btn_layout.addWidget(self.inv_btn)
        
        self.inv_depth_btn = QPushButton("INVERT DEPTH")
        self.inv_depth_btn.setCheckable(True)
        self.inv_depth_btn.setChecked(self.invertDepth == 1)
        self.inv_depth_btn.clicked.connect(self.toggle_inv_depth)
        btn_layout.addWidget(self.inv_depth_btn)
        
        depth_layout.addLayout(btn_layout)
        
        layout.addWidget(depth_group)
        
        # === High Quality Settings ===
        hq_group = QGroupBox("High Quality Settings")
        hq_layout = QVBoxLayout(hq_group)
        
        # Max Parallax Px
        parallax_row = QHBoxLayout()
        parallax_row.addWidget(QLabel("Max Parallax (px):"))
        self.parallax_spin = QDoubleSpinBox()
        self.parallax_spin.setRange(0.0, 32.0)
        self.parallax_spin.setSingleStep(0.25)
        self.parallax_spin.setDecimals(2)
        self.parallax_spin.setValue(self.maxParallaxPx)
        self.parallax_spin.valueChanged.connect(self.update_params)
        parallax_row.addWidget(self.parallax_spin)
        hq_layout.addLayout(parallax_row)
        
        # Combine Near/Far
        nf_row = QHBoxLayout()
        self.near_label = QLabel(f"N: {self.depthNear:.2f}")
        nf_row.addWidget(self.near_label)
        self.near_slider = QSlider(Qt.Horizontal)
        self.near_slider.setRange(0, 100)
        self.near_slider.setValue(int(self.depthNear * 100))
        self.near_slider.valueChanged.connect(self.update_params)
        nf_row.addWidget(self.near_slider)
        
        self.far_label = QLabel(f"F: {self.depthFar:.2f}")
        nf_row.addWidget(self.far_label)
        self.far_slider = QSlider(Qt.Horizontal)
        self.far_slider.setRange(0, 100)
        self.far_slider.setValue(int(self.depthFar * 100))
        self.far_slider.valueChanged.connect(self.update_params)
        nf_row.addWidget(self.far_slider)
        hq_layout.addLayout(nf_row)
        
        # Combine Gamma/Edge/Smooth in two rows
        row2 = QHBoxLayout()
        self.gamma_label = QLabel(f"G: {self.depthGamma:.2f}")
        row2.addWidget(self.gamma_label)
        self.gamma_slider = QSlider(Qt.Horizontal)
        self.gamma_slider.setRange(1, 300)
        self.gamma_slider.setValue(int(self.depthGamma * 100))
        self.gamma_slider.valueChanged.connect(self.update_params)
        row2.addWidget(self.gamma_slider)
        hq_layout.addLayout(row2)
        
        row3 = QHBoxLayout()
        self.edge_label = QLabel(f"E: {self.edgeFade:.2f}")
        row3.addWidget(self.edge_label)
        self.edge_slider = QSlider(Qt.Horizontal)
        self.edge_slider.setRange(0, 100)
        self.edge_slider.setValue(int(self.edgeFade * 100))
        self.edge_slider.valueChanged.connect(self.update_params)
        row3.addWidget(self.edge_slider)
        
        self.smooth_label = QLabel(f"S: {self.depthSmooth:.2f}")
        row3.addWidget(self.smooth_label)
        self.smooth_slider = QSlider(Qt.Horizontal)
        self.smooth_slider.setRange(0, 100)
        self.smooth_slider.setValue(int(self.depthSmooth * 100))
        self.smooth_slider.valueChanged.connect(self.update_params)
        row3.addWidget(self.smooth_slider)
        hq_layout.addLayout(row3)
        
        layout.addWidget(hq_group)
        
        # === Calibration Controls ===
        calib_group = QGroupBox("Calibration (Advanced)")
        calib_layout = QVBoxLayout(calib_group)
        
        # Pitch
        pitch_row = QHBoxLayout()
        pitch_row.addWidget(QLabel("Shader Pitch:"))
        self.pitch_spin = QDoubleSpinBox()
        self.pitch_spin.setRange(1.0, 300.0)
        self.pitch_spin.setSingleStep(0.05)
        self.pitch_spin.setDecimals(3)
        self.pitch_spin.setValue(self.pitch)
        self.pitch_spin.valueChanged.connect(self.update_params)
        pitch_row.addWidget(self.pitch_spin)
        calib_layout.addLayout(pitch_row)
        
        # Tilt
        tilt_row = QHBoxLayout()
        tilt_row.addWidget(QLabel("Shader Tilt:"))
        self.tilt_spin = QDoubleSpinBox()
        self.tilt_spin.setRange(-2.0, 2.0)
        self.tilt_spin.setSingleStep(0.01)
        self.tilt_spin.setDecimals(3)
        self.tilt_spin.setValue(self.tilt)
        self.tilt_spin.valueChanged.connect(self.update_params)
        tilt_row.addWidget(self.tilt_spin)
        calib_layout.addLayout(tilt_row)
        
        # Center
        center_row = QHBoxLayout()
        center_row.addWidget(QLabel("Center:"))
        self.center_spin = QDoubleSpinBox()
        self.center_spin.setRange(-1.0, 1.0)
        self.center_spin.setSingleStep(0.01)
        self.center_spin.setDecimals(3)
        self.center_spin.setValue(self.center)
        self.center_spin.valueChanged.connect(self.update_params)
        center_row.addWidget(self.center_spin)
        calib_layout.addLayout(center_row)
        layout.addWidget(calib_group)
        
        # Debug Mode
        debug_row = QHBoxLayout()
        debug_row.addWidget(QLabel("Render Mode:"))
        self.debug_combo = QComboBox()
        self.debug_combo.addItems(["Normal", "RGB Only (2D)", "Depth Only", "Offset Heatmap"])
        self.debug_combo.currentIndexChanged.connect(self.update_params)
        debug_row.addWidget(self.debug_combo)
        layout.addLayout(debug_row)
        
        # Test Pattern Toggle
        self.test_btn = QPushButton("TOGGLE TEST PATTERN")
        self.test_btn.setCheckable(True)
        self.test_btn.setChecked(self.testPattern == 1)
        self.test_btn.clicked.connect(self.toggle_test_pattern)
        layout.addWidget(self.test_btn)
        
        # Buttons layout
        action_layout = QHBoxLayout()
        self.save_btn = QPushButton("SAVE CALIBRATION")
        self.save_btn.setFixedHeight(45)
        self.save_btn.clicked.connect(self.save_calibration)
        action_layout.addWidget(self.save_btn)
        
        self.reset_btn = QPushButton("RESET TO DEFAULT")
        self.reset_btn.setFixedHeight(45)
        self.reset_btn.clicked.connect(self.reset_defaults)
        action_layout.addWidget(self.reset_btn)
        
        layout.addLayout(action_layout)

    def apply_styles(self):
        self.setStyleSheet("""
            QMainWindow { background-color: #0f172a; }
            #header { color: #38bdf8; font-size: 22px; font-weight: bold; margin-bottom: 5px; }
            QLabel { color: #e2e8f0; font-size: 13px; }
            QGroupBox { 
                color: #94a3b8; font-size: 13px; font-weight: bold;
                border: 1px solid #1e293b; border-radius: 8px;
                margin-top: 8px; padding-top: 16px;
            }
            QGroupBox::title { subcontrol-origin: margin; left: 12px; padding: 0 4px; }
            QSlider::groove:horizontal {
                height: 8px; background: #1e293b; border-radius: 4px;
            }
            QSlider::handle:horizontal {
                background: #38bdf8; width: 20px; height: 20px; margin: -6px 0; border-radius: 10px;
            }
            QSlider::handle:horizontal:hover { background: #7dd3fc; }
            QPushButton { 
                background-color: transparent; border: 2px solid #38bdf8; color: #38bdf8; 
                padding: 8px; border-radius: 6px; font-weight: bold; font-size: 12px;
            }
            QPushButton:checked { background-color: #38bdf8; color: #0f172a; }
            QPushButton:hover { background-color: rgba(56, 189, 248, 0.15); }
            QDoubleSpinBox {
                background-color: #1e293b; color: #e2e8f0; border: 1px solid #334155;
                border-radius: 4px; padding: 4px; font-size: 13px;
            }
            QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {
                background-color: #334155; border: none; width: 16px;
            }
        """)

    def toggle_swap(self):
        self.depthLoc = 3 if self.swap_btn.isChecked() else 2
        self.update_params()

    def toggle_inv(self):
        self.invView = 1 if self.inv_btn.isChecked() else 0
        self.update_params()
        
    def toggle_inv_depth(self):
        self.invertDepth = 1 if self.inv_depth_btn.isChecked() else 0
        self.update_params()
        
    def toggle_test_pattern(self):
        self.testPattern = 1 if self.test_btn.isChecked() else 0
        self.update_params()

    def update_params(self):
        try:
            self.focus = self.focus_slider.value() / 100.0
            self.depthiness = self.depth_slider.value() / 100.0
            self.maxParallaxPx = self.parallax_spin.value()
            self.depthNear = self.near_slider.value() / 100.0
            self.depthFar = self.far_slider.value() / 100.0
            if self.depthFar <= self.depthNear + 0.01:
                self.depthFar = self.depthNear + 0.01
                self.far_slider.setValue(int(self.depthFar * 100))
                
            self.depthGamma = self.gamma_slider.value() / 100.0
            self.edgeFade = self.edge_slider.value() / 100.0
            self.depthSmooth = self.smooth_slider.value() / 100.0
            
            self.pitch = self.pitch_spin.value()
            self.tilt = self.tilt_spin.value()
            self.center = self.center_spin.value()
            self.debugMode = self.debug_combo.currentIndex()
            
            self.focus_label.setText(f"Focus: {self.focus:.2f}")
            self.depth_label.setText(f"D: {self.depthiness:.2f}")
            self.near_label.setText(f"N: {self.depthNear:.2f}")
            self.far_label.setText(f"F: {self.depthFar:.2f}")
            self.gamma_label.setText(f"G: {self.depthGamma:.2f}")
            self.edge_label.setText(f"E: {self.edgeFade:.2f}")
            self.smooth_label.setText(f"S: {self.depthSmooth:.2f}")
        except RuntimeError:
            return # Object already deleted during close

        
        # Send via UDP
        msg = {
            "focus": self.focus,
            "depthiness": self.depthiness,
            "invView": self.invView,
            "depthLoc": self.depthLoc,
            "invertDepth": self.invertDepth,
            "testPattern": self.testPattern,
            "pitch": self.pitch,
            "tilt": self.tilt,
            "center": self.center,
            "flipSubp": self.flipSubp,
            "maxParallaxPx": self.maxParallaxPx,
            "depthNear": self.depthNear,
            "depthFar": self.depthFar,
            "depthGamma": self.depthGamma,
            "edgeFade": self.edgeFade,
            "depthSmooth": self.depthSmooth,
            "debugMode": self.debugMode
        }
        try:
            self.sock.sendto(json.dumps(msg).encode(), (self.udp_ip, self.udp_port))
        except Exception as e:
            print(f"UDP send error: {e}")

    def reset_defaults(self):
        try:
            self.load_calibration()  # Re-load from file
            
            # Block signals to avoid recursive/deleted object calls during update
            self.focus_slider.blockSignals(True)
            self.depth_slider.blockSignals(True)
            self.near_slider.blockSignals(True)
            self.far_slider.blockSignals(True)
            self.gamma_slider.blockSignals(True)
            self.edge_slider.blockSignals(True)
            self.smooth_slider.blockSignals(True)
            self.pitch_spin.blockSignals(True)
            self.tilt_spin.blockSignals(True)
            self.center_spin.blockSignals(True)
            
            self.focus_slider.setValue(50)
            self.depth_slider.setValue(100)
            self.parallax_spin.setValue(self.maxParallaxPx)
            self.near_slider.setValue(int(self.depthNear * 100))
            self.far_slider.setValue(int(self.depthFar * 100))
            self.gamma_slider.setValue(int(self.depthGamma * 100))
            self.edge_slider.setValue(int(self.edgeFade * 100))
            self.smooth_slider.setValue(int(self.depthSmooth * 100))
            
            self.swap_btn.setChecked(True)
            self.inv_btn.setChecked(True)
            self.inv_depth_btn.setChecked(False)
            self.test_btn.setChecked(False)
            self.depthLoc = 3
            self.invView = 1
            self.invertDepth = 0
            self.testPattern = 0
            self.pitch_spin.setValue(self.pitch)
            self.tilt_spin.setValue(self.tilt)
            self.center_spin.setValue(self.center)
            
            self.focus_slider.blockSignals(False)
            self.depth_slider.blockSignals(False)
            self.near_slider.blockSignals(False)
            self.far_slider.blockSignals(False)
            self.gamma_slider.blockSignals(False)
            self.edge_slider.blockSignals(False)
            self.smooth_slider.blockSignals(False)
            self.pitch_spin.blockSignals(False)
            self.tilt_spin.blockSignals(False)
            self.center_spin.blockSignals(False)
            
            self.update_params()
        except RuntimeError:
            pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--monitor", type=int, default=1, help="Monitor index (1 or 2) to control")
    args = parser.parse_args()
    
    app = QApplication(sys.argv)
    window = LKGControlPanel(monitor_index=args.monitor)
    window.show()
    sys.exit(app.exec())
