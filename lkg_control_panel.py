import sys
import json
import socket
import argparse
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QSlider, QLabel, QPushButton, QGroupBox,
                             QDoubleSpinBox)
from PySide6.QtCore import Qt

class LKGControlPanel(QMainWindow):
    def __init__(self, monitor_index=1):
        super().__init__()
        self.setWindowTitle(f"Looking Glass Go - Control Panel (Monitor {monitor_index})")
        self.setFixedWidth(520)
        self.setFixedHeight(700)
        
        # UDP Setup
        self.udp_ip = "127.0.0.1"
        self.udp_port = 5000 + monitor_index
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        
        # Parameters
        self.focus = 0.5
        self.depthiness = 1.0
        self.invView = 1
        self.depthLoc = 3  # 3=Depth on Right (ComfyUI default)
        self.pitch = 47.5
        self.slope = -5.5
        self.center = 0.0
        
        # Load calibration defaults
        self.load_calibration()
        
        self.init_ui()
        self.apply_styles()

    def load_calibration(self):
        """Load calibration defaults from lkg_calibration.json if available."""
        import os
        calib_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lkg_calibration.json")
        if os.path.exists(calib_path):
            try:
                with open(calib_path, 'r') as f:
                    data = json.load(f)
                    config = data.get('configValue', {})
                    self.pitch = config.get('pitch', {}).get('value', self.pitch)
                    self.slope = config.get('slope', {}).get('value', self.slope)
                    self.center = config.get('center', {}).get('value', self.center)
                    inv = config.get('invView', {}).get('value', None)
                    if inv is not None:
                        self.invView = int(inv)
            except Exception as e:
                print(f"Warning: Failed to load calibration: {e}")
        
    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        layout.setContentsMargins(30, 20, 30, 20)
        layout.setSpacing(12)
        
        # Header
        header = QLabel("LKG GO CONTROLS")
        header.setObjectName("header")
        header.setAlignment(Qt.AlignCenter)
        layout.addWidget(header)
        
        # === Depth Controls ===
        depth_group = QGroupBox("Depth Settings")
        depth_layout = QVBoxLayout(depth_group)
        
        # Focus
        self.focus_label = QLabel(f"Focus: {self.focus:.2f}")
        depth_layout.addWidget(self.focus_label)
        self.focus_slider = QSlider(Qt.Horizontal)
        self.focus_slider.setMinimum(0)
        self.focus_slider.setMaximum(100)
        self.focus_slider.setValue(int(self.focus * 100))
        self.focus_slider.valueChanged.connect(self.update_params)
        depth_layout.addWidget(self.focus_slider)
        
        # Depthiness
        self.depth_label = QLabel(f"Depthiness: {self.depthiness:.2f}")
        depth_layout.addWidget(self.depth_label)
        self.depth_slider = QSlider(Qt.Horizontal)
        self.depth_slider.setMinimum(0)
        self.depth_slider.setMaximum(300)
        self.depth_slider.setValue(int(self.depthiness * 100))
        self.depth_slider.valueChanged.connect(self.update_params)
        depth_layout.addWidget(self.depth_slider)
        
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
        depth_layout.addLayout(btn_layout)
        
        layout.addWidget(depth_group)
        
        # === Calibration Controls ===
        calib_group = QGroupBox("Calibration (Advanced)")
        calib_layout = QVBoxLayout(calib_group)
        
        # Pitch
        pitch_row = QHBoxLayout()
        pitch_row.addWidget(QLabel("Pitch:"))
        self.pitch_spin = QDoubleSpinBox()
        self.pitch_spin.setRange(1.0, 300.0)
        self.pitch_spin.setSingleStep(0.05)
        self.pitch_spin.setDecimals(2)
        self.pitch_spin.setValue(self.pitch)
        self.pitch_spin.valueChanged.connect(self.update_params)
        pitch_row.addWidget(self.pitch_spin)
        calib_layout.addLayout(pitch_row)
        
        # Slope
        slope_row = QHBoxLayout()
        slope_row.addWidget(QLabel("Slope:"))
        self.slope_spin = QDoubleSpinBox()
        self.slope_spin.setRange(-10.0, 0.0)
        self.slope_spin.setSingleStep(0.05)
        self.slope_spin.setDecimals(3)
        self.slope_spin.setValue(self.slope)
        self.slope_spin.valueChanged.connect(self.update_params)
        slope_row.addWidget(self.slope_spin)
        calib_layout.addLayout(slope_row)
        
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
        
        # Spacer
        layout.addStretch()
        
        # Reset Button
        self.reset_btn = QPushButton("RESET TO DEFAULT")
        self.reset_btn.setFixedHeight(45)
        self.reset_btn.clicked.connect(self.reset_defaults)
        layout.addWidget(self.reset_btn)

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

    def update_params(self):
        self.focus = self.focus_slider.value() / 100.0
        self.depthiness = self.depth_slider.value() / 100.0
        self.pitch = self.pitch_spin.value()
        self.slope = self.slope_spin.value()
        self.center = self.center_spin.value()
        
        self.focus_label.setText(f"Focus: {self.focus:.2f}")
        self.depth_label.setText(f"Depthiness: {self.depthiness:.2f}")
        
        # Send via UDP
        msg = {
            "focus": self.focus,
            "depthiness": self.depthiness,
            "invView": self.invView,
            "depthLoc": self.depthLoc,
            "pitch": self.pitch,
            "slope": self.slope,
            "center": self.center
        }
        try:
            self.sock.sendto(json.dumps(msg).encode(), (self.udp_ip, self.udp_port))
        except Exception as e:
            print(f"UDP send error: {e}")

    def reset_defaults(self):
        self.load_calibration()  # Re-load from file
        self.focus_slider.setValue(50)
        self.depth_slider.setValue(100)
        self.swap_btn.setChecked(True)
        self.inv_btn.setChecked(True)
        self.depthLoc = 3
        self.invView = 1
        self.pitch_spin.setValue(self.pitch)
        self.slope_spin.setValue(self.slope)
        self.center_spin.setValue(self.center)
        self.update_params()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--monitor", type=int, default=1, help="Monitor index (1 or 2) to control")
    args = parser.parse_args()
    
    app = QApplication(sys.argv)
    window = LKGControlPanel(monitor_index=args.monitor)
    window.show()
    sys.exit(app.exec())
