#!/usr/bin/env python3
import os
import sys
import time
import json
import socket
import threading
import argparse
import subprocess
import re
import math
from dataclasses import dataclass

import numpy as np
import glfw
import OpenGL.GL as GL
from OpenGL.GL import shaders

# --- Phase 3-A: Quilt Configuration ---
@dataclass
class QuiltConfig:
    cols: int = 11
    rows: int = 6
    views: int = 66
    aspect: float = 0.5625
    source: str = "go-default"
    fit: int = 0  # 0=stretch, 1=contain, 2=cover
    flip_rows: int = 1 
    fixed_view: int = -1
    zoom: float = 1.0
    overscan: float = 0.0

def parse_quilt_from_filename(path):
    if not path: return QuiltConfig()
    name = os.path.basename(path)
    m = re.search(r"_qs(\d+)x(\d+)a([0-9]*\.?[0-9]+)", name)
    if m:
        cols, rows, aspect = int(m.group(1)), int(m.group(2)), float(m.group(3))
        return QuiltConfig(cols=cols, rows=rows, views=cols*rows, aspect=aspect, source="filename")
    m = re.search(r"_qs(\d+)x(\d+)", name)
    if m:
        cols, rows = int(m.group(1)), int(m.group(2))
        return QuiltConfig(cols=cols, rows=rows, views=cols*rows, aspect=0.5625, source="filename-no-aspect")
    return QuiltConfig()

def normalize_serial_key(s):
    if not s: return "Unknown"
    return str(s).replace("‑", "-").replace("–", "-").replace("—", "-").strip()

# --- Shaders ---

VERTEX_SHADER = """
#version 330 core
layout (location = 0) in vec2 aPos;
layout (location = 1) in vec2 aTexCoord;
out vec2 TexCoord;
void main() {
    gl_Position = vec4(aPos, 0.0, 1.0);
    TexCoord = aTexCoord;
}
"""

FRAGMENT_SHADER = """
#version 330 core
out vec4 FragColor;
in vec2 TexCoord;

uniform sampler2D texRGBD;
uniform float pitch;
uniform float tilt;
uniform float center;
uniform float subp;
uniform int flipSubp;
uniform int invView;

uniform float focus;
uniform float depthiness;
uniform float parallaxScale;
uniform float depthNear;
uniform float depthFar;
uniform float depthGamma;
uniform float depthSmooth;
uniform float depthContrast;
uniform float edgeFade;
uniform float edgeLow;
uniform float edgeHigh;
uniform int depthLoc;
uniform int invertDepth;
uniform int debugMode;
uniform int testPattern;
uniform vec2 texelSize;

float readDepth(vec2 uv) {
    float d = texture(texRGBD, uv).r;
    if (invertDepth == 1) d = 1.0 - d;
    return d;
}

float processDepth(float d) {
    d = clamp((d - depthNear) / max(depthFar - depthNear, 0.0001), 0.0, 1.0);
    d = clamp(focus + (d - focus) * depthContrast, 0.0, 1.0);
    return pow(d, depthGamma);
}

float getBilateralDepth(vec2 uv) {
    float d = readDepth(uv);
    if (depthSmooth < 0.05) return d;
    float weightSum = 1.0;
    float depthSum = d;
    for(int i=-1; i<=1; i++) {
        for(int j=-1; j<=1; j++) {
            if(i==0 && j==0) continue;
            vec2 off = vec2(float(i), float(j)) * texelSize * 2.0;
            float val = readDepth(uv + off);
            float weight = exp(-float(i*i+j*j)/2.0) * exp(-pow(d-val,2.0)/0.01);
            depthSum += val * weight;
            weightSum += weight;
        }
    }
    return depthSum / weightSum;
}

float getDilatedDepth(vec2 uv) {
    float d = readDepth(uv);
    for(int i=-1; i<=1; i++) {
        for (int j=-1; j<=1; j++) {
            vec2 off = vec2(float(i), float(j)) * texelSize * 2.0;
            d = max(d, readDepth(uv + off));
        }
    }
    return d;
}

void main() {
    if (testPattern == 1) {
        FragColor = vec4(TexCoord.x, TexCoord.y, 0.5, 1.0);
        return;
    }
    
    // Quick Debug Paths
    if (debugMode > 0) {
        vec2 d_uv, r_uv;
        float sample_y = 1.0 - TexCoord.y;
        if (depthLoc == 2) { d_uv = vec2(TexCoord.x*0.5, sample_y); r_uv = vec2(0.5+TexCoord.x*0.5, sample_y); }
        else if (depthLoc == 3) { r_uv = vec2(TexCoord.x*0.5, sample_y); d_uv = vec2(0.5+TexCoord.x*0.5, sample_y); }
        else if (depthLoc == 0) { d_uv = vec2(TexCoord.x, 0.5+sample_y*0.5); r_uv = vec2(TexCoord.x, sample_y*0.5); }
        else { r_uv = vec2(TexCoord.x, 0.5+sample_y*0.5); d_uv = vec2(TexCoord.x, sample_y*0.5); }
        
        if (debugMode == 1) { FragColor = vec4(texture(texRGBD, r_uv).rgb, 1.0); return; } // RGB Only
        if (debugMode == 2) { FragColor = vec4(vec3(processDepth(getBilateralDepth(d_uv))), 1.0); return; } // Smooth Depth
        if (debugMode == 3) { FragColor = vec4(vec3(readDepth(d_uv)), 1.0); return; } // Raw Depth
        if (debugMode == 4) { // Parallax Mask (Intensity x4 for visibility)
            float d = processDepth(getBilateralDepth(d_uv));
            float vis = abs(d - focus) * depthiness * 4.0;
            FragColor = vec4(vec3(clamp(vis, 0.0, 1.0)), 1.0); return; 
        }
        if (debugMode == 5) { // Edge Mask
            float d = processDepth(getBilateralDepth(d_uv));
            float edge = smoothstep(edgeLow, edgeHigh, abs(dFdx(d)) + abs(dFdy(d)));
            FragColor = vec4(vec3(edge), 1.0); return;
        }
    }

    vec3 color = vec3(0.0);
    for (int i = 0; i < 3; i++) {
        float subpixelIndex = float(i);
        if (flipSubp == 1) subpixelIndex = 2.0 - subpixelIndex;
        float phase = (TexCoord.x + (1.0 - TexCoord.y) * tilt) * pitch - center;
        phase += subpixelIndex * subp;
        float view = fract(phase);
        if (invView == 1) view = 1.0 - view;
        
        vec2 rgb_uv, depth_uv;
        float sample_y = 1.0 - TexCoord.y;
        float r_min = 0.0, r_max = 1.0;

        if (depthLoc == 2) { depth_uv = vec2(TexCoord.x*0.5, sample_y); rgb_uv = vec2(0.5+TexCoord.x*0.5, sample_y); r_min=0.5; }
        else if (depthLoc == 3) { rgb_uv = vec2(TexCoord.x*0.5, sample_y); depth_uv = vec2(0.5+TexCoord.x*0.5, sample_y); r_max=0.5; }
        else if (depthLoc == 0) { depth_uv = vec2(TexCoord.x, 0.5+sample_y*0.5); rgb_uv = vec2(TexCoord.x, sample_y*0.5); }
        else { rgb_uv = vec2(TexCoord.x, 0.5+sample_y*0.5); depth_uv = vec2(TexCoord.x, sample_y*0.5); }

        float dRaw = getBilateralDepth(depth_uv);
        float dilatedRaw = getDilatedDepth(depth_uv);
        float d = processDepth(dRaw);
        float dilated = processDepth(dilatedRaw);
        float edge = smoothstep(edgeLow, edgeHigh, abs(dFdx(d)) + abs(dFdy(d)));
        float finalD = mix(d, dilated, 0.5 * depthSmooth);
        float offset = (finalD - focus) * (view - 0.5) * 2.0 * depthiness * parallaxScale;
        offset *= mix(1.0, 1.0 - (edgeFade * 0.5), edge);
        
        vec2 warped_uv = rgb_uv + vec2(offset, 0.0);
        warped_uv.x = clamp(warped_uv.x, r_min, r_max);
        vec3 sampled = texture(texRGBD, warped_uv).rgb;
        if (i == 0) color.r = sampled.r; else if (i == 1) color.g = sampled.g; else color.b = sampled.b;
    }
    FragColor = vec4(color, 1.0);
}
"""

QUILT_FRAGMENT_SHADER = """
#version 330 core
out vec4 FragColor;
in vec2 TexCoord;

uniform sampler2D texQuilt;
uniform float pitch;
uniform float tilt;
uniform float center;
uniform float subp;
uniform int flipSubp;
uniform int invView;

uniform int quiltCols;
uniform int quiltRows;
uniform int quiltViews;
uniform int quiltFlipRows;
uniform int debugFixedView;
uniform float quiltAspect;
uniform float inputAspect;
uniform int quiltFit; // 0=stretch, 1=contain, 2=cover
uniform float quiltZoom;
uniform float overscan;

void main() {
    vec3 color = vec3(0.0);
    for (int i = 0; i < 3; i++) {
        float subpixelIndex = float(i);
        if (flipSubp == 1) subpixelIndex = 2.0 - subpixelIndex;

        float phase = (TexCoord.x + (1.0 - TexCoord.y) * tilt) * pitch - center;
        phase += subpixelIndex * subp;
        float view01 = fract(phase);
        
        int viewIndex;
        if (debugFixedView >= 0) {
            viewIndex = clamp(debugFixedView, 0, quiltViews - 1);
        } else {
            viewIndex = int(floor(view01 * float(quiltViews)));
            viewIndex = clamp(viewIndex, 0, quiltViews - 1);
            if (invView == 1) viewIndex = (quiltViews - 1) - viewIndex;
        }
        
        int col = viewIndex % quiltCols;
        int row = viewIndex / quiltCols;
        if (quiltFlipRows == 1) row = (quiltRows - 1) - row;
        
        vec2 localUV = TexCoord;
        float totalZoom = quiltZoom * (1.0 + overscan);
        if (totalZoom != 1.0) localUV = (localUV - 0.5) / totalZoom + 0.5;

        if (quiltFit == 1) { // Contain
            float s = (inputAspect / quiltAspect);
            if (s > 1.0) localUV.y = (localUV.y - 0.5) / s + 0.5;
            else localUV.x = (localUV.x - 0.5) * s + 0.5;
        } else if (quiltFit == 2) { // Cover
            float s = (inputAspect / quiltAspect);
            if (s > 1.0) localUV.x = (localUV.x - 0.5) / s + 0.5;
            else localUV.y = (localUV.y - 0.5) * s + 0.5;
        }
        
        if (localUV.x < 0.0 || localUV.x > 1.0 || localUV.y < 0.0 || localUV.y > 1.0) {
            if (i == 0) color.r = 0.0; else if (i == 1) color.g = 0.0; else color.b = 0.0;
            continue;
        }

        vec2 texSize = vec2(textureSize(texQuilt, 0));
        vec2 tileSize = texSize / vec2(float(quiltCols), float(quiltRows));
        localUV = clamp(localUV, 0.5 / tileSize, 1.0 - 0.5 / tileSize);
        
        vec2 tileUV = (vec2(float(col), float(row)) + localUV) / vec2(float(quiltCols), float(quiltRows));
        vec3 sampled = texture(texQuilt, tileUV).rgb;
        if (i == 0) color.r = sampled.r; else if (i == 1) color.g = sampled.g; else color.b = sampled.b;
    }
    FragColor = vec4(color, 1.0);
}
"""

class LKGPlayer:
    def __init__(self, args):
        self.args = args
        self.running = True
        self.pipeline = args.pipeline
        self.serial = "Unknown"
        
        self.pitch = 143.6; self.tilt = -0.324; self.center = 0.0; self.subp = 0.0; self.flipSubp = 0
        self.screen_w = 1440.0; self.screen_h = 2560.0
        
        self.focus = args.focus; self.depthiness = args.depthiness; self.maxParallaxPx = args.max_parallax_px
        self.parallaxScale = 0.002; self.depthNear = 0.05; self.depthFar = 0.95; self.depthGamma = 1.2
        self.depthSmooth = 0.5; self.depthContrast = 1.2; self.edgeFade = 0.8; self.edgeLow = 0.02; self.edgeHigh = 0.10
        self.depthLoc = args.depth_loc; self.invertDepth = 0; self.testPattern = 0; self.debugMode = 0; self.invView = args.inv_view
        
        self.quilt_cfg = parse_quilt_from_filename(args.input)
        self.apply_cli_quilt_args(args)
        
        self.udp_port = 5000 + args.monitor
        self.load_calibration()
        self.start_udp_listener()

    def apply_cli_quilt_args(self, args):
        if args.quilt_cols is not None: self.quilt_cfg.cols = args.quilt_cols; self.quilt_cfg.source = "cli"
        if args.quilt_rows is not None: self.quilt_cfg.rows = args.quilt_rows; self.quilt_cfg.source = "cli"
        self.quilt_cfg.views = args.quilt_views or (self.quilt_cfg.cols * self.quilt_cfg.rows)
        if args.quilt_aspect is not None: self.quilt_cfg.aspect = args.quilt_aspect
        FIT_MAP = {"stretch": 0, "contain": 1, "cover": 2}
        self.quilt_cfg.fit = FIT_MAP.get(args.quilt_fit, 0)
        self.quilt_cfg.zoom = args.quilt_zoom; self.quilt_cfg.overscan = args.overscan
        self.quilt_cfg.flip_rows = 1 if args.quilt_flip_rows else 0
        self.quilt_cfg.fixed_view = args.debug_fixed_view

    def get_calib_value(self, config, key, default):
        v = config.get(key, default)
        if isinstance(v, dict): return float(v.get("value", default))
        return float(v)

    def load_calibration(self):
        calib_file = self.args.calib_file or self.discover_factory_calibration()
        if not calib_file or not os.path.exists(calib_file):
            raw_pitch, raw_slope, raw_center, dpi = 49.818, -5.48, 0.157, 491.0
            screen_w, screen_h = 1440.0, 2560.0; raw_serial = "Unknown"
        else:
            with open(calib_file, 'r', encoding='utf-8') as f: config = json.load(f)
            raw_pitch = self.get_calib_value(config, "pitch", 49.818); raw_slope = self.get_calib_value(config, "slope", -5.48)
            raw_center = self.get_calib_value(config, "center", 0.157)
            dpi = self.get_calib_value(config, "DPI", self.get_calib_value(config, "dpi", 491.0))
            screen_w = self.get_calib_value(config, "screenW", 1440.0); screen_h = self.get_calib_value(config, "screenH", 2560.0)
            raw_serial = config.get("serial")
            if not raw_serial:
                base = os.path.splitext(os.path.basename(calib_file))[0]
                parent = os.path.basename(os.path.dirname(calib_file))
                raw_serial = parent if (base.lower() == "visual" and parent) else base
        self.serial = normalize_serial_key(raw_serial)
        self.screen_w, self.screen_h = screen_w, screen_h
        screen_inches = math.sqrt(screen_w**2 + screen_h**2) / dpi
        self.pitch = raw_pitch * screen_inches * math.cos(math.atan(1.0 / raw_slope))
        self.tilt = screen_h / (screen_w * raw_slope); self.center = raw_center; self.update_subp()
        
        script_dir = os.path.dirname(os.path.abspath(__file__))
        override_file = os.path.join(script_dir, "lkg_calibration.json")
        if os.path.exists(override_file):
            try:
                with open(override_file, 'r') as f: ovr = json.load(f)
                ro = ovr.get("runtimeOverride", {})
                runtime_map = {
                    "focus": "focus",
                    "maxParallaxPx": "maxParallaxPx", 
                    "depthNear": "depthNear", 
                    "depthFar": "depthFar", 
                    "depthGamma": "depthGamma", 
                    "depthSmooth": "depthSmooth", 
                    "depthContrast": "depthContrast", 
                    "edgeFade": "edgeFade", 
                    "edgeLow": "edgeLow", 
                    "edgeHigh": "edgeHigh"
                }
                for json_key, attr in runtime_map.items():
                    if json_key in ro: setattr(self, attr, float(ro[json_key]))
                device_overrides = {normalize_serial_key(k): v for k, v in ovr.get("deviceOverride", {}).items()}
                do = device_overrides.get(self.serial, {})
                self.pitch += float(do.get("pitchOffset", 0.0)); self.tilt += float(do.get("tiltOffset", 0.0)); self.center += float(do.get("centerOffset", 0.0))
                self.update_subp()
            except Exception as e: print(f"[CALIB] Override error: {e}")
        self.update_parallax_scale()

    def update_subp(self): self.subp = self.pitch / (3.0 * self.screen_w)
    def update_parallax_scale(self):
        rgb_uv_width = 0.5 if self.depthLoc in (2, 3) else 1.0
        self.parallaxScale = self.maxParallaxPx * rgb_uv_width / float(self.screen_w)

    def discover_factory_calibration(self):
        for p in ["/media", "/mnt"]:
            if not os.path.exists(p): continue
            for root, dirs, files in os.walk(p):
                if "visual.json" in files: return os.path.join(root, "visual.json")
        return None

    def start_udp_listener(self):
        def udp_loop():
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            try: sock.bind(('127.0.0.1', self.udp_port)); sock.settimeout(0.5)
            except: return
            while self.running:
                try:
                    data, _ = sock.recvfrom(2048); msg = json.loads(data.decode()); self.apply_udp_params(msg)
                except socket.timeout: continue
                except Exception as e: print(f"UDP Error: {e}")
        threading.Thread(target=udp_loop, daemon=True).start()

    def apply_udp_params(self, msg):
        if "invView" in msg: self.invView = int(msg["invView"])
        calib_updated = False
        if "pitch" in msg: self.pitch = float(msg["pitch"]); calib_updated = True
        if "tilt" in msg: self.tilt = float(msg["tilt"])
        if "center" in msg: self.center = float(msg["center"])
        if calib_updated: self.update_subp()
        if self.pipeline == "quilt":
            if "quiltFit" in msg: self.quilt_cfg.fit = int(msg["quiltFit"])
            if "flipRows" in msg: self.quilt_cfg.flip_rows = int(msg["flipRows"])
            if "debugFixedView" in msg: self.quilt_cfg.fixed_view = int(msg["debugFixedView"])
            if "quiltZoom" in msg: self.quilt_cfg.zoom = float(msg["quiltZoom"])
            if "overscan" in msg: self.quilt_cfg.overscan = float(msg["overscan"])
        else:
            if "depthiness" in msg: self.depthiness = float(msg["depthiness"])
            if "maxParallaxPx" in msg: self.maxParallaxPx = float(msg["maxParallaxPx"]); self.update_parallax_scale()
            if "focus" in msg: self.focus = float(msg["focus"])
            if "depthContrast" in msg: self.depthContrast = float(msg["depthContrast"])
            if "depthGamma" in msg: self.depthGamma = float(msg["depthGamma"])
            if "depthSmooth" in msg: self.depthSmooth = float(msg["depthSmooth"])
            if "edgeFade" in msg: self.edgeFade = float(msg["edgeFade"])
            if "depthLoc" in msg: self.depthLoc = int(msg["depthLoc"]); self.update_parallax_scale()
            if "invertDepth" in msg: self.invertDepth = int(msg["invertDepth"])
            if "debugMode" in msg: self.debugMode = int(msg["debugMode"])

    def run(self):
        if not glfw.init(): return
        monitors = glfw.get_monitors()
        if not monitors:
            print("[GLFW] No monitors found.")
            glfw.terminate(); return
        monitor_idx = max(0, min(self.args.monitor, len(monitors) - 1))
        monitor = monitors[monitor_idx]
        mode = glfw.get_video_mode(monitor)
        glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
        if self.args.windowed: self.window = glfw.create_window(1280, 720, "LKG Player", None, None)
        else:
            glfw.window_hint(glfw.DECORATED, glfw.FALSE); glfw.window_hint(glfw.AUTO_ICONIFY, glfw.FALSE)
            self.window = glfw.create_window(mode.size.width, mode.size.height, "LKG Player", monitor, None)
        if not self.window: glfw.terminate(); return
        glfw.make_context_current(self.window); glfw.swap_interval(1); glfw.show_window(self.window)
        fb_w, fb_h = glfw.get_framebuffer_size(self.window); GL.glViewport(0, 0, fb_w, fb_h)
        vs = shaders.compileShader(VERTEX_SHADER, GL.GL_VERTEX_SHADER)
        fs_rgbd = shaders.compileShader(FRAGMENT_SHADER, GL.GL_FRAGMENT_SHADER)
        fs_quilt = shaders.compileShader(QUILT_FRAGMENT_SHADER, GL.GL_FRAGMENT_SHADER)
        self.prog_rgbd = shaders.compileProgram(vs, fs_rgbd); self.prog_quilt = shaders.compileProgram(vs, fs_quilt)
        quad = np.array([-1,-1,0,0, 1,-1,1,0, 1,1,1,1, -1,-1,0,0, 1,1,1,1, -1,1,0,1], dtype=np.float32)
        vao = GL.glGenVertexArrays(1); GL.glBindVertexArray(vao); vbo = GL.glGenBuffers(1); GL.glBindBuffer(GL.GL_ARRAY_BUFFER, vbo)
        GL.glBufferData(GL.GL_ARRAY_BUFFER, quad.nbytes, quad, GL.GL_STATIC_DRAW)
        GL.glEnableVertexAttribArray(0); GL.glVertexAttribPointer(0, 2, GL.GL_FLOAT, False, 16, None)
        GL.glEnableVertexAttribArray(1); GL.glVertexAttribPointer(1, 2, GL.GL_FLOAT, False, 16, GL.ctypes.c_void_p(8))
        
        is_static = self.args.input.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))
        probe = subprocess.check_output(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height", "-of", "csv=p=0", self.args.input]).decode().strip()
        vw, vh = map(int, probe.split(','))
        
        print(f"[PIPELINE] {self.pipeline}")
        print(f"[INPUT] file={os.path.basename(self.args.input)} resolution={vw}x{vh} static={is_static}")
        if self.pipeline == "quilt":
            c = self.quilt_cfg; print(f"[QUILT] source={c.source} cols={c.cols} rows={c.rows} views={c.views} aspect={c.aspect} fit={c.fit} flipRows={c.flip_rows} fixedView={c.fixed_view} zoom={c.zoom:.3f} overscan={c.overscan:.3f}")
        else:
            print(f"[RGBD] depthLoc={self.depthLoc} maxParallaxPx={self.maxParallaxPx} parallaxScale={self.parallaxScale:.8f}")
        print(f"[CALIB] serial={self.serial} pitch={self.pitch:.4f} tilt={self.tilt:.4f} center={self.center:.4f} subp={self.subp:.8f}")

        tex = GL.glGenTextures(1); GL.glBindTexture(GL.GL_TEXTURE_2D, tex); f_mode = GL.GL_NEAREST if self.args.nearest else GL.GL_LINEAR
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, f_mode); GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, f_mode)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_CLAMP_TO_EDGE); GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_CLAMP_TO_EDGE)
        GL.glPixelStorei(GL.GL_UNPACK_ALIGNMENT, 1)
        
        proc = None
        if is_static:
            raw_frame = subprocess.run(["ffmpeg", "-i", self.args.input, "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1", "-loglevel", "quiet"], capture_output=True).stdout
            GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_RGB, vw, vh, 0, GL.GL_RGB, GL.GL_UNSIGNED_BYTE, raw_frame)
        else:
            cmd = ["ffmpeg", "-i", self.args.input, "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1", "-loglevel", "quiet"]
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=10**8)
        
        while not glfw.window_should_close(self.window) and self.running:
            if not is_static:
                raw_frame = proc.stdout.read(vw * vh * 3)
                if not raw_frame or len(raw_frame) < (vw * vh * 3):
                    if self.args.loop:
                        proc.terminate()
                        try: proc.wait(timeout=1.0)
                        except: proc.kill()
                        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=10**8); continue
                    else: break
                GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_RGB, vw, vh, 0, GL.GL_RGB, GL.GL_UNSIGNED_BYTE, raw_frame)
            prog = self.prog_quilt if self.pipeline == "quilt" else self.prog_rgbd
            GL.glUseProgram(prog); GL.glActiveTexture(GL.GL_TEXTURE0); GL.glBindTexture(GL.GL_TEXTURE_2D, tex)
            
            def set_f(name, val):
                loc = GL.glGetUniformLocation(prog, name)
                if loc != -1: GL.glUniform1f(loc, val)
            def set_i(name, val):
                loc = GL.glGetUniformLocation(prog, name)
                if loc != -1: GL.glUniform1i(loc, val)
            def set_vec2(name, x, y):
                loc = GL.glGetUniformLocation(prog, name)
                if loc != -1: GL.glUniform2f(loc, x, y)

            set_f("pitch", self.pitch); set_f("tilt", self.tilt); set_f("center", self.center); set_f("subp", self.subp); set_i("flipSubp", self.flipSubp); set_i("invView", self.invView)
            if self.pipeline == "quilt":
                set_i("texQuilt", 0); c = self.quilt_cfg; set_i("quiltCols", c.cols); set_i("quiltRows", c.rows); set_i("quiltViews", c.views); set_f("quiltAspect", c.aspect); set_f("inputAspect", float(vw)/float(vh)); set_i("quiltFit", c.fit); set_i("quiltFlipRows", c.flip_rows); set_i("debugFixedView", c.fixed_view); set_f("quiltZoom", c.zoom); set_f("overscan", c.overscan)
            else:
                set_i("texRGBD", 0); set_f("focus", self.focus); set_f("depthiness", self.depthiness); set_f("parallaxScale", self.parallaxScale); set_f("depthNear", self.depthNear); set_f("depthFar", self.depthFar); set_f("depthGamma", self.depthGamma); set_f("depthSmooth", self.depthSmooth); set_f("depthContrast", self.depthContrast); set_f("edgeFade", self.edgeFade); set_f("edgeLow", self.edgeLow); set_f("edgeHigh", self.edgeHigh); set_i("depthLoc", self.depthLoc); set_i("invertDepth", self.invertDepth); set_i("debugMode", self.debugMode); set_vec2("texelSize", 1.0/float(vw), 1.0/float(vh))
            GL.glClear(GL.GL_COLOR_BUFFER_BIT); GL.glDrawArrays(GL.GL_TRIANGLES, 0, 6)
            glfw.swap_buffers(self.window); glfw.poll_events()
        
        if proc:
            proc.terminate()
            try: proc.wait(timeout=1.0)
            except: proc.kill()
        glfw.terminate()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input"); parser.add_argument("--monitor", type=int, default=1); parser.add_argument("--pipeline", choices=["rgbd", "quilt"], default="rgbd")
    parser.add_argument("--quilt-cols", type=int); parser.add_argument("--quilt-rows", type=int); parser.add_argument("--quilt-views", type=int); parser.add_argument("--quilt-aspect", type=float)
    parser.add_argument("--debug-fixed-view", type=int, default=-1); parser.add_argument("--quilt-fit", choices=["stretch", "contain", "cover"], default="stretch"); parser.add_argument("--quilt-zoom", type=float, default=1.0); parser.add_argument("--overscan", type=float, default=0.0)
    parser.add_argument("--quilt-flip-rows", action="store_true", default=True); parser.add_argument("--no-quilt-flip-rows", dest="quilt_flip_rows", action="store_false")
    parser.add_argument("--focus", type=float, default=0.5); parser.add_argument("--depthiness", type=float, default=1.0); parser.add_argument("--max-parallax-px", type=float, default=3.0); parser.add_argument("--depth-loc", type=int, default=3); parser.add_argument("--inv-view", type=int, default=1); parser.add_argument("--windowed", action="store_true"); parser.add_argument("--calib-file"); parser.add_argument("--loop", action="store_true"); parser.add_argument("--nearest", action="store_true")
    LKGPlayer(parser.parse_args()).run()
