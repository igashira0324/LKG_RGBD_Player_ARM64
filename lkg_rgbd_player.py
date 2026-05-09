#!/usr/bin/env python3
"""
lkg_rgbd_player.py — Standalone RGBD video/image player for Looking Glass Go
Works on ARM64 (aarch64) WITHOUT the Bridge library.
Uses windowed mode with automatic positioning on the LKG display.
"""

import os
import cv2
import json
import socket
import argparse
import sys
import threading
import time
import getpass
import ctypes
import numpy as np
from OpenGL import GL, GLU
from OpenGL.GL import shaders
import glfw
import math
import subprocess

# --- Constants & Shaders ---

# Vertex shader: simple pass-through
VERTEX_SHADER = """
#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec2 aTexCoord;
out vec2 TexCoord;
void main() {
    gl_Position = vec4(aPos, 1.0);
    TexCoord = aTexCoord;
}
"""

# Fragment shader: Lenticular interleaving with backward warping
# Based on official Looking Glass reference implementation.
# Key: offset must be scaled by subp to stay within one lens period.
FRAGMENT_SHADER = """
#version 330 core
out vec4 FragColor;
in vec2 TexCoord;

uniform sampler2D texRGBD;
uniform float pitch;
uniform float tilt;
uniform float center;
uniform float subp;
uniform float focus;
uniform float depthiness;
uniform float parallaxScale;
uniform float depthNear;
uniform float depthFar;
uniform float depthGamma;
uniform float edgeFade;
uniform float edgeLow;
uniform float edgeHigh;
uniform float depthSmooth;
uniform vec2 texelSize;

uniform int depthLoc; // 0=top, 1=bottom, 2=left, 3=right
uniform int invView;
uniform int flipSubp;
uniform int invertDepth;
uniform int testPattern;
uniform int debugMode; // 0=Normal, 1=RGB Only, 2=Depth Only, 3=Offset Heatmap

float readDepth(vec2 uv) {
    float d = texture(texRGBD, uv).r;
    if (invertDepth == 1) d = 1.0 - d;
    return d;
}

float getSmoothDepth(vec2 uv) {
    float c = readDepth(uv);
    float l = readDepth(uv + vec2(-texelSize.x, 0.0));
    float r = readDepth(uv + vec2( texelSize.x, 0.0));
    float u = readDepth(uv + vec2(0.0, -texelSize.y));
    float b = readDepth(uv + vec2(0.0,  texelSize.y));

    float avg = (c * 4.0 + l + r + u + b) / 8.0;
    return mix(c, avg, depthSmooth);
}

void main() {
    vec3 color;
    
    if (testPattern == 1) {
        // High-fidelity test pattern using interleaving logic
        vec3 testColor;
        for (int i = 0; i < 3; i++) {
            float subpixelIndex = float(i);
            if (flipSubp == 1) subpixelIndex = 2.0 - subpixelIndex;

            float phase = (TexCoord.x + (1.0 - TexCoord.y) * tilt) * pitch - center;
            phase += subpixelIndex * subp;
            float v = fract(phase);
            if (invView == 1) v = 1.0 - v;

            float stripe = step(0.5, v);
            if (i == 0) testColor.r = stripe;
            else if (i == 1) testColor.g = stripe;
            else testColor.b = stripe;
        }
        FragColor = vec4(testColor, 1.0);
        return;
    }

    // Looking Glass Go interleaving loop
    for (int i = 0; i < 3; i++) {
        float subpixelIndex = float(i);
        if (flipSubp == 1) {
            subpixelIndex = 2.0 - subpixelIndex;
        }

        float phase = (TexCoord.x + (1.0 - TexCoord.y) * tilt) * pitch - center;
        phase += subpixelIndex * subp;
        
        // Calculate view index (normalized 0.0 to 1.0)
        float view = fract(phase);
        if (invView == 1) view = 1.0 - view;
        
        // Backward warping synthesis
        vec2 rgb_uv;
        vec2 depth_uv;
        
        // Flip Y for image sampling so it renders right-side up
        float sample_y = 1.0 - TexCoord.y;
        
        float rgb_min_x = 0.0;
        float rgb_max_x = 1.0;

        if (depthLoc == 2) { // Left-Right (Depth on Left)
            depth_uv = vec2(TexCoord.x * 0.5, sample_y);
            rgb_uv = vec2(0.5 + TexCoord.x * 0.5, sample_y);
            rgb_min_x = 0.5;
            rgb_max_x = 1.0;
        } else if (depthLoc == 3) { // Left-Right (Depth on Right)
            rgb_uv = vec2(TexCoord.x * 0.5, sample_y);
            depth_uv = vec2(0.5 + TexCoord.x * 0.5, sample_y);
            rgb_min_x = 0.0;
            rgb_max_x = 0.5;
        } else if (depthLoc == 0) { // Top-Bottom (Depth on Top)
            depth_uv = vec2(TexCoord.x, 0.5 + sample_y * 0.5);
            rgb_uv = vec2(TexCoord.x, sample_y * 0.5);
        } else { // Top-Bottom (Depth on Bottom)
            rgb_uv = vec2(TexCoord.x, 0.5 + sample_y * 0.5);
            depth_uv = vec2(TexCoord.x, sample_y * 0.5);
        }
        
        float d = getSmoothDepth(depth_uv);

        if (debugMode == 2) { // Depth Only
            FragColor = vec4(vec3(d), 1.0);
            return;
        }

        // --- Depth Remapping ---
        float nearVal = min(depthNear, depthFar - 0.001);
        float farVal = max(depthFar, depthNear + 0.001);
        d = clamp((d - nearVal) / max(farVal - nearVal, 0.0001), 0.0, 1.0);
        d = pow(d, depthGamma);

        // --- Edge Detection & Fading ---
        float depthGrad = abs(dFdx(d)) + abs(dFdy(d));
        float edge = smoothstep(edgeLow, edgeHigh, depthGrad);

        // --- Parallax Scaling ---
        float depthCentered = clamp((d - focus) * 2.0, -1.0, 1.0);
        float viewCentered = (view - 0.5) * 2.0;
        float offset = depthCentered * viewCentered * depthiness * parallaxScale;

        // Reduce offset at edges to hide artifacts
        offset *= mix(1.0, 1.0 - edgeFade, edge);
        
        if (debugMode == 3) { // Offset Heatmap
            float o = abs(offset) / max(parallaxScale, 0.0001);
            FragColor = vec4(o, 0.0, 1.0 - o, 1.0);
            return;
        }

        vec2 warped_uv = rgb_uv + vec2(offset, 0.0);
        if (debugMode == 1) warped_uv = rgb_uv; // RGB Only (2D)
        
        warped_uv.x = clamp(warped_uv.x, rgb_min_x, rgb_max_x);
        
        if (i == 0) color.r = texture(texRGBD, warped_uv).r;
        else if (i == 1) color.g = texture(texRGBD, warped_uv).g;
        else color.b = texture(texRGBD, warped_uv).b;
    }
    FragColor = vec4(color, 1.0);
}
"""

class LKGPlayer:
    def __init__(self, args):
        self.args = args
        self.window = None
        self.shader_program = None
        self.texture = None
        self.video_proc = None
        self.start_time = 0
        self.running = True
        self.calib = {}
        
        self.focus = args.focus
        self.depthiness = args.depthiness
        self.maxParallaxPx = args.max_parallax_px
        self.parallaxScale = 0.02
        self.invView = args.inv_view
        self.depthNear = 0.05
        self.depthFar = 0.95
        self.depthGamma = 1.2
        self.depthSmooth = 0.5
        self.edgeFade = 0.8
        self.edgeLow = 0.02
        self.edgeHigh = 0.10
        self.depthLoc = args.depth_loc
        self.udp_port = 5000 + args.monitor
        self.debugMode = 0
        
        # Calibration defaults for LKG Go (Shader coordinates)
        self.pitch = 143.6
        self.tilt = -0.324
        self.center = 0.0
        self.subp = self.pitch / (1440.0 * 3.0)
        self.flipSubp = 0
        self.invertDepth = 0
        self.testPattern = 0
        
        self.load_calibration()
        self.start_udp_listener()

    def start_udp_listener(self):
        def udp_loop():
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            try:
                sock.bind(('127.0.0.1', self.udp_port))
            except OSError as e:
                print(f"ERROR: UDP port {self.udp_port} is already in use: {e}")
                return
            sock.settimeout(0.5)
            print(f"UDP listener started on port {self.udp_port}")
            while self.running:
                try:
                    data, addr = sock.recvfrom(1024)
                    msg = json.loads(data.decode())
                    if 'focus' in msg:
                        self.focus = float(msg['focus'])
                    if 'depthiness' in msg:
                        self.depthiness = float(msg['depthiness'])
                    if 'invView' in msg:
                        self.invView = int(msg['invView'])
                    if 'depthLoc' in msg:
                        self.depthLoc = int(msg['depthLoc'])
                    if 'pitch' in msg:
                        self.pitch = float(msg['pitch'])
                    if 'tilt' in msg:
                        self.tilt = float(msg['tilt'])
                    if 'center' in msg:
                        self.center = float(msg['center'])
                    if 'flipSubp' in msg:
                        self.flipSubp = int(msg['flipSubp'])
                    if 'invertDepth' in msg:
                        self.invertDepth = int(msg['invertDepth'])
                    if 'testPattern' in msg:
                        self.testPattern = int(msg['testPattern'])
                    if 'maxParallaxPx' in msg:
                        self.maxParallaxPx = float(msg['maxParallaxPx'])
                        if hasattr(self, 'frame_w') and self.frame_w > 0:
                            self.parallaxScale = self.maxParallaxPx / float(self.frame_w)
                    if 'depthNear' in msg:
                        self.depthNear = float(msg['depthNear'])
                    if 'depthFar' in msg:
                        self.depthFar = float(msg['depthFar'])
                    if 'depthGamma' in msg:
                        self.depthGamma = float(msg['depthGamma'])
                    if 'edgeFade' in msg:
                        self.edgeFade = float(msg['edgeFade'])
                    if 'depthSmooth' in msg:
                        self.depthSmooth = float(msg['depthSmooth'])
                    if 'debugMode' in msg:
                        self.debugMode = int(msg['debugMode'])
                    
                    self.print_runtime_params("UDP")
                except socket.timeout:
                    continue
                except Exception as e:
                    if self.running:
                        print(f"UDP Error: {e}")
            sock.close()
        
        t = threading.Thread(target=udp_loop, daemon=True)
        t.start()
    
    def print_runtime_params(self, prefix="Runtime"):
        print(f"[{prefix}] Mode={self.debugMode} Depthiness={self.depthiness:.2f} MaxParallax={self.maxParallaxPx:.2f} PScale={self.parallaxScale:.5f}")
        print(f"[{prefix}] Near={self.depthNear:.2f} Far={self.depthFar:.2f} Gamma={self.depthGamma:.2f} Smooth={self.depthSmooth:.2f} EdgeFade={self.edgeFade:.2f}")

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
                    # Case 1: drive/LKG_calibration/visual.json
                    full_path = os.path.join(drive_path, "LKG_calibration", "visual.json")
                    if os.path.exists(full_path): return full_path
                    # Case 2: drive/visual.json
                    full_path = os.path.join(drive_path, "visual.json")
                    if os.path.exists(full_path): return full_path
                    
                    # Also check one level deeper in case of user-specific mount points
                    if os.path.isdir(drive_path):
                        for sub in os.listdir(drive_path):
                            full_path = os.path.join(drive_path, sub, "LKG_calibration", "visual.json")
                            if os.path.exists(full_path): return full_path
            except:
                continue
        return None

    def load_calibration(self):
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
            print(f"Using factory calibration: {factory_path}")
        elif self.args.calib_file and os.path.exists(self.args.calib_file):
            calib_file = self.args.calib_file
            print(f"Using specified calibration: {calib_file}")
        else:
            calib_file = override_path
            print(f"Using fallback calibration: {calib_file}")

        calib_data = load_json_if_exists(calib_file)
        override_data = load_json_if_exists(override_path)

        # Base calibration
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
        self.pitch = raw_pitch * screen_inches * math.cos(math.atan(1.0 / raw_slope))
        self.tilt = screen_h / (screen_w * raw_slope)
        self.center = raw_center
        self.subp = self.pitch / (3.0 * screen_w)
        
        # Runtime Overrides
        overrides = override_data.get('runtimeOverride', {})
        self.pitch += float(overrides.get("pitchOffset", 0.0))
        self.tilt += float(overrides.get("tiltOffset", 0.0))
        self.center += float(overrides.get("centerOffset", 0.0))

        self.maxParallaxPx = float(overrides.get("maxParallaxPx", self.maxParallaxPx))
        self.depthNear = float(overrides.get("depthNear", self.depthNear))
        self.depthFar = float(overrides.get("depthFar", self.depthFar))
        self.depthGamma = float(overrides.get("depthGamma", self.depthGamma))
        self.depthSmooth = float(overrides.get("depthSmooth", self.depthSmooth))
        self.edgeFade = float(overrides.get("edgeFade", self.edgeFade))
        
        print(f"Loaded combined calibration. Overrides: {bool(overrides)}")

    def find_lkg_monitors(self):
        """Find all LKG Go monitors via xrandr."""
        monitors = []
        try:
            result = subprocess.check_output(['xrandr', '--current'], text=True)
            for line in result.splitlines():
                if '1440x2560' in line and 'connected' in line:
                    import re
                    m = re.search(r'(\d+)x(\d+)\+(\d+)\+(\d+)', line)
                    if m:
                        monitors.append((int(m.group(3)), int(m.group(4)), int(m.group(1)), int(m.group(2))))
        except:
            pass
        return monitors

    def init_glfw(self):
        if not glfw.init():
            return False
        
        # UDP port is already set in __init__ based on monitor index
        # self.udp_port = 5000 + self.args.monitor
        # self.start_udp_listener() # Already started in __init__

        lkg_monitors = self.find_lkg_monitors()
        lkg_pos = None
        
        if len(lkg_monitors) > 0:
            # Map monitor index (1-based) to the found monitors
            idx = max(0, self.args.monitor - 1)
            if idx < len(lkg_monitors):
                lkg_pos = lkg_monitors[idx]
            else:
                lkg_pos = lkg_monitors[0]
                
        if lkg_pos:
            x, y, w, h = lkg_pos
            print(f"LKG Go display for monitor {self.args.monitor} at position ({x},{y}) size {w}x{h}")
            
            glfw.window_hint(glfw.DECORATED, glfw.FALSE)
            glfw.window_hint(glfw.FLOATING, glfw.TRUE)
            self.window = glfw.create_window(w, h, f"LKG Player {self.args.monitor}", None, None)
            if self.window:
                glfw.set_window_pos(self.window, x, y)
        else:
            monitors = glfw.get_monitors()
            target_mon = monitors[0]
            
            for i, m in enumerate(monitors):
                mode = glfw.get_video_mode(m)
                if mode.size.width == 1440 and mode.size.height == 2560:
                    target_mon = m
                    break
            
            if self.args.windowed:
                self.window = glfw.create_window(1152, 1024, f"LKG Player {self.args.monitor}", None, None)
            else:
                glfw.window_hint(glfw.DECORATED, glfw.FALSE)
                vid_mode = glfw.get_video_mode(target_mon)
                self.window = glfw.create_window(vid_mode.size.width, vid_mode.size.height, f"LKG Player {self.args.monitor}", None, None)
                pos_x, pos_y = glfw.get_monitor_pos(target_mon)
                glfw.set_window_pos(self.window, pos_x, pos_y)
        
        if not self.window:
            glfw.terminate()
            return False
            
        glfw.make_context_current(self.window)
        glfw.swap_interval(1)
        return True

    def init_shaders(self):
        vs = shaders.compileShader(VERTEX_SHADER, GL.GL_VERTEX_SHADER)
        fs = shaders.compileShader(FRAGMENT_SHADER, GL.GL_FRAGMENT_SHADER)
        self.shader_program = shaders.compileProgram(vs, fs)
        
        # Triangles for rendering
        self.vbo = GL.glGenBuffers(1)
        vertices = np.array([
            -1.0, -1.0, 0.0,  0.0, 0.0,
             1.0, -1.0, 0.0,  1.0, 0.0,
             1.0,  1.0, 0.0,  1.0, 1.0,

            -1.0, -1.0, 0.0,  0.0, 0.0,
             1.0,  1.0, 0.0,  1.0, 1.0,
            -1.0,  1.0, 0.0,  0.0, 1.0,
        ], dtype=np.float32)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.vbo)
        GL.glBufferData(GL.GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL.GL_STATIC_DRAW)
        
        self.vao = GL.glGenVertexArrays(1)
        GL.glBindVertexArray(self.vao)
        GL.glVertexAttribPointer(0, 3, GL.GL_FLOAT, GL.GL_FALSE, 20, ctypes.c_void_p(0))
        GL.glEnableVertexAttribArray(0)
        GL.glVertexAttribPointer(1, 2, GL.GL_FLOAT, GL.GL_FALSE, 20, ctypes.c_void_p(12))
        GL.glEnableVertexAttribArray(1)

    def get_video_info(self, path):
        try:
            cmd = [
                'ffprobe', '-v', 'error',
                '-select_streams', 'v:0',
                '-show_entries', 'stream=width,height,r_frame_rate,avg_frame_rate',
                '-of', 'json',
                path
            ]
            out = subprocess.check_output(cmd)
            info = json.loads(out)
            width = float(info['streams'][0]['width'])
            height = float(info['streams'][0]['height'])
            
            # Get FPS if possible
            fps = 30.0
            rate = info['streams'][0].get('avg_frame_rate') or info['streams'][0].get('r_frame_rate', '30/1')
            if '/' in rate:
                num, den = rate.split('/')
                if float(den) > 0:
                    fps = float(num) / float(den)
            
            return int(width), int(height), fps
        except:
            return 2304, 1024, 30.0

    def start_video(self, path):
        if self.video_proc:
            try:
                self.video_proc.terminate()
                self.video_proc.wait(timeout=2)
            except:
                pass
        cmd = [
            'ffmpeg', '-re', '-i', path,
            '-f', 'rawvideo', '-pix_fmt', 'rgb24',
            'pipe:1'
        ]
        self.video_proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        
        if self.args.with_audio:
            audio_cmd = [
                'ffplay', '-nodisp', '-autoexit', '-af', f'volume={self.args.volume}',
                '-ss', '0', 
                path
            ]
            env = os.environ.copy()
            env['AUDIODEV'] = self.args.audio_device
            subprocess.Popen(audio_cmd, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def run(self):
        if not self.init_glfw():
            print("ERROR: Failed to initialize GLFW window")
            return
        self.init_shaders()
        
        # Sync logic
        if self.args.wait_trigger:
            print(f"Waiting for trigger: {self.args.wait_trigger}")
            while not os.path.exists(self.args.wait_trigger):
                time.sleep(0.1)
                glfw.poll_events()
            print("Trigger received!")
            
        # Frame buffer info
        input_lower = self.args.input.lower()
        is_static = input_lower.endswith(('.jpg', '.jpeg', '.png'))
        
        if is_static:
            img = cv2.imread(self.args.input)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            self.frame_h, self.frame_w, _ = img.shape
            raw_frame = img.tobytes()
            fps = 30.0
            self.parallaxScale = self.maxParallaxPx / float(self.frame_w)
        else:
            self.frame_w, self.frame_h, fps = self.get_video_info(self.args.input)
            print(f"Video resolution: {self.frame_w}x{self.frame_h} @ {fps:.2f} FPS")
            self.parallaxScale = self.maxParallaxPx / float(self.frame_w)
            self.print_runtime_params("STARTUP")
            self.start_video(self.args.input)

        frame_w, frame_h = self.frame_w, self.frame_h

        self.start_time = time.time()
        self.frame_index = 0

        self.texture = GL.glGenTextures(1)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self.texture)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_CLAMP_TO_EDGE)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_CLAMP_TO_EDGE)
        
        # Initial texture allocation
        GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_RGB, frame_w, frame_h, 0, GL.GL_RGB, GL.GL_UNSIGNED_BYTE, None)

        frame_size = frame_w * frame_h * 3
        if not is_static:
            raw_frame = None

        while not glfw.window_should_close(self.window) and self.running:
            # Ensure OpenGL context is current (critical for multi-window setups)
            glfw.make_context_current(self.window)
            
            if not is_static and self.video_proc:
                new_frame = self.video_proc.stdout.read(frame_size)
                if not new_frame or len(new_frame) < frame_size:
                    if self.args.loop:
                        if self.args.done_signal:
                            open(self.args.done_signal, 'a').close()
                        if self.args.wait_trigger:
                            self.video_proc.terminate()
                            while not os.path.exists(self.args.wait_trigger):
                                time.sleep(0.01)
                                glfw.poll_events()
                        self.start_video(self.args.input)
                        self.start_time = time.time()
                        self.frame_index = 0
                        continue
                    else:
                        break
                raw_frame = new_frame
            
            if raw_frame is None:
                glfw.poll_events()
                continue
            
            GL.glClear(GL.GL_COLOR_BUFFER_BIT)
            GL.glUseProgram(self.shader_program)
            
            # Update texture using SubImage2D to avoid reallocation
            GL.glBindTexture(GL.GL_TEXTURE_2D, self.texture)
            if raw_frame is not None:
                GL.glTexSubImage2D(GL.GL_TEXTURE_2D, 0, 0, 0, frame_w, frame_h, GL.GL_RGB, GL.GL_UNSIGNED_BYTE, raw_frame)
            
            # Set uniforms
            GL.glUniform1f(GL.glGetUniformLocation(self.shader_program, "pitch"), self.pitch)
            GL.glUniform1f(GL.glGetUniformLocation(self.shader_program, "tilt"), self.tilt)
            GL.glUniform1f(GL.glGetUniformLocation(self.shader_program, "center"), self.center)
            GL.glUniform1f(GL.glGetUniformLocation(self.shader_program, "subp"), self.subp)
            GL.glUniform1f(GL.glGetUniformLocation(self.shader_program, "focus"), self.focus)
            GL.glUniform1f(GL.glGetUniformLocation(self.shader_program, "depthiness"), self.depthiness)
            GL.glUniform1f(GL.glGetUniformLocation(self.shader_program, "parallaxScale"), self.parallaxScale)
            GL.glUniform1f(GL.glGetUniformLocation(self.shader_program, "depthNear"), self.depthNear)
            GL.glUniform1f(GL.glGetUniformLocation(self.shader_program, "depthFar"), self.depthFar)
            GL.glUniform1f(GL.glGetUniformLocation(self.shader_program, "depthGamma"), self.depthGamma)
            GL.glUniform1f(GL.glGetUniformLocation(self.shader_program, "depthSmooth"), self.depthSmooth)
            GL.glUniform2f(GL.glGetUniformLocation(self.shader_program, "texelSize"), 1.0/float(frame_w), 1.0/float(frame_h))
            GL.glUniform1f(GL.glGetUniformLocation(self.shader_program, "edgeFade"), self.edgeFade)
            GL.glUniform1f(GL.glGetUniformLocation(self.shader_program, "edgeLow"), self.edgeLow)
            GL.glUniform1f(GL.glGetUniformLocation(self.shader_program, "edgeHigh"), self.edgeHigh)
            GL.glUniform1i(GL.glGetUniformLocation(self.shader_program, "depthLoc"), self.depthLoc)
            GL.glUniform1i(GL.glGetUniformLocation(self.shader_program, "invView"), self.invView)
            GL.glUniform1i(GL.glGetUniformLocation(self.shader_program, "flipSubp"), self.flipSubp)
            GL.glUniform1i(GL.glGetUniformLocation(self.shader_program, "invertDepth"), self.invertDepth)
            GL.glUniform1i(GL.glGetUniformLocation(self.shader_program, "testPattern"), self.testPattern)
            GL.glUniform1i(GL.glGetUniformLocation(self.shader_program, "debugMode"), self.debugMode)
            
            GL.glBindVertexArray(self.vao)
            GL.glDrawArrays(GL.GL_TRIANGLES, 0, 6)
            
            glfw.swap_buffers(self.window)
            
            # Optional: precise Python-side FPS sync (ffmpeg -re helps, but this is stricter)
            if not is_static and fps > 0:
                self.frame_index += 1
                target_time = self.frame_index / fps
                sleep_time = self.start_time + target_time - time.time()
                if sleep_time > 0:
                    time.sleep(sleep_time)
            
            glfw.poll_events()
            
        self.running = False
        if self.video_proc:
            self.video_proc.terminate()
        glfw.terminate()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("--monitor", type=int, default=1)
    parser.add_argument("--windowed", action="store_true")
    parser.add_argument("--calib-file")
    parser.add_argument("--focus", type=float, default=0.5)
    parser.add_argument("--depthiness", type=float, default=1.0)
    parser.add_argument("--max-parallax-px", type=float, default=2.5)
    parser.add_argument("--depth-loc", type=int, default=3) # 2=Left, 3=Right (ComfyUI outputs Depth on Right)
    parser.add_argument("--inv-view", type=int, default=1)
    parser.add_argument("--wait-trigger")
    parser.add_argument("--done-signal")
    parser.add_argument("--loop", action="store_true")
    parser.add_argument("--with-audio", action="store_true")
    parser.add_argument("--audio-device", default="plughw:0,8")
    parser.add_argument("--volume", type=float, default=0.2)
    
    args = parser.parse_args()
    player = LKGPlayer(args)
    player.run()
