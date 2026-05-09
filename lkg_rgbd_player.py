#!/usr/bin/env python3
"""
lkg_rgbd_player.py — Standalone RGBD video/image player for Looking Glass Go
Works on ARM64 (aarch64) WITHOUT the Bridge library.
Uses windowed mode with automatic positioning on the LKG display.
"""

import os
import sys
import json
import time
import argparse
import subprocess
import threading
import socket
import ctypes
import numpy as np
from OpenGL import GL, GLU
from OpenGL.GL import shaders
import glfw
import math
import cv2

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
uniform int depthLoc; // 0=top, 1=bottom, 2=left, 3=right
uniform int invView;
uniform int flipSubp;
uniform int invertDepth;
uniform int testPattern;

void main() {
    vec3 color;
    
    if (testPattern == 1) {
        // Simple stripe test pattern for pitch calibration
        float phase = TexCoord.x * pitch - center;
        float stripe = step(0.5, fract(phase));
        FragColor = vec4(vec3(stripe), 1.0);
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
        
        float depth = texture(texRGBD, depth_uv).r;
        if (invertDepth == 1) {
            depth = 1.0 - depth;
        }

        float offset = (depth - focus) * depthiness * (view - 0.5);
        
        vec2 warped_uv = rgb_uv + vec2(offset, 0.0);
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
        
        # Real-time control parameters
        self.focus = args.focus
        self.depthiness = args.depthiness
        self.invView = args.inv_view
        self.depthLoc = args.depth_loc
        self.udp_port = 5005
        
        # Calibration defaults for LKG Go (Shader coordinates)
        self.pitch = 143.6
        self.tilt = -0.324
        self.center = 0.0
        self.subp = self.pitch / (1440.0 * 3.0)
        self.flipSubp = 0
        self.invertDepth = 0
        self.testPattern = 0
        
        self.load_calibration()


    def start_udp_listener(self):
        def udp_loop():
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
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
                except socket.timeout:
                    continue
                except Exception as e:
                    if self.running:
                        print(f"UDP Error: {e}")
            sock.close()
        
        t = threading.Thread(target=udp_loop, daemon=True)
        t.start()

    def load_calibration(self):
        calib_file = self.args.calib_file
        
        if not calib_file:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            default_path = os.path.join(script_dir, "lkg_calibration.json")
            if os.path.exists(default_path):
                calib_file = default_path
        
        if calib_file and os.path.exists(calib_file):
            try:
                with open(calib_file, 'r') as f:
                    data = json.load(f)
                    
                    config = data.get('configValue', {})
                    raw_pitch = float(config.get("pitch", {}).get("value", 49.818))
                    raw_slope = float(config.get("slope", {}).get("value", -5.48))
                    raw_center = float(config.get("center", {}).get("value", 0.157))
                    screen_w = float(config.get("screenW", {}).get("value", 1440.0))
                    screen_h = float(config.get("screenH", {}).get("value", 2560.0))
                    dpi = float(config.get("DPI", {}).get("value", 491.0))
                    
                    self.invView = int(config.get("invView", {}).get("value", self.invView))
                    self.flipSubp = int(config.get("flipSubp", {}).get("value", self.flipSubp))
                    
                    screen_inches = screen_w / dpi
                    self.pitch = raw_pitch * screen_inches * math.cos(math.atan(1.0 / raw_slope))
                    self.tilt = screen_h / (screen_w * raw_slope)
                    self.center = raw_center
                    self.subp = self.pitch / (3.0 * screen_w)
                    
                    # Apply runtime overrides if any
                    overrides = data.get('runtimeOverride', {})
                    self.pitch += float(overrides.get("pitchOffset", 0.0))
                    self.tilt += float(overrides.get("tiltOffset", 0.0))
                    self.center += float(overrides.get("centerOffset", 0.0))
                    
                    print(f"Loaded calibration from {calib_file}")
                    print(f"  shader_pitch={self.pitch:.3f}, shader_tilt={self.tilt:.3f}, center={self.center:.3f}, invView={self.invView}")
            except Exception as e:
                print(f"Warning: Failed to load calibration: {e}")
        else:
            print("No calibration file found, using defaults")
            print(f"  shader_pitch={self.pitch:.3f}, shader_tilt={self.tilt:.3f}, center={self.center:.3f}")

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
        
        # Override UDP port based on monitor index so multiple instances don't clash
        self.udp_port = 5000 + self.args.monitor
        self.start_udp_listener()

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
            cmd = ['ffprobe', '-v', 'error', '-select_streams', 'v:0', 
                   '-show_entries', 'stream=width,height', '-of', 'json', path]
            out = subprocess.check_output(cmd)
            info = json.loads(out)
            width = float(info['streams'][0]['width'])
            height = float(info['streams'][0]['height'])
            
            # Get FPS if possible
            fps = 30.0
            r_frame_rate = info['streams'][0].get('r_frame_rate', '30/1')
            if '/' in r_frame_rate:
                num, den = r_frame_rate.split('/')
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
            frame_h, frame_w, _ = img.shape
            raw_frame = img.tobytes()
            fps = 30.0
        else:
            frame_w, frame_h, fps = self.get_video_info(self.args.input)
            print(f"Video resolution: {frame_w}x{frame_h} @ {fps:.2f} FPS")
            self.start_video(self.args.input)

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
            GL.glUniform1i(GL.glGetUniformLocation(self.shader_program, "depthLoc"), self.depthLoc)
            GL.glUniform1i(GL.glGetUniformLocation(self.shader_program, "invView"), self.invView)
            GL.glUniform1i(GL.glGetUniformLocation(self.shader_program, "flipSubp"), self.flipSubp)
            GL.glUniform1i(GL.glGetUniformLocation(self.shader_program, "invertDepth"), self.invertDepth)
            GL.glUniform1i(GL.glGetUniformLocation(self.shader_program, "testPattern"), self.testPattern)
            
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
