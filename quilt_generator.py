import numpy as np
from OpenGL import GL
from OpenGL.GL import shaders

QUILT_VS = """
#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec2 aTexCoord;

out vec2 TexCoord;
out float ViewOffset;

uniform int cols;
uniform int rows;
uniform float depthiness;
uniform float parallaxScale;
uniform float focus;

void main() {
    int totalViews = cols * rows;
    int viewIndex = gl_InstanceID;
    
    // Calculate tile position (0,0 is bottom-left)
    int col = viewIndex % cols;
    int row = viewIndex / cols;
    
    // Normalized tile coordinates
    float tw = 1.0 / float(cols);
    float th = 1.0 / float(rows);
    
    // Map quad to tile
    vec2 pos = aPos.xy * 0.5 + 0.5; // 0..1
    pos.x = (pos.x + float(col)) * tw;
    pos.y = (pos.y + float(row)) * th;
    
    gl_Position = vec4(pos * 2.0 - 1.0, 0.0, 1.0);
    TexCoord = aTexCoord;
    
    // View index normalized to -1..1 for parallax
    ViewOffset = (float(viewIndex) / float(totalViews - 1) - 0.5) * 2.0;
}
"""

QUILT_FS = """
#version 330 core
out vec4 FragColor;

in vec2 TexCoord;
in float ViewOffset;

uniform sampler2D texRGBD;
uniform int depthLoc; // 2=Left, 3=Right, 0=Top, 1=Bottom
uniform float depthiness;
uniform float parallaxScale;
uniform float focus;
uniform float depthContrast;
uniform float depthGamma;
uniform int invertDepth;
uniform float quiltAspect;
uniform float inputAspect;

float readDepth(vec2 uv) {
    float d = texture(texRGBD, uv).r;
    if (invertDepth == 1) d = 1.0 - d;
    return d;
}

// Phase 2: High Quality Depth Processing
uniform float depthSmooth;
uniform float edgeFade;

float getBilateralDepth(vec2 uv) {
    float d = readDepth(uv);
    float sum = 1.0;
    float weighted_d = d;
    
    // Simple 4-tap bilateral filter
    vec2 off = 1.0 / textureSize(texRGBD, 0).xy;
    vec2 steps[4] = vec2[](vec2(1,0), vec2(-1,0), vec2(0,1), vec2(0,-1));
    
    for(int i=0; i<4; i++) {
        vec2 sample_uv = uv + steps[i] * off * 2.0;
        float val = readDepth(sample_uv);
        float weight = exp(-abs(val - d) * 10.0);
        weighted_d += val * weight;
        sum += weight;
    }
    return weighted_d / sum;
}

float getDilatedDepth(vec2 uv) {
    float max_d = readDepth(uv);
    vec2 off = 1.0 / textureSize(texRGBD, 0).xy;
    for(int x=-1; x<=1; x++) {
        for(int y=-1; y<=1; y++) {
            max_d = max(max_d, readDepth(uv + vec2(x,y) * off * 2.0));
        }
    }
    return max_d;
}

void main() {

    // Correct aspect ratio within the tile
    vec2 tileUV = TexCoord;
    float targetAspect = quiltAspect; // e.g., 0.5625
    float srcAspect = inputAspect;    // e.g., 1.0 (assuming SBS RGBD is square-ish)
    
    // Simple center crop logic
    if (srcAspect > targetAspect) {
        float scale = targetAspect / srcAspect;
        tileUV.x = (tileUV.x - 0.5) * scale + 0.5;
    } else {
        float scale = srcAspect / targetAspect;
        tileUV.y = (tileUV.y - 0.5) * scale + 0.5;
    }

    vec2 rgb_uv;
    vec2 depth_uv;
    float sample_y = 1.0 - tileUV.y;

    float rgb_min_x = 0.0, rgb_max_x = 1.0;

    if (depthLoc == 2) { // Left-Right (Depth on Left)
        depth_uv = vec2(tileUV.x * 0.5, sample_y);
        rgb_uv = vec2(0.5 + tileUV.x * 0.5, sample_y);
        rgb_min_x = 0.5; rgb_max_x = 1.0;
    } else if (depthLoc == 3) { // Left-Right (Depth on Right)
        rgb_uv = vec2(tileUV.x * 0.5, sample_y);
        depth_uv = vec2(0.5 + tileUV.x * 0.5, sample_y);
        rgb_min_x = 0.0; rgb_max_x = 0.5;
    } else if (depthLoc == 0) { // Top-Bottom (Depth on Top)
        depth_uv = vec2(tileUV.x, 0.5 + sample_y * 0.5);
        rgb_uv = vec2(tileUV.x, sample_y * 0.5);
    } else { // Top-Bottom (Depth on Bottom)
        rgb_uv = vec2(tileUV.x, 0.5 + sample_y * 0.5);
        depth_uv = vec2(tileUV.x, sample_y * 0.5);
    }

    float smoothedDepth = getBilateralDepth(depth_uv);
    float dilatedDepth = getDilatedDepth(depth_uv);
    
    float d = smoothedDepth;
    d = clamp(focus + (d - focus) * depthContrast, 0.0, 1.0);
    d = pow(d, depthGamma);
    
    // Edge Fading
    float depthGrad = abs(dFdx(d)) + abs(dFdy(d));
    float edge = smoothstep(0.02, 0.10, depthGrad);
    float edgeAtten = mix(1.0, 1.0 - (edgeFade * 0.5), edge);

    float finalDepth = mix(smoothedDepth, dilatedDepth, 0.5 * depthSmooth);
    float depthCentered = (finalDepth - focus) * 2.0;
    
    float offset = depthCentered * ViewOffset * depthiness * parallaxScale * edgeAtten;
    
    vec2 warped_uv = rgb_uv + vec2(offset, 0.0);
    warped_uv.x = clamp(warped_uv.x, rgb_min_x, rgb_max_x);
    
    FragColor = vec4(texture(texRGBD, warped_uv).rgb, 1.0);
}

"""

class QuiltGenerator:
    def __init__(self, cols=8, rows=6, quilt_res=4092):
        self.cols = cols
        self.rows = rows
        self.quilt_res = quilt_res
        self.fbo = None
        self.texture = None
        self.shader = None
        
    def init_gl(self):
        # Shaders
        vs = shaders.compileShader(QUILT_VS, GL.GL_VERTEX_SHADER)
        fs = shaders.compileShader(QUILT_FS, GL.GL_FRAGMENT_SHADER)
        self.shader = shaders.compileProgram(vs, fs)
        
        # FBO
        self.fbo = GL.glGenFramebuffers(1)
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self.fbo)
        
        self.texture = GL.glGenTextures(1)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self.texture)
        GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_RGB, self.quilt_res, self.quilt_res, 0, GL.GL_RGB, GL.GL_UNSIGNED_BYTE, None)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_CLAMP_TO_EDGE)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_CLAMP_TO_EDGE)
        
        GL.glFramebufferTexture2D(GL.GL_FRAMEBUFFER, GL.GL_COLOR_ATTACHMENT0, GL.GL_TEXTURE_2D, self.texture, 0)
        
        if GL.glCheckFramebufferStatus(GL.GL_FRAMEBUFFER) != GL.GL_FRAMEBUFFER_COMPLETE:
            print("ERROR: Quilt FBO incomplete")
            
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)
        
    def generate(self, src_tex, params):
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self.fbo)
        GL.glViewport(0, 0, self.quilt_res, self.quilt_res)
        GL.glClear(GL.GL_COLOR_BUFFER_BIT)
        
        GL.glUseProgram(self.shader)
        
        GL.glActiveTexture(GL.GL_TEXTURE0)
        GL.glBindTexture(GL.GL_TEXTURE_2D, src_tex)
        GL.glUniform1i(GL.glGetUniformLocation(self.shader, "texRGBD"), 0)
        
        GL.glUniform1i(GL.glGetUniformLocation(self.shader, "cols"), self.cols)
        GL.glUniform1i(GL.glGetUniformLocation(self.shader, "rows"), self.rows)
        GL.glUniform1f(GL.glGetUniformLocation(self.shader, "depthiness"), params.get('depthiness', 1.0))
        GL.glUniform1f(GL.glGetUniformLocation(self.shader, "parallaxScale"), params.get('parallaxScale', 0.002))
        GL.glUniform1f(GL.glGetUniformLocation(self.shader, "focus"), params.get('focus', 0.5))
        GL.glUniform1f(GL.glGetUniformLocation(self.shader, "depthContrast"), params.get('depthContrast', 1.2))
        GL.glUniform1f(GL.glGetUniformLocation(self.shader, "depthGamma"), params.get('depthGamma', 1.0))
        GL.glUniform1f(GL.glGetUniformLocation(self.shader, "depthSmooth"), params.get('depthSmooth', 0.5))
        GL.glUniform1f(GL.glGetUniformLocation(self.shader, "edgeFade"), params.get('edgeFade', 0.8))
        GL.glUniform1i(GL.glGetUniformLocation(self.shader, "depthLoc"), params.get('depthLoc', 3))
        GL.glUniform1i(GL.glGetUniformLocation(self.shader, "invertDepth"), params.get('invertDepth', 0))
        GL.glUniform1f(GL.glGetUniformLocation(self.shader, "quiltAspect"), params.get('quiltAspect', 0.5625))
        GL.glUniform1f(GL.glGetUniformLocation(self.shader, "inputAspect"), params.get('inputAspect', 1.0))

        
        # Draw instanced (tiles)
        GL.glDrawArraysInstanced(GL.GL_TRIANGLES, 0, 6, self.cols * self.rows)
        
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)
        return self.texture

