"""
GLSL shaders for the GPU renderer (OpenGL 3.3 core, works on macOS).

Conventions: every pass draws one full-screen triangle. Texture coordinates
are in "image space": v = 0 is the TOP row of the image (the way pygame
stores pixels). Only the final passes that write to the window flip v.
"""

FULLSCREEN_VS = """
#version 330
out vec2 v_uv;
void main() {
    vec2 pos = vec2(float((gl_VertexID << 1) & 2), float(gl_VertexID & 2));
    v_uv = pos;
    gl_Position = vec4(pos * 2.0 - 1.0, 0.0, 1.0);
}
"""

# ---------------------------------------------------------------------------
# Screen-space ambient occlusion (2D). Occluders come from two masks:
# the static arena walls/floor and the moving cars/ball. Each pixel samples
# a golden-angle spiral disk; occluders above a pixel count more because the
# arena is lit from the ceiling (directional occlusion).
# ---------------------------------------------------------------------------
AO_FS = """
#version 330
uniform sampler2D u_static;
uniform sampler2D u_dynamic;
uniform vec2 u_texel;
uniform float u_radius;
uniform int u_taps;
in vec2 v_uv;
out vec4 f_out;
void main() {
    float self_s = texture(u_static, v_uv).r;
    float self_d = texture(u_dynamic, v_uv).r;
    float occ_s = 0.0;
    float occ_d = 0.0;
    float wsum = 0.0;
    for (int i = 0; i < u_taps; i++) {
        float fi = float(i) + 0.5;
        float r = sqrt(fi / float(u_taps));
        float a = fi * 2.39996323;
        vec2 dir = vec2(cos(a), sin(a));
        vec2 offset = dir * r * u_radius * u_texel;
        float w = (1.0 - 0.5 * r) * (1.0 + 0.9 * max(-dir.y, 0.0));
        occ_s += texture(u_static, v_uv + offset).r * w;
        occ_d += texture(u_dynamic, v_uv + offset).r * w;
        wsum += w;
    }
    occ_s /= wsum;
    occ_d /= wsum;
    float ao_static = smoothstep(0.05, 0.70, occ_s) * (1.0 - self_s);
    float ao_dynamic = smoothstep(0.0, 0.35, occ_d) * (1.0 - self_d);
    f_out = vec4(clamp(ao_static * 0.6 + ao_dynamic, 0.0, 1.0));
}
"""

BLUR_FS = """
#version 330
uniform sampler2D u_tex;
uniform vec2 u_step;
in vec2 v_uv;
out vec4 f_out;
void main() {
    float w[5] = float[](0.227027, 0.1945946, 0.1216216, 0.054054, 0.016216);
    vec4 c = texture(u_tex, v_uv) * w[0];
    for (int i = 1; i < 5; i++) {
        c += texture(u_tex, v_uv + u_step * float(i)) * w[i];
        c += texture(u_tex, v_uv - u_step * float(i)) * w[i];
    }
    f_out = c;
}
"""

# ---------------------------------------------------------------------------
# Lighting composite: applies ambient occlusion and a glossy planar floor
# reflection. The reflection mirrors the scene about the floor line, gets
# blurrier with distance (sampling lower mip levels + a horizontal kernel)
# and fades out with a Fresnel-like falloff.
# ---------------------------------------------------------------------------
LIT_FS = """
#version 330
uniform sampler2D u_scene;
uniform sampler2D u_ao;
uniform float u_ao_strength;
uniform float u_floor;
uniform float u_refl_strength;
uniform float u_refl_rough;
uniform float u_refl_falloff;
uniform int u_refl_taps;
uniform vec2 u_texel;
in vec2 v_uv;
out vec4 f_color;

vec3 lit_sample(vec2 uv, float lod) {
    vec3 c = textureLod(u_scene, uv, lod).rgb;
    return c * (1.0 - texture(u_ao, uv).r * u_ao_strength);
}

void main() {
    vec3 col = lit_sample(v_uv, 0.0);
    if (u_refl_strength > 0.0 && v_uv.y > u_floor) {
        float d = v_uv.y - u_floor;
        float dpx = d / u_texel.y;
        vec2 ruv = vec2(v_uv.x, u_floor - d);
        float lod = clamp(dpx * u_refl_rough, 0.0, 6.0);
        vec3 refl = vec3(0.0);
        float wsum = 0.0;
        for (int i = 0; i < u_refl_taps; i++) {
            float t = (u_refl_taps == 1) ? 0.0 : (float(i) / float(u_refl_taps - 1) - 0.5);
            vec2 offset = vec2(t * (2.0 + dpx * 0.45) * u_texel.x, 0.0);
            float w = 1.0 - abs(t);
            refl += lit_sample(ruv + offset, lod * (u_refl_taps > 1 ? 0.55 : 1.0)) * w;
            wsum += w;
        }
        refl /= wsum;
        float f = u_refl_strength * exp(-dpx / u_refl_falloff);
        col = mix(col, refl, f) + refl * f * 0.3;
    }
    f_color = vec4(col, 1.0);
}
"""

# ---------------------------------------------------------------------------
# Bloom: soft-threshold bright pass, then a dual-filter (Kawase) mip chain.
# ---------------------------------------------------------------------------
BRIGHT_FS = """
#version 330
uniform sampler2D u_tex;
uniform vec2 u_texel;
uniform float u_threshold;
uniform float u_knee;
in vec2 v_uv;
out vec4 f_out;
void main() {
    vec3 c = texture(u_tex, v_uv + vec2(-u_texel.x, -u_texel.y) * 0.5).rgb;
    c += texture(u_tex, v_uv + vec2(u_texel.x, -u_texel.y) * 0.5).rgb;
    c += texture(u_tex, v_uv + vec2(-u_texel.x, u_texel.y) * 0.5).rgb;
    c += texture(u_tex, v_uv + vec2(u_texel.x, u_texel.y) * 0.5).rgb;
    c *= 0.25;
    float br = max(c.r, max(c.g, c.b));
    float rq = clamp(br - u_threshold + u_knee, 0.0, 2.0 * u_knee);
    rq = rq * rq / (4.0 * u_knee + 1e-4);
    float contrib = max(rq, br - u_threshold) / max(br, 1e-4);
    f_out = vec4(c * contrib, 1.0);
}
"""

DOWN_FS = """
#version 330
uniform sampler2D u_tex;
uniform vec2 u_texel;
in vec2 v_uv;
out vec4 f_out;
void main() {
    vec3 sum = texture(u_tex, v_uv).rgb * 4.0;
    sum += texture(u_tex, v_uv - u_texel).rgb;
    sum += texture(u_tex, v_uv + u_texel).rgb;
    sum += texture(u_tex, v_uv + vec2(u_texel.x, -u_texel.y)).rgb;
    sum += texture(u_tex, v_uv - vec2(u_texel.x, -u_texel.y)).rgb;
    f_out = vec4(sum / 8.0, 1.0);
}
"""

UP_FS = """
#version 330
uniform sampler2D u_tex;
uniform vec2 u_texel;
in vec2 v_uv;
out vec4 f_out;
void main() {
    vec2 h = u_texel;
    vec3 sum = texture(u_tex, v_uv + vec2(-h.x * 2.0, 0.0)).rgb;
    sum += texture(u_tex, v_uv + vec2(-h.x, h.y)).rgb * 2.0;
    sum += texture(u_tex, v_uv + vec2(0.0, h.y * 2.0)).rgb;
    sum += texture(u_tex, v_uv + vec2(h.x, h.y)).rgb * 2.0;
    sum += texture(u_tex, v_uv + vec2(h.x * 2.0, 0.0)).rgb;
    sum += texture(u_tex, v_uv + vec2(h.x, -h.y)).rgb * 2.0;
    sum += texture(u_tex, v_uv + vec2(0.0, -h.y * 2.0)).rgb;
    sum += texture(u_tex, v_uv + vec2(-h.x, -h.y)).rgb * 2.0;
    f_out = vec4(sum / 12.0, 1.0);
}
"""

# ---------------------------------------------------------------------------
# Light shafts: radial blur of the bright ceiling lights (screen-space god rays).
# ---------------------------------------------------------------------------
SHAFTS_FS = """
#version 330
uniform sampler2D u_tex;
uniform vec2 u_light;
uniform int u_samples;
uniform float u_density;
uniform float u_decay;
uniform float u_weight;
uniform float u_source_limit;
in vec2 v_uv;
out vec4 f_out;
void main() {
    vec2 delta = (v_uv - u_light) * u_density / float(u_samples);
    vec2 uv = v_uv;
    float illum = 1.0;
    vec3 acc = vec3(0.0);
    for (int i = 0; i < u_samples; i++) {
        uv -= delta;
        float source = 1.0 - smoothstep(u_source_limit * 0.7, u_source_limit, uv.y);
        acc += texture(u_tex, uv).rgb * source * illum * u_weight;
        illum *= u_decay;
    }
    f_out = vec4(acc, 1.0);
}
"""

# ---------------------------------------------------------------------------
# Final grade: bloom + shafts, chromatic aberration, highlight tone mapping,
# colour grading, vignette, motion blur and dithering.
# ---------------------------------------------------------------------------
FINAL_FS = """
#version 330
uniform sampler2D u_lit;
uniform sampler2D u_bloom;
uniform sampler2D u_shafts;
uniform sampler2D u_history;
uniform float u_bloom_intensity;
uniform float u_shaft_intensity;
uniform float u_ca;
uniform float u_vignette;
uniform float u_grading;
uniform float u_motion;
uniform float u_time;
in vec2 v_uv;
out vec4 f_color;

float hash(vec2 p) {
    return fract(sin(dot(p, vec2(12.9898, 78.233))) * 43758.5453);
}

vec3 shoulder(vec3 c) {
    // Identity below 0.8, smooth roll-off above: keeps colours, tames bloom.
    vec3 over = max(c - 0.8, 0.0);
    return min(c, vec3(0.8)) + (1.0 - exp(-over * 4.0)) * 0.2;
}

void main() {
    vec3 col;
    if (u_ca > 0.0) {
        vec2 off = (v_uv - 0.5) * u_ca;
        col = vec3(texture(u_lit, v_uv + off).r, texture(u_lit, v_uv).g, texture(u_lit, v_uv - off).b);
    } else {
        col = texture(u_lit, v_uv).rgb;
    }
    col += texture(u_bloom, v_uv).rgb * u_bloom_intensity;
    col += texture(u_shafts, v_uv).rgb * u_shaft_intensity;

    if (u_grading > 0.0) {
        vec3 graded = shoulder(col * 1.04);
        float luma = dot(graded, vec3(0.2126, 0.7152, 0.0722));
        graded = mix(vec3(luma), graded, 1.12);                           // saturation
        graded = mix(graded, graded * vec3(0.96, 1.0, 1.06), 1.0 - luma);  // cool shadows
        graded = (graded - 0.5) * 1.05 + 0.5;                             // contrast
        col = mix(clamp(col, 0.0, 1.0), graded, u_grading);
    }
    if (u_vignette > 0.0) {
        vec2 q = (v_uv - 0.5) * vec2(1.0, 0.9);
        col *= mix(1.0, smoothstep(0.95, 0.3, length(q)), u_vignette * 0.55);
    }
    col = clamp(col, 0.0, 1.0);
    if (u_motion > 0.0) {
        col = mix(col, texture(u_history, v_uv).rgb, u_motion);
    }
    col += (hash(v_uv * 1000.0 + u_time) - 0.5) / 255.0;
    f_color = vec4(col, 1.0);
}
"""

# ---------------------------------------------------------------------------
# Output to the window (v flipped because the window's origin is bottom-left).
# ---------------------------------------------------------------------------
FXAA_FS = """
#version 330
uniform sampler2D u_tex;
uniform vec2 u_texel;
in vec2 v_uv;
out vec4 f_color;
float luma(vec3 c) { return dot(c, vec3(0.299, 0.587, 0.114)); }
void main() {
    vec2 uv = vec2(v_uv.x, 1.0 - v_uv.y);
    vec3 rgbM = texture(u_tex, uv).rgb;
    vec3 rgbNW = texture(u_tex, uv + vec2(-1.0, -1.0) * u_texel).rgb;
    vec3 rgbNE = texture(u_tex, uv + vec2(1.0, -1.0) * u_texel).rgb;
    vec3 rgbSW = texture(u_tex, uv + vec2(-1.0, 1.0) * u_texel).rgb;
    vec3 rgbSE = texture(u_tex, uv + vec2(1.0, 1.0) * u_texel).rgb;
    float lM = luma(rgbM), lNW = luma(rgbNW), lNE = luma(rgbNE), lSW = luma(rgbSW), lSE = luma(rgbSE);
    float lMin = min(lM, min(min(lNW, lNE), min(lSW, lSE)));
    float lMax = max(lM, max(max(lNW, lNE), max(lSW, lSE)));
    vec2 dir = vec2(-((lNW + lNE) - (lSW + lSE)), ((lNW + lSW) - (lNE + lSE)));
    float reduce = max((lNW + lNE + lSW + lSE) * 0.25 * 0.125, 1.0 / 128.0);
    float rcpMin = 1.0 / (min(abs(dir.x), abs(dir.y)) + reduce);
    dir = clamp(dir * rcpMin, vec2(-8.0), vec2(8.0)) * u_texel;
    vec3 rgbA = 0.5 * (texture(u_tex, uv + dir * (1.0 / 3.0 - 0.5)).rgb +
                       texture(u_tex, uv + dir * (2.0 / 3.0 - 0.5)).rgb);
    vec3 rgbB = rgbA * 0.5 + 0.25 * (texture(u_tex, uv + dir * -0.5).rgb +
                                     texture(u_tex, uv + dir * 0.5).rgb);
    float lB = luma(rgbB);
    f_color = vec4((lB < lMin || lB > lMax) ? rgbA : rgbB, 1.0);
}
"""

COPY_FS = """
#version 330
uniform sampler2D u_tex;
in vec2 v_uv;
out vec4 f_color;
void main() {
    f_color = vec4(texture(u_tex, vec2(v_uv.x, 1.0 - v_uv.y)).rgb, 1.0);
}
"""

UI_FS = """
#version 330
uniform sampler2D u_tex;
in vec2 v_uv;
out vec4 f_color;
void main() {
    f_color = texture(u_tex, vec2(v_uv.x, 1.0 - v_uv.y));
}
"""
