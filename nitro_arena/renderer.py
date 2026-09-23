"""
Renderers: turn the frame drawn by pygame into pixels on the screen.

GPURenderer (default)
    Uses OpenGL through `moderngl`. pygame draws the world and the UI into
    off-screen surfaces, which are uploaded to the graphics card every frame
    and run through shader passes: ambient occlusion, glossy floor
    reflections, bloom, light shafts, colour grading, motion blur and FXAA.

CPURenderer (fallback)
    Plain pygame. Used when moderngl is missing, OpenGL cannot start, or
    the player picks it in the graphics settings. Supports reflections only.

Both expose the same methods, so the rest of the game doesn't care which one
is active:
    world_surface()  -> surface to draw the arena, cars and ball on
    finish_world(occluders, fx)
    ui_surface()     -> surface to draw the HUD and menus on (transparent on GPU)
    present()
"""
import sys
import time

import pygame

import settings as S

SIZE = (S.SCREEN_WIDTH, S.SCREEN_HEIGHT)


def create_renderer(user_settings):
    """Create the renderer the player asked for, falling back to the CPU one."""
    fullscreen = user_settings.get("fullscreen", False)
    if user_settings.get("renderer", "GPU") == "GPU":
        try:
            return GPURenderer(fullscreen, user_settings)
        except Exception as exc:  # missing moderngl, no OpenGL 3.3, shader error...
            print(f"[renderer] GPU renderer unavailable, using CPU renderer: {exc}")
            try:
                pygame.display.quit()
                pygame.display.init()
            except pygame.error:
                pass
            renderer = CPURenderer(fullscreen, user_settings)
            renderer.gpu_error = str(exc)
            return renderer
    return CPURenderer(fullscreen, user_settings)


class BaseRenderer:
    gpu = False
    name = "CPU"
    gpu_error = None

    def __init__(self, user_settings):
        self.gfx = user_settings
        self.fullscreen = False

    def window_to_internal(self, pos):
        return pos

    def toggle_fullscreen(self):
        try:
            result = pygame.display.toggle_fullscreen()
        except pygame.error as exc:
            print(f"[renderer] Fullscreen toggle failed: {exc}")
            return self.fullscreen
        if result not in (0, False):
            self.fullscreen = not self.fullscreen
        return self.fullscreen


# ----------------------------------------------------------------------
# CPU renderer
# ----------------------------------------------------------------------
class CPURenderer(BaseRenderer):
    def __init__(self, fullscreen, user_settings):
        super().__init__(user_settings)
        self.screen = self._create_window(fullscreen)
        self._reflection_fade = None

    def _create_window(self, fullscreen):
        attempts = []
        if fullscreen:
            attempts.append(pygame.SCALED | pygame.FULLSCREEN)
        attempts += [pygame.SCALED | pygame.RESIZABLE, 0]
        last_error = None
        for flags in attempts:
            try:
                screen = pygame.display.set_mode(SIZE, flags)
                self.fullscreen = bool(flags & pygame.FULLSCREEN)
                return screen
            except pygame.error as exc:
                last_error = exc
        raise SystemExit(f"Could not open a game window: {last_error}")

    def world_surface(self):
        return self.screen

    def ui_surface(self):
        return self.screen

    def finish_world(self, occluders, fx):
        if self.gfx.get("reflections", 0) > 0:
            self._draw_reflection()

    def _draw_reflection(self):
        """Mirror the strip above the floor onto the floor, fading with depth."""
        height = S.SCREEN_HEIGHT - S.ARENA_FLOOR
        if self._reflection_fade is None:
            fade = pygame.Surface((S.SCREEN_WIDTH, height), pygame.SRCALPHA)
            for y in range(height):
                alpha = int(120 * (1.0 - y / height) ** 2.2)
                pygame.draw.line(fade, (255, 255, 255, alpha), (0, y), (S.SCREEN_WIDTH, y))
            self._reflection_fade = fade
        strip = self.screen.subsurface((0, S.ARENA_FLOOR - height, S.SCREEN_WIDTH, height))
        mirrored = pygame.transform.flip(strip, False, True).convert_alpha()
        mirrored.blit(self._reflection_fade, (0, 0), special_flags=pygame.BLEND_RGBA_MULT)
        self.screen.blit(mirrored, (0, S.ARENA_FLOOR))

    def present(self):
        pygame.display.flip()


# ----------------------------------------------------------------------
# GPU renderer
# ----------------------------------------------------------------------
def _swizzle_for(surface):
    """Work out how pygame's in-memory byte order maps to RGBA."""
    names = {}
    for channel, mask in zip("RGBA", surface.get_masks()):
        if mask:
            names[mask] = channel
    component = {}
    for index, letter in enumerate("RGBA"):
        shift = index * 8 if sys.byteorder == "little" else (3 - index) * 8
        mask = 0xFF << shift
        if mask in names:
            component[names[mask]] = letter
    return component.get("R", "R") + component.get("G", "G") + component.get("B", "B") + component.get("A", "1")


class GPURenderer(BaseRenderer):
    gpu = True
    name = "GPU"

    # Per-level parameters for each effect (index 0 = off).
    AO_LEVELS = [None, (12, 22.0, 0.6), (24, 32.0, 0.75), (48, 42.0, 0.85)]         # taps, radius (half-res px), strength
    REFLECTION_LEVELS = [None, (1, 0.05, 0.42), (3, 0.035, 0.5), (7, 0.03, 0.55)]  # taps, roughness, strength
    BLOOM_LEVELS = [None, (3, 0.55), (5, 0.8), (6, 1.0)]                           # mip levels, intensity
    SHAFT_LEVELS = [None, (40, 0.35), (96, 0.45)]                                  # samples, intensity
    MOTION_LEVELS = [0.0, 0.25, 0.45]

    def __init__(self, fullscreen, user_settings):
        super().__init__(user_settings)
        import moderngl  # optional dependency; ImportError -> CPU fallback
        self.mgl = moderngl

        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MAJOR_VERSION, 3)
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MINOR_VERSION, 3)
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_PROFILE_MASK, pygame.GL_CONTEXT_PROFILE_CORE)
        if sys.platform == "darwin":
            pygame.display.gl_set_attribute(pygame.GL_CONTEXT_FLAGS, pygame.GL_CONTEXT_FORWARD_COMPATIBLE_FLAG)
        flags = pygame.OPENGL | pygame.DOUBLEBUF
        if fullscreen:
            pygame.display.set_mode((0, 0), flags | pygame.FULLSCREEN)
            self.fullscreen = True
        else:
            pygame.display.set_mode(SIZE, flags | pygame.RESIZABLE)
        try:
            self.ctx = moderngl.create_context(require=330)
        except Exception:
            if not sys.platform.startswith("linux"):
                raise
            # Some Linux systems only ship the versioned library name.
            self.ctx = moderngl.create_context(require=330, libgl="libGL.so.1")

        # On Retina screens the OpenGL framebuffer can be larger than the
        # window (in points); remember the ratio for the viewport maths.
        window_w = max(1, pygame.display.get_window_size()[0])
        self.fb_scale = max(1.0, self.ctx.screen.viewport[2] / window_w)

        self.world = pygame.Surface(SIZE, 0, 32)
        self.ui = pygame.Surface(SIZE, pygame.SRCALPHA, 32)
        half = (S.SCREEN_WIDTH // 2, S.SCREEN_HEIGHT // 2)
        self.half = half
        self.dynamic_mask = pygame.Surface(half, 0, 32)
        self._build_programs()
        self._build_targets()
        self.static_mask_ready = False
        self.viewport = (0, 0) + SIZE
        self.start_time = time.perf_counter()
        self.history_valid = False
        print(f"[renderer] GPU renderer: {self.ctx.info.get('GL_RENDERER', 'unknown')} "
              f"(OpenGL {self.ctx.version_code})")

    # --- setup -------------------------------------------------------
    def _build_programs(self):
        import shaders as SH
        self.programs = {}
        self.vaos = {}
        for name, fs in (("ao", SH.AO_FS), ("blur", SH.BLUR_FS), ("lit", SH.LIT_FS),
                         ("bright", SH.BRIGHT_FS), ("down", SH.DOWN_FS), ("up", SH.UP_FS),
                         ("shafts", SH.SHAFTS_FS), ("final", SH.FINAL_FS), ("fxaa", SH.FXAA_FS),
                         ("copy", SH.COPY_FS), ("ui", SH.UI_FS)):
            program = self.ctx.program(vertex_shader=SH.FULLSCREEN_VS, fragment_shader=fs)
            self.programs[name] = program
            self.vaos[name] = self.ctx.vertex_array(program, [])

    def _texture(self, size, components=4, dtype="f1", mipmaps=False):
        tex = self.ctx.texture(size, components, dtype=dtype)
        tex.repeat_x = False
        tex.repeat_y = False
        tex.filter = (self.mgl.LINEAR_MIPMAP_LINEAR if mipmaps else self.mgl.LINEAR, self.mgl.LINEAR)
        return tex

    def _target(self, size, components=4, dtype="f1"):
        tex = self._texture(size, components, dtype)
        return tex, self.ctx.framebuffer(color_attachments=[tex])

    def _build_targets(self):
        W, H = SIZE
        self.scene_tex = self._texture(SIZE, 4, "f1", mipmaps=True)
        self.scene_tex.swizzle = _swizzle_for(self.world)[:3] + "1"
        self.ui_tex = self._texture(SIZE, 4, "f1")
        self.ui_tex.swizzle = _swizzle_for(self.ui)
        self.static_tex = self._texture(self.half, 4, "f1")
        self.dynamic_tex = self._texture(self.half, 4, "f1")
        self.dynamic_tex.swizzle = _swizzle_for(self.dynamic_mask)
        self.ao_tex, self.ao_fbo = self._target(self.half, 1, "f1")
        self.ao_tmp_tex, self.ao_tmp_fbo = self._target(self.half, 1, "f1")
        self.lit_tex, self.lit_fbo = self._target(SIZE, 4, "f2")
        self.bloom = []
        w, h = W // 2, H // 2
        for _ in range(6):
            self.bloom.append(self._target((max(1, w), max(1, h)), 4, "f2"))
            w, h = w // 2, h // 2
        self.shafts_tex, self.shafts_fbo = self._target(self.half, 4, "f2")
        self.post_tex, self.post_fbo = self._target(SIZE, 4, "f1")
        self.history_tex, self.history_fbo = self._target(SIZE, 4, "f1")
        self.black_tex = self._texture((1, 1), 4, "f1")
        self.black_tex.write(bytes(4))
        for fbo in (self.ao_fbo, self.ao_tmp_fbo, self.shafts_fbo):
            fbo.clear(0, 0, 0, 0)

    def _upload(self, texture, surface):
        if surface.get_pitch() == surface.get_width() * 4:
            texture.write(surface.get_view("1"))
        else:
            texture.write(pygame.image.tobytes(surface, "RGBA"))

    def _build_static_mask(self):
        """White = solid arena (walls, floor), black = open air."""
        from arena import Arena
        mask = pygame.Surface(self.half, 0, 32)
        mask.fill((255, 255, 255))
        points = [(p.x * 0.5, p.y * 0.5) for p in Arena().outline]
        pygame.draw.polygon(mask, (0, 0, 0), points)
        self.static_tex.swizzle = _swizzle_for(mask)
        self._upload(self.static_tex, mask)
        self.static_mask_ready = True

    # --- per-frame API -------------------------------------------------
    def world_surface(self):
        return self.world

    def ui_surface(self):
        self.ui.fill((0, 0, 0, 0))
        return self.ui

    def finish_world(self, occluders, fx):
        self._occluders = occluders
        self._fx = fx

    def window_to_internal(self, pos):
        x, y, w, h = self.viewport
        if w <= 0 or h <= 0:
            return pos
        return (int((pos[0] - x) * S.SCREEN_WIDTH / w), int((pos[1] - y) * S.SCREEN_HEIGHT / h))

    def _update_viewport(self):
        win_w, win_h = pygame.display.get_window_size()
        scale = min(win_w / S.SCREEN_WIDTH, win_h / S.SCREEN_HEIGHT)
        w, h = int(S.SCREEN_WIDTH * scale), int(S.SCREEN_HEIGHT * scale)
        # pygame's window coordinates start top-left; OpenGL's start bottom-left.
        x, y = (win_w - w) // 2, (win_h - h) // 2
        self.viewport = (x, y, w, h)
        self.window_size = (win_w, win_h)

    def _set(self, program, name, value):
        try:
            program[name].value = value
        except KeyError:
            pass  # uniform optimised away by the driver

    def _run(self, name, target, textures=(), **uniforms):
        program = self.programs[name]
        target.use()
        for unit, (uniform, texture) in enumerate(textures):
            texture.use(unit)
            self._set(program, uniform, unit)
        for key, value in uniforms.items():
            self._set(program, key, value)
        self.vaos[name].render(self.mgl.TRIANGLES, vertices=3)

    def _draw_dynamic_mask(self):
        mask = self.dynamic_mask
        mask.fill((0, 0, 0))
        for shape in getattr(self, "_occluders", ()):
            if shape[0] == "poly":
                pygame.draw.polygon(mask, (255, 255, 255), [(x * 0.5, y * 0.5) for x, y in shape[1]])
            elif shape[0] == "circle":
                pygame.draw.circle(mask, (255, 255, 255), (shape[1][0] * 0.5, shape[1][1] * 0.5), shape[2] * 0.5)
        self._upload(self.dynamic_tex, mask)

    def present(self):
        g = self.gfx
        fx = getattr(self, "_fx", {}) or {}
        W, H = SIZE
        texel = (1.0 / W, 1.0 / H)
        half_texel = (1.0 / self.half[0], 1.0 / self.half[1])
        if not self.static_mask_ready:
            self._build_static_mask()

        self._upload(self.scene_tex, self.world)
        self.scene_tex.build_mipmaps()

        # 1) Ambient occlusion (half resolution) + separable blur.
        ao_level = self.AO_LEVELS[g.get("ambient_occlusion", 0)]
        if ao_level:
            taps, radius, ao_strength = ao_level
            self._draw_dynamic_mask()
            self._run("ao", self.ao_fbo, (("u_static", self.static_tex), ("u_dynamic", self.dynamic_tex)),
                      u_texel=half_texel, u_radius=radius, u_taps=taps)
            spread = 1.4 if taps > 12 else 1.0
            self._run("blur", self.ao_tmp_fbo, (("u_tex", self.ao_tex),), u_step=(half_texel[0] * spread, 0.0))
            self._run("blur", self.ao_fbo, (("u_tex", self.ao_tmp_tex),), u_step=(0.0, half_texel[1] * spread))
            ao_tex = self.ao_tex
        else:
            ao_strength = 0.0
            ao_tex = self.black_tex

        # 2) Lighting composite: AO + glossy floor reflections.
        refl = self.REFLECTION_LEVELS[g.get("reflections", 0)]
        taps, rough, strength = refl if refl else (1, 0.0, 0.0)
        self._run("lit", self.lit_fbo, (("u_scene", self.scene_tex), ("u_ao", ao_tex)),
                  u_ao_strength=ao_strength, u_floor=S.ARENA_FLOOR / H, u_refl_strength=strength,
                  u_refl_rough=rough, u_refl_falloff=38.0, u_refl_taps=taps, u_texel=texel)

        # 3) Bloom: bright pass + dual-filter mip chain.
        bloom_level = self.BLOOM_LEVELS[g.get("bloom", 0)]
        shafts_level = self.SHAFT_LEVELS[g.get("light_shafts", 0)]
        bloom_tex, bloom_intensity = self.black_tex, 0.0
        if bloom_level or shafts_level:
            first_tex, first_fbo = self.bloom[0]
            self._run("bright", first_fbo, (("u_tex", self.lit_tex),), u_texel=texel,
                      u_threshold=0.62, u_knee=0.25)
        if shafts_level:
            samples, shaft_intensity = shafts_level
            self._run("shafts", self.shafts_fbo, (("u_tex", self.bloom[0][0]),), u_light=(0.5, -0.35),
                      u_samples=samples, u_density=0.55, u_decay=0.975 if samples > 50 else 0.955,
                      u_weight=5.0 / samples, u_source_limit=(S.ARENA_TOP + 60) / H)
            shafts_tex = self.shafts_tex
        else:
            shaft_intensity, shafts_tex = 0.0, self.black_tex
        if bloom_level:
            levels, bloom_intensity = bloom_level
            for i in range(1, levels):
                src_tex = self.bloom[i - 1][0]
                self._run("down", self.bloom[i][1], (("u_tex", src_tex),),
                          u_texel=(1.0 / src_tex.width, 1.0 / src_tex.height))
            self.ctx.enable(self.mgl.BLEND)
            self.ctx.blend_func = self.mgl.ONE, self.mgl.ONE
            for i in range(levels - 1, 0, -1):
                src_tex = self.bloom[i][0]
                self._run("up", self.bloom[i - 1][1], (("u_tex", src_tex),),
                          u_texel=(0.5 / src_tex.width, 0.5 / src_tex.height))
            self.ctx.disable(self.mgl.BLEND)
            bloom_tex = self.bloom[0][0]

        # 4) Final grade into the post buffer.
        motion = self.MOTION_LEVELS[g.get("motion_blur", 0)] if self.history_valid else 0.0
        ca = (0.0012 + 0.006 * fx.get("impact", 0.0)) if g.get("chromatic_aberration", 0) else 0.0
        self._run("final", self.post_fbo,
                  (("u_lit", self.lit_tex), ("u_bloom", bloom_tex), ("u_shafts", shafts_tex),
                   ("u_history", self.history_tex)),
                  u_bloom_intensity=bloom_intensity, u_shaft_intensity=shaft_intensity, u_ca=ca,
                  u_vignette=float(g.get("vignette", 0)), u_grading=float(g.get("color_grading", 0)),
                  u_motion=motion, u_time=(time.perf_counter() - self.start_time) % 100.0)
        if g.get("motion_blur", 0):
            self.ctx.copy_framebuffer(self.history_fbo, self.post_fbo)
            self.history_valid = True
        else:
            self.history_valid = False

        # 5) To the window: letterboxed, anti-aliased, then the UI on top.
        self._update_viewport()
        screen = self.ctx.screen
        screen.use()
        k = self.fb_scale
        win_w, win_h = self.window_size
        screen.viewport = (0, 0, int(win_w * k), int(win_h * k))
        screen.clear(0.0, 0.0, 0.0, 1.0)
        x, y, w, h = self.viewport
        screen.viewport = (int(x * k), int((win_h - y - h) * k), int(w * k), int(h * k))
        if g.get("antialiasing", 0):
            self._set(self.programs["fxaa"], "u_texel", texel)
            self.post_tex.use(0)
            self._set(self.programs["fxaa"], "u_tex", 0)
            self.vaos["fxaa"].render(self.mgl.TRIANGLES, vertices=3)
        else:
            self.post_tex.use(0)
            self._set(self.programs["copy"], "u_tex", 0)
            self.vaos["copy"].render(self.mgl.TRIANGLES, vertices=3)

        self._upload(self.ui_tex, self.ui)
        self.ctx.enable(self.mgl.BLEND)
        self.ctx.blend_func = self.mgl.SRC_ALPHA, self.mgl.ONE_MINUS_SRC_ALPHA
        self.ui_tex.use(0)
        self._set(self.programs["ui"], "u_tex", 0)
        self.vaos["ui"].render(self.mgl.TRIANGLES, vertices=3)
        self.ctx.disable(self.mgl.BLEND)
        pygame.display.flip()

    def read_pixels(self):
        """Return the final image (without letterbox) as a pygame Surface (used for screenshots/tests)."""
        data = self.post_fbo.read(components=3)
        image = pygame.image.frombytes(data, SIZE, "RGB")
        return image
