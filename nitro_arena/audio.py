"""
Audio: every sound effect and the music loop are synthesised in code, so the
game needs no audio files at all.

* You can override any sound by dropping a file with the same name into
  assets/sounds/ (e.g. assets/sounds/goal.wav or assets/sounds/music.ogg).
* If the audio device cannot be opened, the game simply runs silently.
* The music loop is generated in a background thread so the menu appears
  immediately.
"""
import array
import io
import math
import os
import random
import threading
import wave

import pygame

import settings as S

RATE = 22050
TWO_PI = 2 * math.pi


# ----------------------------------------------------------------------
# Tiny synthesiser
# ----------------------------------------------------------------------
def _wave_value(shape, phase):
    x = phase % 1.0
    if shape == "sine":
        return math.sin(TWO_PI * x)
    if shape == "square":
        return 1.0 if x < 0.5 else -1.0
    if shape == "saw":
        return 2.0 * x - 1.0
    return 4.0 * abs(x - 0.5) - 1.0  # triangle


def add_tone(buf, start, duration, freq, shape="sine", volume=0.5, freq_end=None,
             attack=0.005, decay=1.5):
    """Add a tone into buf (list of floats) starting at `start` seconds."""
    first = int(start * RATE)
    count = int(duration * RATE)
    if first + count > len(buf):
        buf.extend([0.0] * (first + count - len(buf)))
    attack_n = max(1, int(attack * RATE))
    phase = 0.0
    f_end = freq if freq_end is None else freq_end
    for i in range(count):
        t = i / count
        phase += (freq + (f_end - freq) * t) / RATE
        env = (1.0 - t) ** decay
        if i < attack_n:
            env *= i / attack_n
        buf[first + i] += _wave_value(shape, phase) * volume * env


def add_noise(buf, start, duration, volume=0.5, smooth=0.5, decay=2.0, rng=None):
    """Add filtered white noise. Higher `smooth` = darker / more muffled."""
    rng = rng or random
    first = int(start * RATE)
    count = int(duration * RATE)
    if first + count > len(buf):
        buf.extend([0.0] * (first + count - len(buf)))
    value = 0.0
    for i in range(count):
        t = i / count
        value = value * smooth + rng.uniform(-1.0, 1.0) * (1.0 - smooth)
        buf[first + i] += value * volume * (1.0 - t) ** decay * 2.0


def to_wav_bytes(samples):
    # Normalise only if the mix would clip.
    peak = max((abs(s) for s in samples), default=0.0)
    scale = 32000 / peak if peak > 1.0 else 32000
    data = array.array("h", (int(s * scale) for s in samples))
    out = io.BytesIO()
    with wave.open(out, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(RATE)
        w.writeframes(data.tobytes())
    return out.getvalue()


def make_loopable(samples, crossfade):
    """Cross-fade the tail into the head so a sound loops without a click."""
    n = len(samples) - crossfade
    out = samples[:n]
    for i in range(crossfade):
        t = i / crossfade
        out[i] = out[i] * t + samples[n + i] * (1.0 - t)
    return out


# ----------------------------------------------------------------------
# Sound designs
# ----------------------------------------------------------------------
def build_sfx():
    rng = random.Random(1)
    sounds = {}

    def new():
        return []

    b = new(); add_tone(b, 0, 0.14, 300, "triangle", 0.5, 640); add_noise(b, 0, 0.07, 0.12, 0.3, rng=rng)
    sounds["jump"] = b
    b = new(); add_tone(b, 0, 0.16, 440, "triangle", 0.5, 900)
    sounds["double_jump"] = b
    b = new(); add_noise(b, 0, 0.28, 0.35, 0.6, 1.2, rng); add_tone(b, 0, 0.2, 220, "triangle", 0.3, 520)
    sounds["dodge"] = b
    b = new(); add_tone(b, 0, 0.12, 180, "sine", 0.7, 90); add_noise(b, 0, 0.05, 0.3, 0.4, rng=rng)
    sounds["hit_soft"] = b
    b = new(); add_tone(b, 0, 0.32, 130, "sine", 1.0, 45, decay=1.2); add_noise(b, 0, 0.2, 0.6, 0.55, 1.5, rng)
    add_tone(b, 0, 0.07, 1100, "square", 0.12, 300)
    sounds["hit_hard"] = b
    b = new(); add_tone(b, 0, 0.1, 120, "sine", 0.55, 70); add_noise(b, 0, 0.04, 0.15, 0.5, rng=rng)
    sounds["bounce"] = b
    b = new(); add_noise(b, 0, 0.09, 0.3, 0.75, rng=rng); add_tone(b, 0, 0.08, 95, "sine", 0.4, 60)
    sounds["land"] = b
    b = new(); add_tone(b, 0, 0.12, 520, "triangle", 0.35, 780)
    sounds["flip_upright"] = b

    # Goal: explosion + rising arpeggio + held chord.
    b = new()
    add_noise(b, 0, 1.1, 0.9, 0.82, 1.4, rng)
    add_tone(b, 0, 0.5, 90, "sine", 0.8, 40, decay=1.0)
    for i, f in enumerate((523.25, 659.25, 783.99, 1046.5)):
        add_tone(b, 0.1 + i * 0.09, 0.25, f, "square", 0.12, decay=1.0)
    for f in (523.25, 659.25, 783.99):
        add_tone(b, 0.46, 0.9, f, "triangle", 0.22, decay=1.3)
    sounds["goal"] = b

    b = new(); add_tone(b, 0, 0.16, 660, "square", 0.22, decay=0.8)
    sounds["countdown"] = b
    b = new(); add_tone(b, 0, 0.45, 990, "square", 0.22, decay=1.0); add_tone(b, 0, 0.45, 1320, "triangle", 0.2)
    sounds["go"] = b
    b = new(); add_tone(b, 0, 0.045, 880, "square", 0.12, decay=2.0)
    sounds["menu_move"] = b
    b = new(); add_tone(b, 0, 0.12, 660, "triangle", 0.4, 1320, decay=1.0)
    sounds["menu_select"] = b
    b = new(); add_tone(b, 0, 0.1, 500, "triangle", 0.35, 250)
    sounds["menu_back"] = b

    b = new()
    for i, f in enumerate((392.0, 523.25, 659.25, 783.99)):
        add_tone(b, i * 0.13, 0.2, f, "square", 0.15, decay=0.8)
    for f in (523.25, 659.25, 783.99, 1046.5):
        add_tone(b, 0.52, 1.1, f, "triangle", 0.2, decay=1.2)
    sounds["win"] = b
    b = new()
    for i, f in enumerate((440.0, 392.0, 349.23, 293.66)):
        add_tone(b, i * 0.18, 0.3, f, "triangle", 0.3, decay=0.9)
    sounds["lose"] = b
    b = new(); add_tone(b, 0, 0.6, 700, "square", 0.18, 690, decay=0.4); add_tone(b, 0, 0.6, 1050, "square", 0.1, decay=0.4)
    sounds["whistle"] = b

    # Boost: continuous rumble + hiss, made loopable.
    b = []
    loop_len = 0.6
    add_noise(b, 0, loop_len + 0.1, 0.5, 0.7, 0.0, rng)
    add_tone(b, 0, loop_len + 0.1, 70, "saw", 0.25, decay=0.0)
    sounds["boost_loop"] = make_loopable(b, int(0.1 * RATE))
    return sounds


def build_music():
    """A 4-bar synth loop at 120 BPM (8 seconds)."""
    step = 0.125  # 16th note
    bars = [
        (110.00, (440.00, 523.25, 659.25, 880.00)),   # Am
        (87.31, (349.23, 440.00, 523.25, 698.46)),    # F
        (130.81, (523.25, 659.25, 783.99, 1046.50)),  # C
        (98.00, (392.00, 493.88, 587.33, 783.99)),    # G
    ]
    total = int(len(bars) * 16 * step * RATE)
    buf = [0.0] * total
    rng = random.Random(5)
    arp_pattern = (0, 1, 2, 3, 2, 1, 2, 3, 0, 2, 1, 3, 2, 1, 0, 1)
    for bar, (root, notes) in enumerate(bars):
        bar_start = bar * 16 * step
        for s in range(16):
            t = bar_start + s * step
            if s % 2 == 0:
                add_tone(buf, t, step * 2, root, "saw", 0.22, decay=0.6, attack=0.01)
                add_tone(buf, t, step * 2, root / 2, "sine", 0.25, decay=0.4, attack=0.01)
            add_tone(buf, t, step, notes[arp_pattern[s]], "triangle", 0.1, decay=1.4)
            if s % 4 == 0:
                add_tone(buf, t, 0.18, 150, "sine", 0.55, 45, decay=1.5)        # kick
            if s % 8 == 4:
                add_noise(buf, t, 0.14, 0.22, 0.35, 2.0, rng)                  # snare
            if s % 2 == 1:
                add_noise(buf, t, 0.04, 0.08, 0.0, 3.0, rng)                   # hi-hat
    return buf[:total]


# ----------------------------------------------------------------------
# Audio manager
# ----------------------------------------------------------------------
class AudioManager:
    MUSIC_CHANNEL = 0
    BOOST_CHANNELS = (1, 2)
    BASE_VOLUME = {
        "hit_soft": 0.7, "hit_hard": 1.0, "bounce": 0.5, "land": 0.4, "jump": 0.5,
        "double_jump": 0.5, "dodge": 0.6, "goal": 0.9, "countdown": 0.5, "go": 0.6,
        "menu_move": 0.4, "menu_select": 0.6, "menu_back": 0.5, "win": 0.7, "lose": 0.7,
        "whistle": 0.6, "flip_upright": 0.4, "boost_loop": 0.35,
    }

    def __init__(self, music_volume=0.5, sfx_volume=0.8):
        self.music_volume = music_volume
        self.sfx_volume = sfx_volume
        self.music_duck = 1.0
        self.sounds = {}
        self.music = None
        self._music_samples = None
        self._music_thread = None
        self.enabled = False
        try:
            if pygame.mixer.get_init() is None:
                pygame.mixer.init(44100, -16, 2, 512)
            pygame.mixer.set_num_channels(20)
            pygame.mixer.set_reserved(3)
            self.enabled = True
        except (pygame.error, NotImplementedError, OSError) as exc:
            print(f"[audio] Sound disabled: {exc}")
            return
        try:
            self._load_sounds()
            self._start_music_loading()
        except Exception as exc:  # never let audio problems stop the game
            print(f"[audio] Could not create sounds: {exc}")
            self.enabled = False

    # --- loading -------------------------------------------------------
    @staticmethod
    def _override_path(name):
        for ext in (".wav", ".ogg", ".mp3"):
            path = os.path.join(S.SOUNDS_DIR, name + ext)
            if os.path.isfile(path):
                return path
        return None

    def _load_sounds(self):
        generated = build_sfx()
        for name, samples in generated.items():
            sound = None
            override = self._override_path(name)
            if override:
                try:
                    sound = pygame.mixer.Sound(override)
                except pygame.error:
                    sound = None
            if sound is None:
                sound = pygame.mixer.Sound(file=io.BytesIO(to_wav_bytes(samples)))
            self.sounds[name] = sound

    def _start_music_loading(self):
        override = self._override_path("music")
        if override:
            try:
                self.music = pygame.mixer.Sound(override)
                self._play_music()
                return
            except pygame.error:
                pass
        self._music_thread = threading.Thread(target=self._generate_music, daemon=True)
        self._music_thread.start()

    def _generate_music(self):
        try:
            self._music_samples = to_wav_bytes(build_music())
        except Exception as exc:
            print(f"[audio] Music generation failed: {exc}")

    def update(self):
        """Call once per frame: starts the music when background generation finishes."""
        if self.enabled and self.music is None and self._music_samples is not None:
            try:
                self.music = pygame.mixer.Sound(file=io.BytesIO(self._music_samples))
                self._play_music()
            except pygame.error:
                pass
            self._music_samples = None

    def _play_music(self):
        channel = pygame.mixer.Channel(self.MUSIC_CHANNEL)
        channel.play(self.music, loops=-1, fade_ms=800)
        channel.set_volume(self.music_volume * self.music_duck)

    # --- playback ------------------------------------------------------
    def play(self, name, volume=1.0):
        if not self.enabled or self.sfx_volume <= 0:
            return
        sound = self.sounds.get(name)
        if sound is None:
            return
        try:
            channel = sound.play()
            if channel is not None:
                channel.set_volume(max(0.0, min(1.0, volume * self.BASE_VOLUME.get(name, 0.6) * self.sfx_volume)))
        except pygame.error:
            pass

    def set_boost(self, index, active, volume=1.0):
        """Start/stop the looping boost sound for car `index` (0 or 1)."""
        if not self.enabled or index >= len(self.BOOST_CHANNELS):
            return
        try:
            channel = pygame.mixer.Channel(self.BOOST_CHANNELS[index])
            if active and self.sfx_volume > 0:
                if not channel.get_busy():
                    channel.play(self.sounds["boost_loop"], loops=-1, fade_ms=60)
                channel.set_volume(self.BASE_VOLUME["boost_loop"] * self.sfx_volume * volume)
            elif channel.get_busy():
                channel.fadeout(120)
        except (pygame.error, KeyError):
            pass

    def stop_loops(self):
        for i in range(len(self.BOOST_CHANNELS)):
            self.set_boost(i, False)

    def set_volumes(self, music_volume, sfx_volume):
        self.music_volume = music_volume
        self.sfx_volume = sfx_volume
        self._apply_music_volume()

    def set_music_duck(self, factor):
        if abs(factor - self.music_duck) > 1e-3:
            self.music_duck = factor
            self._apply_music_volume()

    def _apply_music_volume(self):
        if not self.enabled:
            return
        try:
            pygame.mixer.Channel(self.MUSIC_CHANNEL).set_volume(self.music_volume * self.music_duck)
        except pygame.error:
            pass
