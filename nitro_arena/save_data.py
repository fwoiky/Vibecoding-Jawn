"""Loads and saves user settings and statistics in a small local JSON file."""
import json
import os

import settings as S

DEFAULT_STATS = {
    "matches_played": 0,
    "wins": 0,
    "losses": 0,
    "goals_scored": 0,
    "goals_conceded": 0,
}


class SaveData:
    def __init__(self, path=S.SAVE_FILE):
        self.path = path
        self.settings = dict(S.DEFAULT_USER_SETTINGS)
        self.stats = dict(DEFAULT_STATS)
        self.load()

    def load(self):
        if not os.path.isfile(self.path):
            return
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, ValueError) as exc:
            print(f"[save] Could not read save file, using defaults: {exc}")
            return
        if not isinstance(data, dict):
            return
        # Only accept known keys with the right type so a broken file can't crash the game.
        for key, default in S.DEFAULT_USER_SETTINGS.items():
            value = data.get("settings", {}).get(key)
            if isinstance(value, type(default)) or (isinstance(default, float) and isinstance(value, int)):
                self.settings[key] = value
        for key in DEFAULT_STATS:
            value = data.get("stats", {}).get(key)
            if isinstance(value, int) and value >= 0:
                self.stats[key] = value
        self._sanitize()

    def _sanitize(self):
        s = self.settings
        s["music_volume"] = min(1.0, max(0.0, float(s["music_volume"])))
        s["sfx_volume"] = min(1.0, max(0.0, float(s["sfx_volume"])))
        if s["match_minutes"] not in S.MATCH_LENGTH_OPTIONS:
            s["match_minutes"] = S.DEFAULT_MATCH_MINUTES
        if s["difficulty"] not in S.DIFFICULTIES:
            s["difficulty"] = S.DEFAULT_DIFFICULTY
        if s["renderer"] not in S.RENDERERS:
            s["renderer"] = "GPU"
        if s["graphics_preset"] not in S.PRESET_NAMES:
            s["graphics_preset"] = S.DEFAULT_PRESET
        if s["fps_cap"] not in S.FPS_CAP_OPTIONS:
            s["fps_cap"] = 60
        for key, (_, levels) in S.GRAPHICS_OPTIONS.items():
            s[key] = min(len(levels) - 1, max(0, int(s[key])))

    def save(self):
        try:
            with open(self.path, "w", encoding="utf-8") as f:
                json.dump({"settings": self.settings, "stats": self.stats}, f, indent=2)
        except OSError as exc:
            print(f"[save] Could not write save file: {exc}")

    def record_match(self, player_goals, cpu_goals):
        self.stats["matches_played"] += 1
        self.stats["goals_scored"] += player_goals
        self.stats["goals_conceded"] += cpu_goals
        if player_goals > cpu_goals:
            self.stats["wins"] += 1
        else:
            self.stats["losses"] += 1
        self.save()

    def reset_stats(self):
        self.stats = dict(DEFAULT_STATS)
        self.save()
