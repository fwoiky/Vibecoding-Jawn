"""
Builds nitro_arena_onefile.py: a single self-contained Python file that
unpacks the whole Nitro Arena project, installs its dependencies into the
Python running it, and launches the game. Handy for people who just want to
paste one file into PyCharm and press Run.

    python3 tools/build_onefile.py
"""
import base64
import os
import zlib

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GAME_DIR = os.path.join(ROOT, "nitro_arena")
OUTPUT = os.path.join(ROOT, "nitro_arena_onefile.py")
INCLUDE_EXT = (".py", ".md", ".txt")
SKIP_DIRS = {"__pycache__", ".idea", ".venv", "venv"}

TEMPLATE = '''"""
NITRO ARENA - one-file installer + launcher.

HOW TO USE (PyCharm):
  1. Create a new Python file (e.g. nitro_arena_onefile.py) and paste ALL of
     this text into it - or just open this downloaded file.
  2. Right-click in the editor -> Run.

What it does:
  * unpacks the game into a "nitro_arena" folder next to this file
  * installs pygame (and moderngl for the GPU effects) into the Python that
    PyCharm is using - no Terminal needed
  * starts the game

Run it again any time to play. The game code itself is readable in the
unpacked "nitro_arena" folder (start with nitro_arena/main.py).
"""
import base64
import importlib
import os
import subprocess
import sys
import zlib

FILES = {{
{files}
}}


def unpack(target):
    for rel_path, blob in FILES.items():
        path = os.path.join(target, *rel_path.split("/"))
        os.makedirs(os.path.dirname(path), exist_ok=True)
        data = zlib.decompress(base64.b64decode("".join(blob)))
        with open(path, "wb") as f:
            f.write(data)
    os.makedirs(os.path.join(target, "assets", "sounds"), exist_ok=True)


def pip_install(*packages):
    base = [sys.executable, "-m", "pip", "install", "--disable-pip-version-check"]
    for extra in ([], ["--user"]):
        print("Installing", ", ".join(packages), "...")
        if subprocess.call(base + extra + list(packages)) == 0:
            importlib.invalidate_caches()
            return True
    return False


def have(module):
    try:
        importlib.import_module(module)
        return True
    except ImportError:
        return False


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    target = os.path.join(here, "nitro_arena")
    unpack(target)
    print("Game files are in:", target)

    if not have("pygame"):
        if not pip_install("pygame>=2.1.3") or not have("pygame"):
            print("\\nCould not install pygame automatically.")
            print("In PyCharm: Settings -> Project -> Python Interpreter -> '+' -> search 'pygame' -> Install.")
            print("Then run this file again.")
            sys.exit(1)
    if not have("moderngl"):
        if not pip_install("moderngl>=5.8"):
            print("moderngl could not be installed - the game will use the simpler CPU renderer.")

    print("Starting Nitro Arena...")
    sys.exit(subprocess.call([sys.executable, os.path.join(target, "main.py")], cwd=target))


if __name__ == "__main__":
    main()
'''


def collect():
    files = {}
    for folder, dirs, names in os.walk(GAME_DIR):
        dirs[:] = sorted(d for d in dirs if d not in SKIP_DIRS)
        for name in sorted(names):
            if not name.endswith(INCLUDE_EXT):
                continue
            path = os.path.join(folder, name)
            rel = os.path.relpath(path, GAME_DIR).replace(os.sep, "/")
            files[rel] = open(path, "rb").read()
    return files


def main():
    entries = []
    for rel, data in collect().items():
        encoded = base64.b64encode(zlib.compress(data, 9)).decode("ascii")
        lines = [encoded[i:i + 88] for i in range(0, len(encoded), 88)]
        body = "\n".join(f'        "{line}",' for line in lines)
        entries.append(f'    "{rel}": (\n{body}\n    ),')
    with open(OUTPUT, "w", encoding="utf-8") as f:
        f.write(TEMPLATE.format(files="\n".join(entries)))
    print(f"Wrote {OUTPUT} ({os.path.getsize(OUTPUT) // 1024} KB, {len(entries)} files)")


if __name__ == "__main__":
    main()
