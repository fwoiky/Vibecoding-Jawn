"""JARVIS launcher - paste this whole file into PyCharm and press Run. Nothing else to do.

It downloads jarvis.py into a "jarvis" folder next to this script, then starts it. On the very first run
jarvis.py creates its own virtual environment and installs everything it needs (5-15 min, one time).
"""
import pathlib
import subprocess
import sys
import urllib.request

URL = "https://raw.githubusercontent.com/fwoiky/Vibecoding-Jawn/claude/hopeful-wozniak-kg7811/jarvis.py"

folder = pathlib.Path(__file__).resolve().parent / "jarvis"
folder.mkdir(exist_ok=True)
target = folder / "jarvis.py"
try:
    print("Downloading the latest JARVIS...")
    target.write_bytes(urllib.request.urlopen(urllib.request.Request(URL), timeout=60).read())
except Exception as e:  # offline: reuse the copy we already have
    if not target.exists():
        sys.exit(f"Could not download JARVIS ({e}). Check your internet connection and run again.")
    print(f"Download failed ({e}); using the existing copy.")
sys.exit(subprocess.call([sys.executable, str(target), *sys.argv[1:]]))
