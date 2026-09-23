"""
Nitro Arena - a 2D rocket-car soccer game.

Run this file to play:
    python main.py
(In PyCharm: right-click main.py -> Run 'main'.)
"""
import os
import sys
import traceback

# Make imports work no matter which folder the game is started from.
GAME_DIR = os.path.dirname(os.path.abspath(__file__))
if GAME_DIR not in sys.path:
    sys.path.insert(0, GAME_DIR)


def main():
    try:
        import pygame
    except ImportError:
        print("Pygame is not installed. Install the requirements first:\n"
              "    pip install -r requirements.txt")
        sys.exit(1)

    from game import Game
    try:
        Game().run()
    except KeyboardInterrupt:
        pass
    except Exception:
        traceback.print_exc()
        pygame.quit()
        sys.exit(1)


if __name__ == "__main__":
    main()
