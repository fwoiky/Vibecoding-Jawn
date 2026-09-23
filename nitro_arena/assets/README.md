# Assets

Nitro Arena draws every graphic and synthesises every sound in code, so this
folder is intentionally almost empty and the game never needs a file from it.

## Optional sound overrides

Drop an audio file into `assets/sounds/` with one of these names (`.wav`,
`.ogg` or `.mp3`) to replace the built-in sound:

| Name | When it plays |
|------|---------------|
| `music` | Background music loop |
| `jump`, `double_jump`, `dodge` | Car jumps and flips |
| `hit_soft`, `hit_hard` | Car hits the ball |
| `bounce`, `land` | Ball / car impacts |
| `boost_loop` | While boosting (loops) |
| `goal`, `whistle`, `win`, `lose` | Goals and match end |
| `countdown`, `go` | Kickoff countdown |
| `menu_move`, `menu_select`, `menu_back` | Menus |

Missing or broken files are ignored and the built-in sound is used instead.
