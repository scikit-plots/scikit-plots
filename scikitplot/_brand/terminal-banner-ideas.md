# Terminal Banner Ideas — scikit-plots

Generated with `figlet` (v2.2.5). All banners render the string `scikit-plots`.
Banner files live in `.banners/` and are selected via deterministic day-of-year rotation.

---

## https://www.figlet.org/

```bash
apt install figlet toilet toilet-fonts

# showfigfonts | grep -i shadow
# find /usr/share -iname '*ANSI*'
find /usr/share -iname '*.flf'
```

```bash
# pip install pyfiglet
import pyfiglet

fonts = pyfiglet.FigletFont.getFonts()
```

Check The available .flf (figlet font) files are:

standard Debian/Ubuntu figlet packages:
- banner.flf
- big.flf
- block.flf
- bubble.flf
- digital.flf
- ivrit.flf
- lean.flf
- mini.flf
- mnemonic.flf
- script.flf
- shadow.flf
- slant.flf
- small.flf
- smscript.flf
- smshadow.flf
- smslant.flf
- standard.flf
- term.flf

ANSI Shadow font:

```bash
apt install figlet toilet toilet-fonts
# showfigfonts | grep -i shadow
# find /usr/share -iname '*ANSI*'
find /usr/share -iname '*.flf' | grep -i shadow
```

```bash
# cd /usr/share/figlet/
cd "$(figlet -I2)" && sudo wget https://raw.githubusercontent.com/xero/figlet-fonts/master/ANSI%20Shadow.flf

# figlet -f "ANSI Shadow" "Hello World"
figlet -f "ANSI Shadow" "scikit-plots"
figlet -w 500 -k -f "ANSI Shadow" "SCIKIT-PLOTS"

figlet -k -w 200 -f "ANSI Shadow" "abc"
figlet -k -w 200 -f "ANSI Shadow" "ABC"
```

```bash
sed -n '1,120p' "/usr/share/figlet/ANSI Shadow.flf"
head -120 "/usr/share/figlet/ANSI Shadow.flf"
```

---

## Rotation Overview

| File | Font | Border | Lines | Cols |
|------|------|--------|-------|------|
| `001-standard.txt` | standard | none | 6 | 50 |
| `002-slant.txt` | slant | double-line `╔══╗` | 8 | 59 |
| `003-shadow.txt` | shadow | none | 5 | 51 |
| `004-small.txt` | small | rounded `╭──╮` | 7 | 46 |
| `005-smslant.txt` | smslant | none | 5 | 47 |
| `006-big.txt` | big | solid `▓` background | 10 | 58 |

Rotation formula: `banner_index = day_of_year % total_banners` (deterministic, zero-based).

---

## 001 — Standard (no border)

```text
          _ _    _ _              _       _
 ___  ___(_) | _(_) |_      _ __ | | ___ | |_ ___
/ __|/ __| | |/ / | __|____| '_ \| |/ _ \| __/ __|
\__ \ (__| |   <| | ||_____| |_) | | (_) | |_\__ \
|___/\___|_|_|\_\_|\__|    | .__/|_|\___/ \__|___/
                           |_|
```

- **Font:** `standard`
- **Dimensions:** 6 lines × 50 cols
- **Border:** none

---

## 002 — Slant + Double-Line Box

```text
╔═════════════════════════════════════════════════════════╗
║               _ __   _ __              __      __       ║
║    __________(_) /__(_) /_      ____  / /___  / /______ ║
║   / ___/ ___/ / //_/ / __/_____/ __ \/ / __ \/ __/ ___/ ║
║  (__  ) /__/ / ,< / / /_/_____/ /_/ / / /_/ / /_(__  )  ║
║ /____/\___/_/_/|_/_/\__/     / .___/_/\____/\__/____/   ║
║                             /_/                         ║
╚═════════════════════════════════════════════════════════╝
```

- **Font:** `slant`
- **Dimensions:** 8 lines × 59 cols
- **Border:** double-line `╔══╗`

---

## 003 — Shadow (no border)

```text
          _) |   _) |             |       |
  __|  __| | |  / | __|     __ \  |  _ \  __|  __|
\__ \ (    |   <  | |_____| |   | | (   | |  \__ \
____/\___|_|_|\_\_|\__|     .__/ _|\___/ \__|____/
                           _|
```

- **Font:** `shadow`
- **Dimensions:** 5 lines × 51 cols
- **Border:** none

---

## 004 — Small + Rounded Box

```text
╭────────────────────────────────────────────╮
│         _ _   _ _            _     _       │
│  ___ __(_) |_(_) |_ ___ _ __| |___| |_ ___ │
│ (_-</ _| | / / |  _|___| '_ \ / _ \  _(_-< │
│ /__/\__|_|_\_\_|\__|   | .__/_\___/\__/__/ │
│                        |_|                 │
╰────────────────────────────────────────────╯
```

- **Font:** `small`
- **Dimensions:** 7 lines × 46 cols
- **Border:** rounded `╭──╮`

---

## 005 — Small Slant (no border)

```text
           _ __    _ __           __     __
  ___ ____(_) /__ (_) /________  / /__  / /____
 (_-</ __/ /  '_// / __/___/ _ \/ / _ \/ __(_-<
/___/\__/_/_/\_\/_/\__/   / .__/_/\___/\__/___/
                         /_/
```

- **Font:** `smslant`
- **Dimensions:** 5 lines × 47 cols
- **Border:** none

---

## 006 — Big + Solid Background

```text
▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓
▓            _ _    _ _                _       _         ▓
▓           (_) |  (_) |              | |     | |        ▓
▓   ___  ___ _| | ___| |_ ______ _ __ | | ___ | |_ ___   ▓
▓  / __|/ __| | |/ / | __|______| '_ \| |/ _ \| __/ __|  ▓
▓  \__ \ (__| |   <| | |_       | |_) | | (_) | |_\__ \  ▓
▓  |___/\___|_|_|\_\_|\__|      | .__/|_|\___/ \__|___/  ▓
▓                               | |                      ▓
▓                               |_|                      ▓
▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓
```

- **Font:** `big`
- **Dimensions:** 10 lines × 58 cols
- **Border:** solid `▓` background
---

## Unicode Full Block Background — ANSI Shadow Style

```text
███████╗ ██████╗██╗██╗  ██╗██╗████████╗      ██████╗ ██╗      ██████╗ ████████╗███████╗
██╔════╝██╔════╝██║██║ ██╔╝██║╚══██╔══╝      ██╔══██╗██║     ██╔═══██╗╚══██╔══╝██╔════╝
███████╗██║     ██║█████╔╝ ██║   ██║         ██████╔╝██║     ██║   ██║   ██║   ███████╗
╚════██║██║     ██║██╔═██╗ ██║   ██║         ██╔═══╝ ██║     ██║   ██║   ██║   ╚════██║
███████║╚██████╗██║██║  ██╗██║   ██║         ██║     ███████╗╚██████╔╝   ██║   ███████║
╚══════╝ ╚═════╝╚═╝╚═╝  ╚═╝╚═╝   ╚═╝         ╚═╝     ╚══════╝ ╚═════╝    ╚═╝   ╚══════╝
```

---

## ANSI Colour Wrappers

### Option A — ANSI Background Highlight (blue on white)

```bash
printf '\033[1;97;44m'
cat .banners/001-standard.txt
printf '\033[0m\n'
```

### Option B — Matrix / Cyberpunk (green on black)

```bash
printf '\033[1;32m'
cat .banners/006-big.txt
printf '\033[0m\n'
```

### Option C — lolcat gradient (requires `lolcat`)

```bash
cat .banners/001-standard.txt | lolcat
```

### Option D — rich (requires `rich`)

```python
from rich.console import Console
from rich.text import Text
Console().print(Text(open(".banners/001-standard.txt").read(), style="bold cyan"))
```

---

## Animated Line-by-Line Reveal

```bash
while IFS= read -r line; do
    printf '%s\n' "$line"
    sleep 0.04
done < .banners/006-big.txt
```

---

## Minimal Professional Fallback

```text
=== scikit-plots ===
```

Use when terminal width is < 50 cols or `figlet` is unavailable.

---

## Recommendation Order

| Rank | File | Why |
|------|------|-----|
| 1 | `006-big.txt` | Maximum visual impact, solid `▓` frame |
| 2 | `002-slant.txt` | Clean professional double-line box |
| 3 | `004-small.txt` | Compact, rounded, fits narrow terminals |
| 4 | `001-standard.txt` | Undecorated, timeless, widest compat |
| 5 | `005-smslant.txt` | Small footprint, still readable |
| 6 | `003-shadow.txt` | Unique shadow aesthetic |

---

## Regeneration

```bash
# Requires: apt install figlet  (Debian/Ubuntu)
#           brew install figlet (macOS)
python3 docker/scripts/.bashrc.d/_banner.py   # writes .banners/001-standard.txt … 008-big.txt
python3 -m scikitplot._brand._banner          # writes .banners/001-standard.txt … 008-big.txt
```

```python
# scikitplot._brand._banner — core generation logic
import subprocess, pathlib

BANNER_DIR = pathlib.Path(".banners")
TEXT = "scikit-plots"

def figlet(font: str) -> str:
    return subprocess.check_output(["figlet", "-f", font, TEXT], text=True).rstrip("\n")

def box_double(s: str) -> str:
    lines = s.splitlines()
    w = max(len(l) for l in lines)
    return "\n".join([f"╔{'═'*(w+2)}╗"] + [f"║ {l:<{w}} ║" for l in lines] + [f"╚{'═'*(w+2)}╝"])

def box_rounded(s: str) -> str:
    lines = s.splitlines()
    w = max(len(l) for l in lines)
    return "\n".join([f"╭{'─'*(w+2)}╮"] + [f"│ {l:<{w}} │" for l in lines] + [f"╰{'─'*(w+2)}╯"])

def box_solid(s: str) -> str:
    lines = s.splitlines()
    w = max(len(l) for l in lines)
    bar = "▓" * (w + 6)
    return "\n".join([bar] + [f"▓  {l:<{w}}  ▓" for l in lines] + [bar])
```
