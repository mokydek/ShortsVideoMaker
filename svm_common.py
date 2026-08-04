#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Общие мелочи для всех скриптов ShortsVideoMaker.

Ничего умного здесь нет: вывод в консоль по-русски (в том числе на Windows,
где консоль по умолчанию не UTF-8), чтение config.yaml и поиск ffmpeg.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

CONFIG_NAME = "config.yaml"


# --------------------------------------------------------------- консоль ----
def setup_console() -> None:
    """Заставляем консоль понимать кириллицу и на Windows тоже."""
    if os.name == "nt":
        try:
            import ctypes

            ctypes.windll.kernel32.SetConsoleOutputCP(65001)
            ctypes.windll.kernel32.SetConsoleCP(65001)
        except Exception:
            pass
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


_STAGE = {"n": 0}


def stage(title: str) -> None:
    _STAGE["n"] += 1
    print(f"\n[{_STAGE['n']}] {title}", flush=True)


def info(msg: str) -> None:
    print(f"    {msg}", flush=True)


def warn(msg: str) -> None:
    print(f"    ! {msg}", flush=True)


def fail(msg: str, code: int = 1):
    print(f"\nОШИБКА: {msg}", file=sys.stderr, flush=True)
    sys.exit(code)


def progress(done: int, total: int, prefix: str = "", suffix: str = "") -> None:
    """Однострочный прогресс без внешних библиотек."""
    total = max(1, total)
    frac = min(1.0, done / total)
    width = 28
    filled = int(width * frac)
    bar = "#" * filled + "." * (width - filled)
    line = f"    {prefix}[{bar}] {frac * 100:5.1f}% {suffix}"
    sys.stdout.write("\r" + line + " " * max(0, 78 - len(line)))
    sys.stdout.flush()
    if done >= total:
        sys.stdout.write("\n")
        sys.stdout.flush()


def human_time(sec: float) -> str:
    sec = max(0.0, float(sec))
    h, rem = divmod(int(sec), 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


# ---------------------------------------------------------------- конфиг ----
DEFAULTS = {
    "subtitles": {
        "font": "Arial",
        "font_size": 96,
        "bold": True,
        "color": "#FFFFFF",
        "active_color": "#FFD400",
        "outline_color": "#000000",
        "outline": 6,
        "shadow": 2,
        "position": 0.78,
        "margin_x": 60,
        "words_per_cue": 4,
        "cue_gap": 0.55,
        "highlight_mode": "karaoke",
        "uppercase": False,
    },
    "part_label": {
        "enabled": True,
        "text": "Часть {n}",
        "font_size": 64,
        "color": "#FFFFFF",
        "outline_color": "#000000",
        "outline": 4,
        "margin_top": 90,
        "font_file": "",
    },
    "cut": {"target_len": 60, "min_len": 30, "max_len": 90, "min_pause": 0.6},
    "render": {
        "width": 1080,
        "height": 1920,
        "fps": 30,
        "video_codec": "libx264",
        "preset": "medium",
        "video_bitrate": "8M",
        "audio_codec": "aac",
        "audio_bitrate": "192k",
        "crop": 0.5,
    },
    "asr": {"model": "small", "language": "ru", "vad_filter": True, "beam_size": 5},
    "tools": {"ffmpeg": "", "ffprobe": ""},
}


def _merge(base: dict, over: dict) -> dict:
    out = dict(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path: str | os.PathLike | None = None) -> dict:
    """config.yaml поверх значений по умолчанию. Нет файла — не беда."""
    if path is None:
        path = Path(__file__).resolve().parent / CONFIG_NAME
    path = Path(path)
    if not path.exists():
        return json.loads(json.dumps(DEFAULTS))
    try:
        import yaml
    except ImportError:
        warn("PyYAML не установлен — беру настройки по умолчанию "
             "(pip install -r requirements.txt).")
        return json.loads(json.dumps(DEFAULTS))
    try:
        with open(path, "r", encoding="utf-8") as fh:
            user = yaml.safe_load(fh) or {}
    except Exception as exc:                                   # noqa: BLE001
        warn(f"Не смог прочитать {path.name} ({exc}) — беру значения по умолчанию.")
        return json.loads(json.dumps(DEFAULTS))
    return _merge(DEFAULTS, user)


# ---------------------------------------------------------------- ffmpeg ----
_WIN_GUESSES = [
    r"C:\ffmpeg\bin",
    r"C:\Program Files\ffmpeg\bin",
    r"C:\ProgramData\chocolatey\bin",
]

INSTALL_HINT = (
    "Как поставить ffmpeg:\n"
    "  Windows : winget install Gyan.FFmpeg\n"
    "            (или скачать с https://www.gyan.dev/ffmpeg/builds/ и добавить\n"
    "             папку bin в переменную PATH)\n"
    "  Ubuntu  : sudo apt install ffmpeg\n"
    "  Fedora  : sudo dnf install ffmpeg\n"
    "  macOS   : brew install ffmpeg\n"
    "Либо пропишите полный путь в config.yaml → tools.ffmpeg"
)


def find_tool(name: str, override: str = "") -> str:
    """Ищем ffmpeg/ffprobe: конфиг → переменная окружения → PATH → типовые папки."""
    if override:
        p = Path(os.path.expandvars(os.path.expanduser(override)))
        if p.is_dir():
            p = p / (name + (".exe" if os.name == "nt" else ""))
        if p.exists():
            return str(p)
        fail(f"В config.yaml указан путь к {name}, но там ничего нет: {p}")

    env = os.environ.get(name.upper())
    if env and Path(env).exists():
        return env

    found = shutil.which(name)
    if found:
        return found

    if os.name == "nt":
        for folder in _WIN_GUESSES:
            cand = Path(folder) / f"{name}.exe"
            if cand.exists():
                return str(cand)

    fail(f"Не найден {name}. Он обязателен для работы.\n\n{INSTALL_HINT}")
    return ""                                                  # до сюда не дойдём


def run(cmd: list[str], cwd: str | None = None, quiet: bool = True) -> subprocess.CompletedProcess:
    """Запуск внешней программы. Вывод забираем как UTF-8, чтобы не падать."""
    return subprocess.run(
        cmd,
        cwd=cwd,
        stdout=subprocess.PIPE if quiet else None,
        stderr=subprocess.PIPE if quiet else None,
        encoding="utf-8",
        errors="replace",
        check=False,
    )


def probe_media(ffprobe: str, path: str) -> dict:
    """Длительность и размер кадра. Возвращает {} — значит не смогли."""
    res = run([
        ffprobe, "-v", "error", "-print_format", "json",
        "-show_format", "-show_streams", str(path),
    ])
    if res.returncode != 0:
        return {}
    try:
        data = json.loads(res.stdout or "{}")
    except json.JSONDecodeError:
        return {}

    out = {"duration": 0.0, "width": 0, "height": 0, "has_audio": False, "fps": 0.0}
    try:
        out["duration"] = float(data.get("format", {}).get("duration") or 0.0)
    except (TypeError, ValueError):
        pass
    for st in data.get("streams", []):
        if st.get("codec_type") == "video" and not out["width"]:
            out["width"] = int(st.get("width") or 0)
            out["height"] = int(st.get("height") or 0)
            rate = st.get("avg_frame_rate") or st.get("r_frame_rate") or "0/1"
            try:
                num, den = rate.split("/")
                out["fps"] = float(num) / float(den) if float(den) else 0.0
            except (ValueError, ZeroDivisionError):
                pass
            if not out["duration"]:
                try:
                    out["duration"] = float(st.get("duration") or 0.0)
                except (TypeError, ValueError):
                    pass
        elif st.get("codec_type") == "audio":
            out["has_audio"] = True
    return out


# ------------------------------------------------------------------ цвета ---
def hex_to_ass(color: str, alpha: str = "00") -> str:
    """#RRGGBB → &HAABBGGRR — так цвета записаны в формате ASS."""
    c = str(color).strip().lstrip("#")
    if len(c) == 3:
        c = "".join(ch * 2 for ch in c)
    if len(c) != 6:
        c = "FFFFFF"
    try:
        r, g, b = c[0:2], c[2:4], c[4:6]
        int(c, 16)
    except ValueError:
        r, g, b = "FF", "FF", "FF"
    return f"&H{alpha}{b}{g}{r}".upper()


def find_font_file(explicit: str = "") -> str:
    """TTF для drawtext. На Windows drawtext без fontfile просто не работает."""
    if explicit:
        p = Path(os.path.expandvars(os.path.expanduser(explicit)))
        if p.exists():
            return str(p)
    guesses = []
    if os.name == "nt":
        win = Path(os.environ.get("WINDIR", r"C:\Windows")) / "Fonts"
        guesses += [win / n for n in ("arialbd.ttf", "seguisb.ttf", "segoeuib.ttf", "arial.ttf")]
    guesses += [
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
        Path("/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"),
        Path("/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf"),
        Path("/usr/share/fonts/TTF/DejaVuSans-Bold.ttf"),
        Path("/Library/Fonts/Arial Bold.ttf"),
        Path("/System/Library/Fonts/Supplemental/Arial Bold.ttf"),
    ]
    for g in guesses:
        try:
            if g.exists():
                return str(g)
        except OSError:
            continue
    return ""


def ff_escape(path: str) -> str:
    """Путь внутри строки фильтра ffmpeg: слэши вперёд, двоеточие экранируем."""
    return str(path).replace("\\", "/").replace(":", r"\:").replace("'", r"\'")
