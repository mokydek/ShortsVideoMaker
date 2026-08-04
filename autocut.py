#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ShortsVideoMaker — автонарезка фильма на вертикальные клипы с субтитрами.

Пример:
    python autocut.py "фильм.mp4" --out clips --target-len 60 --lang ru --model small

Что происходит:
    1. из фильма вынимается звук (16 кГц, моно);
    2. faster-whisper распознаёт речь с таймингами по каждому слову;
    3. фильм режется на части ~target-len секунд, но только по паузам в речи;
    4. на каждую часть пишется .ass с караоке-субтитрами;
    5. ffmpeg рендерит части в 1080x1920 с вшитыми субтитрами.

Транскрипт кешируется рядом с фильмом, готовые части не пересчитываются —
повторный запуск продолжает с места остановки.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from svm_common import (
    fail,
    ff_escape,
    find_font_file,
    find_tool,
    hex_to_ass,
    human_time,
    info,
    load_config,
    probe_media,
    progress,
    run,
    setup_console,
    stage,
    warn,
)

TRANSCRIPT_VERSION = 2
PUNCT_END = ".!?…:;"


# =========================================================== разбор аргументов
def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="autocut.py",
        description="Нарезает фильм на короткие вертикальные клипы с субтитрами.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Пример:\n"
               '  python autocut.py "фильм.mp4" --out clips --target-len 60 '
               "--min-len 30 --max-len 90 --lang ru --model small --crop 0.5",
    )
    p.add_argument("video", help="исходный файл: фильм, серия, мультфильм")
    p.add_argument("--out", default="clips", help="папка для готовых клипов (по умолчанию clips)")
    p.add_argument("--target-len", type=float, default=None, help="желаемая длина части, сек")
    p.add_argument("--min-len", type=float, default=None, help="минимальная длина части, сек")
    p.add_argument("--max-len", type=float, default=None, help="максимальная длина части, сек")
    p.add_argument("--lang", default=None, help="язык речи: ru, en, auto…")
    p.add_argument("--model", default=None,
                   help="модель whisper: tiny, base, small, medium, large-v3")
    p.add_argument("--crop", type=float, default=None,
                   help="сдвиг кадра при обрезке в 9:16: 0 — левый край, 0.5 — центр, 1 — правый")
    p.add_argument("--jobs", type=int, default=0,
                   help="сколько частей рендерить одновременно (по умолчанию — по числу ядер)")
    p.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto",
                   help="на чём распознавать речь (по умолчанию auto: CUDA, если есть)")
    p.add_argument("--no-label", action="store_true", help="не рисовать надпись «Часть N»")
    p.add_argument("--no-subs", action="store_true", help="не вшивать субтитры")
    p.add_argument("--force", action="store_true", help="перерисовать уже готовые части")
    p.add_argument("--force-subs", action="store_true",
                   help="перегенерировать .ass, затерев ручные правки")
    p.add_argument("--force-transcribe", action="store_true", help="распознать заново, игнорируя кеш")
    p.add_argument("--keep-audio", action="store_true", help="не удалять извлечённый wav")
    p.add_argument("--limit", type=int, default=0, help="сделать только первые N частей (для пробы)")
    p.add_argument("--config", default=None, help="путь к config.yaml")
    return p.parse_args(argv)


# ================================================================ звук =======
def extract_audio(ffmpeg: str, src: Path, wav: Path) -> None:
    wav.parent.mkdir(parents=True, exist_ok=True)
    res = run([
        ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
        "-i", str(src), "-vn", "-ac", "1", "-ar", "16000",
        "-c:a", "pcm_s16le", str(wav),
    ])
    if res.returncode != 0 or not wav.exists():
        fail("Не удалось извлечь звук из файла.\n"
             f"ffmpeg сказал: {(res.stderr or '').strip()[:600]}")


# ========================================================= распознавание =====
def pick_device(requested: str) -> tuple[str, str]:
    """Возвращает (device, compute_type)."""
    if requested == "cpu":
        return "cpu", "int8"
    try:
        import ctranslate2

        cuda = ctranslate2.get_cuda_device_count() > 0
    except Exception:                                          # noqa: BLE001
        cuda = False
    if requested == "cuda":
        if not cuda:
            warn("CUDA запрошена, но видеокарта не найдена — считаю на процессоре.")
            return "cpu", "int8"
        return "cuda", "float16"
    return ("cuda", "float16") if cuda else ("cpu", "int8")


def transcript_path(video: Path, model: str, lang: str) -> Path:
    return video.with_name(f"{video.stem}.transcript.{model}.{lang}.json")


def load_transcript(path: Path, video: Path, model: str, lang: str) -> dict | None:
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (json.JSONDecodeError, OSError):
        return None
    if data.get("version") != TRANSCRIPT_VERSION:
        return None
    src = data.get("source") or {}
    try:
        st = video.stat()
    except OSError:
        return None
    if src.get("size") != st.st_size or data.get("model") != model or data.get("asked_language") != lang:
        return None
    return data


CUDA_HINT = (
    "Видеокарта есть, но библиотек CUDA для неё нет. Ставится одной командой:\n"
    "      pip install nvidia-cublas-cu12 nvidia-cudnn-cu12\n"
    "    Либо просто работайте на процессоре — это медленнее, но результат тот же."
)


def _is_cuda_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return any(k in msg for k in
               ("cublas", "cudnn", "cuda", "gpu", "libcu", ".dll is not found",
                "cannot be loaded", "out of memory"))


def _wav_seconds(wav: Path) -> float:
    try:
        import wave

        with wave.open(str(wav), "rb") as fh:
            return fh.getnframes() / float(fh.getframerate() or 16000)
    except Exception:                                          # noqa: BLE001
        return 0.0


def _asr_pass(wav: Path, model_name: str, lang: str, device: str, compute: str,
              cfg: dict, total: float) -> tuple[list[dict], list[dict], str]:
    """Одна попытка распознавания целиком, включая перебор сегментов.

    Важно: faster-whisper отдаёт сегменты лениво, поэтому отсутствие cuBLAS
    вылезает не при создании модели, а вот здесь, на первом же сегменте.
    """
    from faster_whisper import WhisperModel

    model = WhisperModel(model_name, device=device, compute_type=compute)
    segments, meta = model.transcribe(
        str(wav),
        language=None if lang == "auto" else lang,
        word_timestamps=True,
        vad_filter=bool(cfg["asr"].get("vad_filter", True)),
        beam_size=int(cfg["asr"].get("beam_size", 5)),
    )
    detected = getattr(meta, "language", None) or lang

    words: list[dict] = []
    seg_list: list[dict] = []
    last_print = 0.0
    for seg in segments:
        seg_list.append({"s": round(seg.start, 3), "e": round(seg.end, 3),
                         "text": (seg.text or "").strip()})
        for w in (seg.words or []):
            text = (w.word or "").strip()
            if not text:
                continue
            words.append({"w": text, "s": round(float(w.start), 3), "e": round(float(w.end), 3)})
        now = time.time()
        if total and (now - last_print > 0.3):
            last_print = now
            progress(min(seg.end, total), total, "распознавание ",
                     f"{human_time(seg.end)} / {human_time(total)}")
    if total:
        progress(total, total, "распознавание ", f"{human_time(total)} / {human_time(total)}")
    return words, seg_list, detected


def transcribe(wav: Path, video: Path, model_name: str, lang: str,
               device_req: str, cfg: dict) -> dict:
    # эти предупреждения huggingface пугают новичков, а делать с ними нечего
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    try:
        import logging

        for name in ("huggingface_hub", "hf_xet", "faster_whisper"):
            logging.getLogger(name).setLevel(logging.ERROR)
    except Exception:                                          # noqa: BLE001
        pass

    try:
        import faster_whisper                                  # noqa: F401
    except ImportError:
        fail("Не установлен faster-whisper.\n"
             "Поставьте зависимости:  pip install -r requirements.txt")

    device, compute = pick_device(device_req)
    total = _wav_seconds(wav)
    t0 = time.time()

    info(f"устройство: {device} ({compute}), модель: {model_name}")
    if device == "cpu":
        info("на процессоре это долго: примерно 30–60 минут на двухчасовой фильм "
             "моделью small. С видеокартой NVIDIA — в разы быстрее.")

    try:
        words, seg_list, detected = _asr_pass(wav, model_name, lang, device, compute, cfg, total)
    except Exception as exc:                                   # noqa: BLE001
        if device == "cuda" and _is_cuda_error(exc):
            print()
            warn(f"CUDA отвалилась: {str(exc).strip()[:160]}")
            warn(CUDA_HINT)
            warn("продолжаю на процессоре — это дольше, но работает.")
            device, compute = "cpu", "int8"
            try:
                words, seg_list, detected = _asr_pass(wav, model_name, lang, device,
                                                      compute, cfg, total)
            except Exception as exc2:                          # noqa: BLE001
                fail(f"Распознавание не удалось и на процессоре: {exc2}")
        elif isinstance(exc, (OSError, ValueError, RuntimeError)):
            fail(f"Распознавание не удалось: {exc}")
        else:
            raise

    if lang == "auto":
        info(f"язык определён как «{detected}»")
    info(f"слов распознано: {len(words)}, время: {human_time(time.time() - t0)}")
    if not words:
        warn("речь не найдена — клипы всё равно будут нарезаны, просто без субтитров.")

    try:
        st = video.stat()
        source = {"name": video.name, "size": st.st_size, "mtime": int(st.st_mtime)}
    except OSError:
        source = {}
    return {
        "version": TRANSCRIPT_VERSION,
        "model": model_name,
        "asked_language": lang,
        "language": detected,
        "device": device,
        "source": source,
        "words": words,
        "segments": seg_list,
    }


# ============================================================== нарезка ======
def split_parts(words: list[dict], duration: float, target: float,
                min_len: float, max_len: float, min_pause: float) -> list[tuple[float, float]]:
    """Режем фильм на последовательные части, стараясь попадать в паузы речи."""
    gaps = []
    for a, b in zip(words, words[1:]):
        g = float(b["s"]) - float(a["e"])
        if g > 0.05:
            gaps.append({"len": g, "cut": (float(a["e"]) + float(b["s"])) / 2.0})

    parts: list[tuple[float, float]] = []
    pos = 0.0
    guard = 0
    while duration - pos > 0.05:
        guard += 1
        if guard > 100000:                                     # страховка от вечного цикла
            break
        if duration - pos <= max_len:
            parts.append((pos, duration))
            break
        lo, hi = pos + min_len, pos + max_len
        window = [g for g in gaps if lo <= g["cut"] <= hi]
        good = [g for g in window if g["len"] >= min_pause]
        if good:
            cut = min(good, key=lambda g: abs(g["cut"] - (pos + target)))["cut"]
        elif window:
            # подходящей паузы нет — берём самую длинную из тех, что есть
            cut = max(window, key=lambda g: g["len"])["cut"]
        else:
            # речи в этом окне вообще нет — режем ровно по времени
            cut = pos + target
        cut = min(max(cut, pos + min_len), pos + max_len, duration)
        parts.append((pos, cut))
        pos = cut

    # хвост короче минимума приклеиваем к предыдущей части, если она это выдержит
    if len(parts) >= 2:
        s, e = parts[-1]
        if e - s < min_len:
            ps, _ = parts[-2]
            if e - ps <= max_len:
                parts[-2] = (ps, e)
                parts.pop()
    return parts


def group_words(words: list[dict], max_words: int, gap: float) -> list[list[dict]]:
    """Слова → реплики по 3–4 штуки, с разрывом на паузах и точках."""
    cues: list[list[dict]] = []
    cur: list[dict] = []
    for i, w in enumerate(words):
        cur.append(w)
        nxt = words[i + 1] if i + 1 < len(words) else None
        text = w["w"].strip()
        ends_sentence = bool(text) and text[-1] in PUNCT_END
        pause = (float(nxt["s"]) - float(w["e"])) if nxt else 1e9
        too_long = sum(len(x["w"]) for x in cur) > 26
        if (nxt is None or len(cur) >= max_words or too_long
                or (ends_sentence and len(cur) >= 2) or pause > gap):
            cues.append(cur)
            cur = []
    if cur:
        cues.append(cur)
    return cues


# ================================================================= ASS =======
def ass_time(t: float) -> str:
    cs = int(round(max(0.0, t) * 100))
    h, cs = divmod(cs, 360000)
    m, cs = divmod(cs, 6000)
    s, cs = divmod(cs, 100)
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


def ass_text(s: str) -> str:
    return (str(s).replace("\\", "/").replace("{", "(").replace("}", ")")
            .replace("\r", " ").replace("\n", " ").strip())


def build_ass(cues: list[list[dict]], cfg: dict) -> str:
    sub = cfg["subtitles"]
    W = int(cfg["render"]["width"])
    H = int(cfg["render"]["height"])
    size = int(sub["font_size"])
    karaoke = str(sub.get("highlight_mode", "karaoke")).lower() != "current"

    # В режиме \k «спетое» слово красится в PrimaryColour, ещё не спетое —
    # в SecondaryColour. В режиме current цвет активного слова ставится
    # прямо в строке тегом \c, поэтому Primary — обычный белый.
    normal = hex_to_ass(sub["color"])
    active = hex_to_ass(sub["active_color"])
    primary = active if karaoke else normal
    secondary = normal if karaoke else active

    margin_v = max(10, int(round(H * (1.0 - float(sub["position"])) - size / 2)))
    bold = -1 if sub.get("bold", True) else 0

    head = (
        "[Script Info]\n"
        "; Сгенерировано ShortsVideoMaker\n"
        "ScriptType: v4.00+\n"
        f"PlayResX: {W}\n"
        f"PlayResY: {H}\n"
        "WrapStyle: 0\n"
        "ScaledBorderAndShadow: yes\n"
        "YCbCr Matrix: TV.709\n"
        "\n"
        "[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, "
        "BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, "
        "BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\n"
        f"Style: Shorts,{sub['font']},{size},{primary},{secondary},"
        f"{hex_to_ass(sub['outline_color'])},&H80000000,{bold},0,0,0,100,100,0,0,1,"
        f"{sub['outline']},{sub['shadow']},2,{sub['margin_x']},{sub['margin_x']},{margin_v},1\n"
        "\n"
        "[Events]\n"
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
    )

    upper = bool(sub.get("uppercase", False))
    lines = []
    for cue in cues:
        if not cue:
            continue
        start = float(cue[0]["s"])
        end = max(float(cue[-1]["e"]), start + 0.2)
        texts = [ass_text(w["w"]) for w in cue]
        if upper:
            texts = [t.upper() for t in texts]

        if karaoke:
            chunks = []
            for i, w in enumerate(cue):
                w_start = max(start, float(w["s"]))
                w_end = max(w_start + 0.05, float(w["e"]))
                dur_cs = max(1, int(round((w_end - w_start) * 100)))
                tail = " " if i < len(cue) - 1 else ""
                chunks.append(f"{{\\k{dur_cs}}}{texts[i]}{tail}")
            lines.append(f"Dialogue: 0,{ass_time(start)},{ass_time(end)},Shorts,,0,0,0,,"
                         + "".join(chunks))
        else:
            for i, w in enumerate(cue):
                w_start = max(start, float(w["s"]))
                w_end = float(cue[i + 1]["s"]) if i + 1 < len(cue) else end
                w_end = max(w_start + 0.05, w_end)
                painted = []
                for j, t in enumerate(texts):
                    painted.append(f"{{\\c{active}}}{t}{{\\c{normal}}}" if j == i else t)
                lines.append(f"Dialogue: 0,{ass_time(w_start)},{ass_time(w_end)},Shorts,,0,0,0,,"
                             + " ".join(painted))

    return head + "\n".join(lines) + "\n"


# ============================================================== рендер ======
def even(n: float) -> int:
    return max(2, int(round(n / 2.0)) * 2)


def crop_rect(src_w: int, src_h: int, out_w: int, out_h: int, shift: float) -> tuple[int, int, int, int]:
    shift = min(1.0, max(0.0, float(shift)))
    target = out_w / float(out_h)
    if src_w / float(src_h) > target:                          # шире, чем надо — режем бока
        cw = min(src_w, even(src_h * target))
        ch = even(src_h)
        ch = min(ch, src_h)
        x = int(round(shift * (src_w - cw)))
        y = 0
    else:                                                      # уже — режем верх и низ
        cw = even(src_w)
        cw = min(cw, src_w)
        ch = min(src_h, even(src_w / target))
        x = 0
        y = int(round(shift * (src_h - ch)))
    x = max(0, min(x, src_w - cw))
    y = max(0, min(y, src_h - ch))
    return cw, ch, x, y


def build_filters(media: dict, cfg: dict, ass_name: str | None,
                  label_file: str | None, font_file: str) -> str:
    r = cfg["render"]
    out_w, out_h = int(r["width"]), int(r["height"])
    cw, ch, x, y = crop_rect(media["width"], media["height"], out_w, out_h, r["crop"])

    chain = [
        f"crop={cw}:{ch}:{x}:{y}",
        f"scale={out_w}:{out_h}:flags=lanczos",
        "setsar=1",
        f"fps={int(r['fps'])}",
    ]
    if ass_name:
        chain.append(f"ass='{ass_name}'")
    if label_file:
        lab = cfg["part_label"]
        col = str(lab["color"]).lstrip("#")
        bcol = str(lab["outline_color"]).lstrip("#")
        # Значения путей обязательно в одинарных кавычках: без них ffmpeg
        # спотыкается об экранированное двоеточие в C\:/Windows/... и решает,
        # что ему передали и text, и textfile сразу.
        opts = [
            f"textfile='{label_file}'",
            f"fontsize={int(lab['font_size'])}",
            f"fontcolor=0x{col}",
            f"borderw={int(lab['outline'])}",
            f"bordercolor=0x{bcol}",
            "x=(w-text_w)/2",
            f"y={int(lab['margin_top'])}",
        ]
        if font_file:
            opts.insert(0, f"fontfile='{ff_escape(font_file)}'")
        chain.append("drawtext=" + ":".join(opts))
    return ",".join(chain)


def render_part(ffmpeg: str, src: Path, out_file: Path, start: float, dur: float,
                filters: str, cfg: dict, has_audio: bool, workdir: Path) -> tuple[bool, str]:
    r = cfg["render"]
    cmd = [ffmpeg, "-y", "-hide_banner", "-loglevel", "error", "-accurate_seek",
           "-ss", f"{start:.3f}", "-i", str(src)]
    if not has_audio:
        cmd += ["-f", "lavfi", "-t", f"{dur:.3f}",
                "-i", "anullsrc=channel_layout=stereo:sample_rate=44100"]
    cmd += ["-t", f"{dur:.3f}", "-vf", filters,
            "-map", "0:v:0", "-map", ("1:a:0" if not has_audio else "0:a:0?"),
            "-c:v", str(r["video_codec"]), "-preset", str(r["preset"]),
            "-b:v", str(r["video_bitrate"]), "-maxrate", str(r["video_bitrate"]),
            "-bufsize", "16M", "-pix_fmt", "yuv420p",
            "-c:a", str(r["audio_codec"]), "-b:a", str(r["audio_bitrate"]), "-ar", "48000",
            "-movflags", "+faststart", "-shortest", str(out_file)]
    res = run(cmd, cwd=str(workdir))
    if res.returncode == 0 and out_file.exists() and out_file.stat().st_size > 1000:
        return True, ""
    return False, (res.stderr or res.stdout or "").strip()[:700]


# ================================================================= main =====
def main(argv=None) -> int:
    setup_console()
    args = parse_args(argv)
    cfg = load_config(args.config)

    # флаги главнее конфига
    if args.target_len is not None:
        cfg["cut"]["target_len"] = args.target_len
    if args.min_len is not None:
        cfg["cut"]["min_len"] = args.min_len
    if args.max_len is not None:
        cfg["cut"]["max_len"] = args.max_len
    if args.lang is not None:
        cfg["asr"]["language"] = args.lang
    if args.model is not None:
        cfg["asr"]["model"] = args.model
    if args.crop is not None:
        cfg["render"]["crop"] = args.crop
    if args.no_label:
        cfg["part_label"]["enabled"] = False

    target = float(cfg["cut"]["target_len"])
    min_len = float(cfg["cut"]["min_len"])
    max_len = float(cfg["cut"]["max_len"])
    if not (0 < min_len <= target <= max_len):
        fail(f"Неправильные длины: нужно 0 < min-len ({min_len}) <= target-len ({target}) "
             f"<= max-len ({max_len}).")
    if not (0.0 <= float(cfg["render"]["crop"]) <= 1.0):
        fail(f"--crop должен быть от 0 до 1, а не {cfg['render']['crop']}.")

    video = Path(args.video).expanduser()
    if not video.exists():
        fail(f"Файл не найден: {video}")
    video = video.resolve()

    out_dir = Path(args.out).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    ffmpeg = find_tool("ffmpeg", cfg["tools"].get("ffmpeg", ""))
    ffprobe = find_tool("ffprobe", cfg["tools"].get("ffprobe", ""))

    print("=" * 64)
    print("  ShortsVideoMaker — автонарезка")
    print("=" * 64)
    info(f"фильм: {video.name}")
    info(f"клипы: {out_dir}")

    # ---------------------------------------------------- 1. что за файл ----
    stage("Читаю файл")
    media = probe_media(ffprobe, str(video))
    if not media or not media.get("width") or not media.get("duration"):
        fail("Не удалось прочитать видео. Возможно, файл битый или это не видео.\n"
             "Проверьте вручную:  ffprobe \"" + str(video) + "\"")
    info(f"{media['width']}x{media['height']}, {human_time(media['duration'])}, "
         f"звук: {'есть' if media['has_audio'] else 'нет'}")
    if media["duration"] < min_len:
        fail(f"Фильм короче минимальной длины части ({human_time(media['duration'])} "
             f"< {min_len:.0f} с). Уменьшите --min-len.")

    # ------------------------------------------------- 2. распознавание ----
    model_name = str(cfg["asr"]["model"])
    lang = str(cfg["asr"]["language"])
    tpath = transcript_path(video, model_name, lang)

    data = None if args.force_transcribe else load_transcript(tpath, video, model_name, lang)
    if data:
        stage("Распознавание речи")
        info(f"беру готовый транскрипт: {tpath.name} (слов: {len(data.get('words', []))})")
    elif not media["has_audio"]:
        stage("Распознавание речи")
        warn("в файле нет звуковой дорожки — субтитров не будет.")
        data = {"words": [], "segments": [], "language": lang}
    else:
        stage("Извлекаю звук")
        wav = out_dir / f".{video.stem}.16k.wav"
        extract_audio(ffmpeg, video, wav)
        info(f"готово: {wav.name} ({wav.stat().st_size / 1048576:.1f} МБ)")

        stage("Распознавание речи")
        data = transcribe(wav, video, model_name, lang, args.device, cfg)
        try:
            with open(tpath, "w", encoding="utf-8") as fh:
                json.dump(data, fh, ensure_ascii=False)
            info(f"транскрипт сохранён: {tpath.name} (повторный запуск его переиспользует)")
        except OSError as exc:
            warn(f"не смог сохранить транскрипт ({exc}) — в следующий раз придётся заново.")
        if not args.keep_audio:
            try:
                wav.unlink()
            except OSError:
                pass

    words = data.get("words") or []

    # ------------------------------------------------------- 3. нарезка ----
    stage("Считаю границы частей")
    parts = split_parts(words, float(media["duration"]), target, min_len, max_len,
                        float(cfg["cut"]["min_pause"]))
    if not parts:
        fail("Не получилось разбить фильм на части — проверьте --min-len/--max-len.")
    if args.limit > 0:
        parts = parts[: args.limit]
        info(f"ограничение --limit: беру только первые {len(parts)}")
    lens = [e - s for s, e in parts]
    info(f"частей: {len(parts)}, длины от {min(lens):.1f} до {max(lens):.1f} с "
         f"(в среднем {sum(lens) / len(lens):.1f} с)")
    off = [i + 1 for i, ln in enumerate(lens) if ln < min_len - 0.5 or ln > max_len + 0.5]
    if off:
        warn(f"вне диапазона [{min_len:.0f}; {max_len:.0f}] с: части {off} — "
             "так бывает на хвосте фильма.")

    # -------------------------------------------------- 4. субтитры .ass ---
    stage("Готовлю субтитры")
    ass_files: dict[int, str] = {}
    if args.no_subs:
        info("субтитры отключены флагом --no-subs")
    else:
        made = kept = 0
        for idx, (s, e) in enumerate(parts, start=1):
            name = f"{idx:03d}.ass"
            path = out_dir / name
            # Уже лежащий рядом .ass не трогаем: его мог поправить руками
            # пользователь, и переписать его было бы свинством.
            if path.exists() and path.stat().st_size > 0 and not args.force_subs:
                ass_files[idx] = name
                kept += 1
                continue
            inside = [w for w in words if float(w["s"]) >= s - 0.05 and float(w["s"]) < e]
            rel = [{"w": w["w"], "s": float(w["s"]) - s, "e": min(float(w["e"]), e) - s}
                   for w in inside]
            rel = [w for w in rel if w["e"] > w["s"]]
            if not rel:
                continue
            cues = group_words(rel, int(cfg["subtitles"]["words_per_cue"]),
                               float(cfg["subtitles"]["cue_gap"]))
            path.write_text(build_ass(cues, cfg), encoding="utf-8")
            ass_files[idx] = name
            made += 1
        if kept:
            info(f"готовые .ass оставлены как есть: {kept} "
                 "(--force-subs, чтобы перегенерировать)")
        info(f"файлов .ass: {len(ass_files)} из {len(parts)}"
             + ("" if words else " — распознанной речи нет"))

    # ------------------------------------------------------- 5. рендер -----
    stage("Рендерю клипы")
    font_file = ""
    if cfg["part_label"]["enabled"]:
        font_file = find_font_file(cfg["part_label"].get("font_file", ""))
        if not font_file:
            warn("не нашёл TTF-шрифт для надписи «Часть N» — попробую системный, "
                 "а если не выйдет, отрисую клип без надписи.")

    jobs = args.jobs if args.jobs > 0 else min(8, max(1, (os.cpu_count() or 2)))
    info(f"параллельно: {jobs}")

    todo = []
    skipped = 0
    for idx, (s, e) in enumerate(parts, start=1):
        out_file = out_dir / f"{idx:03d}.mp4"
        if out_file.exists() and out_file.stat().st_size > 1000 and not args.force:
            skipped += 1
            continue
        todo.append((idx, s, e, out_file))
    if skipped:
        info(f"уже готовы и пропущены: {skipped} (--force, чтобы перерисовать)")

    lock = threading.Lock()
    done = {"n": 0}
    errors: list[str] = []
    no_label: list[int] = []
    no_label_reason = {"why": ""}
    t0 = time.time()

    def work(item):
        idx, s, e, out_file = item
        dur = e - s
        label_file = None
        if cfg["part_label"]["enabled"]:
            label_file = f"cap{idx:03d}.txt"
            text = str(cfg["part_label"]["text"]).replace("{n}", str(idx)).replace(
                "{total}", str(len(parts)))
            (out_dir / label_file).write_text(text, encoding="utf-8")
        filters = build_filters(media, cfg, ass_files.get(idx), label_file, font_file)
        ok, err = render_part(ffmpeg, video, out_file, s, dur, filters, cfg,
                              media["has_audio"], out_dir)
        if not ok and label_file:
            # чаще всего спотыкается именно drawtext (не нашёлся шрифт) —
            # для пользователя лучше клип без надписи, чем никакого клипа
            filters2 = build_filters(media, cfg, ass_files.get(idx), None, "")
            ok2, err2 = render_part(ffmpeg, video, out_file, s, dur, filters2, cfg,
                                    media["has_audio"], out_dir)
            if ok2:
                with lock:
                    no_label.append(idx)
                    if not no_label_reason["why"]:
                        no_label_reason["why"] = (err.splitlines()[0] if err else "без причины")
                ok, err = True, ""
            else:
                err = err or err2
        with lock:
            done["n"] += 1
            if not ok:
                errors.append(f"часть {idx}: {err}")
            progress(done["n"], len(todo), "рендер        ",
                     f"{done['n']}/{len(todo)}")

    if todo:
        with ThreadPoolExecutor(max_workers=jobs) as pool:
            list(pool.map(work, todo))
        info(f"время рендера: {human_time(time.time() - t0)}")
    else:
        info("рендерить нечего — всё уже готово.")

    if no_label:
        warn(f"надпись «Часть N» не отрисовалась на частях {no_label} — сделаны без неё. "
             f"Причина: {no_label_reason['why']}")
        warn("проверьте subtitles/part_label → font_file в config.yaml "
             "или запускайте с --no-label.")
    for e in errors:
        warn(e)

    # ------------------------------------------------------ 6. манифест ----
    stage("Пишу manifest.json")
    manifest = {
        "source": video.name,
        "source_duration": round(float(media["duration"]), 3),
        "language": data.get("language", lang),
        "model": model_name,
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "render": {"width": cfg["render"]["width"], "height": cfg["render"]["height"],
                   "crop": cfg["render"]["crop"]},
        "parts": [],
    }
    ready = 0
    for idx, (s, e) in enumerate(parts, start=1):
        out_file = out_dir / f"{idx:03d}.mp4"
        inside = [w for w in words if float(w["s"]) >= s - 0.05 and float(w["s"]) < e]
        text = " ".join(w["w"] for w in inside).strip()
        exists = out_file.exists() and out_file.stat().st_size > 1000
        ready += 1 if exists else 0
        manifest["parts"].append({
            "n": idx,
            "file": out_file.name,
            "ready": exists,
            "start": round(s, 3),
            "end": round(e, 3),
            "duration": round(e - s, 3),
            "start_hms": human_time(s),
            "text": text,
        })
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    info(f"готовых клипов: {ready} из {len(parts)}")

    # убираем временные подписи
    for f in out_dir.glob("cap*.txt"):
        try:
            f.unlink()
        except OSError:
            pass

    print("\n" + "=" * 64)
    if errors:
        print(f"  Готово с замечаниями: {ready} из {len(parts)} частей в {out_dir}")
    else:
        print(f"  Готово: {ready} частей в {out_dir}")
    print("=" * 64)
    print("\nДальше можно:")
    print(f"  python upload_youtube.py \"{out_dir}\" --title \"Название — часть {{n}} #shorts\"")
    print(f"  python upload_tiktok.py \"{out_dir}\"")
    return 0 if ready else 2


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nПрервано пользователем. Уже готовые части сохранены — "
              "запустите команду ещё раз, чтобы продолжить.")
        sys.exit(130)
