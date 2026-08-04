#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Отправка клипов в черновики TikTok через официальный Content Posting API.

Пример:
    python upload_tiktok.py clips --per-run 3

Как это работает: видео уходит в «инбокс» вашего аккаунта. В приложении TikTok
приходит уведомление, вы открываете черновик, при желании правите описание и
публикуете одним касанием. Прямая публикация без подтверждения возможна только
после того, как TikTok одобрит ваше приложение (audit) — до этого режим
черновиков единственный доступный, и это нормально.

Ключи берутся из переменных окружения TIKTOK_CLIENT_KEY / TIKTOK_CLIENT_SECRET
и TIKTOK_ACCESS_TOKEN, либо из файла tiktok_secret.json рядом со скриптом:

    {
      "client_key": "...",
      "client_secret": "...",
      "access_token": "act...",
      "refresh_token": "rft..."
    }
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

from svm_common import fail, info, setup_console, stage, warn

API_INIT = "https://open.tiktokapis.com/v2/post/publish/inbox/video/init/"
API_STATUS = "https://open.tiktokapis.com/v2/post/publish/status/fetch/"
STATE_NAME = "tiktok_uploaded.json"
SECRET_NAME = "tiktok_secret.json"
CHUNK = 10 * 1024 * 1024                     # TikTok просит куски не меньше 5 МБ

SETUP_HINT = """\
Ключей TikTok нет — это не поломка, просто публикация не настроена.

Готовые клипы никуда не делись: они лежат в папке с клипами, их можно
загрузить в TikTok вручную с телефона или через веб-версию.

Если хотите автоматизировать, коротко:
  1. developers.tiktok.com → Manage apps → создайте приложение;
  2. добавьте продукт «Content Posting API» и включите Direct Post / Inbox;
  3. пройдите Login Kit, получите access_token для своего аккаунта
     (scope: video.upload, а для прямой публикации ещё video.publish);
  4. положите ключи в переменные окружения или в tiktok_secret.json.
Подробнее — в README, раздел «TikTok: настройка с нуля»."""


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        prog="upload_tiktok.py",
        description="Отправляет клипы в черновики TikTok.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Пример:\n  python upload_tiktok.py clips --per-run 3",
    )
    p.add_argument("folder", help="папка с клипами")
    p.add_argument("--per-run", type=int, default=5, help="сколько клипов отправить за запуск")
    p.add_argument("--secret", default=SECRET_NAME, help="путь к файлу с ключами")
    p.add_argument("--dry-run", action="store_true", help="только показать план")
    return p.parse_args(argv)


def load_secrets(path: Path) -> dict:
    """Ключи из окружения, затем из файла. Пусто — значит не настроено."""
    creds = {
        "client_key": os.environ.get("TIKTOK_CLIENT_KEY", ""),
        "client_secret": os.environ.get("TIKTOK_CLIENT_SECRET", ""),
        "access_token": os.environ.get("TIKTOK_ACCESS_TOKEN", ""),
        "refresh_token": os.environ.get("TIKTOK_REFRESH_TOKEN", ""),
    }
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8-sig"))
            for k in creds:
                if not creds[k] and data.get(k):
                    creds[k] = str(data[k])
        except (json.JSONDecodeError, OSError) as exc:
            warn(f"{path.name} не читается ({exc}) — считаю, что ключей нет.")
    return creds


def load_state(path: Path) -> dict:
    if not path.exists():
        return {"uploaded": {}}
    try:
        # utf-8-sig: Блокнот и PowerShell дописывают BOM, а из-за него разбор
        # падал бы и клипы уехали бы в черновики по второму разу
        data = json.loads(path.read_text(encoding="utf-8-sig"))
    except (json.JSONDecodeError, OSError):
        warn(f"{path.name} не читается — считаю, что ничего ещё не отправлено. "
             "Проверьте файл, иначе клипы уйдут в черновики повторно!")
        return {"uploaded": {}}
    data.setdefault("uploaded", {})
    return data


def send_one(requests_mod, token: str, clip: Path, title: str) -> tuple[bool, str]:
    """init → PUT файла кусками. Возвращает (успех, publish_id или ошибка)."""
    size = clip.stat().st_size
    chunk_size = size if size <= CHUNK else CHUNK
    total_chunks = 1 if size <= CHUNK else (size + chunk_size - 1) // chunk_size

    body = {
        "source_info": {
            "source": "FILE_UPLOAD",
            "video_size": size,
            "chunk_size": chunk_size,
            "total_chunk_count": total_chunks,
        }
    }
    try:
        r = requests_mod.post(
            API_INIT,
            headers={"Authorization": f"Bearer {token}",
                     "Content-Type": "application/json; charset=UTF-8"},
            json=body, timeout=60)
    except Exception as exc:                                   # noqa: BLE001
        return False, f"сеть недоступна: {exc}"

    if r.status_code == 401:
        return False, "TOKEN:токен не принят (401). Скорее всего, истёк — получите новый."
    if r.status_code == 403:
        return False, ("SCOPE:доступ запрещён (403). Проверьте, что приложению выдан "
                       "scope video.upload и подключён Content Posting API.")
    try:
        data = r.json()
    except ValueError:
        return False, f"непонятный ответ TikTok ({r.status_code}): {r.text[:200]}"

    err = (data.get("error") or {})
    if err.get("code") not in (None, "ok"):
        return False, f"{err.get('code')}: {err.get('message', '')}"
    upload_url = (data.get("data") or {}).get("upload_url")
    publish_id = (data.get("data") or {}).get("publish_id", "")
    if not upload_url:
        return False, f"TikTok не дал ссылку для загрузки: {json.dumps(data)[:250]}"

    with open(clip, "rb") as fh:
        for idx in range(total_chunks):
            start = idx * chunk_size
            piece = fh.read(chunk_size)
            if not piece:
                break
            end = start + len(piece) - 1
            headers = {
                "Content-Range": f"bytes {start}-{end}/{size}",
                "Content-Length": str(len(piece)),
                "Content-Type": "video/mp4",
            }
            try:
                pr = requests_mod.put(upload_url, headers=headers, data=piece, timeout=600)
            except Exception as exc:                           # noqa: BLE001
                return False, f"обрыв при отправке куска {idx + 1}: {exc}"
            if pr.status_code not in (200, 201, 206):
                return False, f"кусок {idx + 1} отклонён ({pr.status_code}): {pr.text[:200]}"
            if total_chunks > 1:
                print(f"        отправлено {idx + 1}/{total_chunks}", flush=True)
    return True, publish_id


def main(argv=None) -> int:
    setup_console()
    args = parse_args(argv)

    folder = Path(args.folder).expanduser().resolve()
    if not folder.is_dir():
        fail(f"Папка не найдена: {folder}")

    print("=" * 64)
    print("  ShortsVideoMaker — отправка в черновики TikTok")
    print("=" * 64)

    stage("Смотрю, что есть")
    clips = sorted(folder.glob("[0-9][0-9][0-9].mp4"))
    if not clips:
        fail(f"В папке {folder} нет клипов вида 001.mp4.\n"
             "Сначала нарежьте фильм:  python autocut.py \"фильм.mp4\" --out clips")
    state_path = folder / STATE_NAME
    state = load_state(state_path)
    todo = [c for c in clips if c.name not in state["uploaded"]]
    info(f"клипов всего: {len(clips)}, уже отправлено: {len(clips) - len(todo)}")
    if not todo:
        info("новых клипов нет — все уже в черновиках TikTok.")
        return 0

    titles = {}
    man = folder / "manifest.json"
    if man.exists():
        try:
            for p in json.loads(man.read_text(encoding="utf-8-sig")).get("parts", []):
                titles[str(p.get("file"))] = (p.get("text") or "").strip()
        except (json.JSONDecodeError, OSError):
            pass

    todo = todo[: max(0, args.per_run)]
    info(f"в этот заход: {len(todo)}")

    stage("Проверяю ключи")
    creds = load_secrets(Path(args.secret))
    token = creds.get("access_token", "")
    if not token:
        # Это штатная ситуация, а не авария: честно объясняем и не делаем вид,
        # что что-то сломалось.
        print()
        for line in SETUP_HINT.splitlines():
            print("    " + line)
        print()
        info(f"клипы лежат здесь: {folder}")
        return 2
    info("токен найден")

    try:
        import requests
    except ImportError:
        fail("Не установлена библиотека requests.\n"
             "Поставьте зависимости:  pip install -r requirements.txt")

    stage("План")
    for c in todo:
        print(f"    {c.name} ({c.stat().st_size / 1048576:.1f} МБ)")
    if args.dry_run:
        info("это был --dry-run, ничего не отправлено.")
        return 0

    stage("Отправка")
    okc = 0
    for i, clip in enumerate(todo, start=1):
        print(f"    [{i}/{len(todo)}] {clip.name}", flush=True)
        ok, result = send_one(requests, token, clip, titles.get(clip.name, ""))
        if ok:
            okc += 1
            state["uploaded"][clip.name] = {
                "publish_id": result,
                "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            }
            state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2),
                                  encoding="utf-8")
            print("        ушло в черновики — откройте TikTok на телефоне", flush=True)
        elif result.startswith(("TOKEN:", "SCOPE:")):
            warn(result.split(":", 1)[1])
            warn("остальное не отправляю — сначала почините доступ.")
            break
        else:
            warn(f"не отправилось: {result[:300]}")

    print("\n" + "=" * 64)
    print(f"  Отправлено в черновики: {okc} из {len(todo)}")
    print("=" * 64)
    if okc:
        print("\nОткройте TikTok на телефоне: черновики ждут в уведомлениях.")
        print("Опубликовать нужно вручную — это требование самого TikTok "
              "для неодобренных приложений.")
    return 0 if okc else 4


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nПрервано пользователем.")
        sys.exit(130)
