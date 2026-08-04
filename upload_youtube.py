#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Загрузка готовых клипов на YouTube как Shorts.

Пример:
    python upload_youtube.py clips --title "Название — часть {n} #shorts" \
                                   --privacy public --per-day 5

Что важно знать заранее:
  * нужен файл client_secret.json из Google Cloud Console (как его получить —
    подробно расписано в README);
  * пока ваш API-проект не прошёл проверку Google, ролики, загруженные через
    API, остаются приватными, что бы вы ни указали в --privacy;
  * квота YouTube Data API — 10 000 единиц в сутки, одна загрузка стоит 1600,
    то есть примерно 6 роликов в день.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

from svm_common import fail, human_time, info, setup_console, stage, warn

SCOPES = ["https://www.googleapis.com/auth/youtube.upload"]
QUOTA_PER_UPLOAD = 1600
QUOTA_PER_DAY = 10000
STATE_NAME = "uploaded.json"

SETUP_HINT = """\
Нужен файл client_secret.json. Коротко, как его получить:
  1. console.cloud.google.com → создайте проект;
  2. «APIs & Services» → «Library» → включите YouTube Data API v3;
  3. «APIs & Services» → «OAuth consent screen» → тип External, заполните
     обязательные поля, добавьте себя в Test users;
  4. «Credentials» → «Create credentials» → «OAuth client ID» → тип
     «Desktop app» → скачайте JSON;
  5. положите его рядом со скриптом под именем client_secret.json.
Подробнее — в README, раздел «YouTube: настройка с нуля»."""


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        prog="upload_youtube.py",
        description="Заливает клипы из папки на YouTube как Shorts.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='Пример:\n  python upload_youtube.py clips '
               '--title "Название — часть {n} #shorts" --privacy public --per-day 5',
    )
    p.add_argument("folder", help="папка с клипами (та же, что --out у autocut.py)")
    p.add_argument("--title", default="Часть {n} #shorts",
                   help="шаблон заголовка; {n} — номер части, {text} — начало реплик")
    p.add_argument("--description", default="",
                   help="шаблон описания; поддерживает {n} и {text}")
    p.add_argument("--tags", default="shorts,нарезка",
                   help="теги через запятую")
    p.add_argument("--privacy", choices=["public", "unlisted", "private"], default="private",
                   help="доступ к ролику (по умолчанию private — так безопаснее)")
    p.add_argument("--per-day", type=int, default=5,
                   help="сколько роликов заливать за один запуск (квота ~6 в сутки)")
    p.add_argument("--schedule", default="",
                   help="отложенная публикация: шаг вида 3h, 90m, 1d между роликами")
    p.add_argument("--start", default="",
                   help="когда публиковать первый ролик: ГГГГ-ММ-ДД ЧЧ:ММ (по умолчанию через час)")
    p.add_argument("--client-secret", default="client_secret.json",
                   help="путь к client_secret.json")
    p.add_argument("--token", default="youtube_token.json",
                   help="куда сохранить полученный токен доступа")
    p.add_argument("--category", default="24", help="ID категории YouTube (24 — Entertainment)")
    p.add_argument("--dry-run", action="store_true",
                   help="ничего не загружать, только показать план")
    return p.parse_args(argv)


# ------------------------------------------------------------- состояние ----
def load_state(path: Path) -> dict:
    if not path.exists():
        return {"uploaded": {}, "history": []}
    try:
        # utf-8-sig, а не utf-8: Блокнот и PowerShell дописывают в начало BOM,
        # и на обычном utf-8 разбор падал бы — а это значит «залить всё заново»
        data = json.loads(path.read_text(encoding="utf-8-sig"))
    except (json.JSONDecodeError, OSError):
        warn(f"{path.name} не читается — считаю, что ничего ещё не загружено. "
             "Проверьте файл, иначе ролики уйдут на YouTube повторно!")
        return {"uploaded": {}, "history": []}
    data.setdefault("uploaded", {})
    data.setdefault("history", [])
    return data


def save_state(path: Path, state: dict) -> None:
    try:
        path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    except OSError as exc:
        warn(f"не смог записать {path.name}: {exc}")


def used_today(state: dict) -> int:
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    return sum(1 for h in state["history"] if str(h.get("at", "")).startswith(today))


def parse_step(text: str) -> timedelta | None:
    if not text:
        return None
    m = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*([mhd])\s*", text.lower())
    if not m:
        fail(f"Не понял шаг --schedule: «{text}». Примеры: 90m, 3h, 1d")
    value, unit = float(m.group(1)), m.group(2)
    return {"m": timedelta(minutes=value), "h": timedelta(hours=value),
            "d": timedelta(days=value)}[unit]


# ------------------------------------------------------------------ план ----
def build_jobs(folder: Path, args, state: dict) -> list[dict]:
    manifest_path = folder / "manifest.json"
    parts_meta: dict[str, dict] = {}
    if manifest_path.exists():
        try:
            man = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
            for p in man.get("parts", []):
                parts_meta[str(p.get("file"))] = p
        except (json.JSONDecodeError, OSError):
            warn("manifest.json не читается — заголовки будут только по номеру части.")

    clips = sorted(folder.glob("[0-9][0-9][0-9].mp4"))
    if not clips:
        fail(f"В папке {folder} нет клипов вида 001.mp4.\n"
             "Сначала нарежьте фильм:  python autocut.py \"фильм.mp4\" --out clips")

    jobs = []
    for clip in clips:
        if clip.name in state["uploaded"]:
            continue
        meta = parts_meta.get(clip.name, {})
        n = meta.get("n") or int(re.sub(r"\D", "", clip.stem) or 0)
        text = (meta.get("text") or "").strip()
        short = text[:80].rstrip() + ("…" if len(text) > 80 else "")
        title = args.title.replace("{n}", str(n)).replace("{text}", short).strip()
        if len(title) > 100:                                   # ограничение YouTube
            title = title[:99].rstrip() + "…"
        desc = args.description.replace("{n}", str(n)).replace("{text}", text)
        jobs.append({"file": clip, "n": n, "title": title, "description": desc})
    return jobs


# --------------------------------------------------------------- загрузка ---
def get_service(args):
    try:
        from google.auth.transport.requests import Request
        from google.oauth2.credentials import Credentials
        from google_auth_oauthlib.flow import InstalledAppFlow
        from googleapiclient.discovery import build
    except ImportError:
        fail("Не установлены библиотеки Google.\n"
             "Поставьте зависимости:  pip install -r requirements.txt")

    token_path = Path(args.token)
    secret_path = Path(args.client_secret)
    creds = None

    if token_path.exists():
        try:
            creds = Credentials.from_authorized_user_file(str(token_path), SCOPES)
        except Exception as exc:                               # noqa: BLE001
            warn(f"сохранённый токен не подошёл ({exc}) — авторизуюсь заново.")
            creds = None

    if creds and creds.expired and creds.refresh_token:
        try:
            creds.refresh(Request())
            info("токен доступа обновлён")
        except Exception as exc:                               # noqa: BLE001
            warn(f"не удалось обновить токен ({exc}) — авторизуюсь заново.")
            creds = None

    if not creds or not creds.valid:
        if not secret_path.exists():
            fail(f"Не найден {secret_path}.\n\n{SETUP_HINT}")
        info("сейчас откроется браузер — войдите в аккаунт и разрешите доступ")
        flow = InstalledAppFlow.from_client_secrets_file(str(secret_path), SCOPES)
        try:
            creds = flow.run_local_server(port=0)
        except Exception as exc:                               # noqa: BLE001
            fail(f"Авторизация не удалась: {exc}\n"
                 "Проверьте, что тип OAuth-клиента — «Desktop app», а вы добавлены "
                 "в Test users на экране согласия.")
        try:
            token_path.write_text(creds.to_json(), encoding="utf-8")
            info(f"токен сохранён в {token_path.name} — больше входить не придётся")
        except OSError as exc:
            warn(f"не смог сохранить токен: {exc}")

    return build("youtube", "v3", credentials=creds, cache_discovery=False)


def upload_one(youtube, job: dict, args, publish_at: datetime | None) -> tuple[bool, str]:
    from googleapiclient.errors import HttpError
    from googleapiclient.http import MediaFileUpload

    status = {"privacyStatus": args.privacy, "selfDeclaredMadeForKids": False}
    if publish_at is not None:
        # отложенная публикация возможна только для приватного ролика
        status["privacyStatus"] = "private"
        status["publishAt"] = publish_at.astimezone(timezone.utc).isoformat().replace(
            "+00:00", "Z")

    body = {
        "snippet": {
            "title": job["title"],
            "description": job["description"],
            "tags": [t.strip() for t in args.tags.split(",") if t.strip()],
            "categoryId": str(args.category),
        },
        "status": status,
    }
    media = MediaFileUpload(str(job["file"]), chunksize=4 * 1024 * 1024, resumable=True,
                            mimetype="video/mp4")
    try:
        request = youtube.videos().insert(part="snippet,status", body=body, media_body=media)
        response = None
        last = -1
        while response is None:
            chunk_status, response = request.next_chunk()
            if chunk_status:
                pct = int(chunk_status.progress() * 100)
                if pct >= last + 10:
                    last = pct
                    print(f"        загружено {pct}%", flush=True)
        return True, response.get("id", "")
    except HttpError as exc:
        detail = ""
        try:
            detail = json.loads(exc.content.decode("utf-8"))["error"]["message"]
        except Exception:                                      # noqa: BLE001
            detail = str(exc)
        if exc.resp.status == 403 and "quota" in detail.lower():
            return False, "QUOTA:" + detail
        return False, detail
    except Exception as exc:                                   # noqa: BLE001
        return False, str(exc)


# ------------------------------------------------------------------- main ---
def main(argv=None) -> int:
    setup_console()
    args = parse_args(argv)

    folder = Path(args.folder).expanduser().resolve()
    if not folder.is_dir():
        fail(f"Папка не найдена: {folder}")

    print("=" * 64)
    print("  ShortsVideoMaker — публикация на YouTube")
    print("=" * 64)

    state_path = folder / STATE_NAME
    state = load_state(state_path)

    stage("Считаю, что осталось загрузить")
    jobs = build_jobs(folder, args, state)
    already = len(state["uploaded"])
    if already:
        info(f"уже загружено раньше: {already} (список в {STATE_NAME})")
    if not jobs:
        info("новых клипов нет — всё уже на YouTube.")
        return 0
    info(f"готовы к загрузке: {len(jobs)}")

    spent = used_today(state)
    left_by_quota = max(0, (QUOTA_PER_DAY - spent * QUOTA_PER_UPLOAD) // QUOTA_PER_UPLOAD)
    if spent:
        info(f"сегодня уже загружено: {spent}, квота позволяет ещё ~{left_by_quota}")
    limit = min(args.per_day, len(jobs), max(0, left_by_quota) or args.per_day)
    if left_by_quota == 0 and spent:
        warn("дневная квота YouTube API, скорее всего, исчерпана. "
             "Она обнуляется в полночь по тихоокеанскому времени (около 11:00 МСК).")
        return 3
    jobs = jobs[:limit]
    info(f"в этот заход: {len(jobs)}")

    step = parse_step(args.schedule)
    when = None
    if step:
        if args.start:
            try:
                when = datetime.strptime(args.start, "%Y-%m-%d %H:%M").astimezone()
            except ValueError:
                fail(f"Не понял --start: «{args.start}». Нужен формат ГГГГ-ММ-ДД ЧЧ:ММ")
        else:
            when = datetime.now().astimezone() + timedelta(hours=1)
        info(f"отложенная публикация: первый в {when:%Y-%m-%d %H:%M}, "
             f"дальше каждые {args.schedule}")

    stage("План")
    plan_time = when
    for j in jobs:
        stamp = f"  → {plan_time:%d.%m %H:%M}" if plan_time else ""
        print(f"    {j['file'].name}: {j['title']}{stamp}")
        if plan_time and step:
            plan_time = plan_time + step

    if args.dry_run:
        info("это был --dry-run, ничего не загружено.")
        return 0

    if args.privacy == "public" and not step:
        warn("Пока ваш API-проект не прошёл проверку Google, ролик всё равно "
             "останется приватным — это ограничение самого YouTube, а не скрипта. "
             "Как запросить проверку — в README.")

    stage("Авторизация в Google")
    youtube = get_service(args)

    stage("Загрузка")
    okc = 0
    for i, job in enumerate(jobs, start=1):
        size_mb = job["file"].stat().st_size / 1048576
        print(f"    [{i}/{len(jobs)}] {job['file'].name} ({size_mb:.1f} МБ): {job['title']}",
              flush=True)
        ok, result = upload_one(youtube, job, args, when)
        if ok:
            okc += 1
            state["uploaded"][job["file"].name] = {
                "video_id": result,
                "title": job["title"],
                "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "publish_at": when.isoformat() if when else None,
            }
            state["history"].append({
                "file": job["file"].name,
                "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            })
            save_state(state_path, state)
            print(f"        готово: https://youtu.be/{result}", flush=True)
        elif result.startswith("QUOTA:"):
            warn("квота YouTube API кончилась — остальное залью в следующий раз.")
            warn(result[6:][:200])
            break
        else:
            warn(f"не загрузилось: {result[:300]}")
        if when and step:
            when = when + step

    print("\n" + "=" * 64)
    print(f"  Загружено в этот раз: {okc} из {len(jobs)}")
    print("=" * 64)
    if okc:
        print(f"\nСписок загруженного — в {state_path}")
        print("Повторный запуск продолжит с того места, где остановились.")
    return 0 if okc else 4


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nПрервано пользователем. Что успело загрузиться — записано в uploaded.json.")
        sys.exit(130)
