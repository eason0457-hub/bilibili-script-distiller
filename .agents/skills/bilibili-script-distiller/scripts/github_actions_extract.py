#!/usr/bin/env python3
"""Batch-extract existing Bilibili subtitle tracks on a GitHub Actions runner."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
import urllib.request
from pathlib import Path

BV_RE = re.compile(r"\b(BV[0-9A-Za-z]{10})\b")
AV_RE = re.compile(r"\b(?:av|AV)(\d+)\b")
SUPPORTED = {".srt", ".vtt", ".ass", ".ssa"}
NUMBER_PREFIX_RE = re.compile(
    r"^\s*(?:(?:\d+)\s*[.．、)）:]|[（(]\s*\d+\s*[）)])\s*"
)


def now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()


def parse_inputs(raw_input: str) -> list[str]:
    """Preserve line boundaries and normalize optional human list markers."""
    items: list[str] = []
    for raw_line in raw_input.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        line = NUMBER_PREFIX_RE.sub("", line, count=1).strip()
        if line:
            items.append(line)
    return items


def is_supported_input(value: str) -> bool:
    value = value.strip()
    return bool(
        re.match(r"^https?://\S+$", value, re.I)
        or re.fullmatch(r"BV[0-9A-Za-z]{10}", value)
        or re.fullmatch(r"(?:av|AV)\d+", value)
    )


def resolve_input(value: str) -> tuple[str, str | None]:
    value = value.strip()
    bv = BV_RE.search(value)
    if bv:
        video_id = bv.group(1)
        return f"https://www.bilibili.com/video/{video_id}/", video_id
    av = AV_RE.search(value)
    if av:
        video_id = f"av{av.group(1)}"
        return f"https://www.bilibili.com/video/{video_id}/", video_id
    if re.match(r"^https?://(?:www\.)?b23\.tv/", value, re.I):
        request = urllib.request.Request(value, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(request, timeout=30) as response:
            final_url = response.geturl()
        bv = BV_RE.search(final_url)
        av = AV_RE.search(final_url)
        video_id = bv.group(1) if bv else (f"av{av.group(1)}" if av else None)
        return final_url, video_id
    if re.match(r"^https?://(?:www\.)?bilibili\.com/", value, re.I):
        return value, None
    raise ValueError("unsupported Bilibili link or video ID")


def safe_name(value: str) -> str:
    value = re.sub(r'[\\/:*?"<>|\x00-\x1f]', "_", value).strip().strip(".")
    return value[:120] or "unknown-video"


def timestamp(seconds: float) -> str:
    millis = max(0, round(seconds * 1000))
    hours, rem = divmod(millis, 3_600_000)
    minutes, rem = divmod(rem, 60_000)
    secs, ms = divmod(rem, 1000)
    return f"{hours:02}:{minutes:02}:{secs:02}.{ms:03}"


def srt_or_vtt_to_markdown(text: str) -> str:
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    out: list[str] = ["# 原始字幕", ""]
    time_re = re.compile(r"(\d{1,2}:\d{2}:\d{2}[,.]\d{3})\s*-->\s*(\d{1,2}:\d{2}:\d{2}[,.]\d{3})")
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        match = time_re.search(line)
        if not match:
            i += 1
            continue
        start, end = (part.replace(",", ".") for part in match.groups())
        i += 1
        payload: list[str] = []
        while i < len(lines) and lines[i].strip():
            payload.append(lines[i].rstrip())
            i += 1
        out.extend([f"[{start} --> {end}]", "\n".join(payload), ""])
        i += 1
    if len(out) == 2:
        raise ValueError("subtitle file contained no recognized timed cues")
    return "\n".join(out).rstrip() + "\n"


def ass_to_markdown(text: str) -> str:
    out: list[str] = ["# 原始字幕", ""]
    for raw in text.replace("\r\n", "\n").split("\n"):
        if not raw.startswith("Dialogue:"):
            continue
        fields = raw.split(",", 9)
        if len(fields) < 10:
            continue
        start, end, payload = fields[1].strip(), fields[2].strip(), fields[9]
        payload = re.sub(r"\{[^}]*\}", "", payload).replace(r"\N", "\n").replace(r"\n", "\n")
        out.extend([f"[{start} --> {end}]", payload, ""])
    if len(out) == 2:
        raise ValueError("ASS file contained no recognized Dialogue cues")
    return "\n".join(out).rstrip() + "\n"


def convert_track(path: Path) -> str:
    text = path.read_text(encoding="utf-8-sig", errors="replace")
    return ass_to_markdown(text) if path.suffix.lower() in {".ass", ".ssa"} else srt_or_vtt_to_markdown(text)


def track_score(path: Path) -> tuple[int, str, str]:
    name = path.name.lower()
    ai = any(token in name for token in ("ai", "auto", "asr", "自动"))
    chinese = any(token in name for token in ("zh", "chi", "chs", "cht", "中文", "汉语", "漢語"))
    japanese = any(token in name for token in ("ja", "jp", "jpn", "日文", "日本語"))
    if chinese and not ai:
        return 0, "human", "zh"
    if chinese and ai:
        return 1, "ai/platform", "zh"
    if japanese:
        return 2, "human/platform", "ja"
    return 3, "unknown", "unknown"


def extract_title(log: str) -> str | None:
    patterns = [r"(?:视频标题|标题|Title)\s*[:：]\s*(.+)", r"\[P\d+\](.+)"]
    for line in log.splitlines():
        for pattern in patterns:
            match = re.search(pattern, line)
            if match:
                return match.group(1).strip()
    return None


def run_one(raw_input: str, output_root: Path) -> dict:
    status = {
        "original_input": raw_input,
        "final_video_url": None,
        "video_id": None,
        "video_title": None,
        "success": False,
        "subtitle_track_type": None,
        "subtitle_language": None,
        "available_tracks": [],
        "failure_reason": None,
        "processed_at": now_iso(),
    }
    fallback = f"failed-{hashlib.sha256(raw_input.encode()).hexdigest()[:8]}"
    result_dir: Path | None = None
    try:
        if not is_supported_input(raw_input):
            raise ValueError("input must be an http/https URL, BV ID, or AV ID")
        final_url, resolved_id = resolve_input(raw_input)
        status["final_video_url"] = final_url
        status["video_id"] = resolved_id
        with tempfile.TemporaryDirectory(prefix="bili-sub-") as temp_name:
            temp = Path(temp_name)
            command = [
                "BBDown", final_url, "--sub-only", "--skip-ai=false",
                "-F", "<bvid>", "--work-dir", str(temp),
            ]
            process = subprocess.run(command, text=True, capture_output=True, timeout=180, check=False)
            log = (process.stdout or "") + "\n" + (process.stderr or "")
            status["video_title"] = extract_title(log)
            if not status["video_id"]:
                match = BV_RE.search(log) or AV_RE.search(log)
                if match:
                    status["video_id"] = match.group(1) if match.re is BV_RE else f"av{match.group(1)}"
            tracks = sorted(path for path in temp.rglob("*") if path.is_file() and path.suffix.lower() in SUPPORTED)
            ranked = sorted((track_score(path), path) for path in tracks)
            status["available_tracks"] = [
                {"file": path.name, "type": score[1], "language": score[2]}
                for score, path in ranked
            ]
            key = safe_name(status["video_id"] or status["video_title"] or fallback)
            result_dir = output_root / key
            result_dir.mkdir(parents=True, exist_ok=True)
            if not ranked:
                status["failure_reason"] = (log.strip()[-1000:] or f"BBDown exited with code {process.returncode}")
                return status | {"result_dir": str(result_dir)}
            score, selected = ranked[0]
            (result_dir / "subtitle-raw.md").write_text(convert_track(selected), encoding="utf-8")
            status["subtitle_track_type"] = score[1]
            status["subtitle_language"] = score[2]
            status["success"] = True
            return status | {"result_dir": str(result_dir)}
    except Exception as exc:
        status["failure_reason"] = f"{type(exc).__name__}: {exc}"
        result_dir = result_dir or output_root / safe_name(status["video_id"] or status["video_title"] or fallback)
        result_dir.mkdir(parents=True, exist_ok=True)
        return status | {"result_dir": str(result_dir)}
    finally:
        if result_dir:
            serializable = {key: value for key, value in status.items()}
            (result_dir / "extraction-status.json").write_text(
                json.dumps(serializable, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, required=True)
    args = parser.parse_args()
    raw_input = os.environ.get("VIDEO_URLS", "")
    urls = parse_inputs(raw_input)
    print(f"Raw input characters: {len(raw_input)}", flush=True)
    print(f"Parsed inputs: {len(urls)}", flush=True)
    for index, value in enumerate(urls, start=1):
        print(f"Input {index}: {value}", flush=True)
    if not raw_input.strip() or not urls:
        print("::error::VIDEO_URLS is empty", flush=True)
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.summary_json.write_text(
            json.dumps(
                {"processed": 0, "successful": 0, "failed": 0, "items": []},
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        return 2
    args.output_root.mkdir(parents=True, exist_ok=True)
    items = []
    for index, value in enumerate(urls, start=1):
        print(f"Processing {index}/{len(urls)}: {value}", flush=True)
        item = run_one(value, args.output_root)
        items.append(item)
        if item["success"]:
            print(f"Success {index}: {item.get('video_id') or value}", flush=True)
        else:
            reason = item.get("failure_reason") or "unknown extraction error"
            print(f"::error title=Input {index} failed::{reason}", flush=True)
    summary = {
        "processed": len(items),
        "successful": sum(bool(item["success"]) for item in items),
        "failed": sum(not item["success"] for item in items),
        "items": [
            {
                "input": item["original_input"],
                "success": item["success"],
                "video_id": item["video_id"],
                "failure_reason": item["failure_reason"],
                "result_dir": item["result_dir"],
            }
            for item in items
        ],
    }
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0 if items and any(item["success"] for item in items) else 2


if __name__ == "__main__":
    raise SystemExit(main())
