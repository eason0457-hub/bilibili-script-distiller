#!/usr/bin/env python3
"""Batch-extract existing Bilibili subtitle tracks on a GitHub Actions runner."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from hardsub_ocr import run_hardsub_ocr

BV_RE = re.compile(r"\b(BV[0-9A-Za-z]{10})\b")
AV_RE = re.compile(r"\b(?:av|AV)(\d+)\b")
SUPPORTED = {".srt", ".vtt", ".ass", ".ssa", ".json", ".xml"}
NUMBER_PREFIX_RE = re.compile(
    r"^\s*(?:(?:\d+)\s*[.．、)）:]|[（(]\s*\d+\s*[）)])\s*"
)


def now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()


def parse_inputs(raw_input: str) -> list[str]:
    """Parse newline, whitespace, comma, and numbered input without merging URLs."""
    items: list[str] = []
    seen: set[str] = set()
    for raw_line in raw_input.splitlines():
        line = NUMBER_PREFIX_RE.sub("", raw_line.strip(), count=1).strip()
        for token in re.split(r"[\s,，]+", line):
            token = NUMBER_PREFIX_RE.sub("", token.strip(), count=1).strip()
            if not token or re.fullmatch(r"\d+[.．、)）:]?", token):
                continue
            if token not in seen:
                seen.add(token)
                items.append(token)
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
    suffix = path.suffix.lower()
    if suffix in {".ass", ".ssa"}:
        return ass_to_markdown(text)
    if suffix == ".json":
        return json_to_markdown(text)
    if suffix == ".xml":
        return xml_to_markdown(text)
    return srt_or_vtt_to_markdown(text)


def json_to_markdown(text: str) -> str:
    data = json.loads(text)
    cues = data.get("body", data) if isinstance(data, dict) else data
    if not isinstance(cues, list):
        raise ValueError("JSON subtitle did not contain a cue list")
    out = ["# 原始字幕", ""]
    for cue in cues:
        if not isinstance(cue, dict):
            continue
        start = cue.get("from", cue.get("start", cue.get("start_time")))
        end = cue.get("to", cue.get("end", cue.get("end_time")))
        content = cue.get("content", cue.get("text", cue.get("body")))
        if start is None or end is None or content is None:
            continue
        out.extend([f"[{timestamp(float(start))} --> {timestamp(float(end))}]", str(content), ""])
    if len(out) == 2:
        raise ValueError("JSON subtitle contained no recognized timed cues")
    return "\n".join(out).rstrip() + "\n"


def xml_to_markdown(text: str) -> str:
    root = ET.fromstring(text)
    out = ["# 原始字幕", ""]
    for node in root.iter():
        if node.tag.lower().split("}")[-1] not in {"text", "p", "d"}:
            continue
        start = node.attrib.get("start") or node.attrib.get("from")
        duration = node.attrib.get("dur") or node.attrib.get("duration")
        end = node.attrib.get("end") or node.attrib.get("to")
        content = "".join(node.itertext()).strip()
        if not content:
            continue
        if start is None and "p" in node.attrib:
            parts = node.attrib["p"].split(",")
            start = parts[0] if parts else None
        if start is None:
            continue
        start_value = float(start)
        end_value = float(end) if end is not None else start_value + float(duration or 0)
        out.extend([f"[{timestamp(start_value)} --> {timestamp(end_value)}]", content, ""])
    if len(out) == 2:
        raise ValueError("XML subtitle contained no recognized timed cues")
    return "\n".join(out).rstrip() + "\n"


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


def run_one(raw_input: str, output_root: Path, *, hardsub: dict | None = None, command_runner=subprocess.run) -> dict:
    status = {
        "input": raw_input,
        "source_type": "subtitle_track",
        "bvid": None,
        "title": None,
        "success": False,
        "failure_reason": None,
        "bbdown_exit_code": None,
        "subtitle_files_found": [],
        "selected_subtitle_file": None,
        "subtitle_language": None,
        "login_warning": False,
        "processed_at": now_iso(),
    }
    fallback = f"failed-{hashlib.sha256(raw_input.encode()).hexdigest()[:8]}"
    result_dir: Path | None = None
    try:
        if not is_supported_input(raw_input):
            raise ValueError("input must be an http/https URL, BV ID, or AV ID")
        final_url, resolved_id = resolve_input(raw_input)
        status["bvid"] = resolved_id if resolved_id and resolved_id.startswith("BV") else None
        result_dir = output_root / safe_name(resolved_id or fallback)
        result_dir.mkdir(parents=True, exist_ok=True)
        print(f"Normalized BV ID: {status['bvid'] or resolved_id or 'pending BBDown response'}", flush=True)
        with tempfile.TemporaryDirectory(prefix="bili-sub-") as temp_name:
            temp = Path(temp_name)
            command = [
                "BBDown", final_url, "--sub-only", "--skip-ai=false",
                "-F", "<bvid>", "--work-dir", str(temp),
            ]
            print("BBDown command: " + " ".join(shlex.quote(part) for part in command), flush=True)
            process = command_runner(
                command, text=True, capture_output=True, timeout=180, check=False
            )
            stdout = process.stdout or ""
            stderr = process.stderr or ""
            log = stdout + "\n" + stderr
            status["bbdown_exit_code"] = process.returncode
            status["title"] = extract_title(log)
            status["login_warning"] = bool(
                re.search(r"尚未登录|未登录|not logged|login.*required|需要登录", log, re.I)
            )
            if not status["bvid"]:
                match = BV_RE.search(log)
                if match:
                    status["bvid"] = match.group(1)
            print(f"BBDown exit code: {process.returncode}", flush=True)
            tracks = sorted(path for path in temp.rglob("*") if path.is_file() and path.suffix.lower() in SUPPORTED)
            ranked = sorted((track_score(path), path) for path in tracks)
            status["subtitle_files_found"] = [str(path.relative_to(temp)) for _, path in ranked]
            print(f"Found subtitle files: {len(ranked)}", flush=True)
            if process.returncode != 0:
                last_line = next(
                    (line.strip() for line in reversed(log.splitlines()) if line.strip()),
                    "BBDown failed without an error message",
                )
                status["failure_reason"] = f"BBDown exited with code {process.returncode}: {last_line}"
                return status | {"result_dir": str(result_dir)}
            if not ranked:
                status["failure_reason"] = "BBDown completed, but no subtitle file was produced."
                if hardsub and hardsub["enabled"]:
                    print("No subtitle track found; starting enabled hard-subtitle OCR fallback", flush=True)
                    ocr_status = run_hardsub_ocr(
                        bvid=status["bvid"] or safe_name(resolved_id or fallback),
                        title=status["title"],
                        url=final_url,
                        output_dir=result_dir,
                        sample_fps=hardsub["sample_fps"],
                        position=hardsub["position"],
                        language=hardsub["language"],
                        start_time=hardsub["start_time"],
                        end_time=hardsub["end_time"],
                        crop_overrides=hardsub["crop_overrides"],
                    )
                    if ocr_status["success"]:
                        status["source_type"] = "hardcoded_subtitle_ocr"
                        status["success"] = True
                        status["failure_reason"] = None
                        status["subtitle_files_found"] = ["subtitle-ocr.srt"]
                        status["selected_subtitle_file"] = "subtitle-ocr.srt"
                        status["subtitle_language"] = hardsub["language"]
                    else:
                        status["source_type"] = "hardcoded_subtitle_ocr"
                        status["failure_reason"] = ocr_status["failure_reason"] or status["failure_reason"]
                else:
                    status["source_type"] = "no_subtitle_track_user_not_enabled"
                    print("No subtitle track found; hard-subtitle OCR not enabled", flush=True)
                return status | {"result_dir": str(result_dir)}
            score, selected = ranked[0]
            print(f"Selected subtitle file: {selected.name}", flush=True)
            (result_dir / "subtitle-raw.md").write_text(convert_track(selected), encoding="utf-8")
            status["selected_subtitle_file"] = str(selected.relative_to(temp))
            status["subtitle_language"] = score[2]
            status["success"] = True
            return status | {"result_dir": str(result_dir)}
    except Exception as exc:
        status["failure_reason"] = f"{type(exc).__name__}: {exc}"
        result_dir = result_dir or output_root / safe_name(status["bvid"] or fallback)
        result_dir.mkdir(parents=True, exist_ok=True)
        return status | {"result_dir": str(result_dir)}
    finally:
        if result_dir:
            (result_dir / "extraction-status.json").write_text(
                json.dumps(status, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, required=True)
    parser.add_argument("--enable-hardsub-ocr", action="store_true")
    parser.add_argument("--start-time", type=float)
    parser.add_argument("--end-time", type=float)
    parser.add_argument("--subtitle-position", choices=["bottom", "top", "custom"], default="bottom")
    parser.add_argument("--ocr-language", default="ch")
    parser.add_argument("--sample-fps", type=float, default=2.0)
    parser.add_argument("--subtitle-crop-top", type=float)
    parser.add_argument("--subtitle-crop-bottom", type=float)
    parser.add_argument("--subtitle-crop-left", type=float)
    parser.add_argument("--subtitle-crop-right", type=float)
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
    hardsub = {
        "enabled": args.enable_hardsub_ocr,
        "start_time": args.start_time,
        "end_time": args.end_time,
        "position": args.subtitle_position,
        "language": args.ocr_language,
        "sample_fps": min(4.0, max(2.0, args.sample_fps)),
        "crop_overrides": {
            "top": args.subtitle_crop_top,
            "bottom": args.subtitle_crop_bottom,
            "left": args.subtitle_crop_left,
            "right": args.subtitle_crop_right,
        },
    }
    print(f"Hard subtitle OCR enabled: {hardsub['enabled']}", flush=True)
    if hardsub["enabled"]:
        print(
            "Hard subtitle OCR config: "
            f"range={hardsub['start_time']}..{hardsub['end_time']}, "
            f"position={hardsub['position']}, language={hardsub['language']}, fps={hardsub['sample_fps']}",
            flush=True,
        )
    items = []
    for index, value in enumerate(urls, start=1):
        print(f"Processing {index}/{len(urls)}: {value}", flush=True)
        item = run_one(value, args.output_root, hardsub=hardsub)
        items.append(item)
        if item["success"]:
            print(f"Success {index}: {item.get('bvid') or value}", flush=True)
        else:
            reason = item.get("failure_reason") or "unknown extraction error"
            print(f"::error title=Input {index} failed::{reason}", flush=True)
        print(
            f"Final extraction status: {'success' if item['success'] else 'failed'}",
            flush=True,
        )
    summary = {
        "processed": len(items),
        "successful": sum(bool(item["success"]) for item in items),
        "failed": sum(not item["success"] for item in items),
        "items": [
            {
                "input": item["input"],
                "success": item["success"],
                "video_id": item["bvid"],
                "failure_reason": item["failure_reason"],
                "source_type": item["source_type"],
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
