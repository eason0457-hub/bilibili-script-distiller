"""Optional hard-subtitle OCR fallback. Invoked only after BBDown finds no track."""

from __future__ import annotations

import datetime as dt
import difflib
import json
import re
import shutil
import subprocess
import tempfile
from pathlib import Path


def timestamp(seconds: float) -> str:
    milliseconds = max(0, round(seconds * 1000))
    hours, rem = divmod(milliseconds, 3_600_000)
    minutes, rem = divmod(rem, 60_000)
    secs, millis = divmod(rem, 1000)
    return f"{hours:02}:{minutes:02}:{secs:02},{millis:03}"


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", "", text or "")


def similar(left: str, right: str, threshold: float = 0.9) -> bool:
    return difflib.SequenceMatcher(None, normalize_text(left), normalize_text(right)).ratio() >= threshold


def merge_samples(samples: list[dict], frame_interval: float) -> list[dict]:
    segments: list[dict] = []
    for sample in samples:
        text = sample.get("text", "").strip()
        if not text:
            continue
        start = float(sample["time"])
        confidence = float(sample.get("confidence", 0))
        if segments and similar(segments[-1]["text"], text):
            segments[-1]["end"] = start + frame_interval
            segments[-1]["confidence"] = max(segments[-1]["confidence"], confidence)
            continue
        segments.append(
            {"start": start, "end": start + frame_interval, "text": text, "confidence": confidence}
        )
    return segments


def write_srt(segments: list[dict], path: Path) -> None:
    rows: list[str] = []
    for index, item in enumerate(segments, start=1):
        rows.extend([
            str(index),
            f"{timestamp(item['start'])} --> {timestamp(item['end'])}",
            item["text"],
            "",
        ])
    path.write_text("\n".join(rows), encoding="utf-8")


def srt_to_markdown(segments: list[dict]) -> str:
    rows = ["# 原始字幕（硬字幕 OCR）", ""]
    for item in segments:
        rows.extend([
            f"[{timestamp(item['start']).replace(',', '.')} --> {timestamp(item['end']).replace(',', '.')}]",
            item["text"],
            f"<!-- OCR confidence: {item['confidence']:.3f} -->",
            "",
        ])
    return "\n".join(rows).rstrip() + "\n"


def crop_for(position: str, overrides: dict[str, float | None]) -> dict[str, float]:
    presets = {
        "bottom": {"top": 0.70, "bottom": 0.98, "left": 0.0, "right": 1.0},
        "top": {"top": 0.02, "bottom": 0.30, "left": 0.0, "right": 1.0},
    }
    region = dict(presets.get(position, presets["bottom"]))
    for key, value in overrides.items():
        if value is not None:
            region[key] = value
    if not (0 <= region["left"] < region["right"] <= 1 and 0 <= region["top"] < region["bottom"] <= 1):
        raise ValueError("subtitle crop boundaries must be between 0 and 1 and form a non-empty region")
    return region


def extract_ocr_lines(image: Path, ocr) -> tuple[str, float]:
    result = ocr.ocr(str(image), cls=False)
    lines = result[0] if result else []
    text_parts, confidences = [], []
    for line in lines or []:
        try:
            text, confidence = line[1]
        except (IndexError, TypeError):
            continue
        text_parts.append(str(text))
        confidences.append(float(confidence))
    text = "".join(text_parts).strip() or "[无法识别]"
    return text, (sum(confidences) / len(confidences) if confidences else 0.0)


def run_hardsub_ocr(*, bvid: str, title: str | None, url: str, output_dir: Path, sample_fps: float,
                    position: str, language: str, start_time: float | None, end_time: float | None,
                    crop_overrides: dict[str, float | None]) -> dict:
    status = {
        "bvid": bvid,
        "video_title": title,
        "source_type": "hardcoded_subtitle_ocr",
        "downloaded_quality": "360P preferred, 480P fallback",
        "processed_time_range": {"start_time": start_time, "end_time": end_time},
        "crop_region": None,
        "sample_fps": sample_fps,
        "ocr_language": language,
        "average_confidence": 0.0,
        "low_confidence_segments": [],
        "success": False,
        "failure_reason": None,
        "processed_at": dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat(),
    }
    try:
        region = crop_for(position, crop_overrides)
        status["crop_region"] = region
        from paddleocr import PaddleOCR
        with tempfile.TemporaryDirectory(prefix="bili-hardsub-") as temp_name:
            temp = Path(temp_name)
            video_dir, frames = temp / "video", temp / "frames"
            video_dir.mkdir(); frames.mkdir()
            download = ["BBDown", url, "--video-only", "--skip-mux", "-q", "360P 流畅,480P 清晰", "--work-dir", str(video_dir)]
            result = subprocess.run(download, text=True, capture_output=True, timeout=600, check=False)
            if result.returncode != 0:
                raise RuntimeError(f"low-quality video download failed (exit {result.returncode})")
            videos = sorted(path for path in video_dir.rglob("*") if path.suffix.lower() in {".mp4", ".mkv", ".flv", ".webm"})
            if not videos:
                raise RuntimeError("BBDown completed, but no low-quality video file was produced")
            vf = f"fps={sample_fps},crop=iw*{region['right']-region['left']}:ih*{region['bottom']-region['top']}:iw*{region['left']}:ih*{region['top']}"
            frames_cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error"]
            if start_time is not None:
                frames_cmd += ["-ss", str(start_time)]
            frames_cmd += ["-i", str(videos[0])]
            if end_time is not None:
                duration = end_time - (start_time or 0)
                if duration <= 0:
                    raise ValueError("end_time must be greater than start_time")
                frames_cmd += ["-t", str(duration)]
            frames_cmd += ["-vf", vf, str(frames / "frame_%08d.png")]
            subprocess.run(frames_cmd, check=True, timeout=600)
            ocr = PaddleOCR(use_angle_cls=False, lang=language)
            offset = start_time or 0.0
            samples = []
            for index, image in enumerate(sorted(frames.glob("*.png"))):
                text, confidence = extract_ocr_lines(image, ocr)
                samples.append({"time": offset + index / sample_fps, "text": text, "confidence": confidence})
            segments = merge_samples(samples, 1 / sample_fps)
            if not segments or not any(item["text"] != "[无法识别]" for item in segments):
                raise RuntimeError("OCR completed, but no readable subtitle text was produced")
            output_dir.mkdir(parents=True, exist_ok=True)
            write_srt(segments, output_dir / "subtitle-ocr.srt")
            (output_dir / "subtitle-raw.md").write_text(srt_to_markdown(segments), encoding="utf-8")
            confidences = [item["confidence"] for item in segments]
            status["average_confidence"] = sum(confidences) / len(confidences)
            status["low_confidence_segments"] = [item for item in segments if item["confidence"] < 0.65]
            status["success"] = (output_dir / "subtitle-ocr.srt").stat().st_size > 0 and (output_dir / "subtitle-raw.md").stat().st_size > 0
    except Exception as exc:
        status["failure_reason"] = f"{type(exc).__name__}: {exc}"
    (output_dir / "ocr-status.json").write_text(json.dumps(status, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return status
