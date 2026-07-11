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
        speaker = sample.get("speaker") or "旁白"
        start = float(sample["time"])
        confidence = float(sample.get("confidence", 0))
        if segments and segments[-1].get("speaker") == speaker and similar(segments[-1]["text"], text):
            segments[-1]["end"] = start + frame_interval
            segments[-1]["confidence"] = max(segments[-1]["confidence"], confidence)
            continue
        segments.append(
            {
                "start": start,
                "end": start + frame_interval,
                "speaker": speaker,
                "text": text,
                "confidence": confidence,
            }
        )
    return segments


def write_srt(segments: list[dict], path: Path) -> None:
    rows: list[str] = []
    for index, item in enumerate(segments, start=1):
        rows.extend([
            str(index),
            f"{timestamp(item['start'])} --> {timestamp(item['end'])}",
            f"【{item.get('speaker') or '旁白'}】{item['text']}",
            "",
        ])
    path.write_text("\n".join(rows), encoding="utf-8")


def srt_to_markdown(segments: list[dict]) -> str:
    rows = ["# 原始字幕（硬字幕 OCR）", ""]
    for item in segments:
        rows.extend([
            f"[{timestamp(item['start']).replace(',', '.')} --> {timestamp(item['end']).replace(',', '.')}]",
            f"{item.get('speaker') or '旁白'}：",
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


def speaker_crop_region() -> dict[str, float]:
    """Video convention: speaker tags live in the upper or upper-left area."""
    return {"top": 0.0, "bottom": 0.35, "left": 0.0, "right": 1.0}


def build_ocr_engine() -> tuple[object, str]:
    """Use PP-OCR models through ONNX Runtime, avoiding Paddle CPU SIGILL on CI."""
    from rapidocr_onnxruntime import RapidOCR
    return RapidOCR(), "rapidocr_onnxruntime"


def frame_signature(image: Path) -> bytes:
    """Build a small edge mask so unchanged subtitle frames can reuse OCR."""
    from PIL import Image, ImageFilter

    with Image.open(image) as source:
        edges = (
            source.convert("L")
            .resize((96, 24))
            .filter(ImageFilter.FIND_EDGES)
        )
        return bytes(1 if pixel >= 40 else 0 for pixel in edges.getdata())


def signature_distance(left: bytes, right: bytes) -> float:
    if not left or len(left) != len(right):
        return 1.0
    return sum(a != b for a, b in zip(left, right)) / len(left)


def should_reuse_ocr(previous: bytes | None, current: bytes, threshold: float = 0.02) -> bool:
    return previous is not None and signature_distance(previous, current) <= threshold


def normalize_speaker_label(text: str, confidence: float) -> str | None:
    """Keep only a short, high-confidence tag; otherwise treat it as narration."""
    value = re.sub(r"[\s【】\[\]（）()：:·.]+", "", text or "")
    if confidence < 0.78 or not value or value == "无法识别":
        return None
    if len(value) > 12 or re.search(r"\d", value):
        return None
    if not re.search(r"[\u3040-\u30ff\u3400-\u9fffA-Za-z]", value):
        return None
    return value


def extract_ocr_lines(image: Path, ocr) -> tuple[str, float, int]:
    result, _elapsed = ocr(str(image))
    lines = result or []
    text_parts, confidences = [], []
    for line in lines or []:
        try:
            text, confidence = line[1], line[2]
        except (IndexError, TypeError):
            continue
        text_parts.append(str(text))
        confidences.append(float(confidence))
    text = "".join(text_parts).strip() or "[无法识别]"
    return text, (sum(confidences) / len(confidences) if confidences else 0.0), len(lines or [])


def extract_speaker_label(image: Path, ocr) -> tuple[str | None, int]:
    result, _elapsed = ocr(str(image))
    lines = result or []
    candidates: list[tuple[str, float]] = []
    for line in lines:
        try:
            text, confidence = str(line[1]), float(line[2])
        except (IndexError, TypeError, ValueError):
            continue
        label = normalize_speaker_label(text, confidence)
        if label:
            candidates.append((label, confidence))
    if not candidates:
        return None, len(lines)
    label, _confidence = max(candidates, key=lambda item: item[1])
    return label, len(lines)


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
        "speaker_crop_region": None,
        "sample_fps": sample_fps,
        "ocr_language": language,
        "ocr_engine": "rapidocr_onnxruntime",
        "average_confidence": 0.0,
        "low_confidence_segments": [],
        "success": False,
        "failure_reason": None,
        "diagnostics": {
            "video_path": None,
            "video_resolution": None,
            "frame_count": 0,
            "speaker_frame_count": 0,
            "ocr_call_count": 0,
            "ocr_reused_frame_count": 0,
            "speaker_ocr_call_count": 0,
            "speaker_reused_frame_count": 0,
            "speaker_identified_frame_count": 0,
            "raw_ocr_result_count": 0,
            "filtered_result_count": 0,
            "frame_similarity_threshold": 0.02,
        },
        "processed_at": dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat(),
    }
    try:
        region = crop_for(position, crop_overrides)
        speaker_region = speaker_crop_region()
        status["crop_region"] = region
        status["speaker_crop_region"] = speaker_region
        with tempfile.TemporaryDirectory(prefix="bili-hardsub-") as temp_name:
            temp = Path(temp_name)
            video_dir, frames, speaker_frames = temp / "video", temp / "frames", temp / "speaker_frames"
            video_dir.mkdir(); frames.mkdir(); speaker_frames.mkdir()
            download = ["BBDown", url, "--video-only", "--skip-mux", "-q", "360P 流畅,480P 清晰", "--work-dir", str(video_dir)]
            result = subprocess.run(download, text=True, capture_output=True, timeout=600, check=False)
            if result.returncode != 0:
                raise RuntimeError(f"low-quality video download failed (exit {result.returncode})")
            videos = sorted(path for path in video_dir.rglob("*") if path.suffix.lower() in {".mp4", ".mkv", ".flv", ".webm"})
            if not videos:
                raise RuntimeError("BBDown completed, but no low-quality video file was produced")
            status["diagnostics"]["video_path"] = videos[0].name
            probe = subprocess.run(
                ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height", "-of", "json", str(videos[0])],
                text=True, capture_output=True, check=False,
            )
            if probe.returncode == 0:
                stream = (json.loads(probe.stdout).get("streams") or [{}])[0]
                status["diagnostics"]["video_resolution"] = f"{stream.get('width')}x{stream.get('height')}"
            print(f"OCR video path: {videos[0]}", flush=True)
            print(f"OCR video resolution: {status['diagnostics']['video_resolution']}", flush=True)
            vf = f"fps={sample_fps},crop=iw*{region['right']-region['left']}:ih*{region['bottom']-region['top']}:iw*{region['left']}:ih*{region['top']}"
            speaker_vf = f"fps={sample_fps},crop=iw*{speaker_region['right']-speaker_region['left']}:ih*{speaker_region['bottom']-speaker_region['top']}:iw*{speaker_region['left']}:ih*{speaker_region['top']}"
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
            speaker_cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error"]
            if start_time is not None:
                speaker_cmd += ["-ss", str(start_time)]
            speaker_cmd += ["-i", str(videos[0])]
            if end_time is not None:
                speaker_cmd += ["-t", str(end_time - (start_time or 0))]
            speaker_cmd += ["-vf", speaker_vf, str(speaker_frames / "frame_%08d.png")]
            subprocess.run(speaker_cmd, check=True, timeout=600)
            images = sorted(frames.glob("*.png"))
            speaker_images = sorted(speaker_frames.glob("*.png"))
            status["diagnostics"]["frame_count"] = len(images)
            status["diagnostics"]["speaker_frame_count"] = len(speaker_images)
            print(f"OCR crop region: {region}", flush=True)
            print(f"Speaker crop region: {speaker_region}", flush=True)
            print(f"OCR extracted frames: {len(images)}", flush=True)
            if not images or len(images) != len(speaker_images):
                raise RuntimeError("FFmpeg completed, but produced no subtitle-region frames")
            ocr, engine_name = build_ocr_engine()
            status["ocr_engine"] = engine_name
            print(f"OCR engine: {engine_name}", flush=True)
            offset = start_time or 0.0
            samples = []
            previous_signature = None
            previous_result = None
            previous_speaker_signature = None
            previous_speaker_result: tuple[str | None, int] | None = None
            similarity_threshold = status["diagnostics"]["frame_similarity_threshold"]
            for index, (image, speaker_image) in enumerate(zip(images, speaker_images)):
                signature = frame_signature(image)
                if should_reuse_ocr(previous_signature, signature, similarity_threshold) and previous_result:
                    text, confidence, raw_count = previous_result
                    status["diagnostics"]["ocr_reused_frame_count"] += 1
                else:
                    text, confidence, raw_count = extract_ocr_lines(image, ocr)
                    previous_result = (text, confidence, raw_count)
                    status["diagnostics"]["ocr_call_count"] += 1
                    status["diagnostics"]["raw_ocr_result_count"] += raw_count
                previous_signature = signature
                speaker_signature = frame_signature(speaker_image)
                if should_reuse_ocr(previous_speaker_signature, speaker_signature, similarity_threshold) and previous_speaker_result is not None:
                    speaker, speaker_raw_count = previous_speaker_result
                    status["diagnostics"]["speaker_reused_frame_count"] += 1
                else:
                    speaker, speaker_raw_count = extract_speaker_label(speaker_image, ocr)
                    previous_speaker_result = (speaker, speaker_raw_count)
                    status["diagnostics"]["speaker_ocr_call_count"] += 1
                previous_speaker_signature = speaker_signature
                if speaker:
                    status["diagnostics"]["speaker_identified_frame_count"] += 1
                samples.append({
                    "time": offset + index / sample_fps,
                    "speaker": speaker or "旁白",
                    "text": text,
                    "confidence": confidence,
                })
                if (index + 1) % 250 == 0 or index + 1 == len(images):
                    print(
                        f"OCR progress: {index + 1}/{len(images)} frames, "
                        f"calls={status['diagnostics']['ocr_call_count']}, "
                        f"reused={status['diagnostics']['ocr_reused_frame_count']}, "
                        f"speaker_calls={status['diagnostics']['speaker_ocr_call_count']}",
                        flush=True,
                    )
            segments = merge_samples(samples, 1 / sample_fps)
            status["diagnostics"]["filtered_result_count"] = len(segments)
            print(f"OCR calls: {status['diagnostics']['ocr_call_count']}", flush=True)
            print(f"OCR reused frames: {status['diagnostics']['ocr_reused_frame_count']}", flush=True)
            print(f"Speaker OCR calls: {status['diagnostics']['speaker_ocr_call_count']}", flush=True)
            print(f"OCR raw results: {status['diagnostics']['raw_ocr_result_count']}", flush=True)
            print(f"OCR merged segments: {len(segments)}", flush=True)
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
