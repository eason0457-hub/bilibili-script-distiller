"""Optional hard-subtitle OCR fallback. Invoked only after BBDown finds no track."""

from __future__ import annotations

import datetime as dt
import difflib
import json
import os
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
            f"{item.get('speaker') or '旁白'}：{item['text']}",
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
        "bottom": {"top": 0.62, "bottom": 0.96, "left": 0.0, "right": 1.0},
        "top": {"top": 0.02, "bottom": 0.30, "left": 0.0, "right": 1.0},
    }
    region = dict(presets.get(position, presets["bottom"]))
    for key, value in overrides.items():
        if value is not None:
            region[key] = value
    if not (0 <= region["left"] < region["right"] <= 1 and 0 <= region["top"] < region["bottom"] <= 1):
        raise ValueError("subtitle crop boundaries must be between 0 and 1 and form a non-empty region")
    return region


def speaker_crop_regions() -> dict[str, dict[str, float]]:
    """Scan only dialogue-box name areas, never the top watermark band."""
    return {
        "left": {"top": 0.55, "bottom": 0.82, "left": 0.05, "right": 0.45},
        "center": {"top": 0.52, "bottom": 0.80, "left": 0.30, "right": 0.70},
    }


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


def load_character_dictionary(path: Path | None = None) -> dict:
    path = path or Path(__file__).resolve().parents[1] / "references" / "character-name-dictionary.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data.get("characters"), list):
        raise ValueError("character name dictionary must contain a characters list")
    return data


def compact_name_candidate(text: str) -> str:
    return re.sub(r"[\s【】\[\]（）()：:·.、，,!?！？'\"“”]+", "", text or "")


def candidate_shape_is_valid(value: str) -> bool:
    if not value or len(value) > 8 or re.search(r"\d|\d{1,2}:\d{2}", value):
        return False
    latin_count = len(re.findall(r"[A-Za-z]", value))
    if latin_count > 1 or latin_count / len(value) > 0.34:
        return False
    return bool(re.search(r"[\u3040-\u30ff\u3400-\u9fff]", value))


def normalize_name_with_dictionary(value: str, confidence: float, dictionary: dict) -> tuple[str | None, float]:
    """Map only exact or high-confidence near matches to a canonical name."""
    best_name, best_score = None, 0.0
    for character in dictionary.get("characters", []):
        canonical = character.get("canonical")
        for alias in character.get("aliases", []):
            score = difflib.SequenceMatcher(None, value, compact_name_candidate(alias)).ratio()
            if score > best_score:
                best_name, best_score = canonical, score
    if best_score == 1.0 and confidence >= 0.72:
        return best_name, confidence
    if best_score >= 0.74 and confidence >= 0.80:
        return best_name, confidence * best_score
    return None, confidence * best_score


def choose_speaker_candidate(candidates: list[dict], dictionary: dict) -> dict:
    """Select a canonical speaker, distinguish missing labels from doubtful labels."""
    blocked = {str(term).casefold() for term in dictionary.get("blocked_terms", [])}
    detected, rejected, normalized = [], [], []
    saw_name_like = False
    low_confidence_name = False
    for candidate in candidates:
        raw = str(candidate.get("text", "")).strip()
        confidence = float(candidate.get("confidence", 0))
        value = compact_name_candidate(raw)
        if not value:
            continue
        detected.append({"text": raw, "confidence": confidence, "region": candidate.get("region")})
        if any(term in value.casefold() for term in blocked):
            rejected.append({"text": raw, "reason": "blocked watermark/UI term"})
            continue
        if not candidate_shape_is_valid(value):
            rejected.append({"text": raw, "reason": "not a short CJK name"})
            if len(value) <= 8:
                saw_name_like = True
            continue
        saw_name_like = True
        canonical, score = normalize_name_with_dictionary(value, confidence, dictionary)
        if canonical:
            normalized.append({
                "speaker": canonical,
                "confidence": score,
                "raw": raw,
                "region": candidate.get("region"),
            })
        elif confidence < 0.78:
            low_confidence_name = True
            rejected.append({"text": raw, "reason": "low confidence"})
        else:
            rejected.append({"text": raw, "reason": "not matched to character dictionary"})
    if normalized:
        best = max(normalized, key=lambda item: item["confidence"])
        return {
            "status": "identified",
            "speaker": best["speaker"],
            "confidence": best["confidence"],
            "raw": best["raw"],
            "region": best["region"],
            "detected": detected,
            "rejected": rejected,
        }
    if saw_name_like or low_confidence_name:
        return {
            "status": "unknown",
            "speaker": "未知说话者",
            "confidence": 0.0,
            "raw": None,
            "region": None,
            "detected": detected,
            "rejected": rejected,
        }
    return {
        "status": "missing",
        "speaker": "旁白",
        "confidence": 0.0,
        "raw": None,
        "region": None,
        "detected": detected,
        "rejected": rejected,
    }


def extract_ocr_lines(image: Path, ocr, min_body_y_ratio: float = 0.22) -> tuple[str, float, int]:
    from PIL import Image

    result, _elapsed = ocr(str(image))
    lines = result or []
    text_parts, confidences = [], []
    with Image.open(image) as source:
        image_height = source.height
    for line in lines or []:
        try:
            text, confidence = line[1], line[2]
        except (IndexError, TypeError):
            continue
        try:
            box = line[0]
            center_y = sum(float(point[1]) for point in box) / len(box)
            if image_height and center_y / image_height < min_body_y_ratio:
                continue
        except (IndexError, TypeError, ValueError, ZeroDivisionError):
            pass
        text_parts.append(str(text))
        confidences.append(float(confidence))
    text = "".join(text_parts).strip() or "[无法识别]"
    return text, (sum(confidences) / len(confidences) if confidences else 0.0), len(lines or [])


def preprocessed_name_image(image: Path):
    """Upscale and binarize a small colored name tag for a second OCR pass."""
    import cv2

    source = cv2.imread(str(image))
    if source is None:
        raise ValueError(f"unable to read speaker crop: {image}")
    enlarged = cv2.resize(source, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(enlarged, cv2.COLOR_BGR2GRAY)
    enhanced = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8)).apply(gray)
    return cv2.adaptiveThreshold(
        enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 31, 9,
    )


def ocr_name_region(image: Path, ocr, region: str) -> tuple[list[dict], int]:
    candidates, calls = [], 0
    for variant in (str(image), preprocessed_name_image(image)):
        result, _elapsed = ocr(variant)
        calls += 1
        for line in result or []:
            try:
                text, confidence = str(line[1]), float(line[2])
            except (IndexError, TypeError, ValueError):
                continue
            candidates.append({"text": text, "confidence": confidence, "region": region})
    return candidates, calls


def detect_speaker(left_image: Path, center_image: Path, ocr, dictionary: dict) -> tuple[dict, int]:
    left_candidates, left_calls = ocr_name_region(left_image, ocr, "left")
    center_candidates, center_calls = ocr_name_region(center_image, ocr, "center")
    return choose_speaker_candidate(left_candidates + center_candidates, dictionary), left_calls + center_calls


def resolve_speaker_for_frame(
    detection: dict,
    previous_speaker: str | None,
    previous_text: str | None,
    current_text: str,
    frames_since_label: int,
    max_inherit_frames: int = 2,
) -> tuple[str, float, int]:
    if detection["status"] == "identified":
        return detection["speaker"], float(detection["confidence"]), 0
    if detection["status"] == "unknown":
        return "未知说话者", 0.0, max_inherit_frames + 1
    same_dialogue = bool(previous_text and similar(previous_text, current_text, threshold=0.55))
    if previous_speaker not in {None, "旁白", "未知说话者"} and same_dialogue and frames_since_label < max_inherit_frames:
        return previous_speaker, 0.0, frames_since_label + 1
    return "旁白", 0.0, max_inherit_frames + 1


def save_speaker_debug_sample(
    *,
    video: Path,
    time_seconds: float,
    body_image: Path,
    left_image: Path,
    center_image: Path,
    detection: dict,
    resolved_speaker: str,
    regions: dict,
    output_root: Path,
    index: int,
) -> None:
    """Save a small, non-repository debug bundle for changed speaker labels."""
    from PIL import Image, ImageDraw, ImageFont

    sample_dir = output_root / f"sample_{index:02d}_{time_seconds:010.3f}"
    sample_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(body_image, sample_dir / "dialogue-body.png")
    shutil.copy2(left_image, sample_dir / "speaker-left.png")
    shutil.copy2(center_image, sample_dir / "speaker-center.png")
    original = sample_dir / "original-frame.png"
    subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error",
            "-ss", str(time_seconds), "-i", str(video),
            "-frames:v", "1", "-y", str(original),
        ],
        check=True,
        timeout=60,
    )
    with Image.open(original).convert("RGB") as source:
        draw = ImageDraw.Draw(source)
        width, height = source.size
        colors = {"left": "#ff4fa3", "center": "#48b6ff"}
        for name in ("left", "center"):
            region = regions[name]
            draw.rectangle(
                (
                    int(width * region["left"]), int(height * region["top"]),
                    int(width * region["right"]), int(height * region["bottom"]),
                ),
                outline=colors[name],
                width=max(2, width // 320),
            )
        try:
            font = ImageFont.truetype(
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                max(14, width // 32),
            )
        except OSError:
            font = ImageFont.load_default()
        label = f"speaker={resolved_speaker} status={detection['status']}"
        draw.rectangle((0, 0, min(width, len(label) * 12 + 12), 30), fill="black")
        draw.text((6, 5), label, fill="white", font=font)
        source.save(sample_dir / "ocr-overlay.png")
    (sample_dir / "ocr-result.json").write_text(
        json.dumps(
            {
                "time": time_seconds,
                "resolved_speaker": resolved_speaker,
                "detection": detection,
            },
            ensure_ascii=False,
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )


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
        "speaker_crop_regions": None,
        "sample_fps": sample_fps,
        "ocr_language": language,
        "ocr_engine": "rapidocr_onnxruntime",
        "average_confidence": 0.0,
        "low_confidence_segments": [],
        "detected_speaker_labels": [],
        "normalized_speaker_names": [],
        "unknown_speaker_count": 0,
        "narrator_count": 0,
        "rejected_name_candidates": [],
        "speaker_name_confidence": {},
        "success": False,
        "failure_reason": None,
        "diagnostics": {
            "video_path": None,
            "video_resolution": None,
            "frame_count": 0,
            "left_speaker_frame_count": 0,
            "center_speaker_frame_count": 0,
            "ocr_call_count": 0,
            "ocr_reused_frame_count": 0,
            "speaker_ocr_call_count": 0,
            "left_speaker_reused_frame_count": 0,
            "center_speaker_reused_frame_count": 0,
            "speaker_identified_frame_count": 0,
            "speaker_debug_sample_count": 0,
            "raw_ocr_result_count": 0,
            "filtered_result_count": 0,
            "frame_similarity_threshold": 0.02,
        },
        "processed_at": dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat(),
    }
    try:
        region = crop_for(position, crop_overrides)
        name_regions = speaker_crop_regions()
        dictionary = load_character_dictionary()
        status["crop_region"] = region
        status["speaker_crop_regions"] = name_regions
        with tempfile.TemporaryDirectory(prefix="bili-hardsub-") as temp_name:
            temp = Path(temp_name)
            video_dir = temp / "video"
            frames = temp / "frames"
            left_speaker_frames = temp / "left_speaker_frames"
            center_speaker_frames = temp / "center_speaker_frames"
            video_dir.mkdir()
            frames.mkdir()
            left_speaker_frames.mkdir()
            center_speaker_frames.mkdir()
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
            for name, target in (
                ("left", left_speaker_frames),
                ("center", center_speaker_frames),
            ):
                name_region = name_regions[name]
                name_vf = f"fps={sample_fps},crop=iw*{name_region['right']-name_region['left']}:ih*{name_region['bottom']-name_region['top']}:iw*{name_region['left']}:ih*{name_region['top']}"
                name_cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error"]
                if start_time is not None:
                    name_cmd += ["-ss", str(start_time)]
                name_cmd += ["-i", str(videos[0])]
                if end_time is not None:
                    name_cmd += ["-t", str(end_time - (start_time or 0))]
                name_cmd += ["-vf", name_vf, str(target / "frame_%08d.png")]
                subprocess.run(name_cmd, check=True, timeout=600)
            images = sorted(frames.glob("*.png"))
            left_speaker_images = sorted(left_speaker_frames.glob("*.png"))
            center_speaker_images = sorted(center_speaker_frames.glob("*.png"))
            status["diagnostics"]["frame_count"] = len(images)
            status["diagnostics"]["left_speaker_frame_count"] = len(left_speaker_images)
            status["diagnostics"]["center_speaker_frame_count"] = len(center_speaker_images)
            print(f"OCR crop region: {region}", flush=True)
            print(f"Left speaker crop region: {name_regions['left']}", flush=True)
            print(f"Center speaker crop region: {name_regions['center']}", flush=True)
            print(f"OCR extracted frames: {len(images)}", flush=True)
            if not images or not (
                len(images) == len(left_speaker_images) == len(center_speaker_images)
            ):
                raise RuntimeError("FFmpeg completed, but produced no subtitle-region frames")
            ocr, engine_name = build_ocr_engine()
            status["ocr_engine"] = engine_name
            print(f"OCR engine: {engine_name}", flush=True)
            offset = start_time or 0.0
            samples = []
            previous_signature = None
            previous_result = None
            previous_left_signature = None
            previous_center_signature = None
            previous_detection: dict | None = None
            previous_speaker = None
            previous_text = None
            frames_since_label = 99
            detected_labels: list[dict] = []
            rejected_candidates: list[dict] = []
            debug_root = (
                Path(os.environ.get("RUNNER_TEMP", temp))
                / "bilibili-ocr-debug"
                / re.sub(r"[^0-9A-Za-z_-]+", "_", bvid)
            )
            similarity_threshold = status["diagnostics"]["frame_similarity_threshold"]
            for index, (image, left_image, center_image) in enumerate(
                zip(images, left_speaker_images, center_speaker_images)
            ):
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
                left_signature = frame_signature(left_image)
                center_signature = frame_signature(center_image)
                reuse_left = should_reuse_ocr(previous_left_signature, left_signature, similarity_threshold)
                reuse_center = should_reuse_ocr(previous_center_signature, center_signature, similarity_threshold)
                if reuse_left and reuse_center and previous_detection is not None:
                    detection = previous_detection
                    status["diagnostics"]["left_speaker_reused_frame_count"] += 1
                    status["diagnostics"]["center_speaker_reused_frame_count"] += 1
                else:
                    detection, speaker_calls = detect_speaker(
                        left_image, center_image, ocr, dictionary
                    )
                    previous_detection = detection
                    status["diagnostics"]["speaker_ocr_call_count"] += speaker_calls
                previous_left_signature = left_signature
                previous_center_signature = center_signature
                detected_labels.extend(detection["detected"])
                rejected_candidates.extend(detection["rejected"])
                speaker, speaker_confidence, frames_since_label = resolve_speaker_for_frame(
                    detection,
                    previous_speaker,
                    previous_text,
                    text,
                    frames_since_label,
                )
                if detection["status"] == "identified":
                    status["diagnostics"]["speaker_identified_frame_count"] += 1
                samples.append({
                    "time": offset + index / sample_fps,
                    "speaker": speaker,
                    "speaker_confidence": speaker_confidence,
                    "text": text,
                    "confidence": confidence,
                })
                if (
                    speaker != previous_speaker
                    and status["diagnostics"]["speaker_debug_sample_count"] < 20
                ):
                    save_speaker_debug_sample(
                        video=videos[0],
                        time_seconds=offset + index / sample_fps,
                        body_image=image,
                        left_image=left_image,
                        center_image=center_image,
                        detection=detection,
                        resolved_speaker=speaker,
                        regions=name_regions,
                        output_root=debug_root,
                        index=status["diagnostics"]["speaker_debug_sample_count"] + 1,
                    )
                    status["diagnostics"]["speaker_debug_sample_count"] += 1
                previous_speaker = speaker
                previous_text = text
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
            seen_labels = set()
            status["detected_speaker_labels"] = []
            for item in detected_labels:
                key = (item["text"], item.get("region"), round(float(item["confidence"]), 4))
                if key not in seen_labels:
                    seen_labels.add(key)
                    status["detected_speaker_labels"].append(item)
            status["normalized_speaker_names"] = sorted({
                item["speaker"] for item in segments
                if item["speaker"] not in {"旁白", "未知说话者"}
            })
            status["unknown_speaker_count"] = sum(
                item["speaker"] == "未知说话者" for item in segments
            )
            status["narrator_count"] = sum(item["speaker"] == "旁白" for item in segments)
            seen_rejected = set()
            status["rejected_name_candidates"] = []
            for item in rejected_candidates:
                key = (item["text"], item["reason"])
                if key not in seen_rejected:
                    seen_rejected.add(key)
                    status["rejected_name_candidates"].append(item)
            confidence_by_speaker: dict[str, list[float]] = {}
            for item in samples:
                if item["speaker"] not in {"旁白", "未知说话者"} and item["speaker_confidence"] > 0:
                    confidence_by_speaker.setdefault(item["speaker"], []).append(item["speaker_confidence"])
            status["speaker_name_confidence"] = {
                speaker: sum(values) / len(values)
                for speaker, values in confidence_by_speaker.items()
            }
            print(f"OCR calls: {status['diagnostics']['ocr_call_count']}", flush=True)
            print(f"OCR reused frames: {status['diagnostics']['ocr_reused_frame_count']}", flush=True)
            print(f"Speaker OCR calls: {status['diagnostics']['speaker_ocr_call_count']}", flush=True)
            print(f"Normalized speakers: {status['normalized_speaker_names']}", flush=True)
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
