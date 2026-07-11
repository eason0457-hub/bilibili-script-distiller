import importlib.util
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "hardsub_ocr.py"
SPEC = importlib.util.spec_from_file_location("hardsub_ocr", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class HardSubtitleOcrTests(unittest.TestCase):
    def test_similar_adjacent_ocr_samples_merge_into_one_segment(self):
        segments = MODULE.merge_samples(
            [
                {"time": 0.0, "text": "等等", "confidence": 0.92},
                {"time": 0.5, "text": "等 等", "confidence": 0.88},
                {"time": 1.0, "text": "不是", "confidence": 0.95},
            ],
            0.5,
        )
        self.assertEqual(len(segments), 2)
        self.assertEqual(segments[0]["start"], 0.0)
        self.assertEqual(segments[0]["end"], 1.0)

    def test_srt_and_raw_markdown_preserve_text_and_timestamps(self):
        segments = [{"start": 1.0, "end": 2.5, "text": "啊，等等……", "confidence": 0.9}]
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "subtitle-ocr.srt"
            MODULE.write_srt(segments, output)
            srt = output.read_text(encoding="utf-8")
        self.assertIn("00:00:01,000 --> 00:00:02,500", srt)
        self.assertIn("啊，等等……", srt)
        self.assertIn("OCR confidence: 0.900", MODULE.srt_to_markdown(segments))

    def test_default_bottom_crop_is_valid(self):
        crop = MODULE.crop_for("bottom", {"top": None, "bottom": None, "left": None, "right": None})
        self.assertEqual(crop["top"], 0.70)
        self.assertEqual(crop["bottom"], 0.98)

    def test_ocr_backend_uses_onnx_runtime_not_paddle_native_inference(self):
        source = SCRIPT.read_text(encoding="utf-8")
        self.assertIn("rapidocr_onnxruntime", source)
        self.assertIn("RapidOCR", source)
        self.assertNotIn("from paddleocr import PaddleOCR", source)

    def test_ocr_status_has_required_diagnostics(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)
            status = MODULE.run_hardsub_ocr(
                bvid="BV1uknVz9EeN", title="test", url="invalid", output_dir=output,
                sample_fps=2.0, position="bottom", language="ch",
                start_time=0.0, end_time=3.0,
                crop_overrides={"top": None, "bottom": None, "left": None, "right": None},
            )
            saved = (output / "ocr-status.json").read_text(encoding="utf-8")
        self.assertFalse(status["success"])
        self.assertIn("frame_count", saved)
        self.assertIn("ocr_call_count", saved)


if __name__ == "__main__":
    unittest.main()
