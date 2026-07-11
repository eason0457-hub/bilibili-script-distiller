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
    @classmethod
    def setUpClass(cls):
        cls.dictionary = MODULE.load_character_dictionary()

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

    def test_same_text_from_different_speakers_does_not_merge(self):
        segments = MODULE.merge_samples(
            [
                {"time": 0.0, "speaker": "爱音", "text": "嗯", "confidence": 0.9},
                {"time": 0.5, "speaker": "旁白", "text": "嗯", "confidence": 0.9},
            ],
            0.5,
        )
        self.assertEqual(len(segments), 2)

    def test_left_name_label_is_recognized(self):
        result = MODULE.choose_speaker_candidate(
            [{"text": "千早爱音", "confidence": 0.96, "region": "left"}],
            self.dictionary,
        )
        self.assertEqual(result["speaker"], "千早爱音")
        self.assertEqual(result["region"], "left")

    def test_center_name_label_is_recognized(self):
        result = MODULE.choose_speaker_candidate(
            [{"text": "祥子", "confidence": 0.95, "region": "center"}],
            self.dictionary,
        )
        self.assertEqual(result["speaker"], "祥子")
        self.assertEqual(result["region"], "center")

    def test_missing_name_label_outputs_narrator(self):
        detection = MODULE.choose_speaker_candidate([], self.dictionary)
        speaker, _confidence, _age = MODULE.resolve_speaker_for_frame(
            detection, None, None, "旁白正文", 99,
        )
        self.assertEqual(speaker, "旁白")

    def test_watermark_is_rejected_not_used_as_speaker(self):
        result = MODULE.choose_speaker_candidate(
            [{"text": "bilibili", "confidence": 0.99, "region": "left"}],
            self.dictionary,
        )
        self.assertEqual(result["status"], "missing")
        self.assertEqual(result["speaker"], "旁白")
        self.assertEqual(result["rejected"][0]["reason"], "blocked watermark/UI term")

    def test_near_name_maps_to_canonical_character(self):
        for raw, expected in (
            ("干早爱音", "千早爱音"),
            ("样子", "祥子"),
            ("立希I", "立希"),
        ):
            with self.subTest(raw=raw):
                result = MODULE.choose_speaker_candidate(
                    [{"text": raw, "confidence": 0.94, "region": "left"}],
                    self.dictionary,
                )
                self.assertEqual(result["speaker"], expected)

    def test_low_confidence_name_outputs_unknown(self):
        result = MODULE.choose_speaker_candidate(
            [{"text": "灯", "confidence": 0.50, "region": "center"}],
            self.dictionary,
        )
        self.assertEqual(result["status"], "unknown")
        self.assertEqual(result["speaker"], "未知说话者")

    def test_new_high_confidence_label_switches_character(self):
        first = MODULE.choose_speaker_candidate(
            [{"text": "千早爱音", "confidence": 0.95, "region": "left"}],
            self.dictionary,
        )
        second = MODULE.choose_speaker_candidate(
            [{"text": "祥子", "confidence": 0.95, "region": "center"}],
            self.dictionary,
        )
        speaker, _confidence, age = MODULE.resolve_speaker_for_frame(
            first, None, None, "第一句", 99,
        )
        switched, _confidence, _age = MODULE.resolve_speaker_for_frame(
            second, speaker, "第一句", "第二句", age,
        )
        self.assertEqual(speaker, "千早爱音")
        self.assertEqual(switched, "祥子")

    def test_body_ocr_excludes_name_band_without_changing_dialogue(self):
        from PIL import Image

        class FakeOcr:
            def __call__(self, _image):
                return [
                    [[[0, 0], [90, 0], [90, 12], [0, 12]], "千早爱音", 0.99],
                    [[[0, 50], [90, 50], [90, 70], [0, 70]], "正文，保持……", 0.98],
                ], 0.01

        with tempfile.TemporaryDirectory() as temp:
            image = Path(temp) / "body.png"
            Image.new("RGB", (100, 100), "white").save(image)
            text, _confidence, count = MODULE.extract_ocr_lines(image, FakeOcr())
        self.assertEqual(count, 2)
        self.assertEqual(text, "正文，保持……")

    def test_srt_and_raw_markdown_preserve_text_and_timestamps(self):
        segments = [{
            "start": 1.0, "end": 2.5, "speaker": "爱音",
            "text": "啊，等等……", "confidence": 0.9,
        }]
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "subtitle-ocr.srt"
            MODULE.write_srt(segments, output)
            srt = output.read_text(encoding="utf-8")
        self.assertIn("00:00:01,000 --> 00:00:02,500", srt)
        self.assertIn("爱音：", srt)
        self.assertIn("啊，等等……", srt)
        raw = MODULE.srt_to_markdown(segments)
        self.assertIn("爱音：", raw)
        self.assertIn("OCR confidence: 0.900", raw)

    def test_default_bottom_crop_is_valid(self):
        crop = MODULE.crop_for("bottom", {"top": None, "bottom": None, "left": None, "right": None})
        self.assertEqual(crop["top"], 0.62)
        self.assertEqual(crop["bottom"], 0.96)
        regions = MODULE.speaker_crop_regions()
        self.assertEqual(regions["left"], {"top": 0.55, "bottom": 0.82, "left": 0.05, "right": 0.45})
        self.assertEqual(regions["center"], {"top": 0.52, "bottom": 0.80, "left": 0.30, "right": 0.70})

    def test_ocr_backend_uses_onnx_runtime_not_paddle_native_inference(self):
        source = SCRIPT.read_text(encoding="utf-8")
        self.assertIn("rapidocr_onnxruntime", source)
        self.assertIn("RapidOCR", source)
        self.assertNotIn("from paddleocr import PaddleOCR", source)

    def test_similar_frame_signatures_reuse_previous_ocr(self):
        same = bytes([0, 1, 0, 1] * 100)
        slightly_changed = bytearray(same)
        slightly_changed[0] = 1
        different = bytes([1 - value for value in same])
        self.assertTrue(MODULE.should_reuse_ocr(same, bytes(slightly_changed)))
        self.assertFalse(MODULE.should_reuse_ocr(same, different))
        self.assertEqual(MODULE.signature_distance(same, same), 0.0)

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
        self.assertIn("ocr_reused_frame_count", saved)
        self.assertIn("speaker_ocr_call_count", saved)
        self.assertIn("detected_speaker_labels", saved)
        self.assertIn("normalized_speaker_names", saved)
        self.assertIn("unknown_speaker_count", saved)
        self.assertIn("narrator_count", saved)
        self.assertIn("rejected_name_candidates", saved)
        self.assertIn("speaker_name_confidence", saved)


if __name__ == "__main__":
    unittest.main()
