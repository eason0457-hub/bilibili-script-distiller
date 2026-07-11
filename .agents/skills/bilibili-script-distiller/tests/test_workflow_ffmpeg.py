import unittest
from pathlib import Path


WORKFLOW = (
    Path(__file__).parents[2]
    / ".github"
    / "workflows"
    / "bilibili-subtitle-extract.yml"
)


class WorkflowFFmpegTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.text = WORKFLOW.read_text(encoding="utf-8")

    def test_ubuntu_latest_runner_is_preserved(self):
        self.assertIn("runs-on: ubuntu-latest", self.text)

    def test_ffmpeg_is_installed_and_verified(self):
        required = [
            "- name: Install FFmpeg",
            "sudo apt-get update",
            "sudo apt-get install -y ffmpeg",
            "command -v ffmpeg",
            "ffmpeg -version",
        ]
        for value in required:
            self.assertIn(value, self.text)

    def test_ffmpeg_install_precedes_extraction(self):
        checkout = self.text.index("- name: Check out repository")
        install = self.text.index("- name: Install FFmpeg")
        extract = self.text.index("- name: Extract existing subtitles")
        self.assertLess(checkout, install)
        self.assertLess(install, extract)

    def test_extraction_has_path_guard(self):
        self.assertIn(
            "if ! command -v ffmpeg >/dev/null 2>&1; then", self.text
        )
        self.assertIn(
            "::error::FFmpeg is not installed or not available in PATH", self.text
        )

    def test_hardsub_ocr_is_opt_in(self):
        self.assertIn("enable_hardsub_ocr:", self.text)
        self.assertIn("Install PaddleOCR for enabled hard-subtitle fallback", self.text)
        self.assertIn("--enable-hardsub-ocr", self.text)
        self.assertIn("ENABLE_HARDSUB_OCR: ${{ inputs.enable_hardsub_ocr }}", self.text)
        self.assertIn('case "${ENABLE_HARDSUB_OCR,,}" in', self.text)

    def test_dispatch_input_is_validated_and_uses_inputs_context(self):
        self.assertIn("- name: Validate workflow inputs", self.text)
        self.assertIn("VIDEO_URLS: ${{ inputs.video_urls }}", self.text)
        self.assertIn("::error::video_urls is empty", self.text)
        self.assertIn("HH:MM:SS", self.text)

    def test_paddleocr_uses_numpy_one_abi_and_checks_imports(self):
        self.assertIn('"numpy==1.26.4"', self.text)
        self.assertIn('"opencv-python==4.6.0.66"', self.text)
        self.assertIn('"opencv-contrib-python==4.6.0.66"', self.text)
        self.assertIn("--force-reinstall --no-deps", self.text)
        self.assertIn("from paddleocr import PaddleOCR", self.text)
        self.assertIn("PaddleOCR import check: OK", self.text)


if __name__ == "__main__":
    unittest.main()
