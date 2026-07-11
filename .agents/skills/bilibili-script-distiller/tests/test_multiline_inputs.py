import importlib.util
import io
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock


SCRIPT = Path(__file__).parents[1] / "scripts" / "github_actions_extract.py"
SPEC = importlib.util.spec_from_file_location("github_actions_extract", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class MultilineInputTests(unittest.TestCase):
    def test_three_plain_links_remain_three_items(self):
        raw = (
            "https://b23.tv/example1\r\n"
            "https://b23.tv/example2\r\n"
            "https://b23.tv/example3\r\n"
        )
        self.assertEqual(len(MODULE.parse_inputs(raw)), 3)

    def test_cli_reports_three_inputs_before_network_access(self):
        raw = (
            "https://b23.tv/example1\n"
            "https://b23.tv/example2\n"
            "https://b23.tv/example3\n"
        )
        with tempfile.TemporaryDirectory() as temp:
            argv = [
                str(SCRIPT),
                "--output-root",
                str(Path(temp) / "output"),
                "--summary-json",
                str(Path(temp) / "summary.json"),
            ]
            fake_result = {
                "original_input": "mock",
                "success": False,
                "video_id": None,
                "failure_reason": "mocked; no network request",
                "result_dir": str(Path(temp) / "output"),
            }
            output = io.StringIO()
            with (
                mock.patch.dict(os.environ, {"VIDEO_URLS": raw}),
                mock.patch.object(sys, "argv", argv),
                mock.patch.object(MODULE, "run_one", return_value=fake_result),
                redirect_stdout(output),
            ):
                MODULE.main()
        log = output.getvalue()
        self.assertIn("Parsed inputs: 3", log)
        first_processing = log.find("Processing 1/3")
        parsed = log.find("Parsed inputs: 3")
        self.assertGreater(first_processing, parsed)

    def test_numbered_links_are_normalized(self):
        raw = (
            "1. https://b23.tv/example1\n"
            "2、https://b23.tv/example2\n"
            "（3） https://b23.tv/example3\n"
        )
        self.assertEqual(
            MODULE.parse_inputs(raw),
            [
                "https://b23.tv/example1",
                "https://b23.tv/example2",
                "https://b23.tv/example3",
            ],
        )

    def test_validation_is_per_item(self):
        values = MODULE.parse_inputs(
            "BV1uknVz9EeN\nnot a url\nAV170001\nhttps://b23.tv/example1"
        )
        self.assertEqual(
            [MODULE.is_supported_input(value) for value in values],
            [True, False, True, True],
        )

    def test_confirmed_bv_id_is_parsed(self):
        final_url, video_id = MODULE.resolve_input("BV1uknVz9EeN")
        self.assertEqual(video_id, "BV1uknVz9EeN")
        self.assertEqual(final_url, "https://www.bilibili.com/video/BV1uknVz9EeN/")


if __name__ == "__main__":
    unittest.main()
