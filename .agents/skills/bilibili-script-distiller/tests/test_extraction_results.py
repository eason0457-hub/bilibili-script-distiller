import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


SCRIPT = Path(__file__).parents[1] / "scripts" / "github_actions_extract.py"
SPEC = importlib.util.spec_from_file_location("github_actions_extract_results", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def runner_with_file(filename=None, content="", returncode=0, stderr="任务完成"):
    def run(command, **kwargs):
        if filename:
            work_dir = Path(command[command.index("--work-dir") + 1])
            (work_dir / filename).write_text(content, encoding="utf-8")
        return subprocess.CompletedProcess(command, returncode, "视频标题: 测试标题", stderr)

    return run


class ExtractionResultTests(unittest.TestCase):
    def test_zero_exit_without_subtitle_is_failure(self):
        with tempfile.TemporaryDirectory() as temp:
            result = MODULE.run_one(
                "BV1uknVz9EeN", Path(temp), command_runner=runner_with_file()
            )
        self.assertFalse(result["success"])
        self.assertEqual(result["bbdown_exit_code"], 0)
        self.assertEqual(result["subtitle_files_found"], [])
        self.assertEqual(
            result["failure_reason"],
            "BBDown completed, but no subtitle file was produced.",
        )

    def test_found_srt_is_success_and_generates_raw_markdown(self):
        srt = (
            "1\n00:00:01,000 --> 00:00:02,500\n啊，等等……\n\n"
            "2\n00:00:03,000 --> 00:00:04,000\n不是，我是说——\n"
        )
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            result = MODULE.run_one(
                "BV1uknVz9EeN",
                root,
                command_runner=runner_with_file("BV1uknVz9EeN.zh.srt", srt),
            )
            raw = (root / "BV1uknVz9EeN" / "subtitle-raw.md").read_text()
            status = json.loads(
                (root / "BV1uknVz9EeN" / "extraction-status.json").read_text()
            )
        self.assertTrue(result["success"])
        self.assertTrue(status["success"])
        self.assertEqual(result["bbdown_exit_code"], 0)
        self.assertEqual(len(result["subtitle_files_found"]), 1)
        self.assertIn("啊，等等……", raw)
        self.assertIn("[00:00:01.000 --> 00:00:02.500]", raw)

    def test_music_only_track_is_not_a_successful_dialogue_subtitle(self):
        srt = (
            "1\n00:00:01,000 --> 00:00:02,500\n♪ 音乐 ♪\n\n"
            "2\n00:00:03,000 --> 00:00:04,000\n[Music]\n"
        )
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            result = MODULE.run_one(
                "BV1uknVz9EeN",
                root,
                command_runner=runner_with_file("BV1uknVz9EeN.ai-zh.srt", srt),
            )
        self.assertFalse(result["success"])
        self.assertEqual(result["subtitle_cue_count"], 2)
        self.assertEqual(result["dialogue_cue_count"], 0)
        self.assertEqual(
            result["failure_reason"],
            "Selected subtitle track contained only non-dialogue cues (e.g. music).",
        )
        self.assertEqual(result["rejected_subtitle_tracks"], ["BV1uknVz9EeN.ai-zh.srt"])

    def test_main_returns_nonzero_when_all_fail(self):
        failed = {
            "input": "BV1uknVz9EeN",
            "source_type": "subtitle_track",
            "bvid": "BV1uknVz9EeN",
            "success": False,
            "failure_reason": "no subtitle",
            "result_dir": "unused",
        }
        self.assertEqual(self._run_main([failed]), 2)

    def test_main_continues_and_succeeds_when_one_item_succeeds(self):
        failed = {
            "input": "BV1uknVz9EeN",
            "source_type": "subtitle_track",
            "bvid": "BV1uknVz9EeN",
            "success": False,
            "failure_reason": "no subtitle",
            "result_dir": "failed",
        }
        succeeded = {
            "input": "BV1xx411c7mD",
            "source_type": "subtitle_track",
            "bvid": "BV1xx411c7mD",
            "success": True,
            "failure_reason": None,
            "result_dir": "success",
        }
        self.assertEqual(self._run_main([failed, succeeded]), 0)

    def _run_main(self, results):
        with tempfile.TemporaryDirectory() as temp:
            argv = [
                str(SCRIPT),
                "--output-root",
                str(Path(temp) / "output"),
                "--summary-json",
                str(Path(temp) / "summary.json"),
            ]
            raw = "\n".join(item["input"] for item in results)
            with (
                mock.patch.dict(os.environ, {"VIDEO_URLS": raw}),
                mock.patch.object(sys, "argv", argv),
                mock.patch.object(MODULE, "run_one", side_effect=results) as run_one,
            ):
                code = MODULE.main()
            self.assertEqual(run_one.call_count, len(results))
            return code


if __name__ == "__main__":
    unittest.main()
