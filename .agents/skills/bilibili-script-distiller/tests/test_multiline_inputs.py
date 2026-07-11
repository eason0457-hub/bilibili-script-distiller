import importlib.util
import unittest
from pathlib import Path


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
            "BV1xx411c7mD\nnot a url\nAV170001\nhttps://b23.tv/example1"
        )
        self.assertEqual(
            [MODULE.is_supported_input(value) for value in values],
            [True, False, True, True],
        )


if __name__ == "__main__":
    unittest.main()
