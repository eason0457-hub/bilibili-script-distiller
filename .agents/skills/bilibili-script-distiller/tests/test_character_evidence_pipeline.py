import csv
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "character_evidence_pipeline.py"
SPEC = importlib.util.spec_from_file_location("character_pipeline", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


REGISTRY = """characters:
  - canonical_name: 千早爱音
    aliases: [爱音, ano, ano酱]
    ocr_common_misrecognitions: [干早爱音]
    nicknames: [ano酱]
    addressed_by: [小爱音]
    identity_uncertainties: []
  - canonical_name: 高松灯
    aliases: [灯, 高松灯]
    ocr_common_misrecognitions: []
    nicknames: [tomorin]
    addressed_by: [小灯]
    identity_uncertainties: []
"""


class CharacterEvidencePipelineTests(unittest.TestCase):
    def build_project(self):
        temp = tempfile.TemporaryDirectory()
        root = Path(temp.name)
        (root / "references/characters").mkdir(parents=True)
        (root / "references/characters/character-registry.yaml").write_text(REGISTRY, encoding="utf-8")
        video = root / "references/bilibili/BV1test00001"
        video.mkdir(parents=True)
        (video / "extraction-status.json").write_text(json.dumps({"success": True}), encoding="utf-8")
        (video / "subtitle-raw.md").write_text("""# 原始字幕

[00:00:01.000 --> 00:00:02.000]
高松灯：
等一下……
<!-- OCR confidence: 0.990 -->

[00:00:03.000 --> 00:00:04.000]
旁白：
我是千早爱音！
<!-- OCR confidence: 0.990 -->

[00:00:05.000 --> 00:00:06.000]
bilibili：
ano酱？
<!-- OCR confidence: 0.990 -->
""", encoding="utf-8")
        return temp, root, video

    def test_attribution_is_conservative_and_generates_inventory(self):
        temp, root, video = self.build_project()
        with temp:
            stats = MODULE.run(root)
            tagged = (video / "subtitle-speaker-tagged.md").read_text(encoding="utf-8")
            self.assertIn("canonical_name: 高松灯", tagged)
            self.assertIn("assignment_confidence: high", tagged)
            self.assertIn("canonical_name: 千早爱音", tagged)
            self.assertIn("assignment_confidence: medium", tagged)
            self.assertIn("canonical_name: UNKNOWN", tagged)
            self.assertIn("assignment_basis: 无法判断", tagged)
            self.assertEqual(stats["高松灯"]["high"], 1)
            self.assertEqual(stats["千早爱音"]["medium"], 1)
            with (video / "keyword-inventory.raw.csv").open(encoding="utf-8") as handle:
                fields = csv.DictReader(handle).fieldnames
            self.assertEqual(fields, ["character", "expression", "normalized_expression", "type", "count", "contexts", "timestamps", "confidence", "source_bv"])
            self.assertTrue((root / "references/characters/千早爱音/uncertain-evidence.md").exists())

    def test_addressing_a_character_does_not_assign_the_speaker(self):
        registry = MODULE.read_registry(self._registry_path())
        lookup = MODULE.alias_lookup(registry)
        name, confidence, basis = MODULE.attribute_cue("旁白", "ano酱，你醒了吗？", lookup)
        self.assertEqual((name, confidence, basis), ("UNKNOWN", "low", "无法判断"))

    def _registry_path(self):
        temp, root, _video = self.build_project()
        self.addCleanup(temp.cleanup)
        return root / "references/characters/character-registry.yaml"


if __name__ == "__main__":
    unittest.main()
