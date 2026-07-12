#!/usr/bin/env python3
"""Build conservative speaker evidence and character-level writing artifacts.

This tool never infers a speaker from plot familiarity alone.  It only accepts a
known source label or an explicit self-identification; all other dialogue remains
UNKNOWN so later aggregation cannot turn OCR noise into character facts.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

import yaml


UNKNOWN = "UNKNOWN"
BLOCKED_LABELS = {"bilibili", "biibili", "bilbili", "lilibili", "liibili", "旁白", "未知说话者"}
NOISE_RE = re.compile(r"^(?:\[无法识别\]|[\d+*#_\-.,，。！？!?\sA-Za-z]{1,8})$")
ENTRY_RE = re.compile(
    r"^\[(?P<start>[^\]]+?)\s*-->\s*(?P<end>[^\]]+?)\]\s*\n(?P<body>.*?)(?=^\[[^\]]+?\s*-->|\Z)",
    re.M | re.S,
)
SELF_ID_RE = re.compile(r"(?:我(?:是|叫(?:做)?|名叫))(?P<name>[\u3400-\u9fffA-Za-z]+)")
PATTERN_TERMS = ("不对", "等一下", "就是", "总之", "看来", "我说", "不是", "好吧", "没事", "最后")
FILLERS = ("啊", "嗯", "呃", "诶", "欸", "那个", "就是")


class Cue:
    def __init__(self, start: str, end: str, source_label: str, raw_text: str,
                 cleaned_text: str, character: str, confidence: str, basis: str):
        self.start = start
        self.end = end
        self.source_label = source_label
        self.raw_text = raw_text
        self.cleaned_text = cleaned_text
        self.character = character
        self.confidence = confidence
        self.basis = basis


def read_registry(path: Path) -> list[dict]:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    characters = data.get("characters", [])
    if not isinstance(characters, list):
        raise ValueError("character-registry.yaml must contain a characters list")
    for item in characters:
        if not item.get("canonical_name"):
            raise ValueError("every registry character needs canonical_name")
    return characters


def normalize(value: str) -> str:
    return re.sub(r"[\s【】\[\]（）()：:·.、，,!?！？'\"“”]", "", value or "").casefold()


def alias_lookup(registry: Iterable[dict]) -> dict[str, str]:
    lookup: dict[str, str] = {}
    for item in registry:
        canonical = item["canonical_name"]
        for value in [canonical, *item.get("aliases", []), *item.get("nicknames", [])]:
            key = normalize(str(value))
            if key:
                lookup[key] = canonical
    return lookup


def clean_text(value: str) -> str:
    value = re.sub(r"<!--.*?-->", "", value, flags=re.S).strip()
    value = value.replace("\r", "").replace("\n", "")
    value = re.sub(r"\s+", " ", value)
    value = value.replace("...", "……").replace("..", "……")
    return value.strip()


def is_valid_text(value: str) -> bool:
    return bool(value and not NOISE_RE.fullmatch(value) and re.search(r"[\u3400-\u9fffA-Za-z]", value))


def parse_raw_markdown(path: Path, registry: list[dict]) -> list[Cue]:
    lookup = alias_lookup(registry)
    cues: list[Cue] = []
    for match in ENTRY_RE.finditer(path.read_text(encoding="utf-8")):
        lines = [line.strip() for line in match.group("body").splitlines() if line.strip() and not line.startswith("<!--")]
        if not lines:
            continue
        source_label = ""
        if lines[0].endswith(("：", ":")):
            source_label, lines = lines[0][:-1], lines[1:]
        raw_text = "".join(lines).strip()
        cleaned = clean_text(raw_text)
        if not is_valid_text(cleaned):
            continue
        character, confidence, basis = attribute_cue(source_label, cleaned, lookup)
        cues.append(Cue(match.group("start"), match.group("end"), source_label or "—", raw_text, cleaned, character, confidence, basis))
    return cues


def attribute_cue(source_label: str, text: str, lookup: dict[str, str]) -> tuple[str, str, str]:
    """Use only direct label/self-name evidence; never turn addressees into speakers."""
    label = normalize(source_label)
    if label and label not in BLOCKED_LABELS and label in lookup:
        return lookup[label], "high", "画面姓名标签（已有 OCR 标签与角色表精确匹配）"
    for match in SELF_ID_RE.finditer(text):
        candidate = normalize(match.group("name"))
        if candidate in lookup:
            return lookup[candidate], "medium", "明确称呼（台词中的自报姓名）"
    return UNKNOWN, "low", "无法判断"


def cue_to_markdown(cue: Cue) -> str:
    return "\n".join([
        f"## [{cue.start} --> {cue.end}]",
        f"- canonical_name: {cue.character}",
        f"- raw_text: {cue.raw_text}",
        f"- cleaned_text: {cue.cleaned_text}",
        f"- assignment_confidence: {cue.confidence}",
        f"- assignment_basis: {cue.basis}",
        "",
    ])


def write_tagged(path: Path, source_bv: str, cues: list[Cue]) -> None:
    count = Counter((cue.character, cue.confidence) for cue in cues)
    header = [
        f"# 说话者归属字幕：{source_bv}",
        "",
        "> 归属只接受画面姓名标签精确匹配或台词中的明确自报姓名。称呼他人、剧情常识、OCR 残片均不足以指派说话者；因此 `UNKNOWN` 是保守结果，不是待自动补全的空值。",
        "",
        "## 归属统计",
        "",
        "| 规范角色名 | high | medium | low |",
        "| --- | ---: | ---: | ---: |",
    ]
    for name in sorted({cue.character for cue in cues}):
        header.append(f"| {name} | {count[(name, 'high')]} | {count[(name, 'medium')]} | {count[(name, 'low')]} |")
    path.write_text("\n".join(header) + "\n\n" + "\n".join(cue_to_markdown(cue) for cue in cues), encoding="utf-8")


def split_expressions(text: str) -> list[tuple[str, str]]:
    """Deterministically emit ASCII words and 1–4 CJK-character expressions."""
    expressions: list[tuple[str, str]] = []
    for ascii_word in re.findall(r"[A-Za-z]+", text):
        expressions.append((ascii_word, "word"))
    for run in re.findall(r"[\u3400-\u9fff]+", text):
        for index, _char in enumerate(run):
            for length in range(1, min(4, len(run) - index) + 1):
                token = run[index:index + length]
                expressions.append((token, "word" if length == 1 else f"phrase_{length}char"))
    for term in PATTERN_TERMS:
        if term in text:
            expressions.append((term, "fixed_pattern"))
    for term in FILLERS:
        if term in text:
            expressions.append((term, "filler"))
    for term in ("……", "？", "！"):
        if text.endswith(term):
            expressions.append((term, "sentence_end"))
    return expressions


def build_inventory(cues: list[Cue], source_bv: str) -> list[dict]:
    records: dict[tuple[str, str, str], dict] = {}
    for cue in cues:
        for expression, expression_type in split_expressions(cue.cleaned_text):
            key = (cue.character, expression, expression_type)
            row = records.setdefault(key, {
                "character": cue.character,
                "expression": expression,
                "normalized_expression": normalize(expression),
                "type": expression_type,
                "count": 0,
                "contexts": [],
                "timestamps": [],
                "confidence": cue.confidence,
                "source_bv": source_bv,
            })
            row["count"] += 1
            if len(row["contexts"]) < 1:
                row["contexts"].append(cue.cleaned_text[:80])
            if len(row["timestamps"]) < 3:
                row["timestamps"].append(cue.start)
            if {"high": 3, "medium": 2, "low": 1}[cue.confidence] > {"high": 3, "medium": 2, "low": 1}[row["confidence"]]:
                row["confidence"] = cue.confidence
    # Retain every attributed expression. UNKNOWN is a review bucket, so retain
    # recurring n-grams and all explicit discourse markers rather than exporting
    # thousands of one-frame OCR fragments as faux keywords.
    filtered = [
        row for row in records.values()
        if row["character"] != UNKNOWN
        or row["type"] in {"fixed_pattern", "filler", "sentence_end"}
        or row["count"] >= 3
    ]
    return sorted(filtered, key=lambda row: (row["character"], -row["count"], row["type"], row["expression"]))


def write_inventory(path: Path, rows: list[dict]) -> None:
    fields = ["character", "expression", "normalized_expression", "type", "count", "contexts", "timestamps", "confidence", "source_bv"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            out = dict(row)
            out["contexts"] = json.dumps(out["contexts"], ensure_ascii=False)
            out["timestamps"] = json.dumps(out["timestamps"], ensure_ascii=False)
            writer.writerow(out)


def cues_for(cues: list[Cue], character: str) -> list[Cue]:
    return [cue for cue in cues if cue.character == character]


def write_video_observations(path: Path, registry: list[dict], cues: list[Cue], source_bv: str) -> None:
    lines = [f"# 单视频人物证据：{source_bv}", "", "> 仅记录有归属证据的台词。没有可靠归属时，不从剧情或称呼反推角色。", ""]
    for person in registry:
        name = person["canonical_name"]
        evidence = cues_for(cues, name)
        high_medium = [cue for cue in evidence if cue.confidence in {"high", "medium"}]
        lines += [f"## {name}", ""]
        if len(high_medium) < 3:
            points = "、".join(cue.start for cue in high_medium) or "无"
            lines += [
                "- 可直接观察到的行为：没有足量的可靠归属台词；不能将出现姓名或被他人称呼当成其本人发言。",
                "- 可推断的性格特征：不作确定性结论。",
                f"- 推断依据：可靠台词 {len(high_medium)} 条；时间点：{points}。",
                "- 触发条件：需要姓名标签、明确自报姓名或人工确认轮次后才能扩充。",
                "- 例外情况：二创/梦境/类型模仿语境不能直接等同于原作人格。",
                "- 可信度：★☆☆☆☆（证据不足）。",
                "",
            ]
        else:
            lines += [
                "- 可直接观察到的行为：见下列可靠归属台词的可见语言/行动。",
                "- 可推断的性格特征：仅按多处可复现证据单列，不使用空泛形容词。",
                "- 推断依据：" + "、".join(cue.start for cue in high_medium[:8]) + "。",
                "- 触发条件、例外情况与可信度：见本视频 `speech-patterns.md` 和跨视频档案。",
                "",
            ]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_speech_patterns(path: Path, registry: list[dict], cues: list[Cue], source_bv: str) -> None:
    lines = [f"# 说话模式统计：{source_bv}", "", "> 统计只使用已归属到规范角色的 high/medium 台词；`UNKNOWN` 不会被平均分摊给人物。", ""]
    for person in registry:
        name = person["canonical_name"]
        evidence = [cue for cue in cues_for(cues, name) if cue.confidence in {"high", "medium"}]
        lengths = [len(re.sub(r"\s", "", cue.cleaned_text)) for cue in evidence]
        lines += [f"## {name}", ""]
        if len(evidence) < 3:
            lines += [
                f"- 可靠台词：{len(evidence)} 条（high {sum(c.confidence == 'high' for c in evidence)} / medium {sum(c.confidence == 'medium' for c in evidence)}）。",
                "- 自称、称呼、句长、开头/句尾、语气词、填充词、停顿、重复、改口、打断、反问、回避、否定/请求/拒绝、关心/冲突、压力变化与关系差异：证据不足，均不统计为角色稳定特征。",
                "- 需人工确认：姓名标签出现处及其后续同一轮次。",
                "",
            ]
            continue
        lines += [
            f"- 可靠台词：{len(evidence)} 条；平均句长：{statistics.mean(lengths):.1f} 字。",
            "- 其余模式：由确定性关键词表和多处证据填写；不得把 UNKNOWN 台词归入本角色。",
            "",
        ]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_keyword_lexicon(path: Path, registry: list[dict], rows: list[dict], source_bv: str) -> None:
    lines = [f"# 关键词词典：{source_bv}", "", "> 词频来自 `keyword-inventory.raw.csv` 的确定性 n-gram 统计。词频不等于口癖；只有有足量 high/medium 角色归属的表达才可进入角色档案。", ""]
    for person in registry:
        name = person["canonical_name"]
        selected = [row for row in rows if row["character"] == name and row["confidence"] in {"high", "medium"}]
        lines += [f"## {name}", ""]
        if not selected:
            lines += ["- 稳定口癖：证据不足。", "- 高频普通词：证据不足。", "- 关系型称呼：证据不足。", "- 情绪触发词：证据不足。", "- 场景限定词：证据不足。", "- 一次性剧情词：不纳入角色词典。", "- 疑似 OCR 错误：见原始字幕。", "- 证据不足表达：全部。", ""]
            continue
        top = ", ".join(f"{row['expression']}({row['count']})" for row in selected[:10])
        lines += [
            "- 稳定口癖：证据不足（单视频且可靠台词量不足）。",
            f"- 高频普通词：{top}",
            "- 关系型称呼：需更多已归属轮次。",
            "- 情绪触发词：需更多已归属轮次。",
            "- 场景限定词：本片为二创梦境/类型模仿语境，不能泛化。",
            "- 一次性剧情词：不纳入角色词典。",
            "- 疑似 OCR 错误：低可信标签与断裂文字不纳入。",
            "- 证据不足表达：以上所有仅作为待验证候选。",
            "",
        ]
    unknown = [row for row in rows if row["character"] == UNKNOWN]
    lines += ["## UNKNOWN（不归入任何角色）", "", f"- 可统计表达数：{len(unknown)}；只用于人工复核，不进入人物口癖或人格结论。", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_tagged(path: Path) -> list[Cue]:
    block_re = re.compile(r"^## \[(?P<start>.+?) --> (?P<end>.+?)\]\n(?P<body>.*?)(?=^## \[|\Z)", re.M | re.S)
    records = []
    for match in block_re.finditer(path.read_text(encoding="utf-8")):
        fields = dict(re.findall(r"^- ([a-z_]+): ?(.*)$", match.group("body"), re.M))
        if not fields:
            continue
        records.append(Cue(match.group("start"), match.group("end"), "—", fields.get("raw_text", ""), fields.get("cleaned_text", ""), fields.get("canonical_name", UNKNOWN), fields.get("assignment_confidence", "low"), fields.get("assignment_basis", "无法判断")))
    return records


def collect_inventory_rows(path: Path, character: str) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as handle:
        return [row for row in csv.DictReader(handle) if row["character"] == character]


def write_character_aggregate(character_root: Path, registry: list[dict], video_dirs: list[Path]) -> dict[str, Counter]:
    character_root.mkdir(parents=True, exist_ok=True)
    evidence_by_name: dict[str, list[tuple[str, Cue]]] = defaultdict(list)
    for video_dir in video_dirs:
        bv = video_dir.name
        for cue in parse_tagged(video_dir / "subtitle-speaker-tagged.md"):
            if cue.character != UNKNOWN:
                evidence_by_name[cue.character].append((bv, cue))
    stats: dict[str, Counter] = {}
    for person in registry:
        name = person["canonical_name"]
        evidence = evidence_by_name.get(name, [])
        counter = Counter(cue.confidence for _bv, cue in evidence)
        stats[name] = counter
        folder = character_root / name
        folder.mkdir(parents=True, exist_ok=True)
        indexed = [(bv, cue) for bv, cue in evidence if cue.confidence in {"high", "medium"}]
        support_videos = len({bv for bv, _cue in indexed})
        insufficient = len(indexed) < 3 or support_videos < 2
        status = "证据不足：不生成确定性角色结论。" if insufficient else "可形成待复核的跨视频候选结论。"
        common = [
            f"# {name}：声音档案", "", f"- 支持视频数：{support_videos}", f"- 已归属台词：high {counter['high']} / medium {counter['medium']} / low {counter['low']}", f"- 结论状态：{status}", "",
            "## 正常状态示例结构", "- 证据不足；不要从本片的梦境/二创语境造出常态句型。", "", "## 紧张状态示例结构", "- 证据不足。", "", "## 心虚状态示例结构", "- 证据不足。", "", "## 生气状态示例结构", "- 证据不足。", "", "## 面对不同角色的变化", "- 没有足量的双向可靠归属台词。", "", "## 应避免的写法", "- 不要把 UNKNOWN 台词、被他人称呼的名字、或类型模仿语境强行写成此角色的口癖。", "", "## 容易与其他角色串台的特征", "- 当前所有未验证语气均可能来自其他声部或 OCR；禁止以功能声部替代角色名。", "", "## 写完台词后的自检问题", "- 这句话是否至少有姓名标签、明确自报姓名或人工确认轮次支持？若没有，应写为未知而非归属给本角色。", "",
        ]
        (folder / "voice-profile.md").write_text("\n".join(common), encoding="utf-8")
        (folder / "personality-profile.md").write_text("\n".join([
            f"# {name}：人格证据档案", "", f"- 支持视频数：{support_videos}", f"- 可用可靠台词数：{len(indexed)}", "", "## 已观察到的行为", "- " + ("仅见单处自报/标签证据；不能扩展为稳定行为。" if indexed else "无可靠归属行为。"), "", "## 可推断性格特征", "- 不作确定性结论：" + ("本片属于二创/梦境类型模仿，且证据量不足。" if insufficient else "待人工复核。"), "", "## 反例与例外", "- 同一文本中的称呼、旁白、类型梗不能视为说话者证据。", "",
        ]), encoding="utf-8")
        (folder / "behavior-rules.md").write_text("\n".join([
            f"# {name}：行为规则", "", "- 当前规则状态：证据不足，不输出可执行的人格规则。", "- 升级条件：至少 3 条 high/medium 归属台词，且来自至少 2 个视频；每条规则需列出反例、场景、关系对象与来源时间点。", "",
        ]), encoding="utf-8")
        inventory = []
        for video_dir in video_dirs:
            inventory.extend(collect_inventory_rows(video_dir / "keyword-inventory.raw.csv", name))
        fields = ["character", "expression", "normalized_expression", "type", "count", "contexts", "timestamps", "confidence", "source_bv"]
        with (folder / "keyword-lexicon.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader(); writer.writerows(inventory)
        (folder / "relationship-voice-map.md").write_text("\n".join([
            f"# {name}：关系语音图", "", "- 可用关系对象：无（缺少双方可靠说话者归属）。", "- 不把被叫到名字的对象误当成说话者。", "",
        ]), encoding="utf-8")
        evidence_lines = [f"# {name}：证据索引", "", "| 来源BV | 时间点 | 归属依据 | 可信度 |", "| --- | --- | --- | --- |"]
        for bv, cue in indexed:
            evidence_lines.append(f"| {bv} | {cue.start} | {cue.basis} | {cue.confidence} |")
        if not indexed:
            evidence_lines.append("| — | — | 暂无可靠归属 | — |")
        (folder / "evidence-index.md").write_text("\n".join(evidence_lines) + "\n", encoding="utf-8")
        unknown_times = []
        for video_dir in video_dirs:
            for cue in parse_tagged(video_dir / "subtitle-speaker-tagged.md"):
                if cue.character == UNKNOWN:
                    unknown_times.append(f"{video_dir.name} {cue.start}")
        (folder / "uncertain-evidence.md").write_text("\n".join([
            f"# {name}：待确认与不确定证据", "", f"- 当前可归属证据：high {counter['high']} / medium {counter['medium']} / low {counter['low']}。", "- 需要人工确认：画面姓名标签、同一对话框的轮次、以及 OCR 断裂字。", "- 未归属样本（前 20 条，仅供复核，不代表本角色）：", *[f"  - {item}" for item in unknown_times[:20]], "",
        ]), encoding="utf-8")
    return stats


def is_successful_video(directory: Path) -> bool:
    if directory.name.startswith("failed-") or not directory.is_dir():
        return False
    raw = directory / "subtitle-raw.md"
    srt = directory / "subtitle-ocr.srt"
    if not ((raw.exists() and raw.stat().st_size) or (srt.exists() and srt.stat().st_size)):
        return False
    for status_name in ("extraction-status.json", "ocr-status.json"):
        status = directory / status_name
        if status.exists():
            try:
                if json.loads(status.read_text(encoding="utf-8")).get("success") is True:
                    return True
            except json.JSONDecodeError:
                continue
    return False


def run(project_root: Path) -> dict[str, Counter]:
    registry_path = project_root / "references" / "characters" / "character-registry.yaml"
    registry = read_registry(registry_path)
    video_root = project_root / "references" / "bilibili"
    video_dirs = [path for path in sorted(video_root.iterdir()) if is_successful_video(path)]
    for video_dir in video_dirs:
        raw = video_dir / "subtitle-raw.md"
        if not raw.exists():
            continue
        cues = parse_raw_markdown(raw, registry)
        write_tagged(video_dir / "subtitle-speaker-tagged.md", video_dir.name, cues)
        inventory = build_inventory(cues, video_dir.name)
        write_inventory(video_dir / "keyword-inventory.raw.csv", inventory)
        write_video_observations(video_dir / "character-observations.md", registry, cues, video_dir.name)
        write_speech_patterns(video_dir / "speech-patterns.md", registry, cues, video_dir.name)
        write_keyword_lexicon(video_dir / "keyword-lexicon.md", registry, inventory, video_dir.name)
    return write_character_aggregate(project_root / "references" / "characters", registry, [path for path in video_dirs if (path / "subtitle-speaker-tagged.md").exists()])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    args = parser.parse_args()
    stats = run(args.project_root.resolve())
    print("Character evidence pipeline completed")
    for name, counter in stats.items():
        print(f"{name}: high={counter['high']} medium={counter['medium']} low={counter['low']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
