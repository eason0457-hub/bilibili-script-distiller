#!/usr/bin/env python3
"""Generate conservative speaker evidence and character writing references.

Named speakers require an exact registered label or explicit self-introduction.
OCR error names, addressee names and plot familiarity never become character facts.
A video is eligible when it has a real, non-empty dialogue subtitle, even if a stale
status JSON is missing or false. Existing manual tagged files are preserved unless
the source subtitle is newer or --force-rebuild-tagged is supplied.
"""
from __future__ import annotations

import argparse, csv, json, re, statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable
import yaml

UNKNOWN = "UNKNOWN"
BLOCKED = {"bilibili", "biibili", "bilbili", "lilibili", "liibili", "旁白", "未知说话者", "unknown", "tez", "灵", "正", "二正ez"}
NON_DIALOGUE = re.compile(r"^[\s♪♫♬🎵🎶【】\[\]（）()<>—–\-_.:：,，]*(?:音乐|纯音乐|music|bgm|soundtrack|音效|无对白)[\s♪♫♬🎵🎶【】\[\]（）()<>—–\-_.:：,，]*$", re.I)
ENTRY = re.compile(r"^\[(?P<s>[^\]]+?)\s*-->\s*(?P<e>[^\]]+?)\]\s*\n(?P<b>.*?)(?=^\[[^\]]+?\s*-->|\Z)", re.M | re.S)
SRT = re.compile(r"(?:^|\n)(?:\d+\s*\n)?(?P<s>\d{1,2}:\d{2}:\d{2}[,.]\d{3})\s*-->\s*(?P<e>\d{1,2}:\d{2}:\d{2}[,.]\d{3})[^\n]*\n(?P<b>.*?)(?=\n\s*\n|\Z)", re.S)
SELF_ID = re.compile(r"(?:我(?:是|叫(?:做)?|名叫)|我的名字是)(?P<n>[\u3400-\u9fffA-Za-z]{1,12})")
PATTERNS = ("不对", "等一下", "就是", "总之", "看来", "我说", "不是", "好吧", "没事", "最后", "为什么", "怎么办", "真的吗", "等等", "所以", "但是", "可是")
FILLERS = ("啊", "嗯", "呃", "诶", "欸", "那个", "就是", "这个", "嘛", "啦", "吧")
SCORE = {"high": 3, "medium": 2, "low": 1}


class Cue:
    def __init__(self, s: str, e: str, label: str, raw: str, text: str, char: str, conf: str, basis: str):
        self.start, self.end, self.label, self.raw = s, e, label, raw
        self.text, self.character, self.confidence, self.basis = text, char, conf, basis


def norm(v: str) -> str:
    return re.sub(r"[\s【】\[\]（）()：:·.、，,!?！？'\"“”]", "", v or "").casefold()


def registry(path: Path) -> list[dict]:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    chars = data.get("characters", [])
    if not isinstance(chars, list) or any(not x.get("canonical_name") for x in chars):
        raise ValueError("character-registry.yaml must contain characters with canonical_name")
    return chars


def lookup(chars: Iterable[dict]) -> dict[str, str]:
    out = {}
    for item in chars:
        canonical = item["canonical_name"]
        for value in [canonical, *item.get("aliases", []), *item.get("nicknames", [])]:
            key = norm(str(value))
            if key and key not in BLOCKED:
                out[key] = canonical
    return out


def clean(v: str) -> str:
    v = re.sub(r"<!--.*?-->|<[^>]+>", "", v, flags=re.S)
    v = v.replace("\r", "").replace("\\N", "\n")
    v = re.sub(r"\s*\n\s*", " ", v)
    return re.sub(r"\s+", " ", v).replace("...", "……").replace("..", "……").strip()


def valid(v: str) -> bool:
    return bool(v and not NON_DIALOGUE.fullmatch(v) and re.search(r"[\u3400-\u9fffA-Za-z]", re.sub(r"[\s♪♫♬🎵🎶]", "", v)))


def split_body(body: str) -> tuple[str, str]:
    lines = [x.strip() for x in body.splitlines() if x.strip() and not x.lstrip().startswith("<!--")]
    if not lines:
        return "", ""
    if lines[0].endswith(("：", ":")) and len(lines[0]) <= 24:
        return lines[0][:-1], "".join(lines[1:])
    m = re.match(r"^([^：:\n]{1,16})[：:](.*)$", lines[0])
    return (m.group(1), "".join([m.group(2), *lines[1:]])) if m else ("", "".join(lines))


def assign(label: str, text: str, names: dict[str, str]) -> tuple[str, str, str]:
    key = norm(label)
    if key and key not in BLOCKED and key in names:
        return names[key], "high", "画面姓名标签与角色表精确匹配"
    for m in SELF_ID.finditer(text):
        key = norm(m.group("n"))
        if key in names:
            return names[key], "medium", "台词中明确自报姓名"
    return UNKNOWN, "low", "无法可靠判断"


def parse_file(path: Path, chars: list[dict]) -> list[Cue]:
    names, out = lookup(chars), []
    raw = path.read_text(encoding="utf-8-sig", errors="replace").replace("\r\n", "\n")
    pattern = ENTRY if path.suffix.lower() == ".md" else SRT
    for m in pattern.finditer(raw):
        label, source = split_body(m.group("b"))
        text = clean(source)
        if not valid(text):
            continue
        char, conf, basis = assign(label, text, names)
        out.append(Cue(m.group("s").replace(",", "."), m.group("e").replace(",", "."), label or "—", source, text, char, conf, basis))
    return out


def source(video: Path, chars: list[dict]) -> tuple[Path | None, list[Cue]]:
    for name in ("subtitle-raw.md", "subtitle-ocr.srt"):
        path = video / name
        if path.exists() and path.stat().st_size:
            cues = parse_file(path, chars)
            if cues:
                return path, cues
    return None, []


def write_tagged(path: Path, bv: str, cues: list[Cue]) -> None:
    counts = Counter((c.character, c.confidence) for c in cues)
    lines = [f"# 说话者归属字幕：{bv}", "", "> 只接受精确姓名标签或明确自报姓名；OCR 错误姓名和被称呼对象不用于自动归属。", "", "## 归属统计", "", "| 规范角色名 | high | medium | low |", "| --- | ---: | ---: | ---: |"]
    for name in sorted({c.character for c in cues}):
        lines.append(f"| {name} | {counts[(name,'high')]} | {counts[(name,'medium')]} | {counts[(name,'low')]} |")
    for c in cues:
        lines += ["", f"## [{c.start} --> {c.end}]", f"- canonical_name: {c.character}", f"- source_label: {c.label}", f"- raw_text: {c.raw}", f"- cleaned_text: {c.text}", f"- assignment_confidence: {c.confidence}", f"- assignment_basis: {c.basis}"]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_tagged(path: Path) -> list[Cue]:
    blocks = re.compile(r"^## \[(?P<s>.+?) --> (?P<e>.+?)\]\n(?P<b>.*?)(?=^## \[|\Z)", re.M | re.S)
    out = []
    for m in blocks.finditer(path.read_text(encoding="utf-8", errors="replace")):
        f = dict(re.findall(r"^- ([a-z_]+): ?(.*)$", m.group("b"), re.M))
        if f:
            out.append(Cue(m.group("s"), m.group("e"), f.get("source_label", "—"), f.get("raw_text", ""), f.get("cleaned_text", ""), f.get("canonical_name", UNKNOWN), f.get("assignment_confidence", "low"), f.get("assignment_basis", "无法可靠判断")))
    return out


def expressions(text: str) -> list[tuple[str, str]]:
    out = [(x, "word") for x in re.findall(r"[A-Za-z]+", text)]
    for run in re.findall(r"[\u3400-\u9fff]+", text):
        for i in range(len(run)):
            for n in range(1, min(4, len(run)-i)+1):
                out.append((run[i:i+n], "word" if n == 1 else f"phrase_{n}char"))
    out += [(x, "fixed_pattern") for x in PATTERNS if x in text]
    out += [(x, "filler") for x in FILLERS if x in text]
    out += [(x, "sentence_end") for x in ("……", "？", "！", "。", "吧", "啊", "呢", "吗") if text.endswith(x)]
    return out


def inventory(cues: list[Cue], bv: str) -> list[dict]:
    rows = {}
    for c in cues:
        for exp, typ in expressions(c.text):
            row = rows.setdefault((c.character, exp, typ), {"character": c.character, "expression": exp, "normalized_expression": norm(exp), "type": typ, "count": 0, "contexts": [], "timestamps": [], "confidence": c.confidence, "source_bv": bv})
            row["count"] += 1
            if c.text not in row["contexts"] and len(row["contexts"]) < 3: row["contexts"].append(c.text[:100])
            if len(row["timestamps"]) < 5: row["timestamps"].append(c.start)
            if SCORE[c.confidence] > SCORE[row["confidence"]]: row["confidence"] = c.confidence
    keep = [r for r in rows.values() if r["character"] != UNKNOWN or r["type"] in {"fixed_pattern", "filler", "sentence_end"} or r["count"] >= 3]
    return sorted(keep, key=lambda r: (r["character"], -r["count"], r["type"], r["expression"]))


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = ["character", "expression", "normalized_expression", "type", "count", "contexts", "timestamps", "confidence", "source_bv"]
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
        for row in rows:
            out = dict(row); out["contexts"] = json.dumps(out["contexts"], ensure_ascii=False); out["timestamps"] = json.dumps(out["timestamps"], ensure_ascii=False); w.writerow(out)


def reliable(cues: list[Cue], name: str) -> list[Cue]:
    return [c for c in cues if c.character == name and c.confidence in {"high", "medium"}]


def top(cues: list[Cue], types: set[str], limit=10) -> list[tuple[str, int]]:
    counter = Counter(exp for c in cues for exp, typ in expressions(c.text) if typ in types)
    return counter.most_common(limit)


def write_video_files(video: Path, chars: list[dict], cues: list[Cue], rows: list[dict]) -> None:
    bv = video.name
    obs = [f"# 单视频人物证据：{bv}", "", "> 每个视频独立分析；UNKNOWN 不分摊给人物。", ""]
    pat = [f"# 说话模式统计：{bv}", "", "> 仅统计 high/medium 归属台词；词频不自动等于口癖。", ""]
    lex = [f"# 关键词词典：{bv}", "", "> 来自确定性 n-gram 统计，单视频结果只作候选。", ""]
    for item in chars:
        name, ev = item["canonical_name"], reliable(cues, item["canonical_name"])
        obs += [f"## {name}", "", f"- 可靠台词数：{len(ev)}。", "- 性格结论：" + ("存在多条可复核证据，但仍需跨视频确认。" if len(ev) >= 3 else "证据不足，不作确定性结论。"), "- 证据：" + ("；".join(f"{c.start}「{c.text[:32]}」" for c in ev[:5]) or "无"), ""]
        lengths = [len(re.sub(r"\s", "", c.text)) for c in ev]
        pat += [f"## {name}", "", f"- 可靠台词：{len(ev)} 条。", f"- 平均句长：{statistics.mean(lengths):.1f} 字。" if lengths else "- 平均句长：证据不足。", "- 固定表达候选：" + ("、".join(f"{x}({n})" for x,n in top(ev,{"fixed_pattern"},8)) or "无"), "- 填充词候选：" + ("、".join(f"{x}({n})" for x,n in top(ev,{"filler"},8)) or "无"), "- 使用边界：需跨视频重复后才能写入稳定角色规则。", ""]
        selected = [r for r in rows if r["character"] == name and r["confidence"] in {"high", "medium"}]
        candidates = [r for r in selected if r["type"] in {"fixed_pattern", "filler", "sentence_end"}][:15]
        lex += [f"## {name}", "", "- 稳定口癖：单视频不能定论。", "- 话语标记候选：" + ("、".join(f"{r['expression']}({r['count']})" for r in candidates) or "证据不足"), "- OCR 错误姓名与低可信标签不进入正式人物档案。", ""]
    lex += ["## UNKNOWN（不归入任何角色）", "", f"- 候选表达数：{sum(r['character']==UNKNOWN for r in rows)}；只用于人工复核。", ""]
    (video/"character-observations.md").write_text("\n".join(obs), encoding="utf-8")
    (video/"speech-patterns.md").write_text("\n".join(pat), encoding="utf-8")
    (video/"keyword-lexicon.md").write_text("\n".join(lex), encoding="utf-8")


def aggregate(root: Path, chars: list[dict], videos: list[Path]) -> dict[str, Counter]:
    evidence = defaultdict(list)
    for v in videos:
        for c in parse_tagged(v/"subtitle-speaker-tagged.md"):
            if c.character != UNKNOWN: evidence[c.character].append((v.name,c))
    stats = {}
    for item in chars:
        name, data = item["canonical_name"], evidence.get(item["canonical_name"], [])
        count = Counter(c.confidence for _,c in data); stats[name] = count
        good = [(bv,c) for bv,c in data if c.confidence in {"high","medium"}]
        support = len({bv for bv,_ in good}); cues = [c for _,c in good]
        folder = root/name; folder.mkdir(parents=True, exist_ok=True)
        status = "可形成跨视频候选，仍需人工复核。" if len(good)>=5 and support>=2 else "证据不足，不生成确定性人格结论。"
        lengths = [len(re.sub(r"\s", "",c.text)) for c in cues]
        (folder/"voice-profile.md").write_text("\n".join([f"# {name}：声音档案","",f"- 支持视频数：{support}",f"- 台词：high {count['high']} / medium {count['medium']} / low {count['low']}",f"- 状态：{status}","", "## 可量化声音特征", f"- 平均句长：{statistics.mean(lengths):.1f} 字。" if lengths else "- 平均句长：证据不足。", "- 固定表达候选："+("、".join(f"{x}({n})" for x,n in top(cues,{"fixed_pattern"})) or "证据不足"), "- 填充词候选："+("、".join(f"{x}({n})" for x,n in top(cues,{"filler"})) or "证据不足"), "", "## 使用边界", "- 不把 UNKNOWN、OCR 错误姓名或单次二创梗写成稳定口癖。", "- 情绪状态和关系差异需人工场景标注后再补充。", ""]), encoding="utf-8")
        (folder/"personality-profile.md").write_text("\n".join([f"# {name}：人格证据档案","",f"- 支持视频数：{support}",f"- 可靠台词数：{len(good)}","", "## 观察证据", "- "+("；".join(f"{bv} {c.start}「{c.text[:32]}」" for bv,c in good[:8]) or "无可靠归属。"),"", "## 结论", f"- {status}", "- 二创、梦境和类型模仿不能直接等同于原作常态。", ""]), encoding="utf-8")
        (folder/"behavior-rules.md").write_text(f"# {name}：行为规则\n\n- 当前状态：{status}\n- 每条规则必须包含触发条件、表现、反例、关系对象和来源时间点。\n", encoding="utf-8")
        inv = [r for v in videos for r in read_rows(v/"keyword-inventory.raw.csv", name)]
        write_csv(folder/"keyword-lexicon.csv", inv)
        (folder/"relationship-voice-map.md").write_text(f"# {name}：关系语音图\n\n- 自动阶段不根据被称呼对象反推关系语气；需双方轮次可靠归属后再建立。\n", encoding="utf-8")
        idx = [f"# {name}：证据索引","","| 来源BV | 时间点 | 文本摘要 | 归属依据 | 可信度 |","| --- | --- | --- | --- | --- |"] + [f"| {bv} | {c.start} | {c.text.replace('|','｜')[:60]} | {c.basis} | {c.confidence} |" for bv,c in good]
        if not good: idx.append("| — | — | — | 暂无可靠归属 | — |")
        (folder/"evidence-index.md").write_text("\n".join(idx)+"\n", encoding="utf-8")
        unknown = [f"{v.name} {c.start}" for v in videos for c in parse_tagged(v/"subtitle-speaker-tagged.md") if c.character==UNKNOWN][:20]
        (folder/"uncertain-evidence.md").write_text("\n".join([f"# {name}：待确认与不确定证据","", "- 需要人工确认姓名标签区域、轮次和 OCR 断裂文本。", *[f"- {x}" for x in unknown], ""]), encoding="utf-8")
    return stats


def read_rows(path: Path, name: str) -> list[dict]:
    if not path.exists(): return []
    with path.open(encoding="utf-8", newline="") as f: return [r for r in csv.DictReader(f) if r.get("character")==name]


def run(project: Path, force=False) -> dict[str, Counter]:
    chars = registry(project/"references/characters/character-registry.yaml")
    video_root = project/"references/bilibili"
    if not video_root.exists(): raise FileNotFoundError(f"video root not found: {video_root}")
    done = []
    for video in sorted(video_root.iterdir()):
        if not video.is_dir() or video.name.startswith("failed-"): continue
        src, original = source(video, chars)
        if not src or not original: continue
        tagged = video/"subtitle-speaker-tagged.md"
        if force or not tagged.exists() or tagged.stat().st_mtime < src.stat().st_mtime:
            write_tagged(tagged, video.name, original); cues = original
        else:
            cues = parse_tagged(tagged) or original
            if not parse_tagged(tagged): write_tagged(tagged, video.name, original)
        rows = inventory(cues, video.name); write_csv(video/"keyword-inventory.raw.csv", rows); write_video_files(video, chars, cues, rows); done.append(video)
    stats = aggregate(project/"references/characters", chars, done)
    summary = project/"references/distilled/character-evidence-summary.md"; summary.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# 人物证据自动蒸馏总览","","> 确定性统计总览，不替代人工人物分析；无法判断时保持 UNKNOWN。","",f"- 已处理视频：{len(done)}","","| 角色 | high | medium | low |","| --- | ---: | ---: | ---: |"] + [f"| {n} | {c['high']} | {c['medium']} | {c['low']} |" for n,c in stats.items()] + ["","## 视频", *[f"- `{v.name}`" for v in done], "", "- 只有真实对白进入统计，音乐、纯音符和无对白内容不算成功。", ""]
    summary.write_text("\n".join(lines), encoding="utf-8")
    return stats


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__); p.add_argument("--project-root", type=Path, default=Path(".")); p.add_argument("--force-rebuild-tagged", action="store_true"); a=p.parse_args()
    stats = run(a.project_root.resolve(), a.force_rebuild_tagged)
    print("Character evidence pipeline completed")
    for name,c in stats.items(): print(f"{name}: high={c['high']} medium={c['medium']} low={c['low']}")
    return 0

if __name__ == "__main__": raise SystemExit(main())
