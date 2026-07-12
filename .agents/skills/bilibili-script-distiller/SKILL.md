---
name: bilibili-script-distiller
description: Acquire existing subtitles for Bilibili full links, b23.tv short links, BV IDs, or AV IDs either directly when network access works or through the repository's GitHub Actions extractor when it does not; ingest uploaded SRT, ASS, VTT, TXT, MD, and DOCX files; preserve natural spoken irregularities; analyze character voice, relationship-conditioned dialogue, scene rhythm, subtle action/description, and anti-AI mechanisms; produce per-video source cards, speaker-attributed evidence, deterministic keyword inventories, executable writing rules, and cross-video character voice profiles. Use when the user sends Bilibili links or subtitle/text files for dialogue analysis, asks to extract or distill one or more videos into visual-novel writing rules, asks to continue after Action-produced subtitle files arrive, asks to build or verify character voice/personality evidence, or asks to merge/update a multi-video distillation collection.
---

# Bilibili Script Distiller

Build evidence-based visual-novel writing rules from existing Bilibili subtitles or uploaded subtitle/text files. Never invent unavailable video content.

## Accept inputs

- Accept full `bilibili.com/video/...` links, `b23.tv` short links, BV IDs, and AV/av IDs.
- Accept uploaded SRT, ASS, VTT, TXT, MD, and DOCX subtitle or transcript files.
- Prefer inputs in this order: uploaded subtitles; downloadable subtitles exposed by Bilibili/BBDown; audio or video only after explicit user permission.
- Do not download video, install Whisper, run speech recognition, search alternate subtitle sources, install WebGAL, or visit unrelated pages by default. A low-quality temporary video may be downloaded only when the user explicitly enables `enable_hardsub_ocr` and BBDown completed with zero subtitle tracks.

## Load references

- Read `references/distillation-rules.md` before cleaning or analyzing a single source.
- Read `references/source-card-template.md` when creating `source-card.md`.
- Read `references/multi-video-merge-rules.md` only when the user asks to merge/update a collection or supplies multiple similar videos and explicitly requests a merge.
- Read `references/character-name-dictionary.json` when hard-subtitle OCR must identify or normalize speaker labels. Extend aliases instead of hard-coding guessed names in the script.
- Read `PROJECT_ROOT/references/characters/character-registry.yaml` before speaker attribution or character-level aggregation. It is the editable canonical-name, alias, nickname, and OCR-confusion source.

## Use the two-stage architecture

Separate acquisition from distillation while keeping both stages inside this one skill.

### Stage 1: acquire subtitles

1. If `PROJECT_ROOT/references/bilibili/<source>/subtitle-raw.md` already exists, do not re-extract it. Continue to Stage 2.
2. If the current environment can reach Bilibili, direct extraction is allowed using the bundled tool below.
3. If Bilibili/b23.tv access is blocked, do not repeatedly retry and do not ask for a project ZIP. Use `.github/workflows/bilibili-subtitle-extract.yml` through GitHub Actions:
   - Open the repository's **Actions** tab.
   - Run **Extract Bilibili subtitles**.
   - Supply one input per line in `video_urls`.
   - Wait for the workflow to commit only `references/bilibili/` results.
   - If the workflow cannot push, use the named Artifact reported in the run summary and place its `references/bilibili/` contents into the repository.
4. When BBDown completes but no subtitle file exists, treat it as “no subtitle track”, not as proof that the video has no visible subtitles. The Action may automatically download a low-quality temporary video, OCR the configured dialogue/name crops, then write `subtitle-ocr.srt`, `subtitle-raw.md`, and `ocr-status.json`.
5. Do not claim extraction is complete until `subtitle-raw.md` and either a successful `extraction-status.json` or `ocr-status.json` are present in the repository.
5. When the Action result is committed, pull/refresh the current branch before Stage 2. When it is delivered as an Artifact, confirm that its files are present before Stage 2.

The Action resolves b23.tv links externally, processes inputs independently, ranks Chinese human subtitles before Chinese platform/AI subtitles and Japanese subtitles, records all discovered tracks, and continues after individual failures. Hard-subtitle OCR downloads a temporary 360P-preferred/480P-fallback video, samples one frame per second, recognizes the dialogue body separately from left/center speaker tags with RapidOCR ONNX, normalizes reliable names through the editable dictionary, and treats missing labels as narration. It merges near-duplicate frames and deletes temporary video/frames. Up to 20 speaker-change debug bundles are uploaded as an Artifact and are never committed.

### Stage 2: distill available subtitles

For every directory whose `subtitle-raw.md` exists and whose `extraction-status.json` has `success: true`, generate `subtitle-readable.md`, `source-card.md`, and `distilled-writing-rules.md`. Process sources independently. After all requested sources finish, update the multi-video collection only when requested.

## Run the direct extractor when network permits

Treat the repository/workspace root as `PROJECT_ROOT`. The GitHub Actions extractor is `SKILL_ROOT/scripts/github_actions_extract.py`. It requires `BBDown` on `PATH`; the workflow installs BBDown automatically.

Run preflight without pixi:

```bash
command -v BBDown
```

Extract existing subtitles to a temporary directory:

```bash
VIDEO_URLS="<URL_OR_ID>" python3 "$SKILL_ROOT/scripts/github_actions_extract.py" \
  --output-root "$PROJECT_ROOT/references/bilibili" \
  --summary-json "<TEMP_DIR>/extraction-summary.json"
```

Use exit code `0` when at least one input succeeds and `2` when the batch produces no successful subtitles. Read per-video `extraction-status.json` and the batch summary; do not treat logs or warnings as transcript content.

## Process one source

1. Inspect only the submitted link or file and identify its stable source key. For a link, prefer the fetched title; otherwise use the BV/AV ID. Sanitize `/\\:*?\"<>|` and control characters from directory names.
2. For an uploaded file, skip network extraction and parse it locally. Preserve timestamps when present. For DOCX, extract paragraph/table text in reading order; do not alter the uploaded file.
3. For a Bilibili input, choose Stage 1 direct extraction or GitHub Actions based on actual network capability. Do not treat environment blocking as proof that a video has no subtitles.
4. Create/use `PROJECT_ROOT/references/bilibili/<source-key>/`. A GitHub Actions acquisition contains `subtitle-raw.md` and `extraction-status.json`; a failed acquisition may contain only `extraction-status.json`.
5. After successful acquisition, generate:
   - `subtitle-raw.md`
   - `subtitle-readable.md`
   - `source-card.md`
   - `distilled-writing-rules.md`
6. Preserve timecodes and original order in `subtitle-raw.md`. Do not silently correct wording. Mark suspected errors without replacing the original.
7. Clean only according to `distillation-rules.md`; preserve pauses, repetition, fillers, catchphrases, fragments, self-correction, evasions, interruptions, and unanswered questions.
8. Analyze each major speaker separately. Use `说话者A`, `说话者B`, and `未知说话者` when identity is not evidenced. Never guess identities.
9. Convert observations into executable rules with conditions, risks, evidence timestamps, and star confidence. Do not write a generic summary or long plot recap.
10. Do not update the multi-video collection for a single link unless the user explicitly requests a merge/update.

### Stage 2B: attribute speakers and aggregate character evidence

Run `SKILL_ROOT/scripts/character_evidence_pipeline.py --project-root PROJECT_ROOT` after successful source subtitles are available when the user asks for character-level evidence or when this pipeline has been requested for a batch.

1. Scan only successful, non-`failed-` source directories with a nonempty `subtitle-raw.md` or `subtitle-ocr.srt`.
2. Generate `subtitle-speaker-tagged.md`, `character-observations.md`, `speech-patterns.md`, `keyword-inventory.raw.csv`, and `keyword-lexicon.md` separately for each source.
3. Treat a direct reliable name label as `high`; a clear self-identification as `medium`; otherwise use `UNKNOWN` with `low` and `无法判断`. Never assign a speaker from plot familiarity, a name merely being addressed, or an OCR/watermark fragment.
4. Generate the registry-backed `references/characters/<canonical-name>/` profiles only from the per-video structured speaker evidence, not by rereading all raw subtitles. When reliable evidence is sparse, write only uncertainty and required manual checks; do not manufacture personality or stable-voice conclusions.
5. Generate the raw keyword CSV with the deterministic script before interpreting it. Do not present frequency alone as a catchphrase or a personality fact.

## Process a batch

During Stage 1, GitHub Actions processes links sequentially and independently and records each failure. During Stage 2, process every successful source independently before merging. If one source fails, record its reason and continue. Never combine raw subtitles before per-source analysis.

## Merge completed sources

When asked to `合并蒸馏`, `更新总集`, or equivalent:

1. Read only `references/bilibili/*/source-card.md` and `references/bilibili/*/distilled-writing-rules.md` first.
2. Read raw/readable subtitles only for a local evidence check when rules conflict or evidence is insufficient.
3. Create or incrementally update `references/distilled/多视频剧本写作蒸馏总集.md` using `multi-video-merge-rules.md`.
4. Preserve manual edits. Patch affected sections rather than rewriting the whole collection.

## Handle failures

- No downloadable subtitle: report exactly `该视频没有可下载字幕，请上传字幕文件；如需处理音频，请另行明确要求。`
- Current environment blocks Bilibili: route acquisition to `.github/workflows/bilibili-subtitle-extract.yml`; do not present it as a no-subtitle result and do not ask for a project ZIP.
- GitHub Actions push rejected: report the exact Artifact name from the run summary and state that results were not written back automatically.
- Authentication/restricted video: report that BBDown login may be required; do not request or expose cookies in chat.
- Invalid link/ID: identify the rejected input and request a supported Bilibili link, BV ID, AV ID, or uploaded subtitle file.
- Unsupported/invalid file: preserve the file, report the parsing issue, and request SRT, ASS, VTT, TXT, MD, or DOCX.
- Batch failure: record the failed item and continue other items.
- Never fabricate missing dialogue, speakers, timestamps, plot, or evidence.

## Control output and tokens

- Read a source transcript fully only during its first processing pass.
- During merges, prefer source cards and distilled rules; inspect transcript fragments only when necessary.
- Do not paste full subtitles, full source cards, or the full merged collection into chat.
- Report created paths, processed/failed counts, and the next actionable input briefly.
- Stop after setup when no video or subtitle input is supplied.
