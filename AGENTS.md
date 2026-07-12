# WebGAL Character-Pack Instructions

## Load only the needed character evidence

Before writing a scene, identify the on-screen speakers and load only:

1. each participating character's .agents/skills/character-*/SKILL.md;
2. that character's references/mygo-character-pack/<角色>/relationship-voice-map.md for every participating pair;
3. the group-dialogue rules below; and
4. the corresponding voice/personality/behavior/keyword/evidence/uncertainty files only when the Skill requires a check.

Do not preload every character profile for a two-person scene. Never load subtitle-raw.md, subtitle-ocr.srt, extraction status JSON, or OCR diagnostics for routine writing.

## Priority order

当前剧情设定 > 角色 SKILL > 关系语气表 > 人物档案 > 单视频证据 > 通用规则 > 原始 OCR

When two levels conflict, follow the higher level and record the lower-level evidence as non-binding context.

## Group-dialogue rules

- Give every participant a different immediate attention target: event, person, object, practical constraint, or unanswered question. Do not assign 催促者/游离者/校正者 as permanent character labels.
- Let a line be incomplete, delayed, sidelong, or interrupted only when current action and relationship justify it; do not make every character cryptic.
- Do not let every participant deliver a complete explanation of their inner state. Put the explanation burden on scene action, timing, and what remains unanswered.
- Change tone through current relationship status, knowledge asymmetry, turn-taking, and scene pressure before reaching for a catchphrase.
- Keep each character inside their knowledge boundary. A name mentioned in dialogue does not establish that person's voice or knowledge.

## Evidence and safety gates

- Do not randomly scatter catchphrases, honorifics, particles, or nicknames.
- Do not treat a single derivative-video gag as stable personality.
- Do not copy source-video dialogue, continuous joke structures, or OCR wording into new scenes.
- When the character pack says evidence is insufficient, do not synthesize a canonical pattern. Use the scene's explicit project characterization or mark the choice for human review.
- Before final output, check each line for speaker cross-talk, unsupported relationship claims, over-explained emotion, and leakage from raw OCR.

## WebGAL scene handoff

Start each request with: current scene facts, speakers, relationship state, knowledge boundary, emotional pressure, and required plot turn. Then request dialogue in WebGAL form only after these fields are set.
