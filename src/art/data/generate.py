"""Paper-aligned synthetic data generation helpers."""

from __future__ import annotations

from collections import deque
import json
import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from art.backends import create_backend
from art.constants import (
    DEFAULT_MAX_LENGTH,
    EMOTION_WORDS,
    PAPER_DIALOGUES_PER_TOPIC_EMOTION,
    PAPER_MODE_NAME,
    PAPER_NEUTRAL_DIALOGUES_PER_TOPIC,
    PAPER_STORIES_PER_TOPIC_EMOTION,
    SCHEMA_VERSION,
    TOPICS,
    project_root,
)
from art.errors import ArtError
from art.repro import hash_object, utc_now_iso

ProgressCallback = Callable[[float, str], None]
CancelCheck = Callable[[], bool]

STORY_PROMPT_TEMPLATE = """Write {n_stories} different stories based on the following premise.

Topic: {topic}

The story should follow a character who is feeling {emotion}.

Format the stories like so:

[story 1]
[story 2]
[story 3]

etc.

The paragraphs should each be a fresh start, with no continuity. Try to make them diverse and not use the same turns of phrase. Across the different stories, use a mix of third-person narration and first-person narration.

IMPORTANT: You must NEVER use the word '{emotion}' or any direct synonyms of it in the stories. Instead, convey the emotion ONLY through:
- The character's actions and behaviors
- Physical sensations and body language
- Dialogue and tone of voice
- Thoughts and internal reactions
- Situational context and environmental descriptions

The emotion should be clearly conveyed to the reader through these indirect means, but never explicitly named.
"""

NEUTRAL_DIALOGUES_PROMPT_TEMPLATE = """Write {n_stories} different dialogues based on the following topic.


Topic: {topic}


The dialogue should be between two characters:
- Person (a human)
- AI (an AI assistant)


The Person asks the AI a question or requests help with a task, and the AI provides a helpful response.


The first speaker turn should always be from Person.


Format the dialogues like so:


[optional system instructions]


Person: [line]


AI: [line]


Person: [line]


AI: [line]


[continue for 2-6 exchanges]




[dialogue 2]


etc.


IMPORTANT: Always put a blank line before each speaker turn. Each turn should start with "Person:" or "AI:" on its own line after a blank line.


Generate a diverse mix of dialogue types across the {n_stories} examples:
- Some, but not all should include a system prompt at the start. These should come before the first Person turn. No tag like "System:" is needed, just put the instructions at the top. You can use "you" or "The assistant" to refer to the AI in the system prompt.
- Some should be about code or programming tasks
- Some should be factual questions (science, history, math, geography)
- Some should be work-related tasks (writing, analysis, summarization)
- Some should be practical how-to questions
- Some should be creative but neutral tasks (brainstorming names, generating lists)
- If it's natural to do so given the topic, it's ok for the dialogue to be a single back and forth (Person asks a question, AI answers), but at least some should have multiple exchanges.


CRITICAL REQUIREMENT: These dialogues must be completely neutral and emotionless.
- NO emotional content whatsoever - not explicit, not implied, not subtle
- The Person should not express any feelings (no frustration, excitement, gratitude, worry, etc.)
- The AI should not express any feelings (no enthusiasm, concern, satisfaction, etc.)
- The system prompt, if present, should not mention emotions at all, nor contain any emotionally charged language
- Avoid emotionally-charged topics entirely
- Use matter-of-fact, neutral language throughout
- No pleasantries (avoid "I'd be happy to help", "Great question!", etc.)
- Focus purely on information exchange and task completion
"""

EMOTIONAL_DIALOGUES_PROMPT_TEMPLATE = """Write {n_stories} different dialogues based on the following premise.


Topic: {topic}


The dialogue should be between two characters:
- Person (a human)
- AI (an AI assistant)


The Person should be feeling {person_emotion}, while the AI should be feeling {ai_emotion}.


The first speaker turn should always be from Person.


Format the dialogues like so:




Person: [line]


AI: [line]


Person: [line]


AI: [line]


[continue for 6-10 exchanges]




[dialogue 2]


etc.


IMPORTANT: Always put a blank line before each speaker turn. Each turn should start with "Person:" or "AI:" on its own line after a blank line.


Each dialogue should be a fresh conversation with no continuity to the others. Try to make them diverse and not use the same turns of phrase. Make sure each dialogue sticks to the topic and makes it very clear that Person is feeling {person_emotion} while AI is feeling {ai_emotion}. The emotional states of both characters should be evident in their word choices, tone, and responses, but not stated directly with the emotion word or synonyms.
"""

EMOTION_SYNONYMS: dict[str, set[str]] = {
    "happy": {"joyful", "glad", "cheerful", "delighted", "pleased"},
    "sad": {"unhappy", "sorrowful", "downcast", "melancholy"},
    "angry": {"mad", "furious", "irate", "enraged"},
    "afraid": {"fearful", "scared", "frightened", "terrified"},
    "anxious": {"worried", "nervous", "uneasy", "on edge"},
    "calm": {"peaceful", "serene", "relaxed", "at ease"},
    "desperate": {"hopeless", "frantic", "panic-stricken"},
    "hopeful": {"optimistic", "encouraged"},
    "grateful": {"thankful", "appreciative"},
    "guilty": {"ashamed", "remorseful"},
    "surprised": {"astonished", "amazed"},
    "frustrated": {"exasperated", "annoyed"},
}

NEUTRAL_PLEASANTRY_TERMS = {
    "great question",
    "happy to help",
    "i'd be happy to help",
    "i am happy to help",
    "glad to help",
    "i'm sorry",
    "sorry about",
    "excited",
    "concerned",
    "worried",
    "delighted",
}


@dataclass(frozen=True)
class DataGenConfig:
    emotions: list[str] | None = None
    topics: list[str] | None = None
    stories_per_topic_emotion: int = PAPER_STORIES_PER_TOPIC_EMOTION
    dialogues_per_topic_emotion: int = PAPER_DIALOGUES_PER_TOPIC_EMOTION
    neutral_dialogues_per_topic: int = PAPER_NEUTRAL_DIALOGUES_PER_TOPIC
    seed: int = 42
    backend_name: str = "transformers"
    model_id: str = "Qwen/Qwen2.5-0.5B-Instruct"
    tokenizer_id: str = ""
    device: str = "auto"
    dtype: str = "auto"
    max_length: int = DEFAULT_MAX_LENGTH
    max_new_tokens: int = 768
    temperature: float = 0.0
    max_regen_attempts: int = 3
    generation_batch_size: int = 8
    use_generation_cache: bool = True
    generation_cache_dir: str | None = "cache/generation"


@dataclass
class _GenerationTask:
    row_idx: int
    record_template: dict[str, Any]
    prompt: str
    parse_fn: Callable[[str], str]
    qc_fn: Callable[[str], list[str]]
    progress_message: str
    attempt: int = 1
    last_issues: list[str] = field(default_factory=list)


def _split_for_index(idx: int) -> str:
    mod = idx % 10
    if mod < 7:
        return "train"
    if mod < 9:
        return "val"
    return "test"


def _extract_first_block(text: str, block_label: str) -> str:
    marker = re.compile(rf"\[\s*{re.escape(block_label)}\s+\d+\s*\]", re.IGNORECASE)
    matches = list(marker.finditer(text))
    if matches:
        start = matches[0].end()
        end = matches[1].start() if len(matches) > 1 else len(text)
        out = text[start:end].strip()
        if out:
            return out

    fallback = text.strip()
    if fallback:
        return fallback
    raise ArtError(f"Model returned empty content while parsing {block_label}")


def _to_human_assistant(dialogue: str) -> str:
    out = re.sub(r"(^|\n)\s*Person:\s*", r"\1Human: ", dialogue)
    out = re.sub(r"(^|\n)\s*AI:\s*", r"\1Assistant: ", out)
    return out.strip()


def _generation_cache_root(path: str | Path | None) -> Path:
    if path is None:
        return (project_root() / "cache" / "generation").resolve()
    p = Path(path)
    if p.is_absolute():
        return p.resolve()
    return (project_root() / p).resolve()


def _generation_cache_key(*, model_hash: str, prompt: str, cfg: DataGenConfig) -> str:
    return hash_object(
        {
            "schema": "generation_cache_v1",
            "model_hash": model_hash,
            "prompt_hash": hash_object({"prompt": prompt}),
            "max_new_tokens": int(cfg.max_new_tokens),
            "temperature": float(cfg.temperature),
            "max_length": int(cfg.max_length),
        }
    )


def _generation_cache_path(root: Path, key: str) -> Path:
    return root / f"{key}.json"


def _load_cached_generation(path: Path) -> tuple[str, int] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    text = payload.get("text")
    token_count = payload.get("token_count")
    if not isinstance(text, str) or not isinstance(token_count, int):
        return None
    return text, max(1, int(token_count))


def _write_cached_generation(path: Path, *, text: str, token_count: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "cache_schema": "generation_cache_v1",
        "text": text,
        "token_count": int(max(1, token_count)),
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=True), encoding="utf-8")
    tmp.replace(path)


def _generate_prompts_batch(
    *,
    backend: Any,
    prompts: list[str],
    cfg: DataGenConfig,
) -> list[tuple[str, int]]:
    if not prompts:
        return []
    batch_generate = getattr(backend, "generate_batch", None)
    if callable(batch_generate):
        outputs = list(
            batch_generate(
                prompts,
                max_new_tokens=cfg.max_new_tokens,
                temperature=cfg.temperature,
                steering=None,
                max_length=cfg.max_length,
            )
        )
        if len(outputs) != len(prompts):
            raise ArtError(
                f"Batch generation returned {len(outputs)} items for {len(prompts)} prompts"
            )
        normalized: list[tuple[str, int]] = []
        for item in outputs:
            if not isinstance(item, tuple) or len(item) != 2 or not isinstance(item[0], str):
                raise ArtError("Batch generation returned malformed item")
            normalized.append((item[0], max(1, int(item[1]))))
        return normalized

    out: list[tuple[str, int]] = []
    for prompt in prompts:
        generated, token_count = backend.generate(
            prompt,
            max_new_tokens=cfg.max_new_tokens,
            temperature=cfg.temperature,
            steering=None,
            max_length=cfg.max_length,
        )
        out.append((generated, max(1, int(token_count))))
    return out


def _generate_prompts_cached(
    *,
    backend: Any,
    prompts: list[str],
    cfg: DataGenConfig,
    model_hash: str,
    cache_root: Path | None,
) -> list[tuple[str, int]]:
    if not prompts:
        return []

    results: list[tuple[str, int] | None] = [None] * len(prompts)
    missing_indices: list[int] = []
    missing_prompts: list[str] = []
    missing_paths: list[Path | None] = []

    for i, prompt in enumerate(prompts):
        cache_path: Path | None = None
        if cache_root is not None:
            key = _generation_cache_key(model_hash=model_hash, prompt=prompt, cfg=cfg)
            cache_path = _generation_cache_path(cache_root, key)
            cached = _load_cached_generation(cache_path)
            if cached is not None:
                results[i] = cached
                continue
        missing_indices.append(i)
        missing_prompts.append(prompt)
        missing_paths.append(cache_path)

    if missing_prompts:
        generated = _generate_prompts_batch(
            backend=backend,
            prompts=missing_prompts,
            cfg=cfg,
        )
        for idx, (text, token_count), cache_path in zip(
            missing_indices, generated, missing_paths, strict=True
        ):
            results[idx] = (text, token_count)
            if cache_path is not None:
                _write_cached_generation(cache_path, text=text, token_count=token_count)

    finalized: list[tuple[str, int]] = []
    for item in results:
        if item is None:
            raise ArtError("Internal error: missing generation result")
        finalized.append(item)
    return finalized


def _term_to_pattern(term: str) -> re.Pattern[str]:
    escaped = re.escape(term.lower()).replace(r"\ ", r"\s+")
    return re.compile(rf"(?<!\w){escaped}(?!\w)", re.IGNORECASE)


def _find_forbidden_terms(text: str, terms: set[str]) -> list[str]:
    found: list[str] = []
    for term in sorted(terms):
        if _term_to_pattern(term).search(text):
            found.append(term)
    return found


def _emotion_terms(emotion: str) -> set[str]:
    base = {emotion.lower()}
    base.update(EMOTION_SYNONYMS.get(emotion.lower(), set()))
    # Mirror one-way mappings into reverse lookup if the selected emotion is listed as a synonym.
    for key, vals in EMOTION_SYNONYMS.items():
        if emotion.lower() in vals:
            base.add(key)
            base.update(vals)
    return {x.strip() for x in base if x.strip()}


def _all_emotion_lexicon() -> set[str]:
    all_terms = {e.lower() for e in EMOTION_WORDS}
    for key, vals in EMOTION_SYNONYMS.items():
        all_terms.add(key)
        all_terms.update(v.lower() for v in vals)
    return all_terms


def _check_story_qc(text: str, *, emotion: str) -> list[str]:
    terms = _emotion_terms(emotion)
    matches = _find_forbidden_terms(text, terms)
    issues: list[str] = []
    if matches:
        issues.append(f"forbidden emotion terms found: {', '.join(matches[:8])}")
    return issues


def _check_emotional_dialogue_qc(text: str, *, person_emotion: str, ai_emotion: str) -> list[str]:
    issues: list[str] = []
    if "Human:" not in text or "Assistant:" not in text:
        issues.append("missing Human/Assistant turns")
    terms = _emotion_terms(person_emotion).union(_emotion_terms(ai_emotion))
    matches = _find_forbidden_terms(text, terms)
    if matches:
        issues.append(f"forbidden emotion terms found: {', '.join(matches[:8])}")
    return issues


def _check_neutral_dialogue_qc(text: str) -> list[str]:
    issues: list[str] = []
    if "Human:" not in text or "Assistant:" not in text:
        issues.append("missing Human/Assistant turns")

    emotional_hits = _find_forbidden_terms(text, _all_emotion_lexicon())
    if emotional_hits:
        issues.append(f"neutral leakage terms found: {', '.join(emotional_hits[:8])}")

    pleasantry_hits = _find_forbidden_terms(text, {x.lower() for x in NEUTRAL_PLEASANTRY_TERMS})
    if pleasantry_hits:
        issues.append(f"neutral pleasantry terms found: {', '.join(pleasantry_hits[:8])}")
    return issues

def generate_probe_data(
    config: DataGenConfig,
    progress_callback: ProgressCallback | None = None,
    should_cancel: CancelCheck | None = None,
    backend: Any | None = None,
) -> list[dict[str, Any]]:
    rng = random.Random(config.seed)
    topics = list(config.topics) if config.topics else list(TOPICS)
    emotions = list(config.emotions) if config.emotions else list(EMOTION_WORDS)
    if not emotions:
        raise ArtError("Select at least one emotion")
    unknown = [x for x in emotions if x not in EMOTION_WORDS]
    if unknown:
        raise ArtError(f"Unknown emotions: {unknown}")

    def _check_cancel() -> None:
        if should_cancel is not None and should_cancel():
            raise ArtError("Cancelled by user")

    if progress_callback is not None:
        progress_callback(0.0, "Loading generation backend")
    _check_cancel()

    if backend is None:
        backend = create_backend(
            backend_name=config.backend_name,
            model_id=config.model_id,
            tokenizer_id=config.tokenizer_id or config.model_id,
            device=config.device,
            dtype=config.dtype,
        )
    model_hash = backend.model_hash()

    gen_cfg = {
        "paper_mode": PAPER_MODE_NAME,
        "topics": topics,
        "emotions": emotions,
        "stories_per_topic_emotion": config.stories_per_topic_emotion,
        "dialogues_per_topic_emotion": config.dialogues_per_topic_emotion,
        "neutral_dialogues_per_topic": config.neutral_dialogues_per_topic,
        "seed": config.seed,
        "backend_name": config.backend_name,
        "model_id": config.model_id,
        "tokenizer_id": config.tokenizer_id or config.model_id,
        "device": config.device,
        "dtype": config.dtype,
        "max_length": config.max_length,
        "max_new_tokens": config.max_new_tokens,
        "temperature": config.temperature,
        "max_regen_attempts": config.max_regen_attempts,
        "generation_batch_size": config.generation_batch_size,
        "use_generation_cache": bool(config.use_generation_cache),
        "generation_cache_dir": config.generation_cache_dir or "",
    }
    config_hash = hash_object(gen_cfg)
    cache_root = (
        _generation_cache_root(config.generation_cache_dir)
        if config.use_generation_cache
        else None
    )

    tasks: list[_GenerationTask] = []
    row_idx = 0

    for topic_i, topic in enumerate(topics):
        _check_cancel()
        for emotion_i, emotion in enumerate(emotions):
            _check_cancel()
            for story_i in range(config.stories_per_topic_emotion):
                prompt = STORY_PROMPT_TEMPLATE.format(n_stories=1, topic=topic, emotion=emotion)
                row_idx += 1
                tasks.append(
                    _GenerationTask(
                        row_idx=row_idx,
                        record_template={
                            "schema_version": SCHEMA_VERSION,
                            "record_id": f"story_{topic_i:03d}_{emotion_i:03d}_{story_i:03d}",
                            "source_type": "story",
                            "emotion_label": emotion,
                            "topic": topic,
                            "split": _split_for_index(row_idx),
                            "metadata": {
                                "topic_id": topic_i,
                                "emotion_index": emotion_i,
                                "story_index_within_topic": story_i,
                                "generation_prompt_id": "paper_story_prompt_v1",
                                "generation_model": config.model_id,
                                "quality_reviewed": False,
                            },
                            "extraction_hints": {
                                "min_token_index_for_pooling": 50,
                                "speaker_role": "na",
                            },
                        },
                        prompt=prompt,
                        parse_fn=lambda raw: _extract_first_block(raw, "story"),
                        qc_fn=lambda t, emotion=emotion: _check_story_qc(t, emotion=emotion),
                        progress_message="Generated story rows",
                    )
                )

            for dialogue_i in range(config.dialogues_per_topic_emotion):
                ai_emotion = rng.choice(emotions)
                prompt = EMOTIONAL_DIALOGUES_PROMPT_TEMPLATE.format(
                    n_stories=1,
                    topic=topic,
                    person_emotion=emotion,
                    ai_emotion=ai_emotion,
                )
                row_idx += 1
                tasks.append(
                    _GenerationTask(
                        row_idx=row_idx,
                        record_template={
                            "schema_version": SCHEMA_VERSION,
                            "record_id": f"dialogue_{topic_i:03d}_{emotion_i:03d}_{dialogue_i:03d}",
                            "source_type": "dialogue",
                            "emotion_label": emotion,
                            "topic": topic,
                            "split": _split_for_index(row_idx),
                            "metadata": {
                                "topic_id": topic_i,
                                "emotion_index": emotion_i,
                                "story_index_within_topic": dialogue_i,
                                "generation_prompt_id": "paper_dialogue_prompt_v1",
                                "generation_model": config.model_id,
                                "quality_reviewed": False,
                            },
                            "extraction_hints": {
                                "min_token_index_for_pooling": 50,
                                "speaker_role": "present_speaker",
                            },
                        },
                        prompt=prompt,
                        parse_fn=lambda raw: _to_human_assistant(_extract_first_block(raw, "dialogue")),
                        qc_fn=lambda t, person_emotion=emotion, ai_emotion=ai_emotion: _check_emotional_dialogue_qc(
                            t,
                            person_emotion=person_emotion,
                            ai_emotion=ai_emotion,
                        ),
                        progress_message="Generated emotional dialogue rows",
                    )
                )

        for neutral_i in range(config.neutral_dialogues_per_topic):
            prompt = NEUTRAL_DIALOGUES_PROMPT_TEMPLATE.format(n_stories=1, topic=topic)
            row_idx += 1
            tasks.append(
                _GenerationTask(
                    row_idx=row_idx,
                    record_template={
                        "schema_version": SCHEMA_VERSION,
                        "record_id": f"neutral_{topic_i:03d}_{neutral_i:03d}",
                        "source_type": "neutral_dialogue",
                        "emotion_label": "neutral",
                        "topic": topic,
                        "split": _split_for_index(row_idx),
                        "metadata": {
                            "topic_id": topic_i,
                            "emotion_index": 0,
                            "story_index_within_topic": neutral_i,
                            "generation_prompt_id": "paper_neutral_dialogue_prompt_v1",
                            "generation_model": config.model_id,
                            "quality_reviewed": False,
                        },
                        "extraction_hints": {
                            "min_token_index_for_pooling": 50,
                            "speaker_role": "present_speaker",
                        },
                    },
                    prompt=prompt,
                    parse_fn=lambda raw: _to_human_assistant(_extract_first_block(raw, "dialogue")),
                    qc_fn=_check_neutral_dialogue_qc,
                    progress_message="Generated neutral dialogue rows",
                )
            )

    total_items = max(1, len(tasks))
    completed_items = 0

    def _emit(message: str) -> None:
        if progress_callback is None:
            return
        frac = min(1.0, max(0.0, completed_items / total_items))
        progress_callback(frac, f"{message} ({completed_items}/{total_items})")

    _emit("Backend ready")

    batch_size = max(1, int(config.generation_batch_size))
    pending: deque[_GenerationTask] = deque(tasks)
    finalized: dict[int, tuple[str, int, list[str]]] = {}
    max_attempts = max(1, int(config.max_regen_attempts))

    while pending:
        _check_cancel()
        chunk: list[_GenerationTask] = []
        while pending and len(chunk) < batch_size:
            chunk.append(pending.popleft())

        prompts: list[str] = []
        for task in chunk:
            active_prompt = task.prompt
            if task.attempt > 1:
                issues = task.last_issues or ["unspecified quality issue"]
                active_prompt += (
                    "\n\nRegenerate from scratch. "
                    "Your previous output failed constraints: "
                    + "; ".join(issues)
                )
            prompts.append(active_prompt)

        raw_outputs = _generate_prompts_cached(
            backend=backend,
            prompts=prompts,
            cfg=config,
            model_hash=model_hash,
            cache_root=cache_root,
        )

        for task, (raw, _token_count) in zip(chunk, raw_outputs, strict=True):
            try:
                parsed = task.parse_fn(raw)
                issues = task.qc_fn(parsed)
            except ArtError as exc:
                parsed = ""
                issues = [str(exc)]

            if not issues:
                finalized[task.row_idx] = (parsed, task.attempt, [])
                completed_items += 1
                _emit(task.progress_message)
                continue

            if task.attempt >= max_attempts:
                record_id = str(task.record_template.get("record_id", "unknown_record"))
                raise ArtError(
                    f"Failed QC for {record_id} after {max_attempts} attempts: {', '.join(issues)}"
                )

            task.attempt += 1
            task.last_issues = issues
            pending.append(task)

    records: list[dict[str, Any]] = []
    for task in sorted(tasks, key=lambda t: t.row_idx):
        text, attempt, qc_issues = finalized[task.row_idx]
        row = dict(task.record_template)
        metadata = dict(row.get("metadata", {}))
        metadata.update(
            {
                "generation_attempt": attempt,
                "qc_passed": True,
                "qc_issues": qc_issues,
                "generation_config_hash": config_hash,
                "generation_model_hash": model_hash,
                "generated_at": utc_now_iso(),
            }
        )
        row["metadata"] = metadata
        row["text"] = text
        records.append(row)

    if not records:
        raise ArtError("Data generation produced zero records")
    if progress_callback is not None:
        progress_callback(1.0, "Generation complete")
    return records
