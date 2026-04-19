"""Paper-aligned synthetic data generation helpers."""

from __future__ import annotations

from collections import Counter, deque
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

# Terms that are in the global emotion list but are too ambiguous for strict lexical
# neutrality gating (high false-positive rate in neutral task dialogs).
NEUTRAL_LEAKAGE_EXCLUDED_TERMS = {
    "content",
    "hope",
    "kind",
    "patient",
    "safe",
    "sensitive",
    "sorry",
    "stuck",
}

_HUMAN_LABELS = (
    "person",
    "user",
    "human",
    "questioner",
    "speaker 1",
    "speaker1",
)
_ASSISTANT_LABELS = (
    "ai",
    "assistant",
    "model",
    "agent",
    "speaker 2",
    "speaker2",
)


def _speaker_label_pattern(labels: tuple[str, ...]) -> re.Pattern[str]:
    joined = "|".join(re.escape(x) for x in labels)
    return re.compile(
        rf"(^|\n)\s*(?:[-*]\s*|\d+[.)]\s*)?(?:{joined})\s*(?:[:：]|[-–—])\s*",
        re.IGNORECASE,
    )


_HUMAN_LABEL_PATTERN = _speaker_label_pattern(_HUMAN_LABELS)
_ASSISTANT_LABEL_PATTERN = _speaker_label_pattern(_ASSISTANT_LABELS)


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
    max_drop_pct: float = 0.05
    max_drop_pct_per_source_type: float = 0.20
    max_drop_pct_per_emotion: float = 0.20
    min_required_neutral_rows: int | None = None
    min_required_story_emotion_classes: int = 2
    min_required_split_count_per_class: int = 1
    qc_drop_log_path: str | None = "artifacts/qc_dropped_rows.jsonl"
    qc_drop_summary_path: str | None = "artifacts/qc_drop_summary.jsonl"


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
    out = dialogue.replace("\r\n", "\n").replace("\r", "\n")
    out = _HUMAN_LABEL_PATTERN.sub(r"\1Human: ", out)
    out = _ASSISTANT_LABEL_PATTERN.sub(r"\1Assistant: ", out)
    return out.strip()


def _generation_cache_root(path: str | Path | None) -> Path:
    if path is None:
        return (project_root() / "cache" / "generation").resolve()
    p = Path(path)
    if p.is_absolute():
        return p.resolve()
    return (project_root() / p).resolve()


def _qc_drop_log_path(path: str | Path | None) -> Path | None:
    if path is None:
        return None
    p = Path(path)
    if p.is_absolute():
        return p.resolve()
    return (project_root() / p).resolve()


def _qc_drop_summary_path(path: str | Path | None) -> Path | None:
    if path is None:
        return None
    p = Path(path)
    if p.is_absolute():
        return p.resolve()
    return (project_root() / p).resolve()


def generation_config_payload(
    config: DataGenConfig,
    *,
    topics: list[str],
    emotions: list[str],
) -> dict[str, Any]:
    return {
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
        "max_drop_pct": float(config.max_drop_pct),
        "max_drop_pct_per_source_type": float(config.max_drop_pct_per_source_type),
        "max_drop_pct_per_emotion": float(config.max_drop_pct_per_emotion),
        "min_required_neutral_rows": config.min_required_neutral_rows,
        "min_required_story_emotion_classes": int(config.min_required_story_emotion_classes),
        "min_required_split_count_per_class": int(config.min_required_split_count_per_class),
        "qc_drop_log_path": config.qc_drop_log_path or "",
        "qc_drop_summary_path": config.qc_drop_summary_path or "",
    }


def generation_config_hash(
    config: DataGenConfig,
    *,
    topics: list[str],
    emotions: list[str],
) -> str:
    return hash_object(generation_config_payload(config, topics=topics, emotions=emotions))


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
    cache_allowed: list[bool] | None = None,
) -> list[tuple[str, int]]:
    if not prompts:
        return []
    if cache_allowed is None:
        cache_allowed = [True] * len(prompts)
    if len(cache_allowed) != len(prompts):
        raise ArtError(
            f"cache_allowed length {len(cache_allowed)} does not match prompts length {len(prompts)}"
        )

    results: list[tuple[str, int] | None] = [None] * len(prompts)
    missing_indices: list[int] = []
    missing_prompts: list[str] = []
    missing_paths: list[Path | None] = []

    for i, prompt in enumerate(prompts):
        cache_path: Path | None = None
        if cache_root is not None and bool(cache_allowed[i]):
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
    return {term for term in all_terms if term not in NEUTRAL_LEAKAGE_EXCLUDED_TERMS}


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
    if float(config.max_drop_pct) < 0.0 or float(config.max_drop_pct) > 1.0:
        raise ArtError("max_drop_pct must be between 0.0 and 1.0")
    if float(config.max_drop_pct_per_source_type) < 0.0 or float(config.max_drop_pct_per_source_type) > 1.0:
        raise ArtError("max_drop_pct_per_source_type must be between 0.0 and 1.0")
    if float(config.max_drop_pct_per_emotion) < 0.0 or float(config.max_drop_pct_per_emotion) > 1.0:
        raise ArtError("max_drop_pct_per_emotion must be between 0.0 and 1.0")
    if int(config.min_required_story_emotion_classes) < 1:
        raise ArtError("min_required_story_emotion_classes must be >= 1")
    if config.min_required_neutral_rows is not None and int(config.min_required_neutral_rows) < 0:
        raise ArtError("min_required_neutral_rows must be >= 0")
    if int(config.min_required_split_count_per_class) < 0:
        raise ArtError("min_required_split_count_per_class must be >= 0")

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

    config_hash = generation_config_hash(config, topics=topics, emotions=emotions)
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
    dropped_rows: list[dict[str, Any]] = []
    drop_log_path = _qc_drop_log_path(config.qc_drop_log_path)
    drop_summary_path = _qc_drop_summary_path(config.qc_drop_summary_path)
    max_attempts = max(1, int(config.max_regen_attempts))

    while pending:
        _check_cancel()
        chunk: list[_GenerationTask] = []
        while pending and len(chunk) < batch_size:
            chunk.append(pending.popleft())

        prompts: list[str] = []
        cache_allowed: list[bool] = []
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
            cache_allowed.append(task.attempt <= 1)

        raw_outputs = _generate_prompts_cached(
            backend=backend,
            prompts=prompts,
            cfg=config,
            model_hash=model_hash,
            cache_root=cache_root,
            cache_allowed=cache_allowed,
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
                dropped_rows.append(
                    {
                        "schema_version": SCHEMA_VERSION,
                        "event": "generation_qc_drop_v1",
                        "dropped_at": utc_now_iso(),
                        "record_id": record_id,
                        "row_idx": int(task.row_idx),
                        "source_type": str(task.record_template.get("source_type", "")),
                        "emotion_label": str(task.record_template.get("emotion_label", "")),
                        "topic": str(task.record_template.get("topic", "")),
                        "attempts": int(task.attempt),
                        "issues": [str(x) for x in issues],
                    }
                )
                completed_items += 1
                _emit(f"Dropped row after QC failures: {record_id}")
                continue

            task.attempt += 1
            task.last_issues = issues
            pending.append(task)

    if dropped_rows and drop_log_path is not None:
        drop_log_path.parent.mkdir(parents=True, exist_ok=True)
        with drop_log_path.open("a", encoding="utf-8") as f:
            for row in dropped_rows:
                f.write(json.dumps(row, ensure_ascii=True) + "\n")

    dropped_count = len(dropped_rows)
    dropped_pct = dropped_count / float(total_items)
    threshold_errors: list[str] = []
    if dropped_pct > float(config.max_drop_pct):
        threshold_errors.append(
            f"Dropped {dropped_count}/{total_items} rows ({dropped_pct:.2%}) due to QC failures, "
            f"exceeding max_drop_pct={float(config.max_drop_pct):.2%}"
        )

    planned_by_source = Counter(str(t.record_template.get("source_type", "")) for t in tasks)
    dropped_by_source = Counter(str(r.get("source_type", "")) for r in dropped_rows)
    for source_type, planned in sorted(planned_by_source.items()):
        if int(planned) <= 0:
            continue
        dropped_for_source = int(dropped_by_source.get(source_type, 0))
        source_drop_pct = dropped_for_source / float(planned)
        if source_drop_pct > float(config.max_drop_pct_per_source_type):
            threshold_errors.append(
                f"Dropped {dropped_for_source}/{planned} rows for source_type={source_type} "
                f"({source_drop_pct:.2%}), exceeding max_drop_pct_per_source_type="
                f"{float(config.max_drop_pct_per_source_type):.2%}"
            )

    planned_by_emotion = Counter(str(t.record_template.get("emotion_label", "")) for t in tasks)
    dropped_by_emotion = Counter(str(r.get("emotion_label", "")) for r in dropped_rows)
    for emotion_label, planned in sorted(planned_by_emotion.items()):
        if int(planned) <= 0:
            continue
        dropped_for_emotion = int(dropped_by_emotion.get(emotion_label, 0))
        emotion_drop_pct = dropped_for_emotion / float(planned)
        if emotion_drop_pct > float(config.max_drop_pct_per_emotion):
            threshold_errors.append(
                f"Dropped {dropped_for_emotion}/{planned} rows for emotion_label={emotion_label} "
                f"({emotion_drop_pct:.2%}), exceeding max_drop_pct_per_emotion="
                f"{float(config.max_drop_pct_per_emotion):.2%}"
            )

    records: list[dict[str, Any]] = []
    for task in sorted(tasks, key=lambda t: t.row_idx):
        item = finalized.get(task.row_idx)
        if item is None:
            continue
        text, attempt, qc_issues = item
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

    expected_neutral = len(topics) * int(config.neutral_dialogues_per_topic)
    if config.min_required_neutral_rows is None:
        min_neutral_rows = (
            max(1, int(expected_neutral * (1.0 - float(config.max_drop_pct))))
            if expected_neutral > 0
            else 0
        )
    else:
        min_neutral_rows = int(config.min_required_neutral_rows)
    neutral_count = sum(1 for r in records if str(r.get("source_type")) == "neutral_dialogue")

    story_emotion_classes = sorted(
        {
            str(r.get("emotion_label"))
            for r in records
            if str(r.get("source_type")) == "story" and str(r.get("emotion_label")) != "neutral"
        }
    )
    if len(story_emotion_classes) < int(config.min_required_story_emotion_classes):
        threshold_errors.append(
            "Generated story records contain only "
            f"{len(story_emotion_classes)} distinct non-neutral emotion labels; requires at least "
            f"{int(config.min_required_story_emotion_classes)}"
        )

    if neutral_count < min_neutral_rows:
        threshold_errors.append(
            f"Generated only {neutral_count} neutral_dialogue rows; requires at least {min_neutral_rows}"
        )

    min_split_count = int(config.min_required_split_count_per_class)
    if min_split_count > 0:
        planned_split_counts: dict[tuple[str, str], Counter[str]] = {}
        required_class_keys: list[tuple[str, str]] = [("neutral_dialogue", "neutral")]
        required_class_keys.extend(("story", e) for e in emotions)
        for source_type, emotion_label in required_class_keys:
            planned_split_counts[(source_type, emotion_label)] = Counter(
                str(t.record_template.get("split", ""))
                for t in tasks
                if str(t.record_template.get("source_type", "")) == source_type
                and str(t.record_template.get("emotion_label", "")) == emotion_label
            )
        actual_split_counts: dict[tuple[str, str], Counter[str]] = {}
        for source_type, emotion_label in required_class_keys:
            actual_split_counts[(source_type, emotion_label)] = Counter(
                str(r.get("split", ""))
                for r in records
                if str(r.get("source_type", "")) == source_type
                and str(r.get("emotion_label", "")) == emotion_label
            )
        for class_key, planned_counter in planned_split_counts.items():
            source_type, emotion_label = class_key
            actual_counter = actual_split_counts.get(class_key, Counter())
            for split_name in ("train", "val", "test"):
                planned = int(planned_counter.get(split_name, 0))
                if planned <= 0:
                    continue
                required = min(planned, min_split_count)
                actual = int(actual_counter.get(split_name, 0))
                if actual < required:
                    threshold_errors.append(
                        f"Post-drop split check failed for source_type={source_type}, emotion_label={emotion_label}, "
                        f"split={split_name}: have {actual}, require at least {required}"
                    )

    if not records:
        raise ArtError("Data generation produced zero records")

    dropped_by_issue = Counter()
    for row in dropped_rows:
        issues = row.get("issues", [])
        if isinstance(issues, list):
            for issue in issues:
                dropped_by_issue[str(issue)] += 1

    if drop_summary_path is not None:
        summary = {
            "schema_version": SCHEMA_VERSION,
            "event": "generation_qc_drop_summary_v1",
            "generated_at": utc_now_iso(),
            "total_items": int(total_items),
            "retained_items": int(len(records)),
            "dropped_count": int(dropped_count),
            "dropped_pct": float(dropped_pct),
            "dropped_by_source_type": dict(sorted((k, int(v)) for k, v in dropped_by_source.items())),
            "dropped_by_emotion_label": dict(sorted((k, int(v)) for k, v in dropped_by_emotion.items())),
            "dropped_by_reason": dict(sorted((k, int(v)) for k, v in dropped_by_issue.items())),
            "threshold_errors": [str(x) for x in threshold_errors],
        }
        drop_summary_path.parent.mkdir(parents=True, exist_ok=True)
        with drop_summary_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(summary, ensure_ascii=True) + "\n")

    if threshold_errors:
        raise ArtError(threshold_errors[0])

    if progress_callback is not None:
        progress_callback(1.0, "Generation complete")
    return records
