"""Validate documents against JSON schemas."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from jsonschema import Draft202012Validator, FormatChecker

from art.errors import SchemaValidationError
from art.schemas.loader import load_schema


def _format_error(err: Any, schema_filename: str, context: str | None = None) -> str:
    path = ".".join(str(item) for item in err.path)
    where = f" at '{path}'" if path else ""
    prefix = f"[{context}] " if context else ""
    return f"{prefix}{schema_filename}{where}: {err.message}"


def validate_document(
    document: Mapping[str, object],
    schema_filename: str,
    *,
    context: str | None = None,
) -> None:
    validator = Draft202012Validator(load_schema(schema_filename), format_checker=FormatChecker())
    errors = sorted(validator.iter_errors(document), key=lambda e: list(e.path))
    if errors:
        raise SchemaValidationError(_format_error(errors[0], schema_filename, context))


def validate_documents(
    documents: Iterable[Mapping[str, object]],
    schema_filename: str,
    *,
    context_prefix: str | None = None,
) -> None:
    for idx, doc in enumerate(documents):
        context = f"{context_prefix}[{idx}]" if context_prefix else f"row[{idx}]"
        validate_document(doc, schema_filename, context=context)
