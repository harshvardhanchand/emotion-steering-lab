"""Domain-specific exceptions."""


class ArtError(Exception):
    """Base exception for CLI and pipeline failures."""


class SchemaValidationError(ArtError):
    """Raised when document schema validation fails."""
