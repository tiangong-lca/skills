"""Custom exception hierarchy for the workflow."""

from __future__ import annotations


class SpecCodingError(Exception):
    """Base error for the Tiangong spec coding workflow."""


class FlowSearchError(SpecCodingError):
    """Raised when flow search fails."""


class FlowAlignmentError(SpecCodingError):
    """Raised when exchange-to-flow alignment is not possible."""


class ProcessExtractionError(SpecCodingError):
    """Raised when process data cannot be extracted or normalized."""


class TidasValidationError(SpecCodingError):
    """Raised when TIDAS validation fails."""


class ExchangeValidationError(ProcessExtractionError):
    """Raised when Stage 2 exchanges or FlowSearch hints fail strict validation."""

    def __init__(self, process_name: str | None, errors: list[str]) -> None:
        self.process_name = process_name or "Unnamed process"
        self.errors = errors
        message = f"{self.process_name} produced invalid exchanges:\n- " + "\n- ".join(errors)
        super().__init__(message)

    def retry_feedback(self) -> str:
        bullet_list = "\n".join(f"- {issue}" for issue in self.errors)
        return (
            f"The previous extraction for `{self.process_name}` produced invalid exchanges. "
            f"Fix ALL of the following issues while regenerating the complete JSON:\n{bullet_list}\n"
            "Return the entire corrected `processDataSets` output. Do not omit exchanges."
        )
