"""Public facade for running the local TIDAS validation CLI."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path
from typing import Iterable

from tiangong_lca_spec.core.config import Settings, get_settings
from tiangong_lca_spec.core.logging import get_logger
from tiangong_lca_spec.core.models import ProcessDataset, TidasValidationFinding

LOGGER = get_logger(__name__)
ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-9;]*m")
PATH_RE = re.compile(r"(?P<path>(?:\.{0,2}/|/)?[A-Za-z0-9_\-./]+\.json)")


class TidasValidationService:
    def __init__(
        self,
        settings: Settings | None = None,
        *,
        command: Iterable[str] | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._custom_command = list(command) if command is not None else None

    def validate(self, datasets: list[ProcessDataset]) -> list[TidasValidationFinding]:
        LOGGER.info(
            "tidas_validation.start",
            dataset_count=len(datasets),
            input_dir=str(self._settings.artifacts_dir),
        )
        findings = self.validate_directory(self._settings.artifacts_dir)
        LOGGER.info("tidas_validation.complete", finding_count=len(findings))
        return findings

    def validate_directory(self, directory: Path | str) -> list[TidasValidationFinding]:
        path = Path(directory)
        if not path.exists():
            message = f"TIDAS artifacts directory does not exist: {path}"
            LOGGER.error("tidas_validation.missing_directory", path=str(path))
            return [
                TidasValidationFinding(
                    severity="error",
                    message=message,
                    path=str(path),
                    suggestion="Generate ILCD artifacts before running validation.",
                )
            ]

        command = self._build_command(path)
        LOGGER.debug("tidas_validation.invoke_cli", command=command)
        proc = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
        )
        output = _combine_output(proc.stdout, proc.stderr)
        findings = _parse_cli_output(output)

        if proc.returncode != 0:
            LOGGER.error(
                "tidas_validation.cli_exit_nonzero",
                returncode=proc.returncode,
                finding_count=len(findings),
            )
            if not findings:
                suggestion = output or "See console output for details."
                findings.append(
                    TidasValidationFinding(
                        severity="error",
                        message="tidas-validate exited with a non-zero status.",
                        path=str(path),
                        suggestion=suggestion.strip(),
                    )
                )
        return findings

    def close(self) -> None:
        """Compatibility hook; no resources to clean up."""
        return None

    def _build_command(self, directory: Path) -> list[str]:
        if self._custom_command:
            command = list(self._custom_command)
            if "--input-dir" not in command and "-i" not in command:
                command.extend(["-i", str(directory)])
            return [str(part) for part in command]
        return ["uv", "run", "tidas-validate", "-i", str(directory)]


def _combine_output(stdout: str | None, stderr: str | None) -> str:
    pieces = []
    if stdout:
        pieces.append(stdout)
    if stderr:
        pieces.append(stderr)
    return "\n".join(piece for piece in pieces if piece)


def _parse_cli_output(raw_output: str) -> list[TidasValidationFinding]:
    findings: list[TidasValidationFinding] = []
    for raw_line in raw_output.splitlines():
        line = ANSI_ESCAPE_RE.sub("", raw_line).strip()
        if not line:
            continue
        severity = _detect_severity(line)
        path = _extract_path(line)
        findings.append(
            TidasValidationFinding(
                severity=severity,
                message=line,
                path=path,
                suggestion=None,
            )
        )
    return findings


def _detect_severity(message: str) -> str:
    upper = message.upper()
    if "ERROR" in upper or "FAILED" in upper:
        return "error"
    if "WARN" in upper:
        return "warning"
    return "info"


def _extract_path(message: str) -> str | None:
    match = PATH_RE.search(message)
    if match:
        return match.group("path")
    return None


def validate_with_tidas(process_datasets: list[ProcessDataset]) -> list[TidasValidationFinding]:
    service = TidasValidationService()
    try:
        return service.validate(process_datasets)
    finally:
        service.close()
