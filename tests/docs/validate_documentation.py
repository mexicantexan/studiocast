#!/usr/bin/env python3
"""Validate local Markdown links and conservative repository-path references."""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path
from urllib.parse import unquote, urlsplit


INLINE_LINK = re.compile(r"!?\[[^\]\n]*\]\(([^)\n]+)\)")
REFERENCE_LINK = re.compile(
    r"(?m)^[ \t]{0,3}\[[^\]\n]+\]:[ \t]*(<[^>]+>|\S+)"
)
INLINE_CODE = re.compile(r"(?<!`)`([^`\n]+)`(?!`)")
LINE_SUFFIX = re.compile(r":\d+(?:-\d+)?(?:,\d+(?:-\d+)?)*$")

REPOSITORY_PREFIXES = (
    ".github/",
    "cmake/",
    "docs/",
    "installer/",
    "packaging/",
    "resources/",
    "scripts/",
    "src/",
    "tests/",
)
REPOSITORY_FILES = {
    ".gitignore",
    "CMakeLists.txt",
    "CODE_OF_CONDUCT.md",
    "CONTRIBUTING.md",
    "LICENSE",
    "NOTICE",
    "README.md",
    "SECURITY.md",
    "VERSION",
}
FALLBACK_EXCLUDED_PARTS = {
    ".git",
    ".idea",
    ".pytest_cache",
    "__pycache__",
    "dist",
}


def line_number(text: str, offset: int) -> int:
    return text.count("\n", 0, offset) + 1


def repository_markdown_files(source_root: Path) -> list[Path]:
    try:
        result = subprocess.run(
            [
                "git",
                "-C",
                str(source_root),
                "ls-files",
                "--cached",
                "--others",
                "--exclude-standard",
                "-z",
                "--",
                "*.md",
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        files = []
        for path in source_root.rglob("*.md"):
            relative = path.relative_to(source_root)
            if any(part in FALLBACK_EXCLUDED_PARTS for part in relative.parts):
                continue
            if relative.parts and (
                relative.parts[0].startswith("build")
                or relative.parts[0].startswith("cmake-build-")
            ):
                continue
            files.append(path)
        return sorted(files)

    files = []
    for encoded in result.stdout.split(b"\0"):
        if not encoded:
            continue
        path = source_root / encoded.decode("utf-8")
        if path.is_file():
            files.append(path)
    return sorted(files)


def link_destination(raw: str) -> str:
    value = raw.strip()
    if value.startswith("<"):
        closing = value.find(">")
        return value[1:closing] if closing != -1 else value
    match = re.match(r"^(\S+?)(?:\s+(?:\"[^\"]*\"|'[^']*'|\([^)]*\)))?$", value)
    return match.group(1) if match else value


def validate_local_link(
    source_root: Path, markdown: Path, raw: str
) -> str | None:
    destination = link_destination(raw)
    if not destination or destination.startswith("#"):
        return None

    parsed = urlsplit(destination)
    if parsed.scheme or parsed.netloc:
        return None

    local_path = unquote(parsed.path)
    if not local_path:
        return None
    if local_path.startswith("/"):
        return f"absolute local link is not repository-relative: {destination}"

    resolved = (markdown.parent / local_path).resolve()
    try:
        resolved.relative_to(source_root)
    except ValueError:
        return f"local link escapes the repository: {destination}"
    if not resolved.exists():
        return f"missing Markdown link target: {destination}"
    return None


def repository_path_candidate(raw: str) -> str | None:
    candidate = raw.strip().rstrip(".,;")
    if candidate.startswith("./"):
        candidate = candidate[2:]
    candidate = LINE_SUFFIX.sub("", candidate)

    if not candidate or any(ch in candidate for ch in "<>${}|\n\r\t"):
        return None
    if " " in candidate:
        return None
    if candidate not in REPOSITORY_FILES and not candidate.startswith(
        REPOSITORY_PREFIXES
    ):
        return None
    return candidate


def validate_repository_path(source_root: Path, candidate: str) -> str | None:
    if candidate.startswith("/") or ".." in Path(candidate).parts:
        return f"unsafe repository path: {candidate}"

    if any(character in candidate for character in "*?["):
        try:
            matches = list(source_root.glob(candidate))
        except ValueError:
            matches = []
        if not matches:
            return f"backticked repository path pattern has no matches: {candidate}"
        return None

    if not (source_root / candidate).exists():
        return f"missing backticked repository path: {candidate}"
    return None


def validate_file(source_root: Path, markdown: Path) -> list[str]:
    text = markdown.read_text(encoding="utf-8")
    errors: list[str] = []
    relative = markdown.relative_to(source_root)

    for match in INLINE_LINK.finditer(text):
        error = validate_local_link(source_root, markdown, match.group(1))
        if error:
            errors.append(f"{relative}:{line_number(text, match.start())}: {error}")

    for match in REFERENCE_LINK.finditer(text):
        error = validate_local_link(source_root, markdown, match.group(1))
        if error:
            errors.append(f"{relative}:{line_number(text, match.start())}: {error}")

    for match in INLINE_CODE.finditer(text):
        candidate = repository_path_candidate(match.group(1))
        if candidate is None:
            continue
        error = validate_repository_path(source_root, candidate)
        if error:
            errors.append(f"{relative}:{line_number(text, match.start())}: {error}")

    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help="repository source root (defaults to the validator's repository)",
    )
    args = parser.parse_args()
    source_root = args.source_root.resolve()

    markdown_files = repository_markdown_files(source_root)
    if not markdown_files:
        print("documentation validation failed: no Markdown files found", file=sys.stderr)
        return 1

    errors: list[str] = []
    for markdown in markdown_files:
        try:
            errors.extend(validate_file(source_root, markdown))
        except UnicodeDecodeError as exc:
            errors.append(
                f"{markdown.relative_to(source_root)}: invalid UTF-8 Markdown: {exc}"
            )

    if errors:
        print("documentation validation failed:", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        return 1

    print(
        f"Validated {len(markdown_files)} repository Markdown files "
        "(local links and repository paths; network access not used)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
