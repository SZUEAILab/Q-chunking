#!/usr/bin/env python3
"""Validate maintained documentation links and reachability from the main index."""

from __future__ import annotations

import argparse
from pathlib import Path
import re
from urllib.parse import unquote


def link_targets(text: str) -> list[str]:
    """Read Markdown and HTML destinations outside fenced/inline examples."""
    lines = []
    fence = None
    for line in text.splitlines():
        marker = re.match(r"^\s*(`{3,}|~{3,})", line)
        if marker:
            char = marker.group(1)[0]
            if fence is None:
                fence = char
            elif fence == char:
                fence = None
            continue
        if fence is None:
            lines.append(line)
    text = "\n".join(lines)
    # Backticks can occur inside a link label; preserve the destination.
    targets = re.findall(r"\]\(\s*(<[^>]+>|[^\s)]+)", text)
    targets.extend(re.findall(r"(?:src|href)=[\"']([^\"']+)[\"']", text))
    targets.extend(re.findall(r"^\s*\[[^]\n]+\]:\s*(\S+)", text, re.MULTILINE))
    return [value.strip("<>") for value in targets]


def local_target(document: Path, target: str) -> Path | None:
    if re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*:", target) or target.startswith(("#", "/")):
        return None
    if any(char in target for char in ("<", ">", "{", "}")):
        return None
    path = unquote(target.split("#", 1)[0].split("?", 1)[0])
    return (document.parent / path).resolve() if path else None


def maintained_documents(root: Path) -> list[Path]:
    documents = [path for path in (root / "docs").rglob("*.md")
                 if path.relative_to(root / "docs").parts[0] not in {"data", "ds-visualizer"}]
    return sorted(documents)


def validate(root: Path) -> list[str]:
    root = root.resolve()
    documents = maintained_documents(root)
    errors = []
    links = {}
    for document in [*documents, root / "README.md", root / "CLAUDE.md"]:
        if not document.exists():
            continue
        links[document] = set()
        for target in link_targets(document.read_text()):
            path = local_target(document, target)
            if path is None:
                continue
            links[document].add(path)
            if not path.exists():
                errors.append(f"{document.relative_to(root)}: missing {target}")
    index = root / "docs/index.md"
    reachable = set()
    pending = [index]
    while pending:
        document = pending.pop()
        if document in reachable:
            continue
        reachable.add(document)
        pending.extend(path for path in links.get(document, set())
                       if path in links and path not in reachable)
    for document in documents:
        if document.name != "README.md" and document.resolve() not in reachable:
            errors.append(f"{document.relative_to(root)}: not linked from docs/index.md navigation")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    errors = validate(args.root)
    for error in errors:
        print(error)
    print(f"Documentation: {len(maintained_documents(args.root))} Markdown files; {len(errors)} errors")
    raise SystemExit(bool(errors))


if __name__ == "__main__":
    main()
