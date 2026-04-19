from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any

from common import read_json, safe_read_text, write_json

HUNK_RE = re.compile(r"@@\s+-(\d+)(?:,(\d+))?\s+\+(\d+)(?:,(\d+))?\s+@@")

PROMPT_TEMPLATE = """You are given a real historical bug from a Python project.

Your task is to generate a unit test that would expose the bug in the buggy version.

Rules:
- Generate only test code.
- Follow the repository's existing test style.
- Do not modify production code.
- Focus on a minimal regression test.
- Reuse naming, imports, and assertion style from nearby human-written tests when helpful.

Project: {project}
Bug ID: {bug_id}
Slug: {slug}

Relevant buggy source context:
{buggy_source_context}

Bug-fix patch:
{patch_text}

Related human-written test context:
{human_test_context}

Repository test style notes:
{repo_style_notes}
"""


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def locate_repo_root(checkout_parent: Path | None) -> Path | None:
    if not checkout_parent or not checkout_parent.exists():
        return None
    candidates = [p for p in checkout_parent.iterdir() if p.is_dir()]
    if len(candidates) == 1:
        return candidates[0]
    return checkout_parent


def parse_hunk_header(header: str) -> tuple[int, int, int, int] | None:
    m = HUNK_RE.search(header)
    if not m:
        return None
    old_start = int(m.group(1))
    old_len = int(m.group(2) or "1")
    new_start = int(m.group(3))
    new_len = int(m.group(4) or "1")
    return old_start, old_len, new_start, new_len


def extract_line_window(text: str, start_line: int, line_count: int, window: int = 20) -> str:
    lines = text.splitlines()
    if not lines:
        return ""
    start_idx = max(start_line - 1 - window, 0)
    end_idx = min(start_line - 1 + max(line_count, 1) + window, len(lines))
    chunk = []
    for i in range(start_idx, end_idx):
        chunk.append(f"{i+1}: {lines[i]}")
    return "\n".join(chunk)


def build_buggy_source_context(record: dict[str, Any], buggy_repo_root: Path | None) -> str:
    sections: list[str] = []
    if not buggy_repo_root:
        return "(buggy checkout not available)"

    for file_entry in record.get("source_files", []):
        path_in_repo = file_entry.get("display_path") or file_entry.get("old_path") or file_entry.get("new_path")
        if not path_in_repo:
            continue
        full_path = buggy_repo_root / path_in_repo
        if not full_path.exists() or full_path.suffix != ".py":
            continue
        text = safe_read_text(full_path)
        file_sections: list[str] = []
        for hunk in file_entry.get("hunks", []):
            parsed = parse_hunk_header(hunk.get("header", ""))
            if not parsed:
                continue
            old_start, old_len, _, _ = parsed
            file_sections.append(extract_line_window(text, old_start, old_len))
        if not file_sections:
            lines = text.splitlines()[:220]
            file_sections = ["\n".join(f"{i+1}: {line}" for i, line in enumerate(lines))]
        deduped = []
        seen = set()
        for sec in file_sections:
            key = sec.strip()
            if key and key not in seen:
                deduped.append(sec)
                seen.add(key)
        sections.append(f"### FILE: {path_in_repo}\n" + "\n\n".join(deduped[:3]))
    return "\n\n".join(sections) if sections else "(no Python source context extracted)"


def build_human_test_context(record: dict[str, Any], fixed_repo_root: Path | None) -> str:
    chunks: list[str] = []
    human_tests = record.get("human_tests", []) or []
    for test_record in human_tests:
        path_in_repo = test_record.get("file")
        extracted = test_record.get("extracted_functions", []) or []
        if extracted:
            for fn in extracted:
                qualname = fn.get("qualname") or fn.get("name") or "test"
                body = fn.get("body") or ""
                chunks.append(f"### TEST: {qualname} ({path_in_repo})\n{body}")
        elif fixed_repo_root and path_in_repo:
            full_path = fixed_repo_root / path_in_repo
            if full_path.exists() and full_path.suffix == ".py":
                text = safe_read_text(full_path)
                preview = "\n".join(text.splitlines()[:220])
                chunks.append(f"### TEST FILE: {path_in_repo}\n{preview}")

    if not chunks and fixed_repo_root:
        for file_entry in record.get("test_files", [])[:2]:
            path_in_repo = file_entry.get("display_path") or file_entry.get("new_path") or file_entry.get("old_path")
            if not path_in_repo:
                continue
            full_path = fixed_repo_root / path_in_repo
            if full_path.exists() and full_path.suffix == ".py":
                text = safe_read_text(full_path)
                preview = "\n".join(text.splitlines()[:220])
                chunks.append(f"### TEST FILE: {path_in_repo}\n{preview}")

    return "\n\n".join(chunks) if chunks else "(no human test context extracted)"


def infer_repo_style_notes(human_test_context: str, record: dict[str, Any]) -> str:
    notes = []
    if "self.assert" in human_test_context:
        notes.append("Repository appears to use unittest-style assertions in at least some tests.")
    elif "pytest" in human_test_context or any("tests/" in (f.get("file") or "").replace("\\", "/") for f in record.get("human_tests", [])):
        notes.append("Repository appears to use pytest-style test organization.")
    else:
        notes.append("Follow the existing repository test style visible in the provided examples.")
    notes.append("Generate only test code, not explanations.")
    notes.append("Prefer a focused regression test that targets the changed behavior.")
    return "\n".join(f"- {n}" for n in notes)


def keep_row(row: dict[str, str], include_maybe: bool) -> bool:
    status = (row.get("status_final") or row.get("recommended_status") or "").strip().lower()
    return status == "keep" or (include_maybe and status == "maybe")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build one LLM context package per frozen bug. Uses status_final=keep by default."
    )
    parser.add_argument(
        "--frozen-bugs",
        type=Path,
        default=Path("outputs/frozen_bug_set.csv"),
        help="CSV produced by 06_freeze_bug_set.py and optionally edited by hand",
    )
    parser.add_argument(
        "--per-bug-dir",
        type=Path,
        default=Path("outputs/per_bug_json"),
        help="Directory containing per-bug JSON files",
    )
    parser.add_argument(
        "--checkout-manifest",
        type=Path,
        default=Path("outputs/checkout_manifest.json"),
        help="Checkout manifest from step 02",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("llm_context"),
        help="Where to write one context folder per kept bug",
    )
    parser.add_argument(
        "--include-maybe",
        action="store_true",
        help="Include rows marked maybe as well as keep",
    )
    args = parser.parse_args()

    frozen_rows = read_csv_rows(args.frozen_bugs)
    per_bug = {}
    for path in sorted(args.per_bug_dir.glob("*.json")):
        record = read_json(path)
        per_bug[(record["project"], str(record["bug_id"]))] = record

    manifest_items = read_json(args.checkout_manifest) if args.checkout_manifest.exists() else []
    manifest = {(item["project"], str(item["bug_id"])): item for item in manifest_items}

    args.output_dir.mkdir(parents=True, exist_ok=True)

    kept = 0
    for row in frozen_rows:
        if not keep_row(row, include_maybe=args.include_maybe):
            continue

        key = (row["project"], str(row["bug_id"]))
        record = per_bug.get(key)
        checkout = manifest.get(key, {})
        if not record:
            print(f"Skipping {row['project']}_{row['bug_id']}: missing per-bug JSON")
            continue

        slug = record.get("slug") or f"{row['project']}_{row['bug_id']}"
        out_dir = args.output_dir / slug
        out_dir.mkdir(parents=True, exist_ok=True)

        buggy_repo_root = locate_repo_root(Path(checkout["buggy_dir"])) if checkout.get("buggy_dir") else None
        fixed_repo_root = locate_repo_root(Path(checkout["fixed_dir"])) if checkout.get("fixed_dir") else None

        patch_text = safe_read_text(Path(record["patch_path"])) if record.get("patch_path") else ""
        buggy_source_context = build_buggy_source_context(record, buggy_repo_root)
        human_test_context = build_human_test_context(record, fixed_repo_root)
        repo_style_notes = infer_repo_style_notes(human_test_context, record)

        metadata = {
            "project": row["project"],
            "bug_id": str(row["bug_id"]),
            "slug": slug,
            "status_final": row.get("status_final", row.get("recommended_status", "")),
            "recommended_status": row.get("recommended_status", ""),
            "buggy_checkout_path": checkout.get("buggy_dir", ""),
            "fixed_checkout_path": checkout.get("fixed_dir", ""),
            "source_files_changed": record.get("source_files_changed", 0),
            "test_files_changed": record.get("test_files_changed", 0),
            "human_test_names": record.get("human_test_names", []),
            "human_test_files": record.get("human_test_files", []),
            "source_file_paths": [
                f.get("display_path") or f.get("old_path") or f.get("new_path")
                for f in record.get("source_files", [])
            ],
            "test_file_paths": [
                f.get("display_path") or f.get("old_path") or f.get("new_path")
                for f in record.get("test_files", [])
            ],
            "failing_tests": row.get("failing_tests", ""),
            "buggy_commit": row.get("buggy_commit", ""),
            "fixed_commit": row.get("fixed_commit", ""),
        }

        write_json(out_dir / "metadata.json", metadata)
        (out_dir / "fix_patch.diff").write_text(patch_text, encoding="utf-8")
        (out_dir / "buggy_source_context.txt").write_text(buggy_source_context, encoding="utf-8")
        (out_dir / "human_test_context.txt").write_text(human_test_context, encoding="utf-8")
        (out_dir / "repo_style_notes.txt").write_text(repo_style_notes, encoding="utf-8")

        prompt = PROMPT_TEMPLATE.format(
            project=row["project"],
            bug_id=row["bug_id"],
            slug=slug,
            buggy_source_context=buggy_source_context,
            patch_text=patch_text,
            human_test_context=human_test_context,
            repo_style_notes=repo_style_notes,
        )
        (out_dir / "llm_prompt.txt").write_text(prompt, encoding="utf-8")

        kept += 1
        print(f"Built LLM context for {slug}")

    manifest_summary = {
        "kept_context_packages": kept,
        "output_dir": str(args.output_dir.resolve()),
    }
    (args.output_dir / "_manifest.json").write_text(json.dumps(manifest_summary, indent=2), encoding="utf-8")
    print(f"Wrote: {args.output_dir / '_manifest.json'}")


if __name__ == "__main__":
    main()
