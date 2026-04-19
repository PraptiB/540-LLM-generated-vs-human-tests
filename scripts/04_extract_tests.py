from __future__ import annotations

import argparse
import ast
import re
from pathlib import Path
from typing import Any

from common import read_json, safe_read_text, slugify_bug, write_json

PY_PATH_RE = re.compile(r'([A-Za-z0-9_./\\-]+\.py)')
PYTEST_NODE_RE = re.compile(r'([A-Za-z0-9_./\\-]+\.py)(?:::{1,2}([A-Za-z0-9_\.]+))?')
TEST_NAME_RE = re.compile(r'(?<![A-Za-z0-9_])(test_[A-Za-z0-9_]+)')


def locate_repo_root(checkout_parent: Path | None) -> Path | None:
    if not checkout_parent or not checkout_parent.exists():
        return None
    candidates = [p for p in checkout_parent.iterdir() if p.is_dir()]
    if len(candidates) == 1:
        return candidates[0]
    return checkout_parent


def normalize_repo_path(text: str) -> str:
    return text.strip().strip('"\'').replace('\\', '/')


def parse_failing_test_refs(raw: str) -> list[dict[str, Any]]:
    """Extract file/test references from BugsInPy bug.info failing test fields.

    Returns entries like:
      {"file": "pandas/tests/test_x.py", "raw": "...", "requested_tests": ["test_foo", "TestBar.test_baz"]}
    """
    if not raw:
        return []

    refs: dict[str, dict[str, Any]] = {}
    text = raw.strip()

    # First capture pytest-style node ids with optional ::TestClass::test_name
    for m in PYTEST_NODE_RE.finditer(text):
        file_path = normalize_repo_path(m.group(1))
        suffix = (m.group(2) or '').strip(':')
        requested_tests: list[str] = []
        if suffix:
            requested_tests.append(suffix)
            last = suffix.split('.')[-1].split('::')[-1]
            if last.startswith('test_'):
                requested_tests.append(last)
        ref = refs.setdefault(file_path, {"file": file_path, "raw": text, "requested_tests": []})
        ref["requested_tests"].extend(requested_tests)

    # Fallback: any .py path in the command/text
    for m in PY_PATH_RE.finditer(text):
        file_path = normalize_repo_path(m.group(1))
        refs.setdefault(file_path, {"file": file_path, "raw": text, "requested_tests": []})

    # Also collect explicit test_... names mentioned anywhere in the command
    names = sorted(set(TEST_NAME_RE.findall(text)))
    if names and refs:
        for ref in refs.values():
            ref["requested_tests"].extend(names)

    cleaned: list[dict[str, Any]] = []
    for ref in refs.values():
        uniq = []
        seen = set()
        for item in ref["requested_tests"]:
            k = item.strip()
            if k and k not in seen:
                seen.add(k)
                uniq.append(k)
        ref["requested_tests"] = uniq
        cleaned.append(ref)
    return cleaned


def extract_python_tests(file_path: Path) -> dict[str, dict[str, Any]]:
    if not file_path.exists() or file_path.suffix != '.py':
        return {}
    try:
        text = safe_read_text(file_path)
        tree = ast.parse(text)
    except Exception:
        return {}

    lines = text.splitlines()
    result: dict[str, dict[str, Any]] = {}

    class Visitor(ast.NodeVisitor):
        def __init__(self) -> None:
            self.class_stack: list[str] = []

        def visit_ClassDef(self, node: ast.ClassDef) -> Any:
            self.class_stack.append(node.name)
            self.generic_visit(node)
            self.class_stack.pop()

        def visit_FunctionDef(self, node: ast.FunctionDef) -> Any:
            if node.name.startswith('test_'):
                start = getattr(node, 'lineno', None)
                end = getattr(node, 'end_lineno', None)
                body = ''
                if start is not None and end is not None:
                    body = '\n'.join(lines[start - 1:end])
                qualname = '.'.join(self.class_stack + [node.name]) if self.class_stack else node.name
                item = {
                    'name': node.name,
                    'qualname': qualname,
                    'class_name': self.class_stack[-1] if self.class_stack else None,
                    'start_line': start,
                    'end_line': end,
                    'body': body,
                }
                result[node.name] = item
                result[qualname] = item
            self.generic_visit(node)

    Visitor().visit(tree)
    return result


def gather_changed_test_entries(record: dict[str, Any], fixed_repo_root: Path | None) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for file_entry in record.get('test_files', []):
        path_in_repo = file_entry.get('display_path') or file_entry.get('new_path') or file_entry.get('old_path')
        extracted_functions = []
        function_map: dict[str, dict[str, Any]] = {}
        if fixed_repo_root and path_in_repo:
            function_map = extract_python_tests(fixed_repo_root / path_in_repo)

        candidate_names = []
        candidate_names.extend(file_entry.get('added_test_defs', []))
        candidate_names.extend(file_entry.get('deleted_test_defs', []))
        for name in candidate_names:
            fn = function_map.get(name)
            if fn and fn not in extracted_functions:
                extracted_functions.append(fn)

        if not extracted_functions and fixed_repo_root and path_in_repo:
            full_path = fixed_repo_root / path_in_repo
            if full_path.exists() and full_path.suffix == '.py':
                preview = '\n'.join(safe_read_text(full_path).splitlines()[:240])
            else:
                preview = ''
        else:
            preview = ''

        entries.append(
            {
                'source': 'changed_test_file',
                'file': path_in_repo,
                'change_type': file_entry.get('change_type'),
                'added_test_defs': file_entry.get('added_test_defs', []),
                'deleted_test_defs': file_entry.get('deleted_test_defs', []),
                'added_assertions': file_entry.get('added_assertions', []),
                'requested_tests': [],
                'matched_test_names': sorted({fn.get('name', '') for fn in extracted_functions if fn.get('name')}),
                'extracted_functions': extracted_functions,
                'file_preview': preview,
            }
        )
    return entries


def gather_failing_test_entries(record: dict[str, Any], fixed_repo_root: Path | None) -> list[dict[str, Any]]:
    bug_info = record.get('bug_info', {}) or {}
    raw = (
        bug_info.get('failing_test_command')
        or bug_info.get('test_file')
        or bug_info.get('failing_tests')
        or ''
    )
    refs = parse_failing_test_refs(raw)
    entries: list[dict[str, Any]] = []

    for ref in refs:
        path_in_repo = ref['file']
        requested = ref.get('requested_tests', [])
        function_map: dict[str, dict[str, Any]] = {}
        matched: list[dict[str, Any]] = []
        preview = ''

        if fixed_repo_root and path_in_repo:
            full_path = fixed_repo_root / path_in_repo
            function_map = extract_python_tests(full_path)
            if full_path.exists() and full_path.suffix == '.py':
                preview = '\n'.join(safe_read_text(full_path).splitlines()[:260])

        # Match explicit requested tests first
        for req in requested:
            candidates = [req, req.split('.')[-1], req.split('::')[-1]]
            for key in candidates:
                fn = function_map.get(key)
                if fn and fn not in matched:
                    matched.append(fn)

        # If none matched but only one test name was requested, fuzzy-match any qualname ending
        if not matched and requested and function_map:
            last_req = requested[-1].split('.')[-1].split('::')[-1]
            for key, fn in function_map.items():
                if key.endswith(last_req) and fn not in matched:
                    matched.append(fn)

        entries.append(
            {
                'source': 'failing_test_reference',
                'file': path_in_repo,
                'change_type': 'existing',
                'added_test_defs': [],
                'deleted_test_defs': [],
                'added_assertions': [],
                'requested_tests': requested,
                'matched_test_names': sorted({fn.get('name', '') for fn in matched if fn.get('name')}),
                'matched_qualnames': sorted({fn.get('qualname', '') for fn in matched if fn.get('qualname')}),
                'extracted_functions': matched,
                'file_preview': preview,
                'raw_failing_test_reference': raw,
            }
        )
    return entries


def dedupe_human_tests(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: dict[tuple[str, str], dict[str, Any]] = {}
    for item in entries:
        key = (item.get('source', ''), item.get('file', '') or '')
        existing = merged.get(key)
        if not existing:
            merged[key] = item
            continue
        for field in ['added_test_defs', 'deleted_test_defs', 'added_assertions', 'requested_tests', 'matched_test_names', 'matched_qualnames']:
            vals = list(existing.get(field, [])) + list(item.get(field, []))
            deduped = []
            seen = set()
            for v in vals:
                if v and v not in seen:
                    seen.add(v)
                    deduped.append(v)
            existing[field] = deduped
        existing_funcs = existing.get('extracted_functions', [])
        seen_bodies = {f.get('qualname') or f.get('name') for f in existing_funcs}
        for fn in item.get('extracted_functions', []):
            k = fn.get('qualname') or fn.get('name')
            if k not in seen_bodies:
                existing_funcs.append(fn)
                seen_bodies.add(k)
        if not existing.get('file_preview') and item.get('file_preview'):
            existing['file_preview'] = item['file_preview']
    return list(merged.values())


def infer_baseline_quality(entries: list[dict[str, Any]]) -> str:
    if not entries:
        return 'none'
    has_changed_defs = any(e.get('source') == 'changed_test_file' and (e.get('added_test_defs') or e.get('deleted_test_defs')) for e in entries)
    has_matched_fail = any(e.get('source') == 'failing_test_reference' and (e.get('extracted_functions') or e.get('file_preview')) for e in entries)
    if has_changed_defs:
        return 'strong'
    if has_matched_fail:
        return 'partial'
    return 'none'


def main() -> None:
    parser = argparse.ArgumentParser(description='Extract human-written test baseline metadata from changed tests and failing test references.')
    parser.add_argument('--per-bug-dir', type=Path, default=Path('outputs/per_bug_json'), help='Directory with per-bug JSON files from step 03')
    parser.add_argument('--checkout-manifest', type=Path, default=Path('outputs/checkout_manifest.json'), help='Checkout manifest from step 02')
    args = parser.parse_args()

    manifest = {}
    if args.checkout_manifest.exists():
        for item in read_json(args.checkout_manifest):
            manifest[(item['project'], str(item['bug_id']))] = item

    for json_path in sorted(args.per_bug_dir.glob('*.json')):
        record = read_json(json_path)
        key = (record['project'], str(record['bug_id']))
        checkout = manifest.get(key, {})

        fixed_parent = Path(checkout['fixed_dir']) if checkout.get('fixed_dir') else None
        fixed_repo_root = locate_repo_root(fixed_parent) if fixed_parent and fixed_parent.exists() else None

        changed_entries = gather_changed_test_entries(record, fixed_repo_root)
        failing_entries = gather_failing_test_entries(record, fixed_repo_root)
        human_tests = dedupe_human_tests(changed_entries + failing_entries)

        record['human_tests'] = human_tests
        record['human_test_names'] = sorted(
            {
                name
                for item in human_tests
                for name in (
                    list(item.get('matched_test_names', []))
                    + list(item.get('added_test_defs', []))
                    + [fn.get('name', '') for fn in item.get('extracted_functions', []) if fn.get('name')]
                )
                if name
            }
        )
        record['human_test_files'] = sorted({item.get('file', '') for item in human_tests if item.get('file')})
        record['human_test_count'] = len(record['human_test_names'])
        record['human_test_baseline_quality'] = infer_baseline_quality(human_tests)
        record['has_failing_test_reference'] = any(item.get('source') == 'failing_test_reference' for item in human_tests)
        record['has_changed_test_baseline'] = any(item.get('source') == 'changed_test_file' for item in human_tests)

        notes = record.get('notes', '')
        extra_note = f"human_test_baseline_quality={record['human_test_baseline_quality']}"
        if extra_note not in notes:
            record['notes'] = (notes + '; ' + extra_note).strip('; ')

        write_json(json_path, record)
        print(
            f"Enriched human test baseline for {slugify_bug(record['project'], record['bug_id'])} | "
            f"quality={record['human_test_baseline_quality']} | tests={record['human_test_count']}"
        )


if __name__ == '__main__':
    main()
