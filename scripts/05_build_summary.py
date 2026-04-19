from __future__ import annotations

import argparse
from pathlib import Path

from common import read_json, write_csv


def main() -> None:
    parser = argparse.ArgumentParser(description='Build a flat CSV summary from per-bug JSON files, including human test baseline quality.')
    parser.add_argument('--per-bug-dir', type=Path, default=Path('outputs/per_bug_json'), help='Directory with per-bug JSON files')
    parser.add_argument('--output-csv', type=Path, default=Path('outputs/bug_summary.csv'), help='Output CSV path')
    args = parser.parse_args()

    rows = []
    for json_path in sorted(args.per_bug_dir.glob('*.json')):
        record = read_json(json_path)
        rows.append(
            {
                'project': record['project'],
                'bug_id': record['bug_id'],
                'files_changed': record.get('files_changed', 0),
                'source_files_changed': record.get('source_files_changed', 0),
                'test_files_changed': record.get('test_files_changed', 0),
                'human_test_added': record.get('human_test_added', False),
                'human_test_modified': record.get('human_test_modified', False),
                'human_test_count': record.get('human_test_count', 0),
                'human_test_names': '; '.join(record.get('human_test_names', [])),
                'human_test_files': '; '.join(record.get('human_test_files', [])),
                'human_test_baseline_quality': record.get('human_test_baseline_quality', ''),
                'has_changed_test_baseline': record.get('has_changed_test_baseline', False),
                'has_failing_test_reference': record.get('has_failing_test_reference', False),
                'patch_path': record.get('patch_path', ''),
                'buggy_commit': record.get('bug_info', {}).get('buggy_commit_id')
                or record.get('bug_info', {}).get('buggy_commit')
                or '',
                'fixed_commit': record.get('bug_info', {}).get('fixed_commit_id')
                or record.get('bug_info', {}).get('fixed_commit')
                or '',
                'failing_tests': record.get('bug_info', {}).get('failing_test_command')
                or record.get('bug_info', {}).get('test_file')
                or record.get('bug_info', {}).get('failing_tests')
                or '',
                'notes': record.get('notes', ''),
            }
        )

    write_csv(args.output_csv, rows)
    print(f'Wrote: {args.output_csv}')


if __name__ == '__main__':
    main()
