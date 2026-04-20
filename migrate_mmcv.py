#!/usr/bin/env python3
"""
MMCV 1.x → MMEngine 2.x 자동 마이그레이션 스크립트
=====================================================
프로젝트 내 모든 .py 파일을 스캔하여 구버전 mmcv import 구문을
MMEngine 2.x 문법으로 일괄 치환합니다.

사용법:
    python migrate_mmcv.py              # 실제 마이그레이션 (자동 .bak 백업)
    python migrate_mmcv.py --dry-run    # 변경 예정 파일 목록만 출력 (파일 미수정)
    python migrate_mmcv.py --no-backup  # 백업 없이 바로 수정

변환 규칙:
    from mmcv.runner import get_time_str    → 파일 내 로컬 함수 주입
    from mmcv.runner import init_dist       → from mmengine.dist import init_dist
    from mmcv.runner import master_only     → from mmengine.dist import master_only
    from mmcv.runner import get_dist_info   → from mmengine.dist import get_dist_info
    from mmcv.runner import load_checkpoint → from mmengine.runner import load_checkpoint
    from mmcv.runner import save_checkpoint → from mmengine.runner import save_checkpoint
    from mmcv.runner import BaseModule      → from mmengine.model import BaseModule
    from mmcv.runner import auto_fp16       → from mmengine.model import auto_fp16
    from mmcv.utils  import get_logger      → 파일 내 로컬 래퍼 함수 주입
    mmcv.scandir(...)                       → scandir(...)  +  mmengine.utils import
    mmcv.mkdir_or_exist(...)                → mkdir_or_exist(...) + mmengine.utils import
"""
from __future__ import annotations

import os
import re
import sys
import shutil
import textwrap
from collections import defaultdict
from pathlib import Path
from typing import NamedTuple

# ── 프로젝트 루트 ─────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent

# 스캔 제외 디렉토리
SKIP_DIRS: set[str] = {
    '.git', '__pycache__', '.venv', 'venv', 'env',
    'node_modules', '.idea', '.mypy_cache',
}

# 이 스크립트 자신 및 패치 파일은 수정하지 않음
SKIP_FILES: set[str] = {'mmcv_compat.py', 'migrate_mmcv.py'}

# ── mmcv.runner 심볼 → 대상 모듈 라우팅 테이블 ──────────────────────────────
# '_local' 은 파일 내에 로컬 함수/코드를 주입함을 의미
RUNNER_TO_MODULE: dict[str, str] = {
    'get_time_str':    '_local',
    'init_dist':       'mmengine.dist',
    'master_only':     'mmengine.dist',
    'get_dist_info':   'mmengine.dist',
    'load_checkpoint': 'mmengine.runner',
    'save_checkpoint': 'mmengine.runner',
    'BaseModule':      'mmengine.model',
    'auto_fp16':       'mmengine.model',
}

# ── 로컬 주입 코드 스니펫 ─────────────────────────────────────────────────────
_GET_TIME_STR_CODE = textwrap.dedent("""\
    def get_time_str() -> str:
        import datetime
        return datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
""")

_GET_LOGGER_CODE = textwrap.dedent("""\
    def get_logger(name: str, log_file=None, log_level: str = 'INFO', **kwargs):
        \"\"\"mmcv.utils.get_logger 호환 래퍼.\"\"\"
        import logging, os
        logger = logging.getLogger(name)
        if logger.hasHandlers():
            return logger
        level = getattr(logging, log_level.upper(), logging.INFO)
        logger.setLevel(level)
        fmt = logging.Formatter(
            '%(asctime)s.%(msecs)03d - %(levelname)s: %(message)s',
            datefmt='%Y/%m/%d %H:%M:%S',
        )
        sh = logging.StreamHandler()
        sh.setLevel(level)
        sh.setFormatter(fmt)
        logger.addHandler(sh)
        if log_file is not None:
            _log_dir = os.path.dirname(os.path.abspath(log_file))
            if _log_dir:
                os.makedirs(_log_dir, exist_ok=True)
            fh = logging.FileHandler(log_file, 'w')
            fh.setLevel(level)
            fh.setFormatter(fmt)
            logger.addHandler(fh)
        return logger
""")

LOCAL_INJECT_CODE: dict[str, str] = {
    'get_time_str': _GET_TIME_STR_CODE,
    'get_logger':   _GET_LOGGER_CODE,
}


# ── 유틸리티 ──────────────────────────────────────────────────────────────────

def _find_import_block_end(lines: list[str]) -> int:
    """최상위(들여쓰기 없는) import 블록의 마지막 줄 다음 인덱스를 반환.

    들여쓰기된 줄(함수/클래스 내부 import)은 무시합니다.
    """
    last_import_idx = -1
    paren_depth = 0
    for i, line in enumerate(lines):
        # 들여쓰기된 줄은 최상위 import 가 아님 → 건너뜀
        if line and line[0] in (' ', '\t'):
            continue
        stripped = line.strip()
        paren_depth += stripped.count('(') - stripped.count(')')
        paren_depth = max(paren_depth, 0)
        if paren_depth == 0 and re.match(r'^(import |from \S+ import )', stripped):
            last_import_idx = i
    return last_import_idx + 1 if last_import_idx >= 0 else 0


def _already_imported(lines: list[str], module: str, symbol: str) -> bool:
    """파일 내에 이미 해당 import 가 존재하는지 확인 (중복 추가 방지)."""
    pattern = re.compile(
        rf'^from\s+{re.escape(module)}\s+import\s+.*\b{re.escape(symbol)}\b'
    )
    return any(pattern.match(ln.strip()) for ln in lines)


def _function_defined(lines: list[str], func_name: str) -> bool:
    """파일 내에 해당 함수가 이미 정의되어 있는지 확인."""
    pattern = re.compile(rf'^def\s+{re.escape(func_name)}\s*\(')
    return any(pattern.match(ln.strip()) for ln in lines)


# ── 핵심 변환 로직 ────────────────────────────────────────────────────────────

class MigrationResult(NamedTuple):
    source: str
    modified: bool
    summary: list[str]  # 변경 내역 메시지


def migrate_source(source: str, filepath: str = '') -> MigrationResult:
    """소스 코드 문자열을 받아 변환된 문자열과 변경 여부를 반환."""
    lines = source.split('\n')
    summary: list[str] = []
    changed = False

    # 추가할 mmengine import: {module: set(symbols)}
    new_imports: dict[str, set[str]] = defaultdict(set)
    # 추가할 로컬 함수 정의 코드 블록
    local_injects: list[tuple[str, str]] = []  # (name, code)
    injected_names: set[str] = set()

    # ── 패스 1: from mmcv.runner import ... ─────────────────────────────────
    runner_re = re.compile(
        r'^(\s*)from\s+mmcv\.runner\s+import\s+(.+)$'
    )
    new_lines: list[str] = []
    for line in lines:
        m = runner_re.match(line)
        if not m:
            new_lines.append(line)
            continue

        indent = m.group(1)
        symbols_raw = m.group(2).strip().rstrip('\\')
        # 괄호로 묶인 경우 간이 처리 (단일 행 가정)
        symbols_raw = symbols_raw.strip('()')
        symbols = [s.strip().rstrip(',') for s in symbols_raw.split(',')]
        symbols = [s for s in symbols if s]

        new_lines.append(f'{indent}# [migrated from mmcv.runner] {line.strip()}')
        changed = True
        summary.append(f'  mmcv.runner import 제거: {", ".join(symbols)}')

        for sym in symbols:
            target_mod = RUNNER_TO_MODULE.get(sym)
            if target_mod is None:
                new_lines.append(
                    f'{indent}# WARNING: mmcv.runner.{sym} 은 자동 매핑 없음 – 수동 확인 필요'
                )
                summary.append(f'  WARNING: {sym} 수동 확인 필요')
            elif target_mod == '_local':
                if sym not in injected_names and not _function_defined(lines, sym):
                    local_injects.append((sym, LOCAL_INJECT_CODE[sym]))
                    injected_names.add(sym)
                    summary.append(f'  로컬 함수 주입: {sym}()')
            else:
                if not _already_imported(lines, target_mod, sym):
                    new_imports[target_mod].add(sym)
                    summary.append(f'  새 import 추가: from {target_mod} import {sym}')
    lines = new_lines

    # ── 패스 2: from mmcv.utils import get_logger ────────────────────────────
    utils_logger_re = re.compile(
        r'^(\s*)from\s+mmcv\.utils\s+import\s+(.*)$'
    )
    new_lines = []
    for line in lines:
        m = utils_logger_re.match(line)
        if not m:
            new_lines.append(line)
            continue

        indent = m.group(1)
        syms_raw = m.group(2).strip().strip('()')
        syms = [s.strip().rstrip(',') for s in syms_raw.split(',')]
        syms = [s for s in syms if s]

        if 'get_logger' not in syms:
            # get_logger 가 포함되지 않은 mmcv.utils import → 그대로 유지 but warn
            new_lines.append(line)
            continue

        new_lines.append(f'{indent}# [migrated from mmcv.utils] {line.strip()}')
        changed = True

        # get_logger 외 다른 심볼 처리
        other_syms = [s for s in syms if s != 'get_logger']
        if other_syms:
            new_lines.append(
                f'{indent}# WARNING: 다음 심볼은 수동 확인 필요: {", ".join(other_syms)}'
            )

        if 'get_logger' not in injected_names and not _function_defined(lines, 'get_logger'):
            local_injects.append(('get_logger', LOCAL_INJECT_CODE['get_logger']))
            injected_names.add('get_logger')
            summary.append('  로컬 함수 주입: get_logger()')
    lines = new_lines

    # ── 패스 3: mmcv.mkdir_or_exist → mkdir_or_exist ─────────────────────────
    if any('mmcv.mkdir_or_exist' in ln for ln in lines):
        lines = [ln.replace('mmcv.mkdir_or_exist', 'mkdir_or_exist') for ln in lines]
        if not _already_imported(lines, 'mmengine.utils', 'mkdir_or_exist'):
            new_imports['mmengine.utils'].add('mkdir_or_exist')
        changed = True
        summary.append('  mmcv.mkdir_or_exist → mkdir_or_exist')

    # ── 패스 4: mmcv.scandir → scandir (아직 남아있는 경우) ─────────────────
    if any('mmcv.scandir' in ln for ln in lines):
        lines = [ln.replace('mmcv.scandir', 'scandir') for ln in lines]
        if not _already_imported(lines, 'mmengine.utils', 'scandir'):
            new_imports['mmengine.utils'].add('scandir')
        changed = True
        summary.append('  mmcv.scandir → scandir')

    if not changed:
        return MigrationResult(source, False, [])

    # ── 새 import 줄 구성 ───────────────────────────────────────────────────
    insert_idx = _find_import_block_end(lines)

    insertion: list[str] = []

    if new_imports:
        insertion.append('')
        for mod in sorted(new_imports):
            # 이미 존재하는 import 와 합치기
            syms = sorted(new_imports[mod])
            insertion.append(f'from {mod} import {", ".join(syms)}')

    if local_injects:
        insertion.append('')
        for _name, code in local_injects:
            insertion.append(code)

    if insertion:
        lines[insert_idx:insert_idx] = insertion

    return MigrationResult('\n'.join(lines), True, summary)


# ── 파일 수집 ──────────────────────────────────────────────────────────────────

def collect_py_files(root: Path):
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d not in SKIP_DIRS)
        for fname in sorted(filenames):
            if fname.endswith('.py') and fname not in SKIP_FILES:
                yield Path(dirpath) / fname


# ── 메인 ───────────────────────────────────────────────────────────────────────

def main() -> None:
    dry_run   = '--dry-run'   in sys.argv
    no_backup = '--no-backup' in sys.argv

    print('=' * 60)
    print('MMCV 1.x → MMEngine 2.x 자동 마이그레이션')
    print(f'Project root : {PROJECT_ROOT}')
    print(f'Mode         : {"DRY RUN (파일 미수정)" if dry_run else "MIGRATE"}')
    print('=' * 60)
    print()

    modified_count = 0
    error_count    = 0

    for path in collect_py_files(PROJECT_ROOT):
        try:
            source = path.read_text(encoding='utf-8')
            result = migrate_source(source, str(path))

            if not result.modified:
                continue

            modified_count += 1
            rel = path.relative_to(PROJECT_ROOT)
            print(f'[수정] {rel}')
            for msg in result.summary:
                print(msg)

            if not dry_run:
                if not no_backup:
                    shutil.copy2(path, str(path) + '.bak')
                path.write_text(result.source, encoding='utf-8')

        except Exception as exc:
            error_count += 1
            print(f'[오류] {path}: {exc}')

    print()
    print('─' * 60)
    print(f'수정된 파일 수 : {modified_count}')
    if error_count:
        print(f'오류           : {error_count}')
    if dry_run:
        print('DRY RUN - 실제 파일은 변경되지 않았습니다.')
        print('  적용하려면: python migrate_mmcv.py')
    else:
        if not no_backup:
            print('원본 파일은 .bak 확장자로 백업되었습니다.')
        print('마이그레이션 완료.')


if __name__ == '__main__':
    main()
