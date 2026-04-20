"""
MMCV 1.x → MMEngine 2.x Monkey-Patch Compatibility Shim
=========================================================
PyTorch 2.4.1 / MMCV 2.2.0 / MMEngine 환경에서
기존 코드를 수정하지 않고 구동하기 위한 런타임 패치입니다.

사용법 – 메인 스크립트(train.py / test.py / train_refsr.py) 최상단에 삽입:

    import mmcv_compat  # noqa: F401

이 한 줄만으로 아래 모든 리다이렉션이 활성화됩니다.
"""
from __future__ import annotations

import datetime
import logging
import os
import sys
import types

# ── 1. get_time_str ──────────────────────────────────────────────────────────
# mmcv.runner.get_time_str 은 mmengine 에 존재하지 않음 → 직접 구현
def _get_time_str() -> str:
    return datetime.datetime.now().strftime('%Y%m%d_%H%M%S')


# ── 2. master_only ───────────────────────────────────────────────────────────
# mmcv 1.x: mmcv.runner.master_only
# mmengine:  mmengine.dist.master_only  (동일한 데코레이터 인터페이스)
try:
    from mmengine.dist import master_only as _master_only
except ImportError:
    def _master_only(func):  # type: ignore[misc]
        """단일 프로세스 환경용 폴백 – 항상 실행."""
        return func


# ── 3. get_dist_info ─────────────────────────────────────────────────────────
# 반환값 (rank: int, world_size: int) – mmcv 1.x 와 동일
try:
    from mmengine.dist import get_dist_info as _get_dist_info
except ImportError:
    def _get_dist_info():  # type: ignore[misc]
        return 0, 1


# ── 4. init_dist ─────────────────────────────────────────────────────────────
# mmengine.dist.init_dist(launcher, **kwargs) – 시그니처 호환
# mmcv 1.x 와 달리 backend 키워드는 kwargs 로 전달됨 (동일)
try:
    from mmengine.dist import init_dist as _init_dist
except ImportError:
    def _init_dist(launcher: str, **kwargs):  # type: ignore[misc]
        import torch.distributed as dist
        if launcher == 'pytorch':
            dist.init_process_group(backend=kwargs.get('backend', 'nccl'))
        elif launcher == 'slurm':
            dist.init_process_group(backend=kwargs.get('backend', 'nccl'))
        else:
            raise NotImplementedError(
                f"Launcher '{launcher}' not supported in compatibility fallback."
            )


# ── 5. scandir ───────────────────────────────────────────────────────────────
# mmengine.utils.scandir 은 mmcv.scandir 과 동일한 인터페이스
try:
    from mmengine.utils import scandir as _scandir
except ImportError:
    def _scandir(dir_path, suffix=None, recursive=False):  # type: ignore[misc]
        for entry in os.scandir(dir_path):
            if entry.is_file():
                name = entry.name
                if suffix is None:
                    yield name
                elif isinstance(suffix, str) and name.endswith(suffix):
                    yield name
                elif isinstance(suffix, (list, tuple)) and any(
                    name.endswith(s) for s in suffix
                ):
                    yield name


# ── 6. mkdir_or_exist ────────────────────────────────────────────────────────
try:
    from mmengine.utils import mkdir_or_exist as _mkdir_or_exist
except ImportError:
    def _mkdir_or_exist(dir_name: str) -> None:  # type: ignore[misc]
        os.makedirs(dir_name, exist_ok=True)


# ── 7. get_logger (mmcv.utils.get_logger 래퍼) ──────────────────────────────
# mmengine 은 MMLogger 클래스를 사용하지만, 기존 코드는 표준 logging.Logger 를 기대함
# → 표준 Logger 를 반환하는 래퍼로 동작 호환성 유지
def _get_logger(
    name: str,
    log_file: str | None = None,
    log_level: str = 'INFO',
    **kwargs,
) -> logging.Logger:
    """mmcv.utils.get_logger 호환 래퍼."""
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
        log_dir = os.path.dirname(os.path.abspath(log_file))
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        fh = logging.FileHandler(log_file, 'w')
        fh.setLevel(level)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger


# ============================================================================
# mmcv.runner 모듈 패치
# ============================================================================
_runner_mod = types.ModuleType('mmcv.runner')
_runner_mod.get_time_str  = _get_time_str
_runner_mod.master_only   = _master_only
_runner_mod.get_dist_info = _get_dist_info
_runner_mod.init_dist     = _init_dist

try:
    # mmcv 2.x 에서는 mmcv.runner 가 없으므로 ImportError 발생
    import mmcv.runner as _real_runner  # type: ignore[import]
    # 실제 모듈이 있다면 누락된 심볼만 패치
    for _sym, _val in [
        ('get_time_str',  _get_time_str),
        ('master_only',   _master_only),
        ('get_dist_info', _get_dist_info),
        ('init_dist',     _init_dist),
    ]:
        if not hasattr(_real_runner, _sym):
            setattr(_real_runner, _sym, _val)
    sys.modules['mmcv.runner'] = _real_runner
except (ImportError, ModuleNotFoundError, AttributeError):
    # mmcv.runner 자체가 없으면 가짜 모듈을 등록
    sys.modules['mmcv.runner'] = _runner_mod


# ============================================================================
# mmcv.utils 모듈 패치
# ============================================================================
_utils_mod = types.ModuleType('mmcv.utils')
_utils_mod.get_logger = _get_logger  # type: ignore[attr-defined]

try:
    import mmcv.utils as _real_utils  # type: ignore[import]
    if not hasattr(_real_utils, 'get_logger'):
        _real_utils.get_logger = _get_logger
    sys.modules['mmcv.utils'] = _real_utils
except (ImportError, ModuleNotFoundError, AttributeError):
    sys.modules['mmcv.utils'] = _utils_mod


# ============================================================================
# 최상위 mmcv 객체 패치
# ============================================================================
try:
    import mmcv as _mmcv  # type: ignore[import]

    # scandir / mkdir_or_exist 가 없으면 추가
    if not hasattr(_mmcv, 'scandir'):
        _mmcv.scandir = _scandir
    if not hasattr(_mmcv, 'mkdir_or_exist'):
        _mmcv.mkdir_or_exist = _mkdir_or_exist

    # runner / utils 서브모듈 속성 바인딩
    _mmcv.runner = sys.modules.get('mmcv.runner', _runner_mod)
    _mmcv.utils  = sys.modules.get('mmcv.utils',  _utils_mod)

except (ImportError, ModuleNotFoundError):
    # mmcv 자체가 없는 극단적 환경에서도 import 구문이 깨지지 않도록
    _mmcv_stub = types.ModuleType('mmcv')
    _mmcv_stub.runner        = _runner_mod
    _mmcv_stub.utils         = _utils_mod
    _mmcv_stub.scandir       = _scandir
    _mmcv_stub.mkdir_or_exist = _mkdir_or_exist
    sys.modules['mmcv']        = _mmcv_stub
    sys.modules['mmcv.runner'] = _runner_mod
    sys.modules['mmcv.utils']  = _utils_mod
