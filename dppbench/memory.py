"""Best-effort preflight checks, not a replacement for a cgroup hard limit."""
from pathlib import Path
import os


def memory_headroom():
    limits = []
    try:
        import psutil
        limits.append(int(psutil.virtual_memory().available))
    except (ImportError, OSError):
        pass
    # Check ancestors too: a user.slice can be tighter than the experiment scope.
    root = Path('/sys/fs/cgroup')
    try:
        entry = next(line[3:] for line in Path('/proc/self/cgroup').read_text().splitlines()
                     if line.startswith('0::'))
        current = root / entry.lstrip('/')
        while current == root or root in current.parents:
            maximum = (current / 'memory.max').read_text().strip()
            if maximum != 'max':
                used = int((current / 'memory.current').read_text())
                limits.append(max(0, int(maximum) - used))
            if current == root:
                break
            current = current.parent
    except (OSError, ValueError, StopIteration):
        pass
    return min(limits) if limits else None


def require_memory_headroom(additional_bytes, stage):
    available = memory_headroom()
    if os.environ.get('DPPBENCH_MEMORY_TRACE') == '1':
        headroom = 'unknown' if available is None else f'{available / 2**30:.2f}'
        print(f'[memory preflight] stage={stage} additional_gib={additional_bytes / 2**30:.2f} '
              f'headroom_gib={headroom}', flush=True)
    if available is not None and additional_bytes > available:
        raise MemoryError(
            f'{stage}: estimated additional allocation {additional_bytes / 2**30:.2f} GiB '
            f'exceeds available memory headroom {available / 2**30:.2f} GiB. '
            'Refusing dense allocation; pipeline semantics were not changed.')
