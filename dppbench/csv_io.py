"""Bounded retries for transient local/mounted CSV read failures."""
import errno
import time
import warnings

import pandas as pd


def read_csv_retry(path, *, attempts=3, **kwargs):
    """Reopen on transient read failures; never skip rows or hide malformed CSV."""
    if attempts < 1:
        raise ValueError("attempts must be positive")
    transient_errno = {errno.EIO, errno.EINTR, errno.EAGAIN, errno.ETIMEDOUT}
    if hasattr(errno, "ESTALE"):
        transient_errno.add(errno.ESTALE)
    for attempt in range(attempts):
        try:
            return pd.read_csv(path, **kwargs)
        except (OSError, pd.errors.ParserError) as exc:
            retryable = (
                isinstance(exc, OSError) and exc.errno in transient_errno
            ) or (
                isinstance(exc, pd.errors.ParserError)
                and "Calling read(nbytes) on source failed" in str(exc)
            )
            if not retryable or attempt + 1 == attempts:
                raise
            warnings.warn(
                f"CSV read failed for {path!s}; retry {attempt + 1}/{attempts - 1}: {exc}",
                RuntimeWarning, stacklevel=2,
            )
            time.sleep(0.2 * (attempt + 1))
