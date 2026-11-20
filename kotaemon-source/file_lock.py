"""Cross-platform non-blocking advisory file locks."""

from contextlib import contextmanager
import os
from pathlib import Path


class FileLockError(RuntimeError):
    pass


@contextmanager
def exclusive_file_lock(path, unavailable_message=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lease = path.open("a+b")
    acquired = False
    try:
        if os.name == "nt":
            import msvcrt

            lease.seek(0, os.SEEK_END)
            if lease.tell() == 0:
                lease.write(b"\0")
                lease.flush()
            lease.seek(0)
            try:
                msvcrt.locking(lease.fileno(), msvcrt.LK_NBLCK, 1)
            except (BlockingIOError, PermissionError) as error:
                message = unavailable_message or f"Resource is already locked: {path}"
                raise FileLockError(message) from error
        else:
            import fcntl

            try:
                fcntl.flock(lease.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except (BlockingIOError, PermissionError) as error:
                message = unavailable_message or f"Resource is already locked: {path}"
                raise FileLockError(message) from error
        acquired = True
        yield
    finally:
        if acquired:
            if os.name == "nt":
                lease.seek(0)
                msvcrt.locking(lease.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(lease.fileno(), fcntl.LOCK_UN)
        lease.close()
