"""Small local-only learning I/O primitives. No provider or execution authority."""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import tempfile
from pathlib import Path, PurePosixPath


def canonical(value):
    return json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()


def sha256(value):
    return hashlib.sha256(value).hexdigest()


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_bytes())


def relative_path(name):
    if not isinstance(name, str) or not name or "\\" in name or ":" in name:
        raise ValueError("invalid relative path")
    path = PurePosixPath(name)
    if path.is_absolute() or any(p in ("", ".", "..") for p in name.split("/")):
        raise ValueError("path escape")
    return path


def safe_path(root, name):
    relative_path(name)
    root = Path(root).resolve()
    path = root / name
    if any(p.is_symlink() for p in [path, *path.parents] if p != root.parent):
        raise ValueError("symlink not allowed")
    if not path.resolve().is_relative_to(root):
        raise ValueError("path escape")
    return path


def atomic_json(path, value):
    path = Path(path)
    if path.is_symlink():
        raise ValueError("symlink not allowed")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".pending-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(canonical(value) + b"\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, path)
        if os.name == "posix":
            directory = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


@contextlib.contextmanager
def file_lock(path):
    """OS advisory lock across processes; one-byte Windows lock, flock on Unix."""
    path = Path(path)
    if path.is_symlink():
        raise ValueError("lock symlink not allowed")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as stream:
        if os.name == "nt":
            import msvcrt

            stream.seek(0)
            if not stream.read(1):
                stream.write(b"0")
                stream.flush()
            stream.seek(0)
            msvcrt.locking(stream.fileno(), msvcrt.LK_LOCK, 1)
        else:
            import fcntl

            fcntl.flock(stream, fcntl.LOCK_EX)
        try:
            yield
        finally:
            if os.name == "nt":
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream, fcntl.LOCK_UN)


@contextlib.contextmanager
def staged_directory(destination):
    """Publish a complete new directory under a cooperating writer lock.

    Exceptions clean staging; process death leaves a visibly .staging-* orphan
    which is never a deliverable and can be removed after confirming no writer.
    """
    import shutil
    import uuid

    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with file_lock(destination.parent / ("." + destination.name + ".publish.lock")):
        if destination.exists() or destination.is_symlink():
            raise ValueError("destination must be new")
        staging = destination.with_name("." + destination.name + ".staging-" + uuid.uuid4().hex)
        try:
            yield staging
            if destination.exists() or destination.is_symlink():
                raise ValueError("destination must be new")
            os.rename(staging, destination)
        finally:
            if staging.exists():
                shutil.rmtree(staging)
