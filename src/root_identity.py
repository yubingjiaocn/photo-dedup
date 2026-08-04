r"""Normalised photo-root identity: the vocabulary every scope check shares.

Split out of :mod:`src.root_scope` (which owns the *binding* of an output
directory and the fail-closed gates) so that this file can stay pure: no
database, no filesystem, no I/O of any kind. A key must be computable for a path
recorded on another machine, or for a drive that is currently unplugged.

Normalisation rules (Windows-first, because that is the target platform)
-----------------------------------------------------------------------
``E:\Photos\2026``, ``E:/Photos/2026/`` and ``e:\photos\2026`` are one root.
Separators collapse to ``/``; a trailing separator is dropped; on
case-insensitive filesystems (Windows, macOS) the whole key is casefolded, and a
Windows-looking path is always treated that way regardless of the host OS, so a
DB written on Windows stays comparable when inspected from Linux. Prefix
comparisons always append ``/``, so ``E:/Photos/2026`` is never a parent of
``E:/Photos/2026extra``.
"""

from __future__ import annotations

import os
import sys
import uuid
from dataclasses import dataclass
from pathlib import PurePosixPath, PureWindowsPath
from typing import Any, Dict, Optional, Tuple


class ParameterError(ValueError):
    """Invalid combination of parameters. Raised before any mutation."""


# --- normalisation ---------------------------------------------------------

def _windows_like(text: str) -> bool:
    """True when the string looks like a Windows path (drive letter or UNC)."""
    if len(text) >= 2 and text[1] == ":" and text[0].isalpha():
        return True
    return text.startswith("\\\\") or text.startswith("//")


def _case_insensitive() -> bool:
    """True on filesystems that compare names case-insensitively by default."""
    return os.name == "nt" or sys.platform == "darwin"


def normalize(path: str | os.PathLike) -> str:
    """Normalised comparison key for a root or a file path.

    Never touches the filesystem, by design (see the module docstring).
    """
    raw = str(path).strip().strip("\x00")
    if not raw:
        raise ParameterError("path must not be empty")
    windows = _windows_like(raw) or os.name == "nt"
    if windows:
        pure: Any = PureWindowsPath(raw)
        text = str(pure).replace("\\", "/")
    else:
        pure = PurePosixPath(raw)
        text = str(pure)
    # Collapse duplicate separators without eating a UNC/POSIX root prefix.
    lead = "//" if text.startswith("//") else ("/" if text.startswith("/") else "")
    body = "/".join(part for part in text.split("/") if part)
    text = lead + body
    if len(text) > 1 and text.endswith("/"):
        text = text[:-1]
    if windows or _case_insensitive():
        return text.casefold()
    # POSIX: paths are case-sensitive, so the literal text is the key.
    return text


def prefix_of(key: str) -> str:
    """Directory prefix used for containment tests (always ends with ``/``)."""
    return key if key.endswith("/") else key + "/"


def contains(root_key: str, candidate_key: str) -> bool:
    """True when ``candidate_key`` is ``root_key`` itself or below it."""
    return candidate_key == root_key or candidate_key.startswith(prefix_of(root_key))


def relation(requested_key: str, recorded_key: str) -> str:
    """How the requested root relates to the recorded one.

    ``same`` | ``child`` (requested is inside recorded) | ``parent``
    (requested contains recorded) | ``disjoint``.
    """
    if requested_key == recorded_key:
        return "same"
    if contains(recorded_key, requested_key):
        return "child"
    if contains(requested_key, recorded_key):
        return "parent"
    return "disjoint"


def new_run_id() -> str:
    return uuid.uuid4().hex


# --- scope object ----------------------------------------------------------

@dataclass(frozen=True)
class RootScope:
    """The root an output directory is bound to, plus the current run id."""

    key: Optional[str]
    path: Optional[str]
    run_id: Optional[str] = None

    @property
    def bound(self) -> bool:
        return bool(self.key)

    def contains_path(self, path: str | os.PathLike) -> bool:
        if not self.bound:
            return True
        return contains(str(self.key), normalize(path))

    def clause(self, alias: str = "f", name: str = "scope") -> Tuple[str, Dict[str, Any]]:
        """SQL predicate + named params restricting rows to this root.

        Unbound scope yields the constant ``1`` so callers can always inline it.
        A row with a NULL ``path_key`` (impossible after migration, but cheap to
        be explicit about) is treated as out of scope rather than silently kept.
        """
        if not self.bound:
            return "1", {}
        prefix = prefix_of(str(self.key))
        return (
            f"(COALESCE({alias}.path_key, '') = :{name}_key"
            f" OR substr(COALESCE({alias}.path_key, ''), 1, :{name}_len) = :{name}_prefix)",
            {f"{name}_key": self.key, f"{name}_len": len(prefix), f"{name}_prefix": prefix},
        )

    def describe(self) -> str:
        if not self.bound:
            return "unbound (legacy output directory)"
        return f"{self.path} [key={self.key}]"


UNBOUND = RootScope(key=None, path=None, run_id=None)


def requested_scope(root: str | os.PathLike) -> RootScope:
    """The scope a caller is asking for (validated, not yet verified/bound)."""
    return RootScope(key=normalize(root), path=str(root))
