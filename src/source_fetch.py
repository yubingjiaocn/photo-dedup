"""Future cloud original fetch boundary. No OAuth, network or provider implementation.

A fetcher must receive credentials out-of-band and return bytes verified against
content_sha256. A temporary URL is never an asset ID. M2 must verify current
Google Photos/Picker capabilities and explicit user authorization first.
"""
from typing import Protocol


class OriginalFetcher(Protocol):
    def fetch(self, *, provider: str, asset_id: str, content_sha256: str) -> bytes:
        """Fetch only one explicitly requested stable asset, with no library scan."""
        ...


class GooglePhotosFetchUnavailable:
    def fetch(self, *, provider: str, asset_id: str, content_sha256: str) -> bytes:
        raise NotImplementedError(
            "Google Photos fetch requires M2 verification of 2026 API/Picker permissions; no network request made"
        )
