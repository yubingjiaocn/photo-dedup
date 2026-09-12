import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/research"))
from probe_native_dense_tracks import compare, dense


def sample():
    points = np.array(
        [[125 + x * 45, 105 + y * 38] for y in range(10) for x in range(10)], np.float32
    )
    return {"vectors": np.eye(100, dtype=np.float32), "points": points}


def box():
    return {"box": [100, 80, 600, 500]}


def test_dense_native_translation_is_supported_without_identity_authority():
    a = sample()
    b = sample()
    b["points"] = b["points"] + [7, -3]
    result = compare(a, b, box(), box())
    assert result["eligible"] and result["inliers"] == 100
    assert not result["identity_authority"]
    assert result["whole_body_completeness"] == "unknown"


def test_repeated_local_appearance_is_not_unique_identity():
    a = sample()
    b = sample()
    a["vectors"][:] = 0.1
    b["vectors"][:] = 0.1
    result = compare(a, b, box(), box())
    assert not result["eligible"] and result["matches"] == 0


def test_patch_drift_fails_even_when_descriptors_match():
    a = sample()
    b = sample()
    b["points"] = np.random.default_rng(8).permutation(b["points"])
    assert not compare(a, b, box(), box())["eligible"]


def test_tiny_local_fragment_is_not_full_extent_support():
    a = sample()
    b = sample()
    a["points"] /= 20
    b["points"] /= 20
    result = compare(a, b, box(), box())
    assert not result["eligible"]
    assert max(result["coverage"]) < 0.15


def test_capacity_abstains_instead_of_resizing_native_pixels():
    backend = SimpleNamespace(
        torch=None, model=SimpleNamespace(config=SimpleNamespace(patch_size=14))
    )
    image = Image.new("RGB", (3000, 3000))
    value, reason = dense(backend, image, {"box": [0, 0, 3000, 3000]})
    assert value is None and reason == "NATIVE_DENSE_CAPACITY_OR_SIZE"
