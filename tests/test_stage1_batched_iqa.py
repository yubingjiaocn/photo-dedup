"""Batched IQA: fewer model calls, identical per-image results.

The claim under test is narrow and checkable: for one guarded batch the IQA lanes
are called **once per distinct bounded input shape** instead of once per image,
every image still gets its own score and its own
``iqa_input_size``/``iqa_scale``/``sharpness``, and nothing is padded or reordered
to make that happen.

Why a fake metric rather than the real weights: MUSIQ and CLIP-IQA are 400 MB of
downloads and need CUDA, so CI cannot run them. The fake stands in for the *shape
contract* pyiqa actually has -- verified on the real weights and recorded in
``docs/STAGE1_THROUGHPUT.md``: a 4-D ``(B, C, H, W)`` tensor in, one score per row
out, equal shapes only. It is deliberately **position-sensitive** (its score
depends on the image's own pixels), so a backend that shuffled rows, reused one
row's score for the batch, or padded to a common size would fail here.
"""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from src import stage1_backends
from src.config import Config


class FakeMetric:
    """Stands in for a pyiqa metric: 4-D tensor in, one score per row out.

    Mirrors the real constraint set exactly. Equal shapes only (``torch.cat`` is
    impossible otherwise), lists rejected, and each row's score is a function of
    that row's own content, so per-image alignment is verifiable.
    """

    def __init__(self, scale: float = 1.0, offset: float = 0.0) -> None:
        self.scale = scale
        self.offset = offset
        self.calls = 0
        self.batch_sizes: list[int] = []
        self.shapes: list[tuple] = []

    def __call__(self, tensor):
        if not hasattr(tensor, "shape") or len(tensor.shape) != 4:
            raise AssertionError(f"metric needs a 4-D (B,C,H,W) tensor, got {type(tensor)}")
        self.calls += 1
        self.batch_sizes.append(int(tensor.shape[0]))
        self.shapes.append(tuple(int(v) for v in tensor.shape[1:]))
        # Mean of each row -> a per-image value that changes if rows are mixed up.
        means = tensor.reshape(int(tensor.shape[0]), -1).mean(dim=1)
        return (means * self.scale + self.offset).reshape(-1, 1)


class FakeProcessor:
    """Stand-in for ``BitImageProcessor``: fixed-size output, per-image separable.

    Verified against the real ``facebook/dinov2-base`` processor in
    ``docs/STAGE1_THROUGHPUT.md``: preprocessing one image at a time and stacking
    is bitwise identical to preprocessing the whole batch, which is the property
    ``TorchBackend`` checks at load time and this fake preserves.
    """

    def __init__(self, size: int = 8) -> None:
        self.size = size
        self.calls = 0

    def __call__(self, images, return_tensors="pt"):
        import torch

        self.calls += 1
        tensors = []
        for item in images:
            small = np.asarray(item.convert("RGB").resize((self.size, self.size)),
                               dtype=np.float32) / 255.0
            tensors.append(torch.from_numpy(small.transpose(2, 0, 1)).unsqueeze(0))
        return {"pixel_values": torch.cat(tensors, dim=0)}


class FakeYuNet:
    """Records that setInputSize/detect are only ever called from one thread."""

    def __init__(self) -> None:
        self.sizes: list[tuple] = []
        self.threads: set[int] = set()

    def setInputSize(self, size):  # noqa: N802 (OpenCV API name)
        import threading

        self.threads.add(threading.get_ident())
        self.sizes.append(tuple(size))

    def detect(self, _image):
        import threading

        self.threads.add(threading.get_ident())
        return 1, None


def torch_backend(monkeypatch, *, iqa_batch_size=4, iqa_max_long_edge=1920,
                  musiq=True, clipiqa=True, face=None):
    """A ``TorchBackend`` with real torch tensors but fake models.

    Only the weights are faked. ``prepare_cpu``, the shape grouping, the stacking,
    the single host transfer and the metadata assembly are the production code.
    """
    torch = pytest.importorskip("torch")
    backend = stage1_backends.TorchBackend.__new__(stage1_backends.TorchBackend)
    backend.torch = torch
    backend.cfg = Config({"features": {
        "iqa_max_long_edge": iqa_max_long_edge, "yunet_input_width": 640,
    }})
    backend.iqa_max_long_edge = iqa_max_long_edge
    backend.iqa_batch_size = iqa_batch_size
    backend.device = "cpu"
    backend.musiq = FakeMetric(scale=100.0) if musiq else None
    backend.clipiqa = FakeMetric(scale=1.0) if clipiqa else None
    backend._face = face
    backend.telemetry = None
    backend.processor = FakeProcessor()
    backend.model = None
    return backend


def image(width, height, seed=0):
    rng = np.random.default_rng(seed)
    return Image.fromarray(
        (rng.random((height, width, 3)) * 255).astype(np.uint8), mode="RGB")


class CountingSink:
    """Counter-only telemetry sink, so call counts are observable directly."""

    def __init__(self) -> None:
        self.counters: dict[str, int] = {}

    def phase(self, key, count=1):
        return _Null()

    def worker_phase(self, key, count=1):
        return _Null()

    def gpu_event_phase(self, key):
        return _Null()

    def count(self, key, value=1):
        self.counters[key] = self.counters.get(key, 0) + value


class _Null:
    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return False


# --- call counts ------------------------------------------------------------

def test_one_model_call_per_shape_group_not_per_image(monkeypatch):
    """Eight same-shape images at cap 4 -> 2 calls per lane, not 8."""
    backend = torch_backend(monkeypatch, iqa_batch_size=4)
    sink = CountingSink()
    backend.telemetry = sink
    prepared = [backend.prepare_cpu(image(320, 240, seed)) for seed in range(8)]

    results = backend.quality_prepared(prepared)

    assert len(results) == 8
    assert backend.musiq.calls == 2 and backend.musiq.batch_sizes == [4, 4]
    assert backend.clipiqa.calls == 2
    assert sink.counters["iqa_musiq_calls"] == 2
    assert sink.counters["iqa_musiq_images"] == 8
    assert sink.counters["iqa_clipiqa_calls"] == 2
    assert sink.counters["iqa_clipiqa_images"] == 8
    assert sink.counters["iqa_shape_groups"] == 2


def test_scores_match_the_single_image_path_exactly(monkeypatch):
    """Batched output must equal what per-image calls produced, image by image."""
    backend = torch_backend(monkeypatch, iqa_batch_size=4)
    images = [image(320, 240, seed) for seed in range(6)]
    prepared = [backend.prepare_cpu(item) for item in images]

    batched = backend.quality_prepared(prepared)

    # Reconstruct the old path: one metric call per image, batch size 1.
    solo = torch_backend(monkeypatch, iqa_batch_size=1)
    per_image = [solo.quality_prepared([solo.prepare_cpu(item)])[0] for item in images]

    assert solo.musiq.calls == 6 and solo.musiq.batch_sizes == [1] * 6
    for (batch_score, batch_meta), (solo_score, solo_meta) in zip(batched, per_image):
        assert batch_score == pytest.approx(solo_score, abs=1e-5)
        assert batch_meta["clipiqa"] == pytest.approx(solo_meta["clipiqa"], abs=1e-6)
        assert batch_meta["sharpness"] == pytest.approx(solo_meta["sharpness"], rel=1e-9)
        assert batch_meta["iqa_input_size"] == solo_meta["iqa_input_size"]
        assert batch_meta["iqa_scale"] == solo_meta["iqa_scale"]


def test_scores_are_not_shuffled_within_a_group(monkeypatch):
    """Distinct images must keep distinct scores in the batch's own order."""
    backend = torch_backend(monkeypatch, iqa_batch_size=8)
    # Constant-colour images make the expected ordering unambiguous.
    images = [Image.new("RGB", (64, 48), (level, level, level))
              for level in (10, 60, 120, 200, 250)]
    prepared = [backend.prepare_cpu(item) for item in images]

    scores = [score for score, _ in backend.quality_prepared(prepared)]

    assert backend.musiq.calls == 1
    assert scores == sorted(scores)                     # strictly increasing input
    assert len(set(scores)) == len(scores)
    for level, score in zip((10, 60, 120, 200, 250), scores):
        assert score == pytest.approx(100.0 * level / 255.0, abs=0.5)


# --- heterogeneous shapes ---------------------------------------------------

def test_heterogeneous_sizes_are_grouped_never_padded(monkeypatch):
    """Mixed orientations -> one call per shape, each at its own resolution.

    Zero-padding to a common size would be the tempting shortcut; measured on the
    real weights it moves MUSIQ by 1.70 points and CLIP-IQA by 0.116
    (``docs/STAGE1_THROUGHPUT.md``), so the shapes handed to the metric must stay
    exactly the shapes the single-image path used.
    """
    backend = torch_backend(monkeypatch, iqa_batch_size=8)
    sink = CountingSink()
    backend.telemetry = sink
    sizes = [(320, 240), (240, 320), (320, 240), (160, 160), (240, 320)]
    prepared = [backend.prepare_cpu(image(w, h, index))
                for index, (w, h) in enumerate(sizes)]

    results = backend.quality_prepared(prepared)

    assert backend.musiq.calls == 3                      # 3 distinct shapes
    assert sink.counters["iqa_shape_groups"] == 3
    assert sorted(backend.musiq.batch_sizes) == [1, 2, 2]
    # Every call saw a real, unpadded shape from the input set.
    assert set(backend.musiq.shapes) == {(3, 240, 320), (3, 320, 240), (3, 160, 160)}
    for (width, height), (_, meta) in zip(sizes, results):
        assert meta["iqa_input_size"] == [width, height]
        assert meta["iqa_scale"] == 1.0


def test_group_order_is_deterministic_and_traceable():
    """Shape groups appear in first-appearance order; indices keep batch order."""
    prepared = [
        {"iqa_array": np.zeros((4, 6, 3), dtype=np.float32)},     # 0 shape A
        {"iqa_array": np.zeros((6, 4, 3), dtype=np.float32)},     # 1 shape B
        {"iqa_array": np.zeros((4, 6, 3), dtype=np.float32)},     # 2 shape A
        {"iqa_array": np.zeros((4, 6, 3), dtype=np.float32)},     # 3 shape A
    ]
    assert list(stage1_backends._shape_groups(prepared, 4)) == [[0, 2, 3], [1]]
    assert list(stage1_backends._shape_groups(prepared, 2)) == [[0, 2], [3], [1]]
    assert list(stage1_backends._shape_groups(prepared, 1)) == [[0], [2], [3], [1]]


# --- partial batches, single images, disabled lanes -------------------------

def test_partial_final_batch_is_scored_normally(monkeypatch):
    """A last batch of 1 (cap 4) is still exactly one call and one score."""
    backend = torch_backend(monkeypatch, iqa_batch_size=4)
    prepared = [backend.prepare_cpu(image(320, 240, 9))]

    results = backend.quality_prepared(prepared)

    assert len(results) == 1
    assert backend.musiq.calls == 1 and backend.musiq.batch_sizes == [1]
    assert results[0][1]["iqa_input_size"] == [320, 240]


def test_disabled_lanes_are_not_called_and_report_no_clipiqa(monkeypatch):
    backend = torch_backend(monkeypatch, clipiqa=False)
    prepared = [backend.prepare_cpu(image(320, 240, 1))]
    score, meta = backend.quality_prepared(prepared)[0]
    assert backend.clipiqa is None and "clipiqa" not in meta
    assert meta["musiq"] == score and backend.musiq.calls == 1


# --- bounded IQA input is reused, not recomputed ---------------------------

def test_the_bounded_rgb_array_backs_both_the_tensor_and_sharpness(monkeypatch):
    """One bounded resize serves the IQA tensor and the sharpness measure."""
    backend = torch_backend(monkeypatch)
    resizes: list[tuple] = []
    original = Image.Image.resize

    def counting_resize(self, size, *args, **kwargs):
        resizes.append(tuple(size))
        return original(self, size, *args, **kwargs)

    monkeypatch.setattr(Image.Image, "resize", counting_resize)
    prepared = backend.prepare_cpu(image(4000, 3000, 5))

    # One resize for the (fixed-size) embedding preprocess and one for the IQA
    # input, in that order. Sharpness adds none: it reuses the IQA array.
    assert resizes == [(8, 8), (1920, 1440)]
    assert prepared["iqa_array"].shape == (1440, 1920, 3)
    assert prepared["iqa_input_size"] == [1920, 1440]
    expected = float(np.asarray(prepared["iqa_array"]).mean(axis=2).mean())
    assert 0.0 < expected < 1.0                          # array is 0..1 normalised
    assert prepared["sharpness"] > 0


def test_iqa_scale_and_size_survive_the_batch_for_mixed_sources(monkeypatch):
    backend = torch_backend(monkeypatch, iqa_max_long_edge=1000, iqa_batch_size=4)
    prepared = [backend.prepare_cpu(image(4000, 2000, 1)),
                backend.prepare_cpu(image(500, 250, 2))]

    results = backend.quality_prepared(prepared)

    assert [meta["iqa_input_size"] for _, meta in results] == [[1000, 500], [500, 250]]
    assert [meta["iqa_scale"] for _, meta in results] == [0.25, 1.0]


# --- no per-image host synchronisation ------------------------------------

def test_output_crosses_to_the_host_once_per_call_not_once_per_image(monkeypatch):
    """`.item()` per image is what starved the GPU; one transfer per call now."""
    torch = pytest.importorskip("torch")
    backend = torch_backend(monkeypatch, iqa_batch_size=4, clipiqa=False)
    prepared = [backend.prepare_cpu(image(320, 240, seed)) for seed in range(4)]

    transfers: list[int] = []
    original_cpu = torch.Tensor.cpu

    def counting_cpu(self, *args, **kwargs):
        transfers.append(self.numel())
        return original_cpu(self, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "cpu", counting_cpu)
    results = backend.quality_prepared(prepared)

    assert len(results) == 4
    # One transfer for the one MUSIQ call, and it carried all four scores.
    assert transfers == [4]


# --- faces stay on the main thread ---------------------------------------

def test_face_input_is_bounded_and_scale_is_recorded(monkeypatch):
    face = FakeYuNet()
    backend = torch_backend(monkeypatch, face=face)
    prepared = backend.prepare_cpu(image(1280, 960, 1))
    assert prepared["face_bgr"] is not None
    assert max(prepared["face_bgr"].shape[:2]) == 640
    assert prepared["face_scale"] == pytest.approx(0.5)
    backend.faces_prepared(prepared)
    assert face.sizes == [(640, 480)]


# --- stub backend keeps the same contract --------------------------------

def test_stub_backend_exposes_the_same_prepared_api():
    backend = stage1_backends.StubBackend(iqa_max_long_edge=100)
    prepared = [backend.prepare_cpu(image(400, 200, seed)) for seed in range(3)]

    embeddings = backend.embed_prepared(prepared)
    qualities = backend.quality_prepared(prepared)

    assert embeddings.shape == (3, stage1_backends.EMBED_DIM)
    assert len(qualities) == 3
    for _, meta in qualities:
        assert meta["iqa_input_size"] == [100, 50]
        assert meta["iqa_scale"] == 0.25
        assert meta["backend"] == "stub" and meta["clipiqa"] is None


def test_stub_single_image_shims_match_the_batched_path():
    backend = stage1_backends.StubBackend()
    subject = image(320, 240, 4)

    shim_score, shim_meta = backend.quality(subject)
    batch_score, batch_meta = backend.quality_prepared([backend.prepare_cpu(subject)])[0]

    assert shim_score == pytest.approx(batch_score)
    assert shim_meta == batch_meta
    assert backend.embed_batch([subject]).shape == (1, stage1_backends.EMBED_DIM)


def test_stub_embedding_is_unchanged_by_per_image_preparation():
    """Per-image preparation must not alter the stub's embedding semantics."""
    backend = stage1_backends.StubBackend()
    images = [image(320, 240, seed) for seed in range(4)]

    together = backend.embed_prepared([backend.prepare_cpu(item) for item in images])
    apart = np.vstack([backend.embed_prepared([backend.prepare_cpu(item)])
                       for item in images])

    assert np.array_equal(together, apart)
    norms = np.linalg.norm(together, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-6)
