"""Tests for motion-photo detection (embedded + paired)."""

from pathlib import Path

from src import motion_photo as mp


# --- trailing MP4 signature -------------------------------------------------

def test_has_trailing_mp4_true():
    jpeg = b"\xff\xd8\xff\xe0JFIF...image bytes...\xff\xd9"
    appended = b"\x00\x00\x00\x18ftypmp42\x00\x00\x00\x00mp42isom" + b"\x00" * 64
    assert mp.has_trailing_mp4(jpeg + appended) is True


def test_has_trailing_mp4_plain_jpeg_false():
    jpeg = b"\xff\xd8\xff\xe0JFIF...only an image...\xff\xd9"
    assert mp.has_trailing_mp4(jpeg) is False


def test_has_trailing_mp4_pure_mp4_false():
    # ftyp at offset 4 but NO preceding JPEG EOI -> it's a real mp4, not embedded
    pure_mp4 = b"\x00\x00\x00\x18ftypisom" + b"\x00" * 64
    assert mp.has_trailing_mp4(pure_mp4) is False


# --- embedded detection on real files ---------------------------------------

def test_detect_embedded_via_xmp_marker(tmp_path: Path):
    p = tmp_path / "IMG_1.jpg"
    p.write_bytes(b"\xff\xd8" + b"<x:xmpmeta GCamera:MicroVideo=\"1\">" + b"\xff\xd9")
    assert mp.detect_embedded_motion(p) is True


def test_detect_embedded_via_trailing_scan(tmp_path: Path):
    p = tmp_path / "IMG_2.jpg"
    p.write_bytes(b"\xff\xd8\xff\xd9" + b"\x00\x00\x00\x18ftypmp42" + b"\x00" * 32)
    # header has no marker -> only the full scan can find it
    assert mp.detect_embedded_motion(p, full_scan_max_bytes=10_000_000) is True
    # with scan disabled, header-only sees nothing
    assert mp.detect_embedded_motion(p, full_scan_max_bytes=0) is False


def test_detect_embedded_plain_false(tmp_path: Path):
    p = tmp_path / "IMG_3.jpg"
    p.write_bytes(b"\xff\xd8" + b"just a normal jpeg" + b"\xff\xd9")
    assert mp.detect_embedded_motion(p, full_scan_max_bytes=10_000_000) is False


# --- paired detection -------------------------------------------------------

def test_classify_paired_mp4(tmp_path: Path):
    jpg = tmp_path / "IMG_20260101_120000.jpg"
    mp4 = tmp_path / "IMG_20260101_120000.mp4"
    jpg.write_bytes(b"\xff\xd8\xff\xd9")
    mp4.write_bytes(b"\x00\x00\x00\x18ftypmp42")
    dir_files = [jpg, mp4]

    kind_j, partner_j = mp.classify_file(jpg, dir_files)
    kind_m, partner_m = mp.classify_file(mp4, dir_files)
    assert kind_j == "jpg_motion" and Path(partner_j) == mp4
    assert kind_m == "mp4_paired" and Path(partner_m) == jpg


def test_classify_dot_mp_sidecar(tmp_path: Path):
    # Samsung/Xiaomi style: IMG_x.jpg + IMG_x.jpg.MP
    jpg = tmp_path / "PXL_1.jpg"
    side = tmp_path / "PXL_1.jpg.MP"
    jpg.write_bytes(b"\xff\xd8\xff\xd9")
    side.write_bytes(b"\x00\x00\x00\x18ftypmp42")
    dir_files = [jpg, side]
    kind_j, partner_j = mp.classify_file(jpg, dir_files)
    assert kind_j == "jpg_motion" and Path(partner_j) == side


def test_classify_standalone(tmp_path: Path):
    jpg = tmp_path / "solo.jpg"
    vid = tmp_path / "clip.mp4"
    jpg.write_bytes(b"\xff\xd8\xff\xd9")
    vid.write_bytes(b"\x00\x00\x00\x18ftypmp42")
    dir_files = [jpg, vid]
    assert mp.classify_file(jpg, dir_files)[0] == "jpg"
    assert mp.classify_file(vid, dir_files)[0] == "mp4_only"


def test_ambiguous_jpg_jpeg_do_not_claim_same_sidecar(tmp_path: Path):
    jpg = tmp_path / "IMG_1.jpg"
    jpeg = tmp_path / "IMG_1.jpeg"
    video = tmp_path / "IMG_1.mp4"
    for p in (jpg, jpeg, video):
        p.write_bytes(b"x")
    files = [jpg, jpeg, video]
    assert mp.classify_file(jpg, files) == ("jpg", None)
    assert mp.classify_file(jpeg, files) == ("jpg", None)
    assert mp.classify_file(video, files) == ("mp4_only", None)
