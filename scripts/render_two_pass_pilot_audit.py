"""Local human-audit views of REJECTED raw maps, never model inputs/enrichment."""

from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import two_pass_review as tp


def render(root):
    for case in sorted(root.glob("*-G*")):
        packet = json.loads((case / "pass-A-raw.json").read_text())
        if "regions" not in packet:
            continue
        date = case.name[:10]
        inventory = (
            Path(
                "/home/ubuntu/photo-dedup-eval/astra-regime-validation-20260910/development"
            )
            / date
            / "inventory.sqlite"
        )
        images = {}
        with sqlite3.connect(f"file:{inventory}?mode=ro", uri=True) as conn:
            for r in packet["regions"]:
                fid = r["frame"]["file_id"]
                if fid not in images:
                    path = Path(
                        conn.execute(
                            "SELECT path FROM files WHERE id=?", (fid,)
                        ).fetchone()[0]
                    )
                    assert tp.sha(path.read_bytes()) == r["frame"]["content_sha256"]
                    with Image.open(path) as im:
                        images[fid] = ImageOps.exif_transpose(im).convert("RGB")
        assets = []
        for fid, im in images.items():
            shown = im.copy()
            shown.thumbnail((1536, 1536))
            draw = ImageDraw.Draw(shown)
            for r in packet["regions"]:
                if r["frame"]["file_id"] == fid:
                    box = tp._pixels(shown, r["box"])
                    color = "red" if r["role"] == "key_subject" else "yellow"
                    draw.rectangle(box, outline=color, width=3)
                    draw.text(
                        (box[0] + 3, box[1] + 3),
                        f"{r['id']} {r['role']} {r['kind']}",
                        fill=color,
                    )
            path = case / f"audit-raw-map-I{fid}.jpg"
            shown.save(path, quality=92)
            assets.append(tp._asset(path, [], "human_audit_unvalidated_raw_map"))
        lookup = {r["id"]: r for r in packet["regions"]}
        # Draw ALL returned tracks, including forbidden supporting/bystander tracks.
        # No budget waiver: these are diagnostic crops, never CV or B input.
        for track in packet["correspondence"]:
            pictures, cells = [], []
            for rid in track["regions"]:
                r = lookup[rid]
                im = images[r["frame"]["file_id"]]
                crop = im.crop(tp._pixels(im, r["box"]))
                native = list(crop.size)
                crop.thumbnail((700, 700))
                pictures.append(crop)
                cells.append(
                    {
                        "region": rid,
                        "role": r["role"],
                        "frame": tp.identity(r["frame"]),
                        "native_size": native,
                        "actual_image_size": list(crop.size),
                        "upsampled": False,
                        "below_model_minimum": min(native) < 256,
                    }
                )
            strip = Image.new(
                "RGB",
                (sum(p.width for p in pictures), max(p.height for p in pictures) + 55),
                "white",
            )
            draw = ImageDraw.Draw(strip)
            draw.text(
                (5, 3),
                "REJECTED RAW MAP - HUMAN AUDIT ONLY - NOT PASS B INPUT",
                fill="red",
            )
            x = 0
            for picture, cell in zip(pictures, cells):
                strip.paste(picture, (x, 55))
                draw.text(
                    (x + 3, 20),
                    f"{track['id']} {cell['region']} {cell['role']}",
                    fill="black",
                )
                x += picture.width
            path = case / f"audit-raw-strip-{track['id']}.jpg"
            strip.save(path, quality=95)
            assets.append(tp._asset(path, cells, "human_audit_unvalidated_raw_track"))
        (case / "human-audit-assets.json").write_bytes(
            tp.canonical(
                {
                    "authority": "unvalidated_raw_model_map_display_only",
                    "used_by_CV_or_Pass_B": False,
                    "assets": assets,
                }
            )
        )


if __name__ == "__main__":
    render(ROOT / "artifacts/two-pass-real-pilot")
