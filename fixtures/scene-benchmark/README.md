# Synthetic scene fixtures

This directory contains only geometric PNGs generated locally. It intentionally
contains no private images, model weights, EXIF identity data, or actual Motion
Photo sidecars. The manifest is a **coverage checklist**, not semantic ground
truth or a permission to change decisions.

Use `manifest.schema.json` to validate new fixtures. Keep `file` relative,
inside `synthetic/`, and use neutral bucket names. Real calibration samples
must live outside git and be selected explicitly at benchmark invocation.
