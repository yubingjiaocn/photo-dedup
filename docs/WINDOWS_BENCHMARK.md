# Windows RTX 5070 Ti offline benchmark

This is an **opt-in read-only** harness. It never reads `paths.root`, runs
Stage 0, writes a pipeline DB/manifest, changes decisions, moves/deletes files,
downloads a model, or makes a network request. Give it either a small directory
you selected or the checked-in synthetic manifest — never a library root.

## PowerShell

```powershell
cd C:\tools\photo-dedup
.\.venv\Scripts\Activate.ps1
# first validate the selection; this reads no image pixels
python -m src.windows_benchmark --fixture-manifest .\fixtures\scene-benchmark\manifest.json --limit 9 --dry-run --output .\benchmark-dry-run.json
# offline fixture smoke run (uses decode + deterministic local StubBackend)
python -m src.windows_benchmark --fixture-manifest .\fixtures\scene-benchmark\manifest.json --limit 9 --output .\benchmark.json --state .\benchmark-state.json
# real, deliberately selected small sample; use a fresh result/state path
python -m src.windows_benchmark --sample-dir 'D:\BenchmarkSamples\2026-08-03' --limit 100 --output .\benchmark-real.json --state .\benchmark-real.state.json
```

`--limit` is mandatory in recommended real runs (the tool permits an explicit
small directory without it). Output stores opaque path tokens only — never
source paths, basenames, EXIF, or pixels. It records wall-clock throughput,
per-phase errors, peak allocated CUDA memory when torch/CUDA is available,
hardware/software details, config hash, model-runner hash, count, and time.
All JSON is finite (`NaN`/`Infinity` become `null`).

## Provider injection and resume

The built-in runner intentionally reports SigLIP as `SKIPPED`; it neither
loads nor downloads a provider. A future **local, offline** provider can be
injected with `--runner package.module:factory`; its object must expose
`siglip(PIL.Image)` and `stage1(PIL.Image)`, plus optional `model_descriptor`.
Interrupt with Ctrl+C: completed opaque tokens are checkpointed in `--state`;
re-run the same command to resume. State is accepted only when its schema,
config hash, ordered sample set (path token + size + mtime, or manifest hash),
model hash, and runner spec match exactly; any mismatch safely restarts rather
than mixing benchmark populations. JSON output and checkpoints are written via
fsync + atomic replace, so a failure preserves the previous complete file.

Before claiming RTX 5070 Ti results, Willy must run the real selected-sample
command on Windows, confirm `benchmark-real.json` names the GPU and shows
`cuda_available: true`, preserve the JSON, and inspect error counts. This
harness does not validate model quality or authorize any delete action.
