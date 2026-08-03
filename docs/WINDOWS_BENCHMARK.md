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
python -m src.windows_benchmark --sample-dir 'D:\BenchmarkSamples\selected-small' --limit 100 --output .\benchmark-real.json --state .\benchmark-real.state.json
```

`--limit` is mandatory in recommended real runs (the tool permits an explicit
small directory without it). Output stores opaque path tokens only — never
source paths, basenames, EXIF, or pixels. It records wall-clock throughput,
per-phase errors, peak allocated CUDA memory when torch/CUDA is available,
hardware/software details, config hash, model-runner hash, count, and time.
All JSON is finite (`NaN`/`Infinity` become `null`).

## Local SigLIP runner and resume

The built-in runner intentionally reports SigLIP as `SKIPPED`; it neither
loads nor downloads a provider. The real local adapter is explicit opt-in:

```powershell
# Values below are placeholders, not a library path or model ID. All identity
# values are required; absent/invalid values fail before benchmark processing.
$env:PHOTO_DEDUP_SIGLIP_MODEL_PATH = 'C:\models\local-siglip'
$env:PHOTO_DEDUP_SIGLIP_MODEL_REVISION = 'local-build-2026-08'
$env:PHOTO_DEDUP_SIGLIP_MODEL_SHA256 = '<64 lowercase hex artifact hash>'
$env:PHOTO_DEDUP_SIGLIP_PROMPT_BANK_PATH = '.\research\siglip_prompt_bank_v1.yaml'
$env:PHOTO_DEDUP_SIGLIP_PROMPT_BANK_SHA256 = '<64 lowercase hex file hash>'
$env:PHOTO_DEDUP_SIGLIP_DEVICE = 'cuda'
$env:PHOTO_DEDUP_SIGLIP_PRECISION = 'float16'
$env:PHOTO_DEDUP_SIGLIP_BATCH_SIZE = '16'
python -m src.windows_benchmark --sample-dir 'D:\BenchmarkSamples\selected-small' --limit 100 --runner src.windows_siglip_runner:factory --output .\benchmark-siglip.json --state .\benchmark-siglip.state.json
```

Alternatively set only `PHOTO_DEDUP_SIGLIP_RUNNER_CONFIG` to a separate JSON
object containing exactly `model_path`, `model_revision`, `model_sha256`,
`prompt_bank_path`, `prompt_bank_sha256`, `device`, `precision`, and optional
`batch_size`; explicit variables override it. The adapter consumes the one PIL
object decoded by the harness. It does not read the selected directory, reopen
an image, download, or use a model registry. It reports SigLIP `PROCESSED` only
for a complete finite raw-score audit, even though its routing state remains
`UNKNOWN` by design; unavailable/bad providers become errors, never throughput.
Stage 1 is deliberately `StubBackend` and deterministic: it is **not** DINO,
YuNet, IQA, or a claim of real Stage 1 throughput.

Checkpoint identity includes the runner schema, model SHA/revision, prompt-bank
SHA/version, device, precision, batch size, and Stage 1 backend schema. Paths
are never written to output; a changed identity restarts instead of mixing
results. Inspect `hardware.gpu`, `hardware.cuda_available`, each phase's
`processed`/`skipped`/`errors`, and never quote throughput from a phase with
errors or zero processed samples.

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
