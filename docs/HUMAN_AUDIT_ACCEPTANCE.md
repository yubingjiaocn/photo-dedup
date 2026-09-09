# Human audit v1 acceptance — 2026-09-09

## Delivered slice

An independent offline vertical slice: frozen v4 sample → blinded cached-image
phase/acceptable-keeper annotations → versioned JSONL export/import → strict
replay → weighted selective-risk report. The operational Preact workbench,
pipeline stages and deletion authorization are unchanged by this slice.

The main working tree initially pointed at `c65727c`; it was fast-forwarded to
already completed local evaluation commit `182a052` before implementation. No
remote push was performed. Other work under untracked `docs/images/` was not
read, overwritten, removed or staged.

## Real artifact acceptance (no human conclusions)

Authoritative local bundle:
`/home/ubuntu/photo-dedup-eval/human-audit-v1-20260909-r2/`

- Reviewer entry: `reviewer/index.html`.
- 30 frozen probability tasks from a 102-group pre-audit silent population.
- One additional purposive Disney G10 task; all 31 tasks remain **pending**.
- 74 cached thumbnail exports; 31/31 tasks have complete media.
- `pending-report.json`: weighted risk **null**, upper 95% bound **null**,
  `incomplete_no_risk_estimate`, deletion authority **none**.
- G10 fixture pins members 430/431/432/433 and the recorded source DB hash. No
  human phase count, keeper or verdict is prefilled.
- Only v4 metadata and explicit numeric thumbnail caches were read. No original
  media, DB, held-out dataset, embeddings or remote image inference were used.
  Source DB hashes are inherited provenance, not a claim that DBs were re-read.
- Initial r1 development export is retained with a superseded notice; use r2.

## Verification results

- Focused human-audit + risk-coverage: **41 passed**.
- Final fast gate: **313 passed** (12.12 s).
- Final full gate: **732 passed** (108.54 s).
- Ruff, compileall and `git diff --check`: **passed**.
- Real Chrome/Playwright synthetic annotation workflow: **passed**.
- Independent read-only repair re-review: **PASS**, no remaining concrete blocker.

Final logs on this host: `/tmp/photo-human-audit-fast-release.log` and
`/tmp/photo-human-audit-full-release.log`. These are transient run evidence;
the checked-in tests and commands in `HUMAN_AUDIT.md` reproduce the gates.

## Verification coverage

Focused tests cover sample replay/weights, stale fixture, strict label scope and
partitions, uncertainty abstention, deterministic export, cache symlinks, missing
media, source/media/HTML tampering, census and weighted inference, and rejection
of AI packets as human labels. Browser verification uses **synthetic images and
synthetic labels only**; it exercises form entry, download/report, reload/import,
raw duplicate-key/version/NaN rejection, edit invalidation, draft-loss warnings,
import cancellation and local-only asset requests.

Independent code review identified three concrete issues which were repaired:
strict browser/CLI JSON parity; separate draft-loss protection; and sealed HTML
member-to-image mapping. Regression coverage is in the focused and browser gates.
No remote service or real human dataset was used to manufacture passing labels.

## Optional AI visual experiment

`/home/ubuntu/photo-dedup-eval/model-audit-v1-20260909/` contains a reproducible
blinded packet: G10 plus three deterministically selected stratified cases.
`scripts/prepare_model_audit.py` rebuilds it from the verified cached-thumbnail
bundle. `model_provisional` and `selective_risk_eligible: false` are explicit;
the mapping for later unblinding is separate.

**Actual model visual review is blocked, not completed.** The request forbids
network exfiltration, while attaching pixels to the configured remote Astra
model would upload them. A bounded local check found no ollama/llama-server,
common local inference listener or Hugging Face model cache; the repo contains a
face detector, not a narrative VLM. No models were downloaded and no images were
sent to a remote tool. `STATUS.md` records the blocker and next action. There are
no model judgments, unblinded algorithm comparisons or new visually evidenced
failure hypotheses to claim from this unperformed experiment.

## Limits and next step

1. Complete the actual 30 probability-task human labels without prior exposure
   to candidate predictions; G10 is a separate narrative recheck.
2. Replay exported JSONL. Missing/unassessable tasks must not be replaced by AI
   labels; incomplete samples keep overall risk and confidence bounds unavailable.
3. Sparse-stratum Hoeffding bounds are deliberately conservative and may remain
   too loose for a practical safety target. No threshold is certified here.
4. This is development-event inference, not future-event or held-out validation.
   Freeze a new independent evaluation design after any policy fitting.
5. For the optional AI experiment, provide an already installed local vision
   model and run it with networking disabled. Freeze blind provisional output
   before unblinding; never mix it into human safety statistics.
6. No original movement/deletion, no widened `AUTO_REMOVE`, no remote push.
