# Semantic observer research status

Date: 2026-09-30 (Asia/Shanghai). Consumer: photo-dedup review workbench and its structured selection-reason API.

## Outcome boundary

Build and evaluate a group-level VLM deep judgment for human review. The VLM sees the group images, labelled local crops, and every available CV measurement—including weak, default-off, and shadow-only signals—and independently recommends what the human should inspect. It returns at most three evidence-bound discriminators rather than explaining a preselected keeper. It does not change the persisted keeper set, ordering, routing, review queues, or deletion authority. All new outputs remain `shadow_review_only`; missing, conflicting, small-face, profile, glasses/occlusion, or low-confidence observations abstain.

## Success criteria

- CV observations are structured, source-linked, deterministic after model inference, and covered by tests.
- The first real-photo sample contains easy, difficult, multi-face, and known provisional-error groups.
- VLM judgments use a strict schema, cite exact image regions and/or supplied CV references, and distinguish measurements from model preference/judgment.
- The first VLM gate sends at most 8 labelled contact sheets in one Sonnet/Terra-class call; no 143-image expansion without a positive sample gate.
- Reports separate conversion integrity, model observation agreement, provisional-label overlap, runtime, image count, and model-call count.
- No automatic delete, keeper mutation, model/config/plugin reload, or external publication.

## Scope budget

### Must

1. Reuse the existing optional MediaPipe eye pipeline and all existing quality, similarity, face/local, exposure, pose, ownership, completeness/occlusion, and shadow evidence; preserve every supplied signal and its authority in a per-frame CV dossier.
2. Define one shared semantic-observation schema with explicit applicability, confidence kind, evidence source, and abstention reasons.
3. Select at most 8 frozen groups: all provisional-issue groups plus one multi-face/multi-subject boundary group where available.
4. Render one anonymized, labelled contact sheet per group, attach the complete per-frame CV dossier, and invoke one available Sonnet/Terra-class VLM once; record exact model, image count, bytes, CV-context hash, prompt hash, output, latency, and token usage when available.
5. Compare CV/VLM observations to sealed model-provisional labels only as descriptive overlap, never as human-truth accuracy.

### Should

- Feed the VLM's accepted recommendation and at most three grounded discriminators into the human review UI without changing persisted selection.
- Measure coverage and contradiction rates by regime (`single`, `multi`, `no_subject`).
- Independently review evidence semantics and fail-closed behavior.

### Parking lot

- Training a learned ranking model or calibrating thresholds for production.
- Full 37-group VLM expansion, per-photo VLM calls, or premium-model escalation.
- Automatic semantics for narrative importance, identity, or “best composition”.
- Windows performance and packaging certification.

## Current evidence

- Existing code already integrates optional MediaPipe Face Landmarker output under `quality_meta.eye_detection`; model assets are never downloaded implicitly.
- Existing local-quality, face-quality, exposure, phase, and subject-ownership paths are review-only and explicitly deny eye-state/amodal-completeness authority where unavailable.
- Frozen real-photo set is restored at `/home/ubuntu/photo-dedup-eval/astra-instance-recovery-20260910`; its labels are `sealed_blind_model_provisional`, not human truth.
- Available bounded VLM routes include `amazon-bedrock/global.anthropic.claude-sonnet-5-5` and `amazon-bedrock-mantle/global.openai.gpt-5.6-terra`, both text+image. First choice for this experiment: Sonnet 5.5, one call only.

## Stop conditions

Stop the VLM branch after the first call if output violates schema, invents invisible facts, cannot distinguish group members, or adds no useful semantic evidence beyond CV/provisional labels. Do not retry with a stronger model automatically.
