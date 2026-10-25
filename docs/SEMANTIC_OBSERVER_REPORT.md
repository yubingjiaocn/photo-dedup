# CV + bounded VLM semantic observer probe

This is a shadow/review-only experiment. Frozen labels are `sealed_blind_model_provisional`, not human truth.

## Bounded sample

- Local CV: 8 groups / 26 frames, selected as all seven provisional-issue groups plus the only multi group.
- VLM: 5 unique groups, 7 contact-sheet images, 2 Sonnet calls.
- VLM payload: 2067844 bytes; 23145 input + 2916 output tokens; catalog-rate estimate `$0.0754`.
- Resolution gate: use 2048px. 1024px retained broad pose/action but weakened subtle expression comparison.

## CV result

- Eye statuses over the 26 frames: `{"ALL_OPEN": 1, "HAS_CLOSED_EYES": 3, "HAS_MAYBE": 11, "NO_FACE": 3, "UNKNOWN": 8}`.
- Against the six eye-specific model-provisional frame labels only: CLOSED 3/3 detected; OPEN 0/3 promoted, with 3 abstentions and 0 false CLOSED flags.
- This is descriptive overlap, not calibrated accuracy. The detector is deliberately asymmetric: it can flag likely closed eyes while abstaining on many open-eye frames.

## VLM result

- Accepted medium/high direct-visual claims: 11.
- Rejected low-confidence, invalid, or delegated-CV claims: 5.
- Sonnet recovered useful pose/action and scene-information differences on the five CV-unresolved groups.
- It also caught a real routing defect: in one group YuNet cropped an illustrated character instead of the photographed person. Resolution cannot repair the wrong subject crop.

## Revised decision

The first probe established that 2048px sheets and bounded Sonnet calls are viable, but its role was too narrow. The target architecture makes the VLM the group-level deep judge for human review: it sees the images, labelled crops, and the complete CV dossier, including weak and shadow-only signals. CV measurements retain their source and authority; the VLM may combine them but cannot silently promote them to truth. The output is an independent review recommendation plus at most three grounded discriminators, each citing an image region and/or exact CV reference. It is not a post-hoc explanation of the persisted keeper. Use one 2048px labelled sheet per reviewed group and preserve human confirmation. No result expands deletion or persisted-keeper authority.
