"""Offline, opt-in large-group v4 candidate. No transport or decision-write API.

Two-frame requests and responses delegate unchanged to v3. The extra ledger is
observable evidence, not hidden reasoning and not a claim of subject identity.
"""

from __future__ import annotations

import copy
import re

import jsonschema

from . import cloud_judge_batch as batch
from . import two_pass_review as tp

CONTRACT = "large-group-v4-candidate-1"
MAX_FRAMES = 16
MAX_OUTPUT_TOKENS = 8192
MAX_OUTPUT_BYTES = 48000
PRIVACY_INSTRUCTION = (
    " Return JSON only, without fences. No names, identity guesses, OCR, personal "
    "details or demographic/relationship inference. Use anonymous region IDs. "
    "Do not claim individual coverage from distant stage views. "
    "所有面向用户的描述使用简体中文。"
)
PROMPT = """Review-only single judge, large groups (3–16 complete frames).
Return only the supplied JSON schema. All user-facing text must be 简体中文;
keys, enums, frame identities and region IDs stay English. Read EVERY complete
frame before supplements. Local hints and CV scores are not preference truth.

First populate phase_census: exactly one short observable entry per frame,
with its expression/action/interaction/formation/lighting/composition state,
readability (usable/unreadable/uncertain), evidence and cluster ID. This is a
visible-state ledger, not chain-of-thought. Describe what is actually visible;
do not invent fine eye states or individual coverage from distant stage views.
Then partition EVERY frame into substitutability clusters. Two usable frames
belong together only if keeping one loses no meaningful expression, gesture,
action stage, interaction or composition. Compare all cluster members directly
to the representative; do not merge by transitive similarity alone. Near-equal
pixels, same subject, same body pose or adjacent timestamps do not prove that
expressions or gestures are interchangeable. Unknown/equal timestamps establish
no causal order. A changed head direction, mouth expression, paw/hand action,
pre-contact/contact/release or coordinated stage gesture can be worth keeping.
Do not dismiss release, closed-mouth, looking-away or slightly soft but readable
moments merely because a sharper peak exists. Judge ensemble and image-level
changes without claiming unnamed individuals are fully covered.

A cluster's relation is distinct, uncertain_distinct, or unassessable. Distinct
means a visible useful state not replaced by another cluster. Uncertain_distinct
means the state is readable with medium/high visual evidence but whether another
readable state substitutes for it is uncertain: conservatively keep one here.
It does NOT upgrade low visual confidence. Unreadable input belongs in an
unassessable cluster and makes the group attention_only/abstain, not select.
Every omitted frame needs a same-cluster representative and a concise comparison
explaining why its useful state is preserved. Keep one best usable representative
per distinct/uncertain_distinct cluster. Select all representatives, even if this
is every frame; never impose a target count or an eight-frame shortlist cap.
Within a genuinely substitutable cluster prefer expression/gesture readability
and framing before minor global CV score gaps. Conversely camera jitter, slight
crop/brightness, background movement, timestamp separation or an intrusive hand
alone do not create a useful new phase. Do not split all near-duplicates as a
precaution. A failed substitutability comparison is different from unreadability.

For select every representative needs medium/high own-frame evidence and global
confidence medium/high. recommended equals the cluster representatives exactly.
Keep at most three concise user discriminators; their count is NOT the phase or
shortlist count. multi_select_reason explains the complementary useful states and
binds every selected frame. irreplaceable remains empty: deterministic regions
are frame views, not semantic subject identity. Respect the supplied decision
ceiling; on incomplete or unreadable coverage return attention_only/abstain with
no recommendations. Never alter stored decisions, keeper or deletion state.
"""


def contract_for(member_count):
    if type(member_count) is not int or member_count < 2:
        raise tp.ReviewError("at least two members required")
    return "v3" if member_count == 2 else CONTRACT


def response_schema():
    return batch.read(batch.ROOT / "schemas/large-group-judge-v4.schema.json")


def build_request(v3_request, rows):
    """Pure opt-in adapter; v3 live runner remains untouched."""
    if contract_for(len(rows)) == "v3":
        return copy.deepcopy(v3_request)
    batch.verify(len(rows) <= MAX_FRAMES, "large-group complete-frame budget exceeded")
    members = {tp.identity(r) for r in rows}
    batch.verify(len(members) == len(rows), "duplicate group member")
    result = copy.deepcopy(v3_request)
    packet = result["context"]["visual_packet"]
    batch.verify(set(packet["complete_frames"]) == members, "incomplete group packet")
    full = [a for a in result["images"] if a["kind"] == "full_group_page"]
    cells = [c for a in full for c in a["cells"]]
    batch.verify(
        len(cells) == len(rows) and {c["frame"] for c in cells} == members,
        "complete images required for every frame",
    )
    lookup = {v["id"]: v for v in packet["views"]}
    for cell in cells:
        view = lookup.get(cell["region"], {})
        batch.verify(
            view.get("purpose") == "complete_frame"
            and view.get("box") == [0, 0, 1, 1]
            and tp.identity(view["frame"]) == cell["frame"],
            "complete image region binding mismatch",
        )
    result.update(
        judgment_contract=CONTRACT,
        prompt=PROMPT,
        response_schema=response_schema(),
        max_output_tokens=MAX_OUTPUT_TOKENS,
        privacy_instruction=PRIVACY_INSTRUCTION,
    )
    # Same wire budget, including per-image metadata and privacy instruction.
    wire_bytes = len(tp.canonical({k: v for k, v in result.items() if k != "images"}))
    wire_bytes += sum(len(tp.canonical(a)) for a in result["images"])
    wire_bytes += len(PRIVACY_INSTRUCTION.encode())
    batch.verify(wire_bytes <= batch.VISUAL_BUDGET["text_bytes"], "v4 wire text budget")
    return result


def _chinese(text):
    # Observable language gate, not a claim to detect every traditional character.
    batch.verify(bool(re.search(r"[\u4e00-\u9fff]", text)), "user text must be Chinese")
    batch.verify(
        not tp._PRIVACY.search(text)
        and not re.search(r"电话号码|身份证|家庭住址|真实姓名|邮箱地址", text),
        "privacy prose tripwire",
    )


def validate_response(answer, rows, packet, validation, l2):
    if contract_for(len(rows)) == "v3":
        jsonschema.validate(
            answer, batch.read(batch.ROOT / "schemas/cloud-scene-judge-v1.schema.json")
        )
        return tp.validate_judgment(
            answer, rows, packet, validation, l2, allow_visual_views=True
        )
    jsonschema.validate(answer, response_schema())
    batch.verify(len(rows) <= MAX_FRAMES, "large-group frame budget")
    batch.verify(len(tp.canonical(answer)) <= MAX_OUTPUT_BYTES, "v4 output byte budget")
    members = {tp.identity(r) for r in rows}
    batch.verify(len(members) == len(rows), "duplicate group member")
    batch.verify(set(packet["complete_frames"]) == members, "incomplete group packet")
    lookup = validation["lookup"]

    def evidence(items, frame, strong=False):
        supported = False
        for e in items:
            view = lookup.get(e["region"])
            batch.verify(
                view is not None
                and tp.identity(view["frame"]) == frame
                and view["role"] == "frame_view",
                "evidence frame binding mismatch",
            )
            supported |= e["confidence"] != "low" and view["confidence"] != "low"
        batch.verify(
            not strong or supported, "representative requires medium/high evidence"
        )

    census = {}
    for item in answer["phase_census"]:
        key = tp.identity(item["frame"])
        batch.verify(
            key in members and key not in census, "census identity/duplicate mismatch"
        )
        _chinese(item["state"])
        evidence(item["evidence"], key)
        census[key] = item
    batch.verify(set(census) == members, "census must cover every frame")
    assigned, cluster_ids, representatives = set(), set(), set()
    selecting = answer["decision"] == "select"
    for cluster in answer["clusters"]:
        cid = cluster["id"]
        batch.verify(cid not in cluster_ids, "duplicate cluster")
        cluster_ids.add(cid)
        keys = [tp.identity(f) for f in cluster["members"]]
        batch.verify(
            len(set(keys)) == len(keys)
            and set(keys) <= members
            and not assigned.intersection(keys),
            "cluster partition mismatch",
        )
        assigned.update(keys)
        batch.verify(
            all(census[k]["cluster"] == cid for k in keys), "census cluster mismatch"
        )
        _chinese(cluster["reason"])
        rep = cluster["representative"]
        if cluster["relation"] == "unassessable":
            batch.verify(
                rep is None and not selecting, "unassessable coverage forbids select"
            )
            batch.verify(
                any(census[k]["readability"] != "usable" for k in keys),
                "unassessable requires uncertain/unreadable evidence",
            )
        else:
            batch.verify(rep is not None, "assessable cluster requires representative")
            key = tp.identity(rep)
            batch.verify(
                key in keys and census[key]["readability"] == "usable",
                "representative must be usable cluster member",
            )
            evidence(census[key]["evidence"], key, strong=True)
            representatives.add(key)
        comparisons = cluster["omissions"]
        omitted = {tp.identity(o["frame"]) for o in comparisons}
        expected = set(keys) - ({tp.identity(rep)} if rep else set())
        batch.verify(
            len(omitted) == len(comparisons) and omitted == expected,
            "each omitted member needs a comparison",
        )
        for item in comparisons:
            _chinese(item["reason"])
        if selecting:
            batch.verify(
                all(census[k]["readability"] == "usable" for k in keys),
                "uncertain visual coverage forbids select",
            )
    batch.verify(assigned == members, "cluster partition incomplete")
    selected = {tp.identity(r["frame"]) for r in answer["recommended"]}
    batch.verify(
        not selecting or selected == representatives, "select must cover every cluster"
    )
    for item in answer["discriminators"]:
        _chinese(item["text"])
    if answer["multi_select_reason"]:
        _chinese(answer["multi_select_reason"]["text"])
    # Reuse v3 evidence semantics without its eight-item multi-reason ceiling.
    # Validate all recommendation bindings together; multi-reason is checked here.
    core = {
        k: copy.deepcopy(answer[k])
        for k in (
            "decision",
            "confidence",
            "recommended",
            "multi_select_reason",
            "irreplaceable",
            "discriminators",
        )
    }
    reason = core["multi_select_reason"]
    if len(selected) > 1:
        batch.verify(reason is not None, "multi-select reason required")
        bound = set()
        for e in reason["evidence"]:
            view = lookup.get(e["region"])
            batch.verify(
                view is not None
                and e["confidence"] != "low"
                and view["confidence"] != "low",
                "weak multi-select evidence",
            )
            key = tp.identity(view["frame"])
            evidence([e], key, strong=True)
            bound.add(key)
        batch.verify(
            bound == selected, "multi-select reason must bind every recommended frame"
        )
        # One-frame slices preserve v3's recommendation/confidence/ceiling checks.
        for rec in core["recommended"]:
            one = {**core, "recommended": [rec], "multi_select_reason": None}
            tp.validate_judgment(
                one, rows, packet, validation, l2, allow_visual_views=True
            )
        batch.verify(
            len(selected) == len(core["recommended"]), "duplicate recommendation"
        )
    else:
        tp.validate_judgment(
            core, rows, packet, validation, l2, allow_visual_views=True
        )
    return copy.deepcopy(answer)
