"""Run only independently planned B groups with real exposure savings.

Unchanged groups reuse the exact A result and input, with zero extra calls.
Failed local exposure recall never reaches a provider. No retries.
"""

from scripts import run_large_group_development as a
from scripts.prepare_large_group_candidate import OUT
from src import cloud_judge_batch as b


def main():
    plan = b.read(OUT / "exposure-b-plans.json")
    report = b.read(OUT / "exposure-b-evaluation.json")
    policy = "balanced"
    eligibility = next(r for r in report["routes"] if r["policy"] == policy)
    b.verify(
        eligibility["human_selected_exposure_recall"] == 1.0,
        "B exposure recall failed; no inference",
    )
    b.verify(
        (a.RUN / "canary-evaluation.json").exists(), "A canary must validate first"
    )
    dev = a.allowed_groups()
    import boto3
    from botocore.config import Config

    client = boto3.Session().client(
        "bedrock-runtime",
        region_name="us-east-1",
        config=Config(
            retries={"total_max_attempts": 1, "mode": "standard"},
            connect_timeout=20,
            read_timeout=300,
        ),
    )
    for group in plan["plans"][policy]:
        if len(group["exposed"]) == group["members"]:
            continue
        state = a.run_group(
            group["group"],
            dev,
            client,
            run_root=OUT / "development-run-b-r2",
            exposure_plan=group,
        )
        print(
            group["group"],
            state["status"],
            "members",
            state["member_count"],
            "exposed",
            state["exposure"]["unique_source_photos"],
            flush=True,
        )


if __name__ == "__main__":
    main()
