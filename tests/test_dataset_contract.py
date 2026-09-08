import pytest

from src.dataset_contract import (
    DatasetManifest,
    Purpose,
    Split,
    authorize,
    validate_manifest,
    validate_split_contract,
)


def manifest(dataset_id="willy-dev", split="train", events=("event-a",)):
    return validate_manifest({
        "schema_version": 1, "dataset_id": dataset_id, "split": split,
        "owner_scope": "synthetic-fixture", "events": list(events),
    })


def test_event_level_split_leakage_rejected():
    train = manifest(events=("same-event",))
    held = manifest("lin-final", "held_out", ("same-event",))
    with pytest.raises(ValueError, match="event leakage"):
        validate_split_contract([train, held])


def test_held_out_tuning_purposes_fail_closed():
    held = manifest("lin-final", "held_out", ("lin-event",))
    for purpose in (Purpose.TRAIN, Purpose.THRESHOLD_SEARCH,
                    Purpose.PROMPT_SELECTION, Purpose.PRESET_SELECTION):
        with pytest.raises(PermissionError, match="report-only"):
            authorize(held, purpose)


def test_held_out_final_evaluation_is_report_only_zero_authority():
    held = DatasetManifest("lin-final", Split.HELD_OUT, "lin", ("lin-event",))
    contract = authorize(held, Purpose.FINAL_EVALUATION, labels_requested=True)
    assert contract["report_only"] is True
    assert contract["policy_writeback_allowed"] is False
    assert contract["delete_trash_authority"] == "none"


def test_labels_cannot_be_read_during_tuning():
    tune = manifest("willy-tune", "tune", ("event-t",))
    with pytest.raises(PermissionError, match="labels"):
        authorize(tune, Purpose.THRESHOLD_SEARCH, labels_requested=True)
