import pytest
import torch

from scripts.geodro_lejepa_eval import (
    EvalDataset,
    WATERBIRDS_GROUP_NAMES,
    aggregate_imagenetc_metrics,
    build_imagenet100_to_imagenet1k_map,
    invert_label_map,
    parse_int_list,
    parse_int_set,
    parse_str_list,
    summarize_controlled_counts,
    summarize_group_counts,
    waterbirds_group_fn,
)


pytestmark = pytest.mark.unit


def test_parse_int_set_supports_ranges_and_values():
    assert parse_int_set("0-3,7,10-11") == {0, 1, 2, 3, 7, 10, 11}


def test_parse_lists_use_defaults_for_empty_specs():
    assert parse_str_list("", ("a", "b")) == ["a", "b"]
    assert parse_int_list(None, (1, 2)) == [1, 2]
    assert parse_str_list("fog, snow", ()) == ["fog", "snow"]
    assert parse_int_list("1, 5", ()) == [1, 5]


def test_group_summary_reports_worst_group():
    metrics = summarize_group_counts(
        {0: 8, 1: 2},
        {0: 10, 1: 5},
        {0: "background", 1: "coherent"},
        prefix="eval",
    )

    assert metrics["eval/background_acc"] == 0.8
    assert metrics["eval/coherent_acc"] == 0.4
    assert metrics["eval/worst_group_acc"] == 0.4


def test_controlled_summary_adds_worst_subset_alias():
    metrics = summarize_controlled_counts(
        {0: 9, 1: 2},
        {0: 10, 1: 4},
        prefix="imagenet100ctrl/val",
    )

    assert metrics["imagenet100ctrl/val/background_acc"] == 0.9
    assert metrics["imagenet100ctrl/val/coherent_acc"] == 0.5
    assert metrics["imagenet100ctrl/val/worst_subset_acc"] == 0.5


def test_imagenetc_aggregation_by_corruption_and_severity():
    metrics = aggregate_imagenetc_metrics(
        {
            "fog/severity_1": {"acc": 0.8, "correct": 8, "total": 10},
            "fog/severity_2": {"acc": 0.6, "correct": 6, "total": 10},
            "snow/severity_1": {"acc": 0.4, "correct": 4, "total": 10},
        }
    )

    assert metrics["imagenetc/mean_acc"] == pytest.approx(0.6)
    assert metrics["imagenetc/mean_error"] == pytest.approx(0.4)
    assert metrics["imagenetc/fog/mean_acc"] == pytest.approx(0.7)
    assert metrics["imagenetc/severity_1/mean_acc"] == pytest.approx(0.6)


def test_imagenet100_to_imagenet1k_mapping_uses_text_aliases():
    mapping = build_imagenet100_to_imagenet1k_map(
        {
            0: "tench, Tinca tinca",
            1: "American coot, marsh hen, mud hen",
            2: "rooster",
        },
        ["goldfish", "tench", "American coot", "cock"],
    )

    assert mapping == {0: 1, 1: 2, 2: 3}
    assert invert_label_map(mapping) == {1: 0, 2: 1, 3: 2}


def test_eval_dataset_filters_and_remaps_labels():
    class FakeImage:
        def convert(self, mode):
            return self

    class FakeDataset:
        column_names = ["image", "label"]

        def __init__(self):
            self.rows = [{"image": FakeImage(), "label": 0}, {"image": FakeImage(), "label": 5}]

        def __len__(self):
            return len(self.rows)

        def __getitem__(self, idx):
            if idx == "label":
                return [row["label"] for row in self.rows]
            return self.rows[idx]

    dataset = EvalDataset(
        FakeDataset(),
        lambda image: torch.zeros(3, 4, 4),
        label_map={5: 1},
        skip_unmapped=True,
    )

    assert len(dataset) == 1
    item = dataset[0]
    assert item["label"].item() == 1
    assert item["group"].item() == -1


def test_waterbirds_group_ids_are_stable():
    assert WATERBIRDS_GROUP_NAMES[0] == "landbird_land"
    assert waterbirds_group_fn({"place": 0}, raw_label=0, mapped_label=0) == 0
    assert waterbirds_group_fn({"place": 1}, raw_label=0, mapped_label=0) == 1
    assert waterbirds_group_fn({"place": 0}, raw_label=1, mapped_label=1) == 2
    assert waterbirds_group_fn({"place": 1}, raw_label=1, mapped_label=1) == 3
