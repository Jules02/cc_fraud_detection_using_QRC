#%%
import numpy as np
import pytest

from qrc.decision import (
    card_precision_top_k,
    card_precision_top_ks,
    select_threshold_min_fp,
    select_thresholds_min_fp,
)

#%%
def test_threshold_minimizes_false_positives_subject_to_recall():
    # To keep recall >= 0.8 with two frauds, both frauds must be detected.
    # Threshold 0.6 does that with one false positive; lower thresholds add
    # false positives, while higher thresholds miss one fraud.
    y = np.array([1, 1, 0, 0, 0])
    scores = np.array([0.9, 0.6, 0.7, 0.5, 0.1])

    threshold = select_threshold_min_fp(y, scores, min_recall=0.8)

    assert threshold == pytest.approx(0.6)

    predictions = (scores >= threshold)

    assert np.sum((predictions == 1) & (y == 0)) == 1
    assert np.sum((predictions == 1) & (y == 1)) == 2

#%%
def test_threshold_can_trade_recall_for_zero_false_positives():
    y = np.array([1, 1, 0, 0, 0])
    scores = np.array([0.9, 0.6, 0.7, 0.5, 0.1])

    threshold = select_threshold_min_fp(y, scores, min_recall=0.5)

    assert threshold == pytest.approx(0.9)

#%%
def test_threshold_handles_tied_scores_as_one_candidate():
    y = np.array([1, 0, 1, 0])
    scores = np.array([0.9, 0.8, 0.8, 0.2])

    threshold = select_threshold_min_fp(y, scores, min_recall=1.0)

    assert threshold == pytest.approx(0.8)

#%%
def test_threshold_prefers_higher_recall_when_false_positives_tie():
    y = np.array([1, 1, 1, 0, 0])
    scores = np.array([0.9, 0.8, 0.7, 0.6, 0.5])

    threshold = select_threshold_min_fp(y, scores, min_recall=2/3)

    # 0.8 satisfies the recall constraint with 0 FP,
    # but 0.7 also has 0 FP and gives higher recall.
    assert threshold == pytest.approx(0.7)

#%%
def test_multiple_thresholds_match_individual_threshold_selection():
    y = np.array([1, 1, 1, 0, 0, 0])
    scores = np.array([0.95, 0.80, 0.55, 0.90, 0.40, 0.10])

    min_recalls = [0.3, 0.6, 0.8, 1.0]

    thresholds = select_thresholds_min_fp(y, scores, min_recalls=min_recalls)

    for min_recall in min_recalls:
        expected_threshold = select_threshold_min_fp(y, scores, min_recall=min_recall)
        assert thresholds[min_recall] == pytest.approx(expected_threshold)

#%%
def test_multiple_thresholds_remove_duplicate_recall_constraints():
    y = np.array([1, 1, 0])
    scores = np.array([0.9, 0.8, 0.1])

    result = select_thresholds_min_fp(y, scores, min_recalls=[0.5, 0.5, 1.0])

    assert set(result) == {0.5, 1.0}

#%%
def test_threshold_validation():
    with pytest.raises(ValueError):
        select_threshold_min_fp(
            [0, 1],
            [0.1, 0.9],
            min_recall=0.0,
        )

    with pytest.raises(ValueError):
        select_threshold_min_fp(
            [0, 1],
            [0.1],
            min_recall=0.8,
        )

    with pytest.raises(ValueError):
        select_threshold_min_fp(
            [0, 0],
            [0.1, 0.2],
            min_recall=0.8,
        )

    with pytest.raises(ValueError):
        select_threshold_min_fp(
            [0, 1],
            [0.1, np.nan],
            min_recall=0.8,
        )

    with pytest.raises(ValueError):
        select_thresholds_min_fp(
            [0, 1],
            [0.1, 0.9],
            min_recalls=[],
        )


#%%
def test_card_precision_top_k_groups_multiple_transactions_from_same_card():
    y = np.array([1, 0, 0, 1, 0, 1, 1])
    scores = np.array([0.9, 0.8, 0.7, 0.6, 0.2, 0.9, 0.8])
    customers = np.array(["A", "A", "B", "C", "A", "D", "E"])
    timestamps = np.array(
        [
            "2026-01-01 09:00", "2026-01-01 10:00", "2026-01-01 11:00", "2026-01-01 12:00",
            "2026-01-02 09:00", "2026-01-02 10:00", "2026-01-02 11:00",
        ],
        dtype="datetime64[m]",
    )

    # Day 1: card A (fraud, max score .9) and B (genuine, .7) are top 2 -> .5. card C (fraud, .6)
    # Day 2: cards D (fraud, 0.9) and E (fraud, 0.8) are top 2 -> 1.0. card A (genuine, .2)
    # Mean CP@2 = .75
    result = card_precision_top_k(
        y, scores, customers, timestamps, top_k=2
    )

    assert result == pytest.approx(0.75)

#%%
def test_card_precision_top_k_uses_available_cards_when_fewer_than_k():
    y = np.array([1, 0])
    scores = np.array([0.9, 0.8])
    customers = np.array(["A", "B"])

    timestamps = np.array(
        [
            "2026-01-01", "2026-01-01",
        ],
        dtype="datetime64[D]",
    )

    result = card_precision_top_k(y, scores, customers, timestamps, top_k=10)

    assert result == pytest.approx(0.5)

#%%
def test_card_precision_top_k_is_deterministic_for_tied_scores():
    y = np.array([1, 0])
    scores = np.array([0.8, 0.8])
    customers = np.array(["A", "B"])

    timestamps = np.array(
        [
            "2026-01-01", "2026-01-01",
        ],
        dtype="datetime64[D]",
    )

    result = card_precision_top_k(y, scores, customers, timestamps, top_k=1)

    assert result == pytest.approx(1.0)

#%%
def test_multiple_card_precision_top_ks_match_individual_calls():
    y = np.array([1, 0, 0, 1, 0, 1, 1])
    scores = np.array([0.9, 0.8, 0.7, 0.6, 0.2, 0.9, 0.8])
    customers = np.array(["A", "A", "B", "C", "A", "D", "E"])

    timestamps = np.array(
        [
            "2026-01-01 09:00",
            "2026-01-01 10:00",
            "2026-01-01 11:00",
            "2026-01-01 12:00",
            "2026-01-02 09:00",
            "2026-01-02 10:00",
            "2026-01-02 11:00",
        ],
        dtype="datetime64[m]",
    )

    top_ks = [1, 2, 3, 10]

    results = card_precision_top_ks(y, scores, customers, timestamps, top_ks=top_ks)

    for top_k in top_ks:
        expected = card_precision_top_k(y, scores, customers, timestamps, top_ks=top_ks)

        assert results[top_k] == pytest.approx(expected)

#%%
def test_card_precision_multiple_k_expected_values():
    y = np.array([1, 0, 0, 1, 0, 1, 1])

    scores = np.array([0.9, 0.8, 0.7, 0.6, 0.2, 0.9, 0.8])

    customers = np.array(["A", "A", "B", "C", "A", "D", "E"])

    timestamps = np.array(
        [
            "2026-01-01 09:00",
            "2026-01-01 10:00",
            "2026-01-01 11:00",
            "2026-01-01 12:00",
            "2026-01-02 09:00",
            "2026-01-02 10:00",
            "2026-01-02 11:00",
        ],
        dtype="datetime64[m]",
    )

    results = card_precision_top_ks(y, scores, customers, timestamps, top_ks=[1, 2, 10])

    assert results[1] == pytest.approx(1.0)
    assert results[2] == pytest.approx(0.75)

    # There are 3 cards on each day.
    # Day 1: A fraud, B genuine, C fraud -> 2/3
    # Day 2: D fraud, E fraud, A genuine -> 2/3
    assert results[10] == pytest.approx(2 / 3)

#%%
def test_card_precision_empty_input_returns_nan():
    result = card_precision_top_k([], [], [], [], top_k=5)

    assert np.isnan(result)

#%%
def test_multiple_card_precision_empty_input_returns_nan():
    result = card_precision_top_ks([], [], [], [], top_ks=[1, 5, 10])

    assert set(result) == {1, 5, 10}
    assert all(np.isnan(v) for v in result.values())

#%%
def test_card_precision_validation():
    # Single-K validation
    with pytest.raises(ValueError):
        card_precision_top_k(
            [0],
            [0.1],
            [1],
            ["2026-01-01"],
            top_k=0,
        )

    with pytest.raises(ValueError):
        card_precision_top_k(
            [0, 1],
            [0.1],
            [1, 2],
            ["2026-01-01", "2026-01-02"],
            top_k=1,
        )

    with pytest.raises(ValueError):
        card_precision_top_k(
            [0],
            [np.inf],
            [1],
            ["2026-01-01"],
            top_k=1,
        )

    # Multi-K validation
    with pytest.raises(ValueError):
        card_precision_top_ks(
            [0],
            [0.1],
            [1],
            ["2026-01-01"],
            top_ks=[],
        )

    with pytest.raises(ValueError):
        card_precision_top_ks(
            [0],
            [0.1],
            [1],
            ["2026-01-01"],
            top_ks=[1, 0, 5],
        )