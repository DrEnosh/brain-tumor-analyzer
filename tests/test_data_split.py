import hashlib

import numpy as np
import pandas as pd
import pytest

from ml.config import CLASSES, NEAR_DUPLICATE_THRESHOLD, PROJECT_ROOT, SEED, SPLIT_FRACTIONS, SPLITS_DIR
from ml.prepare_data import assign_splits, connected_groups, dhash_of, hex_to_hashes, near_duplicate_pairs


@pytest.fixture(scope="module")
def splits() -> dict[str, pd.DataFrame]:
    if not (SPLITS_DIR / "train.csv").exists():
        pytest.skip("run `python -m ml.prepare_data` first")
    return {name: pd.read_csv(SPLITS_DIR / f"{name}.csv") for name in SPLIT_FRACTIONS}


@pytest.mark.parametrize("column", ["path", "md5", "group_id"])
def test_no_file_content_or_group_shared_between_splits(splits, column):
    names = list(splits)
    for a in range(len(names)):
        for b in range(a + 1, len(names)):
            shared = set(splits[names[a]][column]) & set(splits[names[b]][column])
            assert not shared, f"{column} shared between {names[a]} and {names[b]}: {list(shared)[:3]}"


def test_no_near_duplicate_pair_crosses_splits(splits):
    frame = pd.concat(splits.values(), ignore_index=True)
    i, j = near_duplicate_pairs(hex_to_hashes(frame["dhash"]), NEAR_DUPLICATE_THRESHOLD)
    split = frame["split"].to_numpy()
    assert len(i) > 0
    assert int((split[i] != split[j]).sum()) == 0


def test_split_sizes_are_70_15_15_overall_and_per_class(splits):
    total = sum(len(frame) for frame in splits.values())
    for name, target in SPLIT_FRACTIONS.items():
        assert abs(len(splits[name]) / total - target) < 0.01
        for cls in CLASSES:
            class_total = sum(int((frame["label"] == cls).sum()) for frame in splits.values())
            assert abs(int((splits[name]["label"] == cls).sum()) / class_total - target) < 0.02


def test_stored_hashes_match_the_image_files(splits):
    sample = pd.concat(splits.values()).sample(n=30, random_state=0)
    for rel, stored_md5, stored_dhash in zip(sample["path"], sample["md5"], sample["dhash"]):
        assert hashlib.md5((PROJECT_ROOT / rel).read_bytes()).hexdigest() == stored_md5
        assert dhash_of(rel).tobytes().hex() == stored_dhash


def test_removed_copies_are_byte_identical_to_kept_images():
    removed_csv = SPLITS_DIR / "removed_exact_duplicates.csv"
    if not removed_csv.exists():
        pytest.skip("run `python -m ml.prepare_data` first")
    removed = pd.read_csv(removed_csv).head(25)
    for rel, kept in zip(removed["path"], removed["kept_copy"]):
        assert (PROJECT_ROOT / rel).read_bytes() == (PROJECT_ROOT / kept).read_bytes()


def _synthetic(n_groups: int = 600, seed: int = 0):
    rng = np.random.default_rng(seed)
    labels, groups = [], []
    for g in range(n_groups):
        size = int(rng.choice([1, 1, 1, 2, 3, 8]))
        cls = str(rng.choice(CLASSES))
        labels += [cls] * size
        groups += [g] * size
    return np.array(labels), np.array(groups)


def test_assign_splits_is_deterministic_and_depends_on_seed():
    labels, groups = _synthetic()
    first = assign_splits(labels, groups, seed=SEED)
    assert (first == assign_splits(labels, groups, seed=SEED)).all()
    assert (first != assign_splits(labels, groups, seed=SEED + 1)).any()


def test_assign_splits_keeps_each_group_in_one_split():
    labels, groups = _synthetic()
    result = pd.DataFrame({"group": groups, "split": assign_splits(labels, groups)})
    assert result.groupby("group")["split"].nunique().max() == 1


def test_assign_splits_rejects_groups_with_mixed_labels():
    with pytest.raises(ValueError):
        assign_splits(np.array(["glioma", "notumor"]), np.array([0, 0]))


def test_connected_groups_are_transitive():
    groups = connected_groups(5, np.array([0, 1]), np.array([1, 2]))
    assert groups[0] == groups[1] == groups[2]
    assert len({groups[0], groups[3], groups[4]}) == 3
