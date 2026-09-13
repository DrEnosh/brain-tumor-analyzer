"""Deduplicate the raw MRI folders and write a leakage-safe, stratified 70/15/15 split."""
import hashlib
import json
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
from PIL import Image

from ml.config import (
    CLASSES, NEAR_DUPLICATE_THRESHOLD, PROJECT_ROOT, RAW_DIRS, SEED, SPLIT_FRACTIONS, SPLITS_DIR,
)
from ml.preprocessing import load_grayscale

SPLIT_NAMES = tuple(SPLIT_FRACTIONS)


def scan_raw_images() -> pd.DataFrame:
    rows = []
    for source in RAW_DIRS:
        for label in CLASSES:
            for path in sorted((PROJECT_ROOT / source / label).iterdir()):
                if path.is_file():
                    rows.append({"path": path.relative_to(PROJECT_ROOT).as_posix(),
                                 "label": label, "source_split": source})
    return pd.DataFrame(rows)


def md5_of(rel_path: str) -> str:
    return hashlib.md5((PROJECT_ROOT / rel_path).read_bytes()).hexdigest()


def dhash_of(rel_path: str) -> np.ndarray:
    """256-bit difference hash, packed into 32 bytes. Robust to resizing, re-encoding and contrast changes."""
    gray = load_grayscale(PROJECT_ROOT / rel_path).resize((17, 16), Image.BILINEAR)
    g = np.asarray(gray, dtype=np.int16)
    return np.packbits((g[:, 1:] > g[:, :-1]).flatten())


def hex_to_hashes(hex_values) -> np.ndarray:
    return np.stack([np.frombuffer(bytes.fromhex(h), dtype=np.uint8) for h in hex_values])


def near_duplicate_pairs(hashes: np.ndarray, threshold: int, chunk: int = 256):
    n = len(hashes)
    out_i, out_j = [], []
    for start in range(0, n, chunk):
        block = hashes[start:start + chunk]
        dist = np.bitwise_count(block[:, None, :] ^ hashes[None, :, :]).sum(axis=2, dtype=np.uint16)
        ii, jj = np.nonzero(dist <= threshold)
        ii += start
        keep = jj > ii
        out_i.append(ii[keep])
        out_j.append(jj[keep])
    return np.concatenate(out_i), np.concatenate(out_j)


def connected_groups(n: int, pairs_i, pairs_j) -> np.ndarray:
    parent = np.arange(n)

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i, j in zip(pairs_i, pairs_j):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[max(ri, rj)] = min(ri, rj)
    roots = np.array([find(i) for i in range(n)])
    _, group_ids = np.unique(roots, return_inverse=True)
    return group_ids


def assign_splits(labels, groups, seed: int = SEED, fractions=SPLIT_FRACTIONS) -> np.ndarray:
    """Per class, place whole groups (largest first) into the split furthest below its target share."""
    labels = np.asarray(labels)
    groups = np.asarray(groups)
    names = list(fractions)
    result = np.empty(len(labels), dtype=object)
    rng = np.random.default_rng(seed)

    members = defaultdict(list)
    for idx, g in enumerate(groups):
        members[g].append(idx)
    group_label = {}
    for g, idxs in members.items():
        group_labels = set(labels[idxs])
        if len(group_labels) != 1:
            raise ValueError(f"group {g} mixes labels {sorted(group_labels)}")
        group_label[g] = group_labels.pop()

    for cls in sorted(set(labels)):
        cls_groups = [g for g in sorted(members) if group_label[g] == cls]
        order = rng.permutation(len(cls_groups))
        cls_groups = sorted((cls_groups[k] for k in order), key=lambda g: -len(members[g]))
        total = sum(len(members[g]) for g in cls_groups)
        target = {s: fractions[s] * total for s in names}
        filled = dict.fromkeys(names, 0)
        for g in cls_groups:
            best = max(names, key=lambda s: (target[s] - filled[s]) / target[s])
            filled[best] += len(members[g])
            result[members[g]] = best
    return result


def main() -> None:
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    df = scan_raw_images()
    n_raw = len(df)
    print(f"scanned {n_raw} images")

    with ThreadPoolExecutor(max_workers=16) as pool:
        df["md5"] = list(pool.map(md5_of, df["path"]))

    dup_groups = df.groupby("md5")
    crossing_exact = int(sum(1 for _, g in dup_groups if len(g) > 1 and g["source_split"].nunique() > 1))
    conflicting_exact = int(sum(1 for _, g in dup_groups if g["label"].nunique() > 1))
    if conflicting_exact:
        raise ValueError(f"{conflicting_exact} byte-identical groups carry different labels")
    keep_mask = ~df.duplicated("md5", keep="first")
    kept_by_md5 = df[keep_mask].set_index("md5")["path"]
    removed = df[~keep_mask].assign(kept_copy=lambda d: d["md5"].map(kept_by_md5))
    df = df[keep_mask].reset_index(drop=True)
    print(f"removed {len(removed)} byte-identical copies -> {len(df)} unique images")

    with ThreadPoolExecutor(max_workers=16) as pool:
        hashes = np.stack(list(pool.map(dhash_of, df["path"])))
    df["dhash"] = [h.tobytes().hex() for h in hashes]

    pi, pj = near_duplicate_pairs(hashes, NEAR_DUPLICATE_THRESHOLD)
    df["group_id"] = connected_groups(len(df), pi, pj)
    group_sizes = df["group_id"].value_counts()
    crossing_near = int(df.groupby("group_id")["source_split"].nunique().gt(1).sum())

    df["split"] = assign_splits(df["label"].to_numpy(), df["group_id"].to_numpy(), SEED)

    columns = ["path", "label", "split", "group_id", "source_split", "md5", "dhash"]
    for name in SPLIT_NAMES:
        df.loc[df["split"] == name, columns].to_csv(SPLITS_DIR / f"{name}.csv", index=False)
    removed[["path", "label", "source_split", "kept_copy"]].to_csv(
        SPLITS_DIR / "removed_exact_duplicates.csv", index=False)

    summary = {
        "seed": SEED,
        "raw_images": n_raw,
        "exact_duplicate_copies_removed": int(len(removed)),
        "exact_duplicate_groups_crossing_original_training_testing": crossing_exact,
        "unique_images": int(len(df)),
        "near_duplicate_threshold_bits_of_256": NEAR_DUPLICATE_THRESHOLD,
        "near_duplicate_pairs": int(len(pi)),
        "groups": int(len(group_sizes)),
        "groups_with_multiple_images": int((group_sizes > 1).sum()),
        "images_in_multi_image_groups": int(group_sizes[group_sizes > 1].sum()),
        "largest_group": int(group_sizes.max()),
        "near_duplicate_groups_crossing_original_training_testing": crossing_near,
        "splits": {},
    }
    for name in SPLIT_NAMES:
        part = df[df["split"] == name]
        summary["splits"][name] = {
            "images": int(len(part)),
            "fraction": round(len(part) / len(df), 4),
            "groups": int(part["group_id"].nunique()),
            "per_class": {c: int(v) for c, v in Counter(part["label"]).items()},
        }
    (SPLITS_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
