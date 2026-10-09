"""Extract data.zip and build dataset/{train,val,test}/<grade> layout.

Splits by patient (image names look like 9001695L.png / 9001695R.png) so the
same patient never appears in two splits.
"""

import os
import random
import shutil
import zipfile
from collections import defaultdict

SEED = 12049
RATIOS = (0.70, 0.15, 0.15)  # train, val, test
SRC_ZIP = "_dl/data.zip"
RAW = "dataset/_raw"
OUT_ROOT = "dataset"


def extract():
    if os.path.isdir(RAW):
        return
    os.makedirs(RAW, exist_ok=True)
    with zipfile.ZipFile(SRC_ZIP) as z:
        z.extractall(RAW)


def load_groups():
    raw_data = os.path.join(RAW, "data")
    groups = defaultdict(list)
    for grade in sorted(os.listdir(raw_data)):
        gdir = os.path.join(raw_data, grade)
        if not os.path.isdir(gdir):
            continue
        for fn in sorted(os.listdir(gdir)):
            patient = fn.rsplit(".", 1)[0][:-1]  # drop trailing L/R
            groups[patient].append((grade, fn, gdir))
    return groups


def split(groups):
    rng = random.Random(SEED)
    keys = sorted(groups)
    rng.shuffle(keys)

    # primary grade of each patient = most frequent grade among its knees
    primary = {}
    for k in keys:
        counts = defaultdict(int)
        for grade, _, _ in groups[k]:
            counts[grade] += 1
        primary[k] = max(counts, key=lambda g: (counts[g], g))

    # stratify the shuffled patients on their primary grade
    by_grade = defaultdict(list)
    for k in keys:
        by_grade[primary[k]].append(k)

    assignment = {}
    for grade, members in sorted(by_grade.items()):
        n = len(members)
        n_train = round(n * RATIOS[0])
        n_val = round(n * RATIOS[1])
        for i, k in enumerate(members):
            if i < n_train:
                assignment[k] = "train"
            elif i < n_train + n_val:
                assignment[k] = "val"
            else:
                assignment[k] = "test"
    return assignment


def materialise(groups, assignment):
    for split_name in ("train", "val", "test"):
        for grade in ("0", "1", "2", "3", "4"):
            os.makedirs(os.path.join(OUT_ROOT, split_name, grade), exist_ok=True)

    stats = defaultdict(lambda: defaultdict(int))
    for patient, items in groups.items():
        split_name = assignment[patient]
        for grade, fn, gdir in items:
            dst = os.path.join(OUT_ROOT, split_name, grade, fn)
            shutil.copy2(os.path.join(gdir, fn), dst)
            stats[split_name][grade] += 1

    for split_name in ("train", "val", "test"):
        row = stats[split_name]
        total = sum(row.values())
        print(f"{split_name:6s} total={total:5d}  " +
              "  ".join(f"{g}={row[g]:5d}" for g in ("0", "1", "2", "3", "4")))


if __name__ == "__main__":
    extract()
    groups = load_groups()
    print("patients:", len(groups), "images:", sum(len(v) for v in groups.values()))
    assignment = split(groups)
    materialise(groups, assignment)
    shutil.rmtree(RAW, ignore_errors=True)
    print("done -> dataset/{train,val,test}")
