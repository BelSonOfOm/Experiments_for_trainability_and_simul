"""Build data_final/ from the raw harvest, by stated rules. The raw harvest is never written to.

Rules (approved by Charbel, 9 Oct 2026):

  R1  Exclusion: sat-generator-2:XOR-3:n=10:count-high is excluded (overall class balance 0.19 < 0.2:
      the median-tie rule had only checked TRAIN).
  R2  Group-aware split ("adapted B"):
      1. samples whose inputs are identical (same canonical key, harvest/dedupe.py) form one group;
         groups chain through units (union-find): two units that share an input join one group;
      2. a problem is re-split ONLY if some group has samples on both sides (TRAIN and TEST); the
         re-split is the harvest's own split (harvest/split.py: 30 % TEST, stratified, seed 0) applied
         to the groups instead of the units;
      3. everything that depends on TRAIN is recomputed for a re-split problem: the median cut of
         score labels (labels re-derived from the recorded scores) and the task operator M̄;
      4. a re-split problem must pass every floor again: >= 100 samples, class balance in
         [0.2, 0.8] overall, in TRAIN and in TEST, >= 30 distinct TEST inputs; otherwise excluded.
  R3  Weights: every sample of every problem gets weight 1 / (number of copies of its input on its
      side of the split), so a weighted TEST score counts each distinct input once. Stored in
      duplicates.json next to the problem (with the group of each sample and the ambiguous samples:
      identical inputs carrying different labels). Problems without duplicates get weights 1.
  R4  Flag (no change): problems with fewer than 50 distinct TEST inputs are listed in the changelog.
  R5  Sample exclusion (approved 9 Oct 2026): molecule samples whose Hartree-Fock calculation did not
      converge (info["hf_converged"] is False) are removed: their Hartree-Fock energy, hence their
      score, is not trustworthy. Applied BEFORE R2-R4. The other samples keep their side of the split
      (removing samples cannot create a leak); what depends on TRAIN is recomputed (median cut of
      score labels, M̄); the problem's counts are recomputed (samples, TRAIN/TEST, class balance,
      distinct inputs, conflict rate, equation checks) and it must pass the floors again (>= 100
      samples, >= 50 distinct inputs, conflict rate <= 0.2, class balance in [0.2, 0.8] overall, in
      TRAIN and in TEST, >= 30 distinct TEST inputs), otherwise it is excluded.

  R6  Sample flags (no change, approved 9 Oct 2026): every problem gets flags.json listing samples the
      analysis must be able to set aside for a sensitivity check (results with and without them):
        - uninformative_operator (COVARIANCE): Ledoit-Wolf shrinkage >= 0.99, so the operator is the
          identity matrix and carries no information about the recording (LSST, RacketSports);
        - degenerate_ground_state (QUANTUM): gap to the next level < 1e-6 (molecules, e.g. the square
          H4 rectangle) or a degeneracy left inside the symmetry sector (spin chains,
          info["residual_degeneracy"]).
      The per-sample shrinkage and gap are stored too, so another threshold can be used later.

Spin-chain problems are read from data_v3/QUANTUM (re-harvested 9-10 Oct 2026 with symmetry-sector
ground states); data_v2/QUANTUM is superseded and no longer read.

Every file not changed by R2 or R5 is copied byte for byte; each problem folder also gets two new
files, duplicates.json (R3) and flags.json (R6). The raw folders are fingerprinted (sha256 of every file)
before and after the build; the two fingerprints must be equal, and both are written to the changelog.

Usage (from COURSEWORK/TME4/, with the project's virtual environment):

    python tools/build_final_dataset.py            # builds data_final/ (refuses if it already holds data)
    python tools/build_final_dataset.py --rebuild  # deletes data_final/ contents first, then builds
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import shutil
import sqlite3
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
TME4 = HERE.parent
PROJECT = TME4.parents[1]                                    # .../DALAS
sys.path.insert(0, str(PROJECT / "code"))                     # the shared harvest code, imported read-only

from harvest.config import HarvestConfig  # noqa: E402
from harvest.dedupe import canonical_key, distinct_and_conflicts  # noqa: E402
from harvest.label_after_split import label_by_train_median  # noqa: E402
from harvest.records import Instance  # noqa: E402
from harvest.split import split_by_unit  # noqa: E402
from harvest.store import SCHEMA  # noqa: E402
from harvest.task_operator import task_operator  # noqa: E402

RAW_FOLDERS = ["data_real/GRAPH-NETWORKS", "data_real/GRAPH-MOLECULES",
               "data_real/EEG-n4", "data_real/EEG-n6", "data_real/EEG-n8", "data_real/EEG-n10",
               "data_real/EEG-n12", "data_online_uea",
               "data_v2/DIAGONAL", "data_v3/QUANTUM", "data_v2/QUANTUM-CHEM"]
EXCLUDE = {"sat-generator-2:XOR-3:n=10:count-high":
           "R1: overall class balance 0.19 < 0.2 (the median-tie rule had only checked TRAIN)"}
MIN_DISTINCT_TEST = 30
FLAG_DISTINCT_TEST = 50
IDENTITY_SHRINKAGE = 0.99
DEGENERATE_GAP = 1e-6
FILES = ["instances.pkl", "operators.pkl", "split.json", "equation.json", "provenance.json"]


# ----------------------------------------------------------------------------------------
# Fingerprint of the raw data
# ----------------------------------------------------------------------------------------

def fingerprint(folders: list) -> dict:
    """WHAT: sha256 of every file under the given folders, then one sha256 over all of them.
    OUTPUT: {"files": number of files, "bytes": total size, "sha256": the combined hash}.
    WHY: proves the raw data are unchanged by the build (computed before and after)."""
    total = hashlib.sha256()
    n_files = n_bytes = 0
    for folder in folders:
        for path in sorted(p for p in folder.rglob("*") if p.is_file()):
            h = hashlib.sha256()
            with open(path, "rb") as f:
                for chunk in iter(lambda: f.read(1 << 20), b""):
                    h.update(chunk)
            total.update(str(path.relative_to(PROJECT)).encode() + h.digest())
            n_files += 1
            n_bytes += path.stat().st_size
    return {"files": n_files, "bytes": n_bytes, "sha256": total.hexdigest()}


# ----------------------------------------------------------------------------------------
# R2 / R3: groups, weights, re-split
# ----------------------------------------------------------------------------------------

def groups_of(instances: list, keys: list) -> list:
    """WHAT: the group of each sample: samples are in one group when they share a unit or an
    identical input, directly or through a chain (union-find).
    HOW: start with one set per unit; for every input key seen before, merge the current unit's set
    with the set of the unit where the key was first seen. Each set is named by one of its units.
    OUTPUT: list of group names, one per sample.
    WHY: rule R2.1."""
    parent = {}

    def find(a):
        root = a
        while parent.setdefault(root, root) != root:
            root = parent[root]
        while parent[a] != root:                              # path compression
            parent[a], a = root, parent[a]
        return root

    first_unit_of_key = {}
    for inst, key in zip(instances, keys):
        find(inst.unit)
        if key in first_unit_of_key:
            a, b = find(inst.unit), find(first_unit_of_key[key])
            if a != b:
                parent[a] = b
        else:
            first_unit_of_key[key] = inst.unit
    return [find(inst.unit) for inst in instances]


def straddling(groups: list, split: list) -> int:
    """WHAT: the number of groups with samples on both sides of the split."""
    sides = defaultdict(set)
    for g, s in zip(groups, split):
        sides[g].add(s)
    return sum(1 for s in sides.values() if len(s) > 1)


def weights_of(keys: list, split: list) -> list:
    """WHAT: rule R3: weight of a sample = 1 / (copies of its input on its side of the split)."""
    copies = Counter(zip(keys, split))
    return [1.0 / copies[(k, s)] for k, s in zip(keys, split)]


def ambiguous_samples(keys: list, labels: list) -> list:
    """WHAT: indices of samples whose input occurs elsewhere in the problem with the other label."""
    labels_of = defaultdict(set)
    for k, y in zip(keys, labels):
        labels_of[k].add(y)
    return [i for i, k in enumerate(keys) if len(labels_of[k]) > 1]


def sample_flags(problem_type: str, instances: list, operators: list) -> dict:
    """WHAT: rule R6: the flags.json content of one problem (indices refer to instances.pkl).
    OUTPUT: {"rule", "uninformative_operator": [i], "degenerate_ground_state": [i],
             "shrinkage": [per sample] (COVARIANCE only), "gap": [per sample] (QUANTUM only)}.
    HOW: COVARIANCE: shrinkage read from the operator (operators.pkl factors) and compared with
    IDENTITY_SHRINKAGE; QUANTUM: gap read from the sample's info, compared with DEGENERATE_GAP, or the
    residual_degeneracy flag set by the symmetry-sector solver. Other types: empty lists."""
    flags = {"rule": f"R6: uninformative_operator = Ledoit-Wolf shrinkage >= {IDENTITY_SHRINKAGE}; "
                     f"degenerate_ground_state = gap < {DEGENERATE_GAP} or residual degeneracy in the symmetry sector",
             "uninformative_operator": [], "degenerate_ground_state": []}
    if problem_type == "COVARIANCE":
        shrinkage = [float(op.factors["shrinkage"]) for op in operators]
        flags["shrinkage"] = shrinkage
        flags["uninformative_operator"] = [i for i, v in enumerate(shrinkage) if v >= IDENTITY_SHRINKAGE]
    if problem_type == "QUANTUM":
        gaps = [float(inst.info.get("gap", np.inf)) for inst in instances]
        flags["gap"] = gaps
        flags["degenerate_ground_state"] = [i for i, (inst, g) in enumerate(zip(instances, gaps))
                                            if g < DEGENERATE_GAP or inst.info.get("residual_degeneracy")]
    return flags


def balance(labels) -> float:
    """Fraction of +1 among labels (nan if empty)."""
    return float(np.mean([y == 1 for y in labels])) if len(labels) else float("nan")


def floor_problems(labels: list, split: list, keys: list, cfg: HarvestConfig, distinct_and_conflict=None) -> list:
    """WHAT: the floors a changed problem must pass again; returns the list of failures (empty = pass).
    CHECKS: >= cfg.m_min samples; class balance in [balance_low, balance_high] overall, in TRAIN and in
    TEST; >= MIN_DISTINCT_TEST distinct TEST inputs; if (n_distinct, conflict_rate) is given, also
    S6 (>= cfg.min_distinct distinct inputs) and S7 (conflict rate <= cfg.max_conflict).
    WHY: R2.4 and R5: a problem changed by the cleanup must still meet every rule of the harvest."""
    problems = []
    if len(labels) < cfg.m_min:
        problems.append(f"{len(labels)} samples")
    for name, side in (("overall", None), ("TRAIN", "TRAIN"), ("TEST", "TEST")):
        values = labels if side is None else [y for y, s in zip(labels, split) if s == side]
        b = balance(values)
        if not cfg.balance_low <= b <= cfg.balance_high:
            problems.append(f"balance {name} {b:.2f}")
    distinct_test = len({k for k, s in zip(keys, split) if s == "TEST"})
    if distinct_test < MIN_DISTINCT_TEST:
        problems.append(f"{distinct_test} distinct TEST inputs")
    if distinct_and_conflict is not None:
        n_distinct, conflict = distinct_and_conflict
        if n_distinct < cfg.min_distinct:
            problems.append(f"S6: {n_distinct} distinct inputs")
        if conflict > cfg.max_conflict:
            problems.append(f"S7: conflict rate {conflict:.2f}")
    return problems


def drop_unconverged(row: dict, eq_row: dict, instances: list, split: list, ops: dict, cfg: HarvestConfig) -> tuple:
    """WHAT: rule R5 on one problem.
    OUTPUT: (instances, split, ops, row, eq_row, record) — unchanged objects and record None when no
    sample has info["hf_converged"] False; otherwise the reduced problem and a record
    {"r5_dropped": [units], "r5_label_flips", "r5_old_cut", "r5_new_cut", "r5_failures": [...]}.
    HOW: keep the samples whose hf_converged is not False, with their split side, operator and
    equation check; for score labels, cut again at the median of the remaining TRAIN scores
    (harvest's own label_by_train_median); recompute M̄ from the remaining TRAIN operators; update
    the database rows (counts, balance, distinct, conflicts, checks, M̄ rule); check the floors."""
    keep = [i for i, inst in enumerate(instances) if inst.info.get("hf_converged") is not False]
    if len(keep) == len(instances):
        return instances, split, ops, row, eq_row, None
    record = {"r5_dropped": [instances[i].unit for i in range(len(instances)) if i not in set(keep)]}
    instances = [instances[i] for i in keep]
    split = [split[i] for i in keep]
    ops = {**ops, "operators": [ops["operators"][i] for i in keep], "checks": [ops["checks"][i] for i in keep]}
    row, eq_row = dict(row), dict(eq_row)
    flips = 0
    if "TRAIN median" in row["label_rule"]:
        old = instances
        instances, cut_rule, _ = label_by_train_median(old, split)
        flips = sum(1 for a, b in zip(old, instances) if a.y != b.y)
        record["r5_old_cut"], record["r5_new_cut"] = old[0].info.get("cut"), instances[0].info["cut"]
        base_rule = row["label_rule"].split("; score > TRAIN median")[0]
        row["label_rule"] = f"{base_rule}; {cut_rule} [final: R5 removed {len(record['r5_dropped'])} unconverged samples]"
    else:
        row["label_rule"] += f" [final: R5 removed {len(record['r5_dropped'])} unconverged samples]"
    record["r5_label_flips"] = flips
    identity = not ops["task_operator"].rule.startswith("sites have no identity")
    ops["task_operator"] = task_operator(ops["operators"], split, identity, cfg)
    labels = [inst.y for inst in instances]
    n_distinct, conflict = distinct_and_conflicts(row["problem_type"], instances)
    row.update(n_samples=len(instances), n_train=split.count("TRAIN"), n_test=split.count("TEST"),
               class_balance=balance(labels), n_distinct=n_distinct, conflict_rate=conflict)
    eq_row.update(task_operator_rule=ops["task_operator"].rule,
                  task_operator_support_size=len(ops["task_operator"].support),
                  check_max_residual=max(c.residual for c in ops["checks"]),
                  n_failed=sum(1 for c in ops["checks"] if not c.passed))
    keys = [canonical_key(row["problem_type"], inst.x) for inst in instances]
    record["r5_failures"] = floor_problems(labels, split, keys, cfg, (n_distinct, conflict))
    return instances, split, ops, row, eq_row, record


# ----------------------------------------------------------------------------------------
# The build
# ----------------------------------------------------------------------------------------

def process_problem(row: dict, eq_row: dict, raw_dir: Path, out_dir: Path, cfg: HarvestConfig) -> dict:
    """WHAT: apply R2-R4 to one problem and write it into out_dir.
    OUTPUT: a change record {"task_id", "action": "copied" | "weights" | "re-split" | "excluded", ...}
    and, unless excluded, the updated database rows under "row" and "eq_row".
    HOW: read instances and split; compute keys, groups, weights; if a group straddles the split,
    re-split by groups (harvest's split_by_unit), recompute median-cut labels and M̄, check the
    floors; write the problem folder (files copied byte for byte unless they changed) and
    duplicates.json."""
    task_id = row["task_id"]
    instances = pickle.load(open(raw_dir / "instances.pkl", "rb"))
    split = json.loads((raw_dir / "split.json").read_text())
    record = {"task_id": task_id}
    # R5 first (only molecule problems carry hf_converged; every other problem passes through unchanged)
    r5_ops = None
    if any(inst.info.get("hf_converged") is False for inst in instances):
        ops = pickle.load(open(raw_dir / "operators.pkl", "rb"))
        instances, split, r5_ops, row, eq_row, r5 = drop_unconverged(row, eq_row, instances, split, ops, cfg)
        record.update(r5)
        if r5["r5_failures"]:
            record["action"] = "excluded"
            record["reason"] = "R5 after removing unconverged samples: " + "; ".join(r5["r5_failures"])
            return record
    r5_changed = r5_ops is not None
    keys = [canonical_key(row["problem_type"], inst.x) for inst in instances]
    groups = groups_of(instances, keys)
    record.update(n_samples=len(instances), distinct=len(set(keys)))
    n_straddling = straddling(groups, split)
    new_split, new_instances, new_ops = split, instances, r5_ops
    row, eq_row = dict(row), dict(eq_row)

    if n_straddling:
        record["action"] = "re-split"
        record["straddling_groups"] = n_straddling
        record["old_train_test"] = [split.count("TRAIN"), split.count("TEST")]
        grouped = [Instance(x=None, y=inst.y, unit=g) for inst, g in zip(instances, groups)]
        new_split = split_by_unit(grouped, cfg.test_fraction, cfg.seed)
        record["samples_changing_side"] = sum(1 for a, b in zip(split, new_split) if a != b)
        flips = 0
        if "TRAIN median" in row["label_rule"]:
            scores = [inst.info["score"] for inst in instances]
            cut = float(np.median([s for s, side in zip(scores, new_split) if side == "TRAIN"]))
            new_instances = []
            for inst, s in zip(instances, scores):
                y = 1 if s > cut else -1
                flips += y != inst.y
                new_instances.append(Instance(x=inst.x, y=y, unit=inst.unit, info={**inst.info, "cut": cut}))
            base_rule = row["label_rule"].split("; score > TRAIN median")[0]
            row["label_rule"] = f"{base_rule}; score > TRAIN median ({cut:.6g}) [final: recomputed after group-aware re-split]"
            record["old_cut"] = instances[0].info.get("cut")
            record["new_cut"] = cut
        record["label_flips"] = flips
        ops = r5_ops if r5_changed else pickle.load(open(raw_dir / "operators.pkl", "rb"))
        old_rule = ops["task_operator"].rule
        identity = not old_rule.startswith("sites have no identity")
        ops["task_operator"] = task_operator(ops["operators"], new_split, identity, cfg)
        new_ops = ops
        record["new_train_test"] = [new_split.count("TRAIN"), new_split.count("TEST")]
        # R2.4: floors
        labels = [inst.y for inst in new_instances]
        problems = floor_problems(labels, new_split, keys, cfg)
        if problems:
            record["action"] = "excluded"
            record["reason"] = "R2.4 after re-split: " + "; ".join(problems)
            return record
        row["n_train"], row["n_test"] = new_split.count("TRAIN"), new_split.count("TEST")
        row["class_balance"] = balance(labels)
        row["balance_rule"] = row["balance_rule"] + " [final: group-aware re-split]"
        eq_row["task_operator_rule"] = ops["task_operator"].rule
        eq_row["task_operator_support_size"] = len(ops["task_operator"].support)
    elif r5_changed:
        record["action"] = "samples-removed"
    elif len(set(keys)) < len(keys):
        record["action"] = "weights"
    else:
        record["action"] = "copied"

    labels = [inst.y for inst in new_instances]
    weights = weights_of(keys, new_split)
    ambiguous = ambiguous_samples(keys, labels)
    record["distinct_test"] = len({k for k, s in zip(keys, new_split) if s == "TEST"})
    record["ambiguous"] = len(ambiguous)
    record["test_balance"] = balance([y for y, s in zip(labels, new_split) if s == "TEST"])

    target = out_dir / "problems" / Path(row["problem_dir"]).name
    target.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        shutil.copy2(raw_dir / name, target / name)
    if new_split is not split or r5_changed:
        (target / "split.json").write_text(json.dumps(new_split))
    if new_instances is not instances or r5_changed:
        with open(target / "instances.pkl", "wb") as f:
            pickle.dump(new_instances, f)
    if new_ops is not None:
        with open(target / "operators.pkl", "wb") as f:
            pickle.dump(new_ops, f)
    if r5_changed:                                            # equation.json carries the check summary
        equation = json.loads((raw_dir / "equation.json").read_text())
        equation["check_max_residual"], equation["n_failed"] = eq_row["check_max_residual"], eq_row["n_failed"]
        (target / "equation.json").write_text(json.dumps(equation, indent=2))
    if row["problem_type"] == "COVARIANCE":
        operators = (new_ops or pickle.load(open(raw_dir / "operators.pkl", "rb")))["operators"]
    else:
        operators = []
    flags = sample_flags(row["problem_type"], new_instances, operators)
    (target / "flags.json").write_text(json.dumps(flags))
    record["flag_uninformative"] = len(flags["uninformative_operator"])
    record["flag_degenerate"] = len(flags["degenerate_ground_state"])
    record["n_samples"] = len(new_instances)
    group_index = {g: i for i, g in enumerate(dict.fromkeys(groups))}
    (target / "duplicates.json").write_text(json.dumps({
        "rule": "R3: weight = 1 / copies of the same input on the same side of the split",
        "weights": weights, "group": [group_index[g] for g in groups], "ambiguous_samples": ambiguous}))
    row["problem_dir"] = str(target.relative_to(out_dir))
    record["row"], record["eq_row"] = row, eq_row
    return record


def write_changelog(path: Path, records: list, before: dict, after: dict, counts: dict) -> None:
    """WHAT: CHANGELOG.md: the rules, the raw fingerprints, and every changed or excluded problem."""
    changed = [r for r in records if r["action"] in ("re-split", "weights")]
    excluded = [r for r in records if r["action"] == "excluded"]
    flagged = [r for r in records if r["action"] != "excluded" and r.get("distinct_test", 999) < FLAG_DISTINCT_TEST]
    lines = ["# data_final — changelog", "",
             f"Built {datetime.now(timezone.utc).isoformat(timespec='seconds')} by `tools/build_final_dataset.py` "
             "from the raw harvest. Rules R1-R6 are stated in the script's header and approved by Charbel (9 Oct 2026).", "",
             "## Raw data untouched", "",
             "| | files | bytes | sha256 of all raw files |", "|---|---|---|---|",
             f"| before the build | {before['files']} | {before['bytes']} | `{before['sha256']}` |",
             f"| after the build | {after['files']} | {after['bytes']} | `{after['sha256']}` |", "",
             "Equal: yes" if before == after else "**NOT EQUAL: the raw data changed during the build. Do not use data_final.**", "",
             "## Counts", "",
             "| | problems |", "|---|---|"]
    lines += [f"| {k} | {v} |" for k, v in counts.items()]
    lines += ["", "## Excluded", "", "| problem | reason |", "|---|---|"]
    lines += [f"| {r['task_id']} | {r['reason']} |" for r in excluded]
    lines += ["", "## Re-split (R2): groups of identical inputs had straddled TRAIN and TEST", "",
              "| problem | straddling groups | TRAIN/TEST before | after | samples changing side | label flips | median cut before → after | distinct TEST inputs | TEST balance | ambiguous samples |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    for r in changed:
        if r["action"] != "re-split":
            continue
        cut = f"{r['old_cut']:.6g} → {r['new_cut']:.6g}" if "new_cut" in r else "—"
        lines.append(f"| {r['task_id']} | {r['straddling_groups']} | {r['old_train_test'][0]}/{r['old_train_test'][1]} | "
                     f"{r['new_train_test'][0]}/{r['new_train_test'][1]} | {r['samples_changing_side']} | {r['label_flips']} | "
                     f"{cut} | {r['distinct_test']} | {r['test_balance']:.2f} | {r['ambiguous']} |")
    lines += ["", "## Weights only (R3): duplicates all on one side, split unchanged", "",
              "| problem | samples | distinct inputs | ambiguous samples |", "|---|---|---|---|"]
    lines += [f"| {r['task_id']} | {r['n_samples']} | {r['distinct']} | {r['ambiguous']} |"
              for r in changed if r["action"] == "weights"]
    lines += ["", "## Samples removed (R5): Hartree-Fock not converged", "",
              "| problem | removed samples | label flips | median cut before → after | outcome |", "|---|---|---|---|---|"]
    for r in records:
        if "r5_dropped" in r:
            cut = (f"{r['r5_old_cut']:.6g} → {r['r5_new_cut']:.6g}" if r.get("r5_new_cut") is not None else "fixed cut")
            outcome = r["reason"] if r["action"] == "excluded" else "kept"
            lines.append(f"| {r['task_id']} | {', '.join(r['r5_dropped'])} | {r['r5_label_flips']} | {cut} | {outcome} |")
    lines += ["", f"## Flagged (R4): fewer than {FLAG_DISTINCT_TEST} distinct TEST inputs", "",
              "| problem | distinct TEST inputs |", "|---|---|"]
    lines += [f"| {r['task_id']} | {r['distinct_test']} |" for r in flagged]
    lines += ["", "## Flagged samples (R6): kept, listed in each problem's flags.json for a sensitivity check", "",
              "| problem | samples | uninformative operator | degenerate ground state |", "|---|---|---|---|"]
    lines += [f"| {r['task_id']} | {r['n_samples']} | {r['flag_uninformative']} | {r['flag_degenerate']} |"
              for r in records if r["action"] != "excluded" and (r["flag_uninformative"] or r["flag_degenerate"])]
    kept = [r for r in records if r["action"] != "excluded"]
    lines += [f"| **total** | {sum(r['n_samples'] for r in kept)} | {sum(r['flag_uninformative'] for r in kept)} | "
              f"{sum(r['flag_degenerate'] for r in kept)} |"]
    lines += ["", "All other problems are copied byte for byte from the raw harvest, with a duplicates.json "
              "holding weights 1 and a flags.json with empty lists."]
    path.write_text("\n".join(lines) + "\n")


def main():
    """WHAT: command line (see the module docstring)."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default=str(TME4 / "data_final"))
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument("--raw", nargs="*", default=RAW_FOLDERS, help=argparse.SUPPRESS)
    args = parser.parse_args()
    out = Path(args.out)
    raw_folders = [Path(f) if Path(f).is_absolute() else PROJECT / f for f in args.raw]
    if (out / "harvest.sqlite").exists() or (out / "problems").exists():
        if not args.rebuild:
            raise SystemExit(f"{out} already holds data; use --rebuild to replace it")
        shutil.rmtree(out / "problems", ignore_errors=True)
        (out / "harvest.sqlite").unlink(missing_ok=True)
    out.mkdir(parents=True, exist_ok=True)
    cfg = HarvestConfig()

    print("fingerprinting the raw data (before) ...", flush=True)
    before = fingerprint(raw_folders)
    db_out = sqlite3.connect(out / "harvest.sqlite")
    db_out.executescript(SCHEMA)
    db_out.execute("CREATE TABLE IF NOT EXISTS final_changes (task_id TEXT, action TEXT, detail_json TEXT)")
    db_out.execute("CREATE TABLE IF NOT EXISTS raw_source (task_id TEXT PRIMARY KEY, raw_folder TEXT)")
    records = []
    for folder in raw_folders:
        db = sqlite3.connect(f"file:{folder / 'harvest.sqlite'}?mode=ro", uri=True)   # read-only
        db.row_factory = sqlite3.Row
        eq_rows = {r["task_id"]: dict(r) for r in db.execute("SELECT * FROM equation")}
        for log in db.execute("SELECT source, dataset, stage, reason, n_samples, logged_at FROM harvest_log"):
            db_out.execute("INSERT INTO harvest_log (source, dataset, stage, reason, n_samples, logged_at) "
                           "VALUES (?,?,?,?,?,?)", tuple(log))
        rows = [dict(r) for r in db.execute("SELECT * FROM task")]
        print(f"{folder.relative_to(PROJECT)}: {len(rows)} problems", flush=True)
        for row in rows:
            raw_dir = folder / "problems" / Path(row["problem_dir"]).name
            if row["task_id"] in EXCLUDE:
                record = {"task_id": row["task_id"], "action": "excluded", "reason": EXCLUDE[row["task_id"]]}
            else:
                record = process_problem(row, eq_rows[row["task_id"]], raw_dir, out, cfg)
            if record["action"] != "excluded":
                r, e = record.pop("row"), record.pop("eq_row")
                db_out.execute(f"INSERT INTO task VALUES ({','.join('?' * len(r))})", tuple(r.values()))
                db_out.execute(f"INSERT INTO equation VALUES ({','.join('?' * len(e))})", tuple(e.values()))
                db_out.execute("INSERT INTO raw_source VALUES (?, ?)", (row["task_id"], str(folder.relative_to(PROJECT))))
            db_out.execute("INSERT INTO final_changes VALUES (?, ?, ?)",
                           (record["task_id"], record["action"], json.dumps(record, default=str)))
            records.append(record)
        db_out.commit()
    db_out.close()
    print("fingerprinting the raw data (after) ...", flush=True)
    after = fingerprint(raw_folders)
    counts = Counter(r["action"] for r in records)
    counts = {"in the raw harvest": len(records), "copied unchanged (weights 1)": counts["copied"],
              "weights only (R3)": counts["weights"], "re-split (R2)": counts["re-split"],
              "samples removed (R5)": sum(1 for r in records if "r5_dropped" in r and r["action"] != "excluded"),
              "excluded (R1, R2.4, R5)": counts["excluded"],
              "with flagged samples (R6)": sum(1 for r in records if r["action"] != "excluded"
                                               and (r["flag_uninformative"] or r["flag_degenerate"])),
              "in data_final": len(records) - counts["excluded"]}
    write_changelog(out / "CHANGELOG.md", records, before, after, counts)
    print(json.dumps(counts, indent=1))
    print("raw data unchanged:", before == after)


if __name__ == "__main__":
    main()
