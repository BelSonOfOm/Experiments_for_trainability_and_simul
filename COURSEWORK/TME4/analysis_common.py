"""analysis_common.py — shared loading and bookkeeping for the whole study (Q1–Q4, Q6).

The specification is `QnA_what_to_do.md` in this folder; section numbers below refer to it. Every
function says WHAT it does, HOW and WHY (the project's convention).

WHAT THIS FILE PROVIDES
1. Where things are (`paths`): the project root `DALAS/`, the harvest package `DALAS/code/harvest`,
   the dataset `COURSEWORK/TME4/data_final/`, its audit report, the raw harvest folders (only for the
   funnel, Q1 1.4), and the output folders `COURSEWORK/TME4/results/` (tables) and `results/figures/`.
2. Making the harvest package importable (`import_harvest`). This is required: `instances.pkl` and
   `operators.pkl` are Python pickles of objects defined in `harvest/records.py` (Instance, Operator),
   and a pickle can only be read back when those classes can be imported.
3. The master table (`load_index`, piece 1.1): one row per problem of data_final, read from
   `data_final/harvest.sqlite` (table `task`). Every question starts from it.
4. Reading one problem completely (`load_problem`, the `Problem` class): instances, operators, split,
   R3 weights (`duplicates.json`), R6 flags (`flags.json`), provenance, equation.
5. Distinct input sets (decision D4.1, spec 2.3): problems that share exactly the same inputs are
   merged into one input set, because structure features never use labels. One key per problem, then
   a table input set -> problems.
6. Small shared helpers used by several questions: the tier of a problem, the Holm correction, saving
   tables and figures in one place with one naming rule, the colours of the four types.

FILES THE ANALYSIS READS (format, from the harvest code and tools/build_final_dataset.py)
  data_final/harvest.sqlite          tables `task` (one row per problem: task_id, problem_type, n_qubits,
                                     n_samples, n_train, n_test, class_balance, n_distinct, group_id,
                                     origin, source, dataset, modelling_map_json, problem_dir, ...),
                                     `equation`, `harvest_log`
  data_final/problems/<folder>/instances.pkl    list[harvest.records.Instance] (x, y, unit, info)
  data_final/problems/<folder>/operators.pkl    {"operators": list[Operator], "checks": ..., "task_operator": ...}
  data_final/problems/<folder>/split.json       ["TRAIN" | "TEST", ...], one per instance
  data_final/problems/<folder>/duplicates.json  {"rule": ..., "weights": [...], ...}   (R3)
  data_final/problems/<folder>/flags.json       {"rule", "uninformative_operator": [indices],
                                                 "degenerate_ground_state": [indices], "gap": [...], ...} (R6)
  data_final/problems/<folder>/provenance.json  source, dataset, group_id, origin, modelling_map, ...
  data_final/problems/<folder>/equation.json    form, operator kind, check_max_residual, n_failed, ...
"""
from __future__ import annotations

import functools
import hashlib
import json
import pickle
import sqlite3
import sys
from contextlib import closing
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Optional

TYPES = ("GRAPH", "COVARIANCE", "DIAGONAL", "QUANTUM")
"""The four problem types, always in this order in tables and figures."""

TYPE_COLOURS = {"GRAPH": "#2a78d6", "COVARIANCE": "#eb6834", "DIAGONAL": "#1baf7a", "QUANTUM": "#eda100"}
"""One fixed colour per type, shared by every figure of Q1-Q4 so a type looks the same everywhere: the first
four slots of the validated categorical palette of the dataviz guidance (blue, orange, aqua, yellow), checked
with its validator (colour-blind separation PASS; two colours below 3:1 contrast on white, so every figure
also names the types in a legend or label, never by colour alone)."""

INK = {"primary": "#1f1f1e", "secondary": "#5f5e5a", "grid": "#e4e3df"}
"""Text and grid colours of the figures (text never wears a series colour)."""


def figure_style() -> None:
    """WHAT: set matplotlib defaults shared by every figure: thin 2 px lines, light grid behind the data, no
    top/right frame, text in INK colours, a clean sans-serif font.
    WHY: every figure of the report looks like one system."""
    import matplotlib
    matplotlib.use("Agg")                                   # files only, no window
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.dpi": 100, "savefig.facecolor": "white", "axes.facecolor": "white",
        "axes.edgecolor": INK["secondary"], "axes.labelcolor": INK["primary"], "axes.titlecolor": INK["primary"],
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "axes.axisbelow": True,
        "grid.color": INK["grid"], "grid.linewidth": 0.8, "xtick.color": INK["secondary"],
        "ytick.color": INK["secondary"], "text.color": INK["primary"], "lines.linewidth": 2,
        "lines.markersize": 5, "font.size": 9, "axes.titlesize": 10, "legend.frameon": False})

FLAG_KINDS = ("uninformative_operator", "degenerate_ground_state")
"""The two R6 flag kinds stored in flags.json."""

EXPECTED_PROBLEMS = 1368
"""Number of problems in data_final (build of 9 Oct 2026, audited: 0 FAIL)."""


@dataclass(frozen=True)
class Paths:
    """FIELDS: root = DALAS/; code = DALAS/code (contains the `harvest` package); data_final =
    COURSEWORK/TME4/data_final; audit = the audit report of data_final
    (DALAS/audit/data_final/TME4__data_final.json, written by code/scripts/audit_dataset.py); raw_folders =
    the raw harvest folders, the same list as RAW_FOLDERS in tools/build_final_dataset.py, as absolute
    paths; results = COURSEWORK/TME4/results; figures = results/figures."""
    root: Path
    code: Path
    data_final: Path
    audit: Path
    raw_folders: tuple
    results: Path
    figures: Path


@functools.lru_cache(maxsize=None)
def paths() -> Paths:
    """WHAT: the Paths of the project, computed once (lru_cache stores the first result).
    HOW: tme4 = the folder of this file (Path(__file__).resolve().parent); root = tme4.parents[1] (the
    same pattern as tools/build_final_dataset.py, lines 67-69); build every other path from these two;
    create results/ and results/figures/ if missing. RAW_FOLDERS is imported from
    tools/build_final_dataset.py rather than retyped, so the two lists can never disagree (the import
    does not start a build: the build only runs under `if __name__ == "__main__"`).
    WHY: one place for every path; nothing else in the analysis writes a path by hand."""
    tme4 = Path(__file__).resolve().parent                 # .../DALAS/COURSEWORK/TME4 (local only)
    root = tme4.parents[1]                                  # .../DALAS
    if str(tme4) not in sys.path:                           # so `tools` is importable from any working folder
        sys.path.insert(0, str(tme4))
    from tools.build_final_dataset import RAW_FOLDERS      # the raw list, defined once in the build script
    results = tme4 / "results"
    figures = results / "figures"
    figures.mkdir(parents=True, exist_ok=True)              # also creates results/; no error if present
    return Paths(root=root,
                 code=root / "code",
                 data_final=tme4 / "data_final",
                 audit=root / "audit" / "data_final" / "TME4__data_final.json",
                 raw_folders=tuple(root / folder for folder in RAW_FOLDERS),
                 results=results,
                 figures=figures)


def import_harvest() -> None:
    """WHAT: make `import harvest` work.
    HOW: if str(paths().code) is not in sys.path, insert it at position 0; then `import harvest` once to
    fail early with a clear message ("harvest package not found at ...") if it is missing.
    WHY: instances.pkl / operators.pkl contain harvest.records objects; unpickling needs the classes
    (pickle stores a class by its module name, which must be importable when reading back)."""
    code = str(paths().code)
    if code not in sys.path:
        sys.path.insert(0, code)
    try:
        import harvest  # noqa: F401  (imported only to check it is there)
    except ImportError as error:
        raise ImportError(f"harvest package not found at {code}: {error}") from error


@dataclass
class Problem:
    """One problem of data_final, fully loaded.

    FIELDS: task_id; problem_type (one of TYPES); n (number of sites = n_qubits in the database);
    group_id (the independent source); origin (REAL, REAL_INPUT_COMPUTED_LABEL, SYNTHETIC); source;
    dataset; folder (Path of the problem folder); instances (list of harvest Instance); operators (list of
    harvest Operator, same order as instances, or None if not loaded); split (list of "TRAIN"/"TEST");
    weights (R3 weights, list of floats); flags (the flags.json dict); provenance (dict); equation (dict).
    Index i means the same sample in instances, operators, split, weights and the flag lists.
    """
    task_id: str
    problem_type: str
    n: int
    group_id: str
    origin: str
    source: str
    dataset: str
    folder: Path
    instances: list = field(default_factory=list)
    operators: Optional[list] = None
    split: list = field(default_factory=list)
    weights: list = field(default_factory=list)
    flags: dict = field(default_factory=dict)
    provenance: dict = field(default_factory=dict)
    equation: dict = field(default_factory=dict)

    def train_indices(self) -> list:
        """WHAT: indices i with split[i] == "TRAIN". HOW: list comprehension. WHY: TRAIN-only steps."""
        return [i for i, side in enumerate(self.split) if side == "TRAIN"]

    def test_indices(self) -> list:
        """WHAT: indices i with split[i] == "TEST". HOW: list comprehension (compared to "TEST" explicitly,
        so a corrupted value is not silently counted as TEST)."""
        return [i for i, side in enumerate(self.split) if side == "TEST"]

    def labels(self) -> list:
        """WHAT: the labels y (+1 / -1) of all instances, in order. HOW: [inst.y for inst in instances]."""
        return [instance.y for instance in self.instances]

    def flagged(self, kinds: tuple = FLAG_KINDS) -> list:
        """WHAT: a list of booleans, True for every instance flagged by R6 under one of `kinds`.
        HOW: start all False; for each kind in kinds, set True at the indices listed in flags[kind]
        (flags.json stores positions, tools/build_final_dataset.py lines 179-187). An unknown kind raises
        (a typo would otherwise silently exclude nothing); a missing key counts as nothing flagged.
        WHY: the sensitivity checks (2.7, 3.8, 4.6) recompute everything without these samples."""
        mask = [False] * len(self.instances)
        for kind in kinds:
            if kind not in FLAG_KINDS:
                raise ValueError(f"unknown flag kind {kind!r}; expected one of {FLAG_KINDS}")
            for i in self.flags.get(kind, []):
                mask[i] = True
        return mask


def _connect() -> sqlite3.Connection:
    """WHAT: a connection to data_final/harvest.sqlite. Use with contextlib.closing (a sqlite3
    connection used in `with` only commits, it does not close)."""
    return sqlite3.connect(paths().data_final / "harvest.sqlite")


def tier_of(origin: str, source: str) -> int:
    """WHAT: the data tier 1, 2 or 3 of a problem (tiers defined in docs/LOGBOOK.md, 8 Oct night).
    HOW: REAL -> 1; REAL_INPUT_COMPUTED_LABEL -> 2; SYNTHETIC -> 3, EXCEPT the QUANTUM molecules
    (source "chem-generator"), which are tier 2 (real molecules, computed labels) although their stored
    origin is SYNTHETIC (inconsistency found in the 9 Oct audit; to be aligned).
    WHY: Q6 and the slides split results by tier."""
    if origin == "REAL":
        return 1
    if origin == "REAL_INPUT_COMPUTED_LABEL":
        return 2
    if origin == "SYNTHETIC":
        return 2 if source == "chem-generator" else 3
    raise ValueError(f"unknown origin {origin!r}")


def load_index() -> "pandas.DataFrame":
    """WHAT (piece 1.1): the master table, one row per problem of data_final.
    COLUMNS: task_id, problem_type, n, n_samples, n_train, n_test, class_balance, n_distinct, group_id,
    origin, tier, source, dataset, subfamily, modelling_map (dict), folder.
    HOW: read table `task` of data_final/harvest.sqlite with sqlite3 + pandas.read_sql; rename n_qubits -> n;
    folder = data_final/<problem_dir>; subfamily = the `dataset` field as stored (e.g. "TFIM-open",
    "XOR-3", "Cho2017"); tier = tier_of(origin, source). Check that the number of rows is
    EXPECTED_PROBLEMS and that every folder exists; raise if not. (The domain of Q3b is added by
    q3_classify.domain_labels; the input set by attach_input_sets.)
    WHY: every later step (counts, features, models) joins on this table."""
    import pandas as pd
    with closing(_connect()) as db:
        index = pd.read_sql("SELECT task_id, problem_type, n_qubits AS n, n_samples, n_train, n_test, "
                            "class_balance, n_distinct, group_id, origin, source, dataset, "
                            "modelling_map_json, problem_dir FROM task ORDER BY task_id", db)
    if len(index) != EXPECTED_PROBLEMS:
        raise ValueError(f"data_final has {len(index)} problems, expected {EXPECTED_PROBLEMS}")
    index["subfamily"] = index["dataset"]
    index["tier"] = [tier_of(o, s) for o, s in zip(index["origin"], index["source"])]
    index["modelling_map"] = [json.loads(m) for m in index["modelling_map_json"]]
    index["folder"] = [paths().data_final / d for d in index["problem_dir"]]
    missing = [str(f) for f in index["folder"] if not f.is_dir()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} problem folders missing, e.g. {missing[:3]}")
    return index.drop(columns=["modelling_map_json", "problem_dir"])


def load_problem(task_id: str, with_operators: bool = True) -> Problem:
    """WHAT: one Problem, fully loaded from its folder.
    HOW: read its row from data_final/harvest.sqlite (table task) directly (not through load_index, which
    reads the whole table: calling it once per problem would repeat that work 1 368 times); read
    instances.pkl, operators.pkl ["operators"] (if with_operators), split.json, duplicates.json (weights;
    all 1.0 if absent: R3 gives weight 1 to problems without duplicates), flags.json (empty if absent),
    provenance.json, equation.json. Check that every per-sample list has the same length and that no flag
    points past the last sample; raise if not.
    WHY: one reader for the whole study, with the consistency checks in one place."""
    import_harvest()                                        # the pickles contain harvest.records objects
    data = paths().data_final
    with closing(_connect()) as db:
        row = db.execute("SELECT problem_type, n_qubits, group_id, origin, source, dataset, problem_dir "
                         "FROM task WHERE task_id = ?", (task_id,)).fetchone()
    if row is None:
        raise KeyError(f"no problem {task_id!r} in {data / 'harvest.sqlite'}")
    problem_type, n, group_id, origin, source, dataset, problem_dir = row
    folder = data / problem_dir

    def read_json(name: str, default=None):
        """Read one JSON file of the folder; a missing file raises unless a default is given."""
        path = folder / name
        if not path.exists():
            if default is None:
                raise FileNotFoundError(path)
            return default
        return json.loads(path.read_text())

    with open(folder / "instances.pkl", "rb") as f:
        instances = pickle.load(f)
    operators = None
    if with_operators:
        with open(folder / "operators.pkl", "rb") as f:
            operators = pickle.load(f)["operators"]
    split = read_json("split.json")
    weights = read_json("duplicates.json", {}).get("weights", [1.0] * len(instances))
    flags = read_json("flags.json", {})

    sizes = {"instances": len(instances), "split": len(split), "weights": len(weights)}
    if operators is not None:
        sizes["operators"] = len(operators)
    if len(set(sizes.values())) != 1:
        raise ValueError(f"{task_id}: per-sample lists of different lengths {sizes}")
    for kind in FLAG_KINDS:
        if any(i >= len(instances) for i in flags.get(kind, [])):
            raise ValueError(f"{task_id}: flags.json {kind} points past the last sample")

    return Problem(task_id=task_id, problem_type=problem_type, n=int(n), group_id=group_id, origin=origin,
                   source=source, dataset=dataset, folder=folder, instances=instances,
                   operators=operators, split=split, weights=weights, flags=flags,
                   provenance=read_json("provenance.json"), equation=read_json("equation.json"))


def iter_problems(types: tuple = TYPES, with_operators: bool = True) -> Iterator[Problem]:
    """WHAT: every problem of the given types, one at a time (a generator), in task_id order.
    HOW: read the task ids from the database (ORDER BY task_id, so every run has the same order); `yield`
    one load_problem at a time, so the previous problem can be freed from memory.
    WHY: the dataset is 7.5 GB; never hold all problems in memory at once."""
    with closing(_connect()) as db:
        rows = db.execute("SELECT task_id, problem_type FROM task ORDER BY task_id").fetchall()
    for task_id, problem_type in rows:
        if problem_type in types:
            yield load_problem(task_id, with_operators)


def input_set_key(problem: Problem) -> str:
    """WHAT (D4.1): a key that is equal for two problems exactly when they have the same inputs.
    HOW: k_i = harvest.dedupe.canonical_key(problem_type, instance.x) for every instance (the harvest's
    own notion of 'same input', also used by the audit); sort the k_i (order ignored, repeats kept, labels
    ignored); return the sha1 of the keys joined by newlines (the separator prevents two different lists
    from gluing into the same string).
    WHY: structure features depend only on the inputs; problems sharing inputs (e.g. the score labels of
    one network at one n) must count as one point in Q2 and Q4.
    NOTE: canonical_key treats two SAT formulas equal up to variable renaming as DIFFERENT (its documented
    choice); keep it, and say so in the report. Problems whose inputs only partly overlap (property labels
    balanced after drawing; a molecule problem that lost samples under R5) are NOT merged."""
    import_harvest()
    from harvest.dedupe import canonical_key
    keys = sorted(canonical_key(problem.problem_type, instance.x) for instance in problem.instances)
    return hashlib.sha1("\n".join(keys).encode()).hexdigest()


def attach_input_sets(index: "pandas.DataFrame") -> tuple:
    """WHAT: the index with a column `input_set` (the key above), and a second table, one row per input
    set: input_set, problem_type, n, group_id, subfamily, tier, n_problems, task_ids (";"-joined).
    HOW: compute input_set_key for every problem (instances only, no operators); cache the keys in
    results/input_set_keys.csv (recomputed if data_final changes: the cache stores the modification time of
    data_final/harvest.sqlite); group by key. Check that an input set never mixes two group_ids or two
    values of n; raise if it does.
    OUTPUT: (index_with_column, input_sets_table); input_sets_table is saved to results/input_sets.csv.
    WHY: the unit of Q2 and Q4 (spec 2.3)."""
    import pandas as pd
    cache = paths().results / "input_set_keys.csv"
    stamp = str((paths().data_final / "harvest.sqlite").stat().st_mtime)
    keys = {}
    if cache.exists():
        cached = pd.read_csv(cache)
        if len(cached) and str(cached["stamp"].iloc[0]) == stamp:
            keys = dict(zip(cached["task_id"], cached["input_set"]))
    missing = [t for t in index["task_id"] if t not in keys]
    for k, task_id in enumerate(missing):
        keys[task_id] = input_set_key(load_problem(task_id, with_operators=False))
        if (k + 1) % 100 == 0 or k + 1 == len(missing):
            pd.DataFrame({"task_id": list(keys), "input_set": list(keys.values()), "stamp": stamp}).to_csv(
                cache, index=False)
            print(f"input-set keys: {len(keys)}/{len(index)}", flush=True)
    index = index.copy()
    index["input_set"] = index["task_id"].map(keys)
    rows = []
    for key, part in index.groupby("input_set", sort=False):
        if part["group_id"].nunique() != 1 or part["n"].nunique() != 1:
            raise ValueError(f"input set {key} mixes groups or sizes: {list(part['task_id'])}")
        first = part.iloc[0]
        rows.append({"input_set": key, "problem_type": first["problem_type"], "n": int(first["n"]),
                     "group_id": first["group_id"], "subfamily": first["subfamily"], "tier": int(first["tier"]),
                     "n_problems": len(part), "task_ids": ";".join(part["task_id"])})
    sets = pd.DataFrame(rows).sort_values(["problem_type", "subfamily", "n", "input_set"]).reset_index(drop=True)
    sets.to_csv(paths().results / "input_sets.csv", index=False)
    return index, sets


def holm(pvalues: list) -> list:
    """WHAT: Holm-corrected p-values, same order as the input; NaN inputs stay NaN and are not counted.
    HOW: sort ascending; the k-th smallest (k = 0, 1, ...) is multiplied by (m - k), m = number of tests;
    then enforce monotonicity (each corrected value >= the previous one in sorted order); cap at 1.
    WHY: many features are tested; Holm keeps the chance of even one false 'difference' at 5 %.
    TEST: [0.01, 0.04, 0.03] -> [0.03, 0.06, 0.06]."""
    import math
    valid = [(p, i) for i, p in enumerate(pvalues) if p is not None and not math.isnan(p)]
    valid.sort()
    m = len(valid)
    corrected = [float("nan")] * len(pvalues)
    running = 0.0
    for k, (p, i) in enumerate(valid):
        running = max(running, min(1.0, (m - k) * p))
        corrected[i] = running
    return corrected


def _markdown(table: "pandas.DataFrame") -> str:
    """WHAT: a Markdown version of a table, without extra dependencies (floats with 4 significant digits)."""
    def cell(v):
        if isinstance(v, float):
            return "" if v != v else f"{v:.4g}"
        return str(v).replace("|", "\\|")
    header = "| " + " | ".join(str(c) for c in table.columns) + " |"
    rule = "|" + "---|" * len(table.columns)
    body = ["| " + " | ".join(cell(v) for v in row) + " |" for row in table.itertuples(index=False)]
    return "\n".join([header, rule, *body]) + "\n"


def save_table(table: "pandas.DataFrame", name: str, folder: Optional[Path] = None) -> Path:
    """WHAT: write a table to results/<name>.csv AND results/<name>.md (Markdown, for the chapters).
    HOW: DataFrame.to_csv and a small Markdown writer; return the csv path. `folder` overrides results/
    (used by --quick runs, which write to results/quick/).
    WHY: every number in the report comes from a saved file, never from a notebook cell."""
    folder = folder or paths().results
    folder.mkdir(parents=True, exist_ok=True)
    csv_path = folder / f"{name}.csv"
    table.to_csv(csv_path, index=False)
    (folder / f"{name}.md").write_text(_markdown(table))
    return csv_path


def save_figure(figure: Any, name: str, folder: Optional[Path] = None) -> Path:
    """WHAT: save a matplotlib figure to results/figures/<name>.png (dpi 200) and .pdf, then close it.
    WHY: one naming rule for the slides and the chapters."""
    import matplotlib.pyplot as plt
    folder = folder or paths().figures
    folder.mkdir(parents=True, exist_ok=True)
    png = folder / f"{name}.png"
    figure.savefig(png, dpi=200, bbox_inches="tight")
    figure.savefig(folder / f"{name}.pdf", bbox_inches="tight")
    plt.close(figure)
    return png
