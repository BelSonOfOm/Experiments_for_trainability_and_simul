"""Writing problems to disk and to the SQLite database; the harvest log.

On disk, one folder per problem (docs/HARVEST_SCALE.md §5.2):

    <data_dir>/problems/<task folder>/instances.pkl       the instances (inputs, labels, units, info)
    <data_dir>/problems/<task folder>/operators.pkl       M(x) for every instance, and M̄
    <data_dir>/problems/<task folder>/split.json          "TRAIN"/"TEST" per instance
    <data_dir>/problems/<task folder>/equation.json       ℰₙ
    <data_dir>/problems/<task folder>/provenance.json     where it comes from
    <data_dir>/harvest.sqlite                               one row per problem, per equation, per log event

Why pickle for instances and operators: they mix numbers, arrays, dictionaries and lists, and
pickle stores them as they are with one line of code. It is Python-only and not meant for
sharing outside the project; replacing it (e.g. with HDF5 or npz) is a later, efficiency task.

The SQLite tables follow docs/DATA_STRUCTURE.md §6.2 with these differences, all for this
first version: `task` gains group_id, origin, n_distinct, conflict_rate, label_rule,
balance_rule, modelling_map_json and problem_dir; `harvest_log` gets an automatic id and a
time stamp so that every event is kept (the original primary key (source, dataset, stage)
kept only one event per stage).
"""
from __future__ import annotations

import json
import pickle
import re
import sqlite3
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

SCHEMA = """
CREATE TABLE IF NOT EXISTS task (
  task_id TEXT PRIMARY KEY,
  problem_type TEXT NOT NULL CHECK (problem_type IN ('GRAPH','COVARIANCE','DIAGONAL','QUANTUM')),
  n_qubits INTEGER NOT NULL,
  n_samples INTEGER NOT NULL,
  n_train INTEGER NOT NULL,
  n_test INTEGER NOT NULL,
  class_balance REAL NOT NULL,
  label_rule TEXT NOT NULL,
  balance_rule TEXT NOT NULL,
  n_distinct INTEGER NOT NULL,
  conflict_rate REAL NOT NULL,
  source TEXT NOT NULL, dataset TEXT NOT NULL, source_url TEXT, source_version TEXT,
  licence TEXT, citation TEXT, raw_hash TEXT,
  group_id TEXT NOT NULL,
  origin TEXT NOT NULL CHECK (origin IN ('REAL','REAL_INPUT_COMPUTED_LABEL','SYNTHETIC')),
  modelling_map_json TEXT NOT NULL,
  problem_dir TEXT NOT NULL,
  seed INTEGER NOT NULL,
  created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS equation (
  task_id TEXT PRIMARY KEY REFERENCES task,
  form TEXT NOT NULL, operator TEXT NOT NULL, storage TEXT NOT NULL,
  params_json TEXT NOT NULL, nonlinearity_json TEXT NOT NULL,
  task_operator_rule TEXT NOT NULL, task_operator_support_size INTEGER NOT NULL,
  check_method TEXT NOT NULL, check_max_residual REAL, n_failed INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS harvest_log (
  log_id INTEGER PRIMARY KEY AUTOINCREMENT,
  source TEXT NOT NULL, dataset TEXT NOT NULL,
  stage TEXT NOT NULL CHECK (stage IN ('LISTED','SCREENED','FETCHED','CONVERTED','CHECKED','KEPT','MERGED','REJECTED')),
  reason TEXT, n_samples INTEGER, logged_at TEXT NOT NULL
);
"""


def now() -> str:
    """The current time, UTC, as text (ISO format)."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def folder_name(task_id: str) -> str:
    """WHAT: turn a task id into a safe folder name (letters, digits, '-', '_', '.', '=' only)."""
    return re.sub(r"[^A-Za-z0-9_.=-]+", "_", task_id)


def to_json(value):
    """WHAT: make a value writable as JSON (numpy numbers and arrays become plain numbers and
    lists, tuples become lists). Used for equation.json and provenance.json."""
    if isinstance(value, dict):
        return {str(k): to_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_json(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, complex):
        return [value.real, value.imag]
    return value


class ProblemStore:
    """WHAT: the place where harvested problems and the harvest log are written.
    USE:
        store = ProblemStore("path/to/data", seed=0)
        store.log("SAT", "3-SAT n=8", "REJECTED", "S6: ...", 120)
        store.save(problem)
    WHY: one class owns all writing, so the on-disk layout is defined in one place."""

    def __init__(self, data_dir, seed: int = 0):
        self.data_dir = Path(data_dir)
        self.problems_dir = self.data_dir / "problems"
        self.problems_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.data_dir / "harvest.sqlite"
        self.seed = seed
        with sqlite3.connect(self.db_path) as db:
            db.executescript(SCHEMA)

    def log(self, source: str, dataset: str, stage: str, reason: str = "", n_samples: int = 0) -> None:
        """WHAT: add one event to the harvest log (one row of `harvest_log`).
        WHY: every decision (listed, screened, rejected and why, kept, merged) is kept, so the
        funnel can be reported and every rejection audited."""
        with sqlite3.connect(self.db_path) as db:
            db.execute("INSERT INTO harvest_log (source, dataset, stage, reason, n_samples, logged_at) "
                       "VALUES (?, ?, ?, ?, ?, ?)", (source, dataset, stage, reason, int(n_samples), now()))

    def save(self, problem) -> Path:
        """WHAT: write one problem's folder and its `task` and `equation` rows.
        Overwrites a problem with the same task_id (re-running the harvest replaces it)."""
        folder = self.problems_dir / folder_name(problem.task_id)
        folder.mkdir(parents=True, exist_ok=True)
        with open(folder / "instances.pkl", "wb") as f:
            pickle.dump(problem.instances, f)
        with open(folder / "operators.pkl", "wb") as f:
            pickle.dump({"operators": problem.operators, "checks": problem.checks,
                         "task_operator": problem.task_operator}, f)
        (folder / "split.json").write_text(json.dumps(problem.split))
        (folder / "equation.json").write_text(json.dumps(to_json(asdict(problem.equation)), indent=2))
        (folder / "provenance.json").write_text(json.dumps(to_json(asdict(problem.provenance)), indent=2))

        p, e, t = problem.provenance, problem.equation, problem.task_operator
        with sqlite3.connect(self.db_path) as db:
            db.execute("INSERT OR REPLACE INTO task VALUES (" + ",".join("?" * 24) + ")", (
                problem.task_id, problem.problem_type, problem.n, problem.n_samples,
                problem.split.count("TRAIN"), problem.split.count("TEST"),
                problem.class_balance, problem.label_rule, problem.balance_rule,
                problem.n_distinct, problem.conflict_rate,
                p.source, p.dataset, p.url, p.version, p.licence, p.citation, p.raw_hash,
                p.group_id, p.origin, json.dumps(to_json(p.modelling_map)),
                str(folder.relative_to(self.data_dir)), self.seed, now()))
            db.execute("INSERT OR REPLACE INTO equation VALUES (?,?,?,?,?,?,?,?,?,?,?)", (
                problem.task_id, e.form, e.operator_kind, e.storage,
                json.dumps(to_json(e.parameters)), json.dumps(to_json(e.nonlinearity)),
                t.rule, len(t.support), e.check_method, e.check_max_residual, e.n_failed))
        return folder

    def load_instances(self, task_id: str) -> list:
        """WHAT: read back a stored problem's instances (for later stages and for checking)."""
        with open(self.problems_dir / folder_name(task_id) / "instances.pkl", "rb") as f:
            return pickle.load(f)

    def funnel(self) -> list:
        """WHAT: how many log events of each stage, per source: the harvest funnel.
        OUTPUT: rows (source, stage, count)."""
        with sqlite3.connect(self.db_path) as db:
            return db.execute("SELECT source, stage, COUNT(*) FROM harvest_log "
                              "GROUP BY source, stage ORDER BY source, stage").fetchall()
