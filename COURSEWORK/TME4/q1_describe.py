"""q1_describe.py — Q1: what does the dataset contain? (spec: QnA_what_to_do.md, section Q1)

WHAT THIS FILE PRODUCES
Q1 is pure description, no test. From data_final (1 368 problems, 58 groups) it produces:
  1.2 `count_table`   problems, groups, samples per type -> sub-family -> group, and per n.
  1.3 `balance_table` per problem: +1 fraction in TRAIN, TEST and overall, computed WITH the R3 weights
                      (each distinct input counts once); the minimum per type.
  1.4 `funnel_table`  per source: settings listed, rejected (by reason), kept, from the `harvest_log`
                      tables of the RAW folders; then the final-build changes (R1 excluded, R2 re-split,
                      R3 re-weighted, R5 samples removed, R6 flagged) from data_final/CHANGELOG.md.
  1.5 `checks_table`  per type: number of failed equation checks (expected 0) and worst residual.
  1.6 `flags_table`   per problem: number of samples flagged by R6, by kind.
  1.7 `figures`       (a) problems per group, one panel per type (horizontal bars; a stacked bar split by
                      sub-family would need more than 8 colours, which the figure guidance forbids, so each
                      group is named on its own bar instead); (b) problems per n, one panel per type;
                      (c) problems, groups and input sets per tier.
  plus the audit summary (data_final audited 9 Oct 18:30: 0 FAIL, 2 WARN) quoted from the audit report.
All tables go through analysis_common.save_table, all figures through save_figure.
Outputs: results/q1_*.csv / .md, results/figures/fig_q1_*.png / .pdf.
"""
from __future__ import annotations

import json
import re
import sqlite3
from contextlib import closing

import analysis_common as ac


def count_table(index: "pandas.DataFrame") -> dict:
    """WHAT (1.2): the counts.
    OUTPUT: a dict of three DataFrames:
      "by_type"      type | problems | groups | samples | tier-1/2/3 problems
      "by_subfamily" type | subfamily | group_id | problems | n values present (e.g. "4,6,8,10,12")
      "by_n"         type x n -> number of problems (a pivot table, n = 4..12 as columns)
    HOW: pandas groupby on index columns problem_type, subfamily, group_id, n; samples = sum of n_samples.
    CHECK: total problems 1 368; groups 58; GRAPH 345, COVARIANCE 404, DIAGONAL 358, QUANTUM 261.
    WHY: a finding is only as broad as the data behind it; gaps (few groups, missing sizes) must be visible."""
    import pandas as pd
    by_type = index.groupby("problem_type").agg(problems=("task_id", "size"), groups=("group_id", "nunique"),
                                                samples=("n_samples", "sum")).reindex(ac.TYPES)
    tiers = index.pivot_table(index="problem_type", columns="tier", values="task_id", aggfunc="size",
                              fill_value=0).reindex(ac.TYPES).fillna(0).astype(int)
    for t in (1, 2, 3):
        by_type[f"tier_{t}"] = tiers[t] if t in tiers.columns else 0
    by_type.loc["TOTAL"] = [len(index), index["group_id"].nunique(), int(index["n_samples"].sum()),
                            *(int((index["tier"] == t).sum()) for t in (1, 2, 3))]
    by_type = by_type.reset_index().rename(columns={"index": "problem_type"})
    expected = {"GRAPH": 345, "COVARIANCE": 404, "DIAGONAL": 358, "QUANTUM": 261}
    for t, count in expected.items():
        got = int((index["problem_type"] == t).sum())
        if got != count:
            raise ValueError(f"{t}: {got} problems, expected {count} (audit of 9 Oct)")
    by_sub = (index.groupby(["problem_type", "subfamily", "group_id"])
              .agg(problems=("task_id", "size"), samples=("n_samples", "sum"),
                   n_values=("n", lambda s: ",".join(str(v) for v in sorted(set(s)))))
              .reset_index())
    by_n = index.pivot_table(index="problem_type", columns="n", values="task_id", aggfunc="size",
                             fill_value=0).reindex(ac.TYPES).fillna(0).astype(int).reset_index()
    by_n.columns = [str(c) for c in by_n.columns]
    return {"by_type": by_type, "by_subfamily": by_sub, "by_n": by_n}


def weighted_balance(labels: list, weights: list) -> float:
    """WHAT: fraction of +1 labels where each sample counts with its R3 weight.
    HOW: sum(w for y, w in zip(labels, weights) if y == 1) / sum(weights). Raise on empty input.
    TEST: labels [1, 1, -1], weights [0.5, 0.5, 1] -> 0.5."""
    total = sum(weights)
    if not labels or total <= 0:
        raise ValueError("weighted_balance of an empty sample")
    return sum(w for y, w in zip(labels, weights) if y == 1) / total


def balance_table(index: "pandas.DataFrame") -> "pandas.DataFrame":
    """WHAT (1.3): one row per problem: task_id, type, balance_train, balance_test, balance_all (all with
    R3 weights), test_outside (TEST balance outside [0.2, 0.8]); then one summary row per type (min / max).
    HOW: for each problem (analysis_common.iter_problems, with_operators=False): split the labels and
    weights by TRAIN / TEST and call weighted_balance. Expected test_outside rows: exactly 2 (XOR-3 n=9
    count-high and 1-IN-3 n=8 maxsat, TEST balance 0.17; audit of 9 Oct).
    WHY: shows the labels are balanced where it matters, and why scores are balanced accuracy."""
    import pandas as pd
    rows = []
    for p in ac.iter_problems(with_operators=False):
        y, w = p.labels(), p.weights
        tr, te = p.train_indices(), p.test_indices()
        b_te = weighted_balance([y[i] for i in te], [w[i] for i in te])
        rows.append({"task_id": p.task_id, "problem_type": p.problem_type,
                     "balance_train": weighted_balance([y[i] for i in tr], [w[i] for i in tr]),
                     "balance_test": b_te, "balance_all": weighted_balance(y, w),
                     "test_outside": not (0.2 <= b_te <= 0.8)})
    table = pd.DataFrame(rows)
    summary = table.groupby("problem_type")[["balance_train", "balance_test", "balance_all"]].agg(["min", "max"])
    summary.columns = [f"{a}_{b}" for a, b in summary.columns]
    table.attrs["summary"] = summary.reindex(ac.TYPES).reset_index()
    return table


def funnel_table() -> tuple:
    """WHAT (1.4): (funnel per source, final-build counts).
    Funnel: one row per (raw folder, source, dataset group): listed, rejected (one column per reason class),
    merged, kept, from the `harvest_log` table of every raw harvest.sqlite. Reasons are normalised by
    replacing numbers with '#' (as code/scripts/audit_dataset.py does), so 'balance 0.19' and 'balance 0.17'
    count together.
    Build counts: the "Counts" table of data_final/CHANGELOG.md (a Markdown table '| label | number |'),
    read, not typed: R1 excluded, R2 re-split, R3 weights only, R5 samples removed, R6 problems flagged.
    WHY: the professor must see what was rejected and why, not only what was kept."""
    import pandas as pd
    rows = []
    for folder in ac.paths().raw_folders:
        db_path = folder / "harvest.sqlite"
        if not db_path.exists():
            continue
        with closing(sqlite3.connect(db_path)) as db:
            for source, stage, reason, count in db.execute(
                    "SELECT source, stage, COALESCE(reason, ''), COUNT(*) FROM harvest_log "
                    "GROUP BY source, stage, reason"):
                rows.append({"folder": folder.relative_to(ac.paths().root).as_posix(), "source": source,
                             "stage": stage, "reason": re.sub(r"[0-9.]+", "#", reason)[:60], "count": count})
    log = pd.DataFrame(rows)
    stages = log.pivot_table(index=["folder", "source"], columns="stage", values="count", aggfunc="sum",
                             fill_value=0).reset_index()
    rejected = log[log["stage"] == "REJECTED"].groupby(["folder", "source", "reason"])["count"].sum().reset_index()
    stages.attrs["rejections"] = rejected.sort_values(["folder", "count"], ascending=[True, False])
    text = (ac.paths().data_final / "CHANGELOG.md").read_text()
    counts_block = text.split("## Counts", 1)[1].split("\n## ", 1)[0]
    build = []
    for line in counts_block.splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) == 2 and cells[1].isdigit():
            build.append({"step": cells[0], "problems": int(cells[1])})
    return stages, pd.DataFrame(build)


def checks_table(index: "pandas.DataFrame") -> "pandas.DataFrame":
    """WHAT (1.5): per type: problems, total failed equation checks (sum of n_failed, expected 0), worst
    check_max_residual and the problem it belongs to, and the check method(s) used.
    HOW: read table `equation` of data_final/harvest.sqlite joined with `task` on task_id.
    WHY: every input satisfies its equation; this is the dataset's correctness claim."""
    import pandas as pd
    with closing(ac._connect()) as db:
        eq = pd.read_sql("SELECT t.task_id, t.problem_type, e.check_method, e.check_max_residual, e.n_failed "
                         "FROM equation e JOIN task t ON e.task_id = t.task_id", db)
    rows = []
    for t in ac.TYPES:
        part = eq[eq["problem_type"] == t]
        worst = part.loc[part["check_max_residual"].idxmax()]
        rows.append({"problem_type": t, "problems": len(part), "failed_checks": int(part["n_failed"].sum()),
                     "worst_residual": float(worst["check_max_residual"]), "worst_problem": worst["task_id"],
                     "check_methods": "; ".join(sorted(set(part["check_method"])))})
    return pd.DataFrame(rows)


def flags_table(index: "pandas.DataFrame") -> "pandas.DataFrame":
    """WHAT (1.6): one row per problem with at least one R6 flag: task_id, type, n_samples,
    n_uninformative_operator, n_degenerate_ground_state; plus a TOTAL row.
    HOW: read flags.json of every problem (no need to load instances).
    CHECK against CLEANUP_LOG.md: 475 uninformative UEA samples (LSST 183, RacketSports 292), 3 degenerate
    H4-rectangle samples, and the remaining flagged spin samples of data_v3 (count read here).
    WHY: the sensitivity checks of Q2-Q4 depend on these samples; their number must be reported."""
    import pandas as pd
    rows = []
    for _, r in index.iterrows():
        path = r["folder"] / "flags.json"
        flags = json.loads(path.read_text()) if path.exists() else {}
        counts = {k: len(flags.get(k, [])) for k in ac.FLAG_KINDS}
        if any(counts.values()):
            rows.append({"task_id": r["task_id"], "problem_type": r["problem_type"], "n_samples": r["n_samples"],
                         **{f"n_{k}": v for k, v in counts.items()}})
    table = pd.DataFrame(rows)
    total = {"task_id": "TOTAL", "problem_type": "", "n_samples": int(table["n_samples"].sum())}
    total.update({f"n_{k}": int(table[f"n_{k}"].sum()) for k in ac.FLAG_KINDS})
    return pd.concat([table, pd.DataFrame([total])], ignore_index=True)


def audit_summary() -> dict:
    """WHAT: the audit result of data_final as a small dict: problems, groups, n_fail, n_warn, warnings
    (task_id -> messages), origins.
    HOW: read the audit report (analysis_common.paths().audit, written by code/scripts/audit_dataset.py).
    WHY: quoted in the Q1 chapter; never re-typed by hand."""
    report = json.loads(ac.paths().audit.read_text())
    return {"problems": report["problems"], "groups": report["groups"], "n_fail": len(report["fail"]),
            "n_warn": len(report["warn"]), "warnings": report["warn"], "origins": report["origins"]}


def figures(index: "pandas.DataFrame", input_sets: "pandas.DataFrame" = None) -> list:
    """WHAT (1.7): three figures, returned as the list of saved paths.
      fig_q1_groups: four panels (one per type), one horizontal bar per group = its number of problems,
                     sorted, in the type's colour, the count written at the bar's end.
      fig_q1_sizes:  four panels (one per type), x = n (4..12), y = number of problems.
      fig_q1_tiers:  for tiers 1, 2, 3: problems, input sets and groups, side by side (grey shades: these
                     are three measures of the same tier, not identities).
    HOW: matplotlib with analysis_common.figure_style and TYPE_COLOURS; save with save_figure.
    WHY: the slides show the dataset in three pictures."""
    import matplotlib.pyplot as plt
    import numpy as np
    ac.figure_style()
    saved = []
    fig, axes = plt.subplots(1, 4, figsize=(14, 6.5))
    for ax, t in zip(axes, ac.TYPES):
        counts = index[index["problem_type"] == t]["group_id"].value_counts().sort_values()
        labels = [g.split(":", 1)[1] for g in counts.index]
        ax.barh(labels, counts.values, color=ac.TYPE_COLOURS[t], height=0.7)
        for y, v in enumerate(counts.values):
            ax.text(v, y, f" {v}", va="center", fontsize=7, color=ac.INK["secondary"])
        ax.set_title(f"{t}: {counts.sum()} problems, {len(counts)} groups")
        ax.tick_params(axis="y", labelsize=7)
        ax.grid(axis="y", visible=False)
        ax.set_xlabel("problems")
    fig.tight_layout()
    saved.append(ac.save_figure(fig, "fig_q1_groups"))

    fig, axes = plt.subplots(1, 4, figsize=(13, 3), sharey=True)
    for ax, t in zip(axes, ac.TYPES):
        counts = index[index["problem_type"] == t]["n"].value_counts().reindex(range(4, 13), fill_value=0)
        ax.bar(counts.index, counts.values, color=ac.TYPE_COLOURS[t], width=0.7)
        ax.set_title(t)
        ax.set_xticks(range(4, 13))
        ax.set_xlabel("n (sites)")
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("problems")
    fig.tight_layout()
    saved.append(ac.save_figure(fig, "fig_q1_sizes"))

    fig, ax = plt.subplots(figsize=(6, 3))
    tiers = (1, 2, 3)
    measures = {"problems": [int((index["tier"] == t).sum()) for t in tiers]}
    if input_sets is not None:
        measures["input sets"] = [int((input_sets["tier"] == t).sum()) for t in tiers]
    measures["groups"] = [index[index["tier"] == t]["group_id"].nunique() for t in tiers]
    shades = ["#3d3d3a", "#8a8984", "#c3c2b7"]
    width = 0.8 / len(measures)
    for k, (name, values) in enumerate(measures.items()):
        x = np.arange(3) + (k - (len(measures) - 1) / 2) * width
        ax.bar(x, values, width=width * 0.92, color=shades[k], label=name)
        for xi, v in zip(x, values):
            ax.text(xi, v, str(v), ha="center", va="bottom", fontsize=7, color=ac.INK["secondary"])
    ax.set_xticks(range(3), ["tier 1\nmeasured", "tier 2\nreal input,\ncomputed label", "tier 3\nstandard models"])
    ax.set_yscale("log")
    ax.set_ylabel("count (log scale)")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=len(measures))
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    saved.append(ac.save_figure(fig, "fig_q1_tiers"))
    return saved


def run_q1() -> None:
    """WHAT: produce every Q1 output.
    HOW: index = load_index(); save_table for count_table (3 tables), balance_table (+ its summary),
    funnel_table (+ rejections, build counts), checks_table, flags_table, audit summary; figures(index);
    print a short summary (problems, groups, per type).
    WHY: one call regenerates the whole chapter's material."""
    import pandas as pd
    index = ac.load_index()
    _, input_sets = ac.attach_input_sets(index)
    counts = count_table(index)
    for name, table in counts.items():
        ac.save_table(table, f"q1_counts_{name}")
    balance = balance_table(index)
    ac.save_table(balance, "q1_balance")
    ac.save_table(balance.attrs["summary"], "q1_balance_summary")
    stages, build = funnel_table()
    ac.save_table(stages, "q1_funnel")
    ac.save_table(stages.attrs["rejections"], "q1_funnel_rejections")
    ac.save_table(build, "q1_build_counts")
    ac.save_table(checks_table(index), "q1_checks")
    ac.save_table(flags_table(index), "q1_flags")
    audit = audit_summary()
    ac.save_table(pd.DataFrame([{k: v for k, v in audit.items() if k not in ("warnings", "origins")}]), "q1_audit")
    sets = input_sets.groupby("problem_type").agg(input_sets=("input_set", "size"), problems=("n_problems", "sum"))
    ac.save_table(sets.reindex(ac.TYPES).reset_index(), "q1_input_sets")
    figures(index, input_sets)
    print(counts["by_type"].to_string(index=False))
    print(f"TEST balance outside [0.2, 0.8]: {int(balance['test_outside'].sum())} problems; "
          f"audit: {audit['n_fail']} FAIL, {audit['n_warn']} WARN")


if __name__ == "__main__":
    run_q1()
