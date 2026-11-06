"""q4_scaling.py — Q4: how does structure change with size n? (spec: QnA_what_to_do.md, section Q4;
decisions D4.1-D4.4)

WHAT THIS FILE PRODUCES
Within each LINE (one sub-family followed across n), for every feature (instance features, type-specific
features and the spin-chain per-side features), on results/features.csv (one row per distinct input set):
  4.1 which lines exist: sub-families with at least MIN_SIZES sizes. Left out and said so: QUANTUM molecules
      (fixed qubit count per molecule), UEA datasets (fixed channel count), and any sub-family with fewer
      sizes (e.g. ZINC, some small networks).
  4.2 tier 1, always: the slope against log n with its 95 % interval -> increases / decreases / flat
      (mixed model with a random intercept per unit; OLS when the line has a single unit).
  4.3 tier 2: forced and predictable cells with a written prediction (forced_table.predicted_value) tested
      against it: forced cells by exact comparison (a deviation is a bug); predictable cells by
      mixed_models.test_against_law (a deviation is a finding), Holm over the predictable tests.
  4.4 tier 3: found cells: BIC among constant, a+b/n, a+b·log n, power, exponential; a law is named only if
      ΔBIC > 10 (Kass & Raftery 1995), else "undetermined on n = a-b". Never an asymptotic claim.
  4.5 entanglement and gap laws on the spin chains, per (family, cut, window, side of p_c) — laws are
      assigned per SIDE because some cuts have a whole critical phase on one side (XXZ "neel" and "ferro":
      critical XY side; J1-J2: the critical side below 0.2411):
        a critical side, or the near window of a cut whose transition has a cited central charge:
          entanglement ~ a + (c/6) log n for open chains, (c/3) log n for periodic chains
          (Calabrese & Cardy 2004); in-sector gap ~ 1/n (slope -1 of log gap against log n);
        a gapped side in the far window: entanglement bounded (area law, Hastings 2007), gap constant
          (slope 0 of log gap against log n).
      Central charges used: TFIM c = 1/2, XXZ critical phase c = 1 (the two approved on 9 Oct). Other
      families: no c is used until one is looked up and cited (the slope is still reported). The near
      window is an interval around p_c, so its prediction is approximate: the fitted slope is reported
      against c/6, not required to equal it.
  4.6 the R6 sensitivity run (degenerate ground states have an arbitrary entanglement).
  4.7 figures.
Role (stated in the report): these laws are checks that the extracted structure is right, inputs to the
circuit step; they do not answer the trainability / dequantization question.
Outputs: results/q4_lines.*, q4_trends.*, q4_theory.*, q4_laws.*, q4_entanglement.*, q4_sensitivity.*,
figures/fig_q4_*.
"""
from __future__ import annotations

import math
import re

import numpy as np

import analysis_common as ac
import forced_table as ftb
import mixed_models as mm
from features import INSTANCE_FEATURES, TYPE_SPECIFIC_FEATURES

MIN_SIZES = 4
"""A line needs at least this many distinct n values (a 3-df spline plus intercept needs 4). PROPOSED."""

Q4_FEATURES = INSTANCE_FEATURES + TYPE_SPECIFIC_FEATURES
"""Features followed along every line (the per-side spin features are handled in entanglement_study)."""

CENTRAL_CHARGE = {"TFIM": 0.5, "XXZ": 1.0}
"""Central charges with a citation approved on 9 Oct (TFIM: Ising transition; XXZ: critical XY phase /
Heisenberg point). Calabrese & Cardy 2004 for the log law."""


def line_name(row) -> str:
    """WHAT: the line a feature-table row belongs to: problem_type + subfamily (e.g. "QUANTUM/TFIM-open",
    "DIAGONAL/3-SAT", "GRAPH/collins_yeast", "COVARIANCE/Cho2017")."""
    return f"{row['problem_type']}/{row['subfamily']}"


def unit_of(row) -> str:
    """WHAT: the random-effect unit of a row: the first task id of its input set with ":n=<n>" removed, so the
    same spin cut and window, the same SAT label or the same EEG subject-session is followed across n; for
    real networks and molecule datasets the group_id (their input sets merge several labels, so a task name
    is not a stable identity across n)."""
    if str(row["group_id"]).startswith(("NET:", "MOL:")):
        return row["group_id"]
    return re.sub(r":n=\d+", "", row["task_ids"].split(";")[0])


def prepare(features: "pandas.DataFrame") -> "pandas.DataFrame":
    """WHAT: the feature table with columns `line` and `unit` added, and the DIAGONAL constraint ratio
    (forced_table.add_ratio, needed by the predicted values of the SAT lines)."""
    ac.import_harvest()
    out = ftb.add_ratio(features, ac.load_index()) if "ratio" not in features.columns else features.copy()
    out["line"] = [line_name(r) for _, r in out.iterrows()]
    out["unit"] = [unit_of(r) for _, r in out.iterrows()]
    return out


def lines(features: "pandas.DataFrame") -> "pandas.DataFrame":
    """WHAT (4.1): one row per line: line, problem_type, subfamily, n values, input sets, units, included,
    reason if excluded (molecules / UEA: fixed n by construction; otherwise fewer than MIN_SIZES sizes)."""
    import pandas as pd
    rows = []
    for name, part in features.groupby("line"):
        sizes = sorted(set(part["n"]))
        first = part.iloc[0]
        reason = ""
        if str(first["group_id"]).startswith("CHEM:"):
            reason = "molecule: fixed number of qubits"
        elif str(first["group_id"]).startswith("TS:"):
            reason = "UEA dataset: fixed number of channels"
        elif len(sizes) < MIN_SIZES:
            reason = f"only {len(sizes)} sizes"
        rows.append({"line": name, "problem_type": first["problem_type"], "subfamily": first["subfamily"],
                     "n_values": ",".join(str(int(v)) for v in sizes), "input_sets": len(part),
                     "units": part["unit"].nunique(), "included": reason == "", "reason": reason})
    return pd.DataFrame(rows)


def line_frame(part: "pandas.DataFrame", feature: str):
    """WHAT: mixed_models.model_frame of one line and one feature (columns y, n, unit, ...)."""
    return mm.model_frame(part, feature)


def trend(frame) -> dict:
    """WHAT (4.2, tier 1): {"slope_log_n", "ci_low", "ci_high", "reading"} of y against log n
    (mixed_models.slope_vs_log_n)."""
    slope, lo, hi = mm.slope_vs_log_n(frame, "unit")
    return {"slope_log_n": slope, "ci_low": lo, "ci_high": hi, "reading": mm.reading(lo, hi)}


def line_status(feature: str, part):
    """WHAT: the status of (feature, the line's cell) from forced_table, None if the feature does not exist."""
    return ftb.status(feature, ftb.cell_of(part.iloc[0]))


def theory_test(part, feature: str, status) -> dict:
    """WHAT (4.3, tier 2): compare a line with its written law (forced_table.predicted_value, per row).
    HOW: FORCED cells: exact comparison, max |computed - predicted| <= 1e-6 x max(1, |predicted|) on every row
    (the rule of forced_table.pipeline_check). A statistical test is meaningless here: an exact law leaves
    zero residual variance, so its p-values are degenerate (found 11 Oct: the first run marked all 28 forced
    checks as deviations although every value matched exactly). PREDICTABLE cells (expected values):
    mixed_models.test_against_law. None when a row of the line has no written prediction."""
    column = f"{feature}_median"
    keep = part[np.isfinite(part[column])].copy()
    keep["_pred"] = [ftb.predicted_value(feature, r) for _, r in keep.iterrows()]
    if keep["_pred"].isna().any() or len(keep) < 3:
        return None
    deviation = (keep[column] - keep["_pred"]).abs()
    if status is ftb.Status.FORCED:
        worst = float((deviation / np.maximum(1.0, keep["_pred"].abs())).max())
        return {"test": "exact", "max_relative_deviation": worst,
                "reading": "exact match" if worst <= 1e-6 else "BUG (forced)"}
    frame = line_frame(keep, feature)
    return {"test": "statistical", **mm.test_against_law(frame, keep["_pred"].to_numpy(), "unit")}


def law_choice(frame) -> dict:
    """WHAT (4.4, tier 3): mixed_models.choose_law."""
    return mm.choose_law(frame, "unit")


def run_lines(features: "pandas.DataFrame") -> tuple:
    """WHAT: tiers 1-3 for every included line and every feature present on it.
    OUTPUT (trends, theory, laws) DataFrames; Holm over all tier-2 p-values (shape and offset separately);
    reading of each tier-2 test: consistent, BUG (forced cell) or deviates (finding, predictable cell)."""
    import pandas as pd
    table = lines(features)
    trends, theory, laws = [], [], []
    for name in table.loc[table["included"], "line"]:
        part = features[features["line"] == name]
        for feature in Q4_FEATURES:
            column = f"{feature}_median"
            if part[column].notna().sum() < MIN_SIZES or part[column].nunique() < 2:
                continue
            st = line_status(feature, part)
            frame = line_frame(part, feature)
            if frame["n"].nunique() < MIN_SIZES:
                continue
            base = {"line": name, "feature": feature, "status": st.value if st else "", "rows": len(frame),
                    "units": frame["unit"].nunique()}
            try:
                trends.append({**base, **trend(frame)})
            except Exception as error:                     # noqa: BLE001  (reported, not hidden)
                trends.append({**base, "reading": f"fit failed: {type(error).__name__}"})
            if st in (ftb.Status.FORCED, ftb.Status.PREDICTABLE):
                try:
                    test = theory_test(part, feature, st)
                except Exception as error:                 # noqa: BLE001
                    test = {"error": type(error).__name__}
                if test is not None:
                    theory.append({**base, **test})
            elif st is ftb.Status.FOUND:
                choice = law_choice(frame)
                laws.append({**base, "verdict": choice["verdict"], "best": choice["best"],
                             "delta_bic": choice["delta_bic"], **{f"bic_{k}": v for k, v in choice["bic"].items()}})
    theory = pd.DataFrame(theory)
    stat = theory["test"] == "statistical" if len(theory) else []
    if len(theory) and stat.any():
        theory.loc[stat, "p_shape_holm"] = ac.holm(list(theory.loc[stat, "p_shape"]))
        theory.loc[stat, "p_offset_holm"] = ac.holm(list(theory.loc[stat, "p_offset"]))
        theory.loc[stat, "reading"] = [
            "test not possible" if (a != a or b != b) else ("consistent" if (a >= 0.05 and b >= 0.05) else "deviates (finding)")
            for a, b in zip(theory.loc[stat, "p_shape_holm"], theory.loc[stat, "p_offset_holm"])]
    return pd.DataFrame(trends), theory, pd.DataFrame(laws)


def spin_cuts() -> dict:
    """WHAT: {(family, cut name): Cut} from harvest.sources.spin_more.FAMILIES (p_c, parameter, phases)."""
    ac.import_harvest()
    from harvest.sources.spin_more import FAMILIES
    return {(f, c.name): c for f, fam in FAMILIES.items() for c in fam.cuts}


def side_prediction(family: str, cut, window: str, side: int, periodic: bool) -> dict:
    """WHAT (4.5): the predicted entanglement and gap behaviour on one side of one cut.
    HOW: phase = cut.phases[0] below p_c, cut.phases[1] above (spin_more.py label rule). If the phase name
    contains "critical": entanglement 'log', gap slope -1. Else in the near window: 'log' and gap slope -1
    (transition), with c only where CENTRAL_CHARGE cites one for that transition (TFIM; XXZ "neel" cut, whose
    transition point Δ = 1 is the Heisenberg point). Else (gapped side, far window): entanglement 'bounded'
    (area law), gap slope 0.
    OUTPUT {"phase", "entanglement_law", "c", "predicted_log_slope" (c/6 open, c/3 periodic, 0 bounded,
    NaN unknown), "predicted_gap_slope"}."""
    phase = cut.phases[0] if side < 0 else cut.phases[1]
    critical_phase = "critical" in phase
    if critical_phase or window == "near":
        cited = critical_phase or family == "TFIM" or (family == "XXZ" and cut.name == "neel")
        c = CENTRAL_CHARGE.get(family) if cited else None
        slope = (c / 3 if periodic else c / 6) if c is not None else math.nan
        return {"phase": phase, "entanglement_law": "log", "c": c if c is not None else math.nan,
                "predicted_log_slope": slope, "predicted_gap_slope": -1.0}
    return {"phase": phase, "entanglement_law": "bounded", "c": math.nan, "predicted_log_slope": 0.0,
            "predicted_gap_slope": 0.0}


def _spins(features: "pandas.DataFrame") -> "pandas.DataFrame":
    """WHAT: the spin-chain rows with columns cut and window (from the first task id, ".../<cut>:<window>")."""
    spins = features[(features["problem_type"] == "QUANTUM") & ~features["group_id"].str.startswith("CHEM:")].copy()
    tails = spins["task_ids"].str.split(";").str[0].str.split(":")
    spins["cut"], spins["window"] = tails.str[-2], tails.str[-1]
    return spins


def entanglement_study(features: "pandas.DataFrame") -> "pandas.DataFrame":
    """WHAT (4.5): one row per (spin sub-family, cut, window, side): the predicted law, the fitted slope of
    entanglement against log n and of log(in-sector gap) against log n (OLS: one input set per n), their
    95 % intervals, and whether the prediction lies inside the interval (None where no c is cited).
    HOW: side columns entanglement_below/above and gap_below/above from features.py."""
    import pandas as pd
    import statsmodels.formula.api as smf
    cuts = spin_cuts()
    rows = []
    for (sub, cut_name, window), part in _spins(features).groupby(["subfamily", "cut", "window"]):
        family, periodic = sub.split("-")[0], sub.endswith("periodic")
        cut = cuts[(family, cut_name)]
        for side, ent_col, gap_col in ((-1, "entanglement_below", "gap_below"), (1, "entanglement_above", "gap_above")):
            pred = side_prediction(family, cut, window, side, periodic)
            row = {"subfamily": sub, "cut": cut_name, "window": window, "side": "below p_c" if side < 0 else "above p_c",
                   **pred, "sizes": int(part["n"].nunique())}
            for name, column, transform, target in (("entanglement", ent_col, "none", pred["predicted_log_slope"]),
                                                    ("gap", gap_col, "log", pred["predicted_gap_slope"])):
                data = part[["n", column]].rename(columns={column: "y"}).dropna()
                if transform == "log":
                    data = data[data["y"] > 0].assign(y=lambda d: np.log(d["y"]))
                if data["n"].nunique() < 3:
                    continue
                fit = smf.ols("y ~ np.log(n)", data).fit()
                slope = float(fit.params["np.log(n)"])
                lo, hi = fit.conf_int().loc["np.log(n)"].tolist()
                row.update({f"{name}_slope": slope, f"{name}_ci_low": lo, f"{name}_ci_high": hi,
                            f"{name}_prediction_inside": (lo <= target <= hi) if target == target else None})
            rows.append(row)
    return pd.DataFrame(rows)


def sensitivity_flags(trends_with: "pandas.DataFrame") -> "pandas.DataFrame":
    """WHAT (4.6): the tier-1 readings that change when the R6-flagged samples are removed (merged on line and
    feature); an empty table means no reading depends on the flagged samples."""
    import pandas as pd
    without = prepare(pd.read_csv(ac.paths().results / "features_noflags.csv"))
    t_without, _, _ = run_lines(without)
    merged = trends_with.merge(t_without, on=["line", "feature"], suffixes=("_with_flagged", "_without_flagged"))
    changed = merged[merged["reading_with_flagged"] != merged["reading_without_flagged"]]
    return changed[["line", "feature", "slope_log_n_with_flagged", "reading_with_flagged",
                    "slope_log_n_without_flagged", "reading_without_flagged"]]


def figures(features: "pandas.DataFrame", trends: "pandas.DataFrame", folder=None) -> list:
    """WHAT (4.7): (a) per spin family, entanglement (per side) against log n, near and far windows side by
    side, with a reference line of slope c/6 where a central charge is cited; (b) a summary heat map: lines x
    features, coloured by the tier-1 reading, with the reading written in each cell (+, 0, −) so colour is
    never the only carrier."""
    import matplotlib.pyplot as plt
    import pandas as pd
    ac.figure_style()
    saved = []
    spins = _spins(features)
    for family in sorted(set(s.split("-")[0] for s in spins["subfamily"])):
        part = spins[spins["subfamily"].str.startswith(family + "-")]
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.4), sharey=True)
        for ax, window in zip(axes, ("near", "far")):
            styles = {"entanglement_below": ("#2a78d6", "below p_c"), "entanglement_above": ("#eb6834", "above p_c")}
            for (sub, cut), seg in part[part["window"] == window].groupby(["subfamily", "cut"]):
                seg = seg.sort_values("n")
                for col, (colour, label) in styles.items():
                    ls = "--" if sub.endswith("periodic") else "-"
                    ax.plot(np.log(seg["n"]), seg[col], ls, color=colour, marker="o",
                            label=f"{label}, cut {cut}, {sub.split('-')[1]}")
            c = CENTRAL_CHARGE.get(family)
            if c is not None and len(part):
                x = np.log(np.array([4, 12]))
                ax.plot(x, part["entanglement_median"].min() + c / 6 * (x - x[0]), color="#1f1f1e", linewidth=1,
                        label=f"reference slope c/6 = {c / 6:.3f}")
            ax.set_xlabel("log n")
            ax.set_title(f"{family}: {window} window")
        axes[0].set_ylabel("half-chain entanglement (median)")
        axes[1].legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=7)
        fig.tight_layout()
        saved.append(ac.save_figure(fig, f"fig_q4_entanglement_{family}", folder))
    if len(trends):
        pivot = trends.pivot_table(index="line", columns="feature", values="reading", aggfunc="first")
        code = pivot.apply(lambda s: pd.to_numeric(s.map({"increases": 1, "flat": 0, "decreases": -1}), errors="coerce"))
        fig, ax = plt.subplots(figsize=(0.55 * len(pivot.columns) + 3, 0.28 * len(pivot) + 1.5))
        ax.imshow(code.to_numpy(dtype=float), cmap="coolwarm", vmin=-1.5, vmax=1.5, aspect="auto")
        for i in range(pivot.shape[0]):
            for j in range(pivot.shape[1]):
                v = pivot.iat[i, j]
                if isinstance(v, str):
                    ax.text(j, i, {"increases": "+", "decreases": "−", "flat": "0"}.get(v, "?"), ha="center",
                            va="center", fontsize=7)
        ax.set_xticks(range(pivot.shape[1]), pivot.columns, rotation=60, ha="right", fontsize=7)
        ax.set_yticks(range(pivot.shape[0]), pivot.index, fontsize=7)
        ax.grid(False)
        ax.set_title("tier 1: trend against log n (+ increases, − decreases, 0 flat, ? fit failed)")
        fig.tight_layout()
        saved.append(ac.save_figure(fig, "fig_q4_trends", folder))
    return saved


def run_q4(folder=None) -> None:
    """WHAT: lines -> tiers 1-3 for every included line and feature -> entanglement study -> sensitivity ->
    figures; save every table."""
    import pandas as pd
    features = prepare(pd.read_csv(ac.paths().results / "features.csv"))
    ac.import_harvest()
    ac.save_table(lines(features), "q4_lines", folder)
    trends, theory, laws = run_lines(features)
    ac.save_table(trends, "q4_trends", folder)
    ac.save_table(theory, "q4_theory", folder)
    ac.save_table(laws, "q4_laws", folder)
    ac.save_table(entanglement_study(features), "q4_entanglement", folder)
    ac.save_table(sensitivity_flags(trends), "q4_sensitivity", folder)
    figures(features, trends, folder)
    print(trends["reading"].value_counts().to_string())
    if len(theory):
        print(theory["reading"].value_counts().to_string())
    if len(laws):
        print(laws["verdict"].str.startswith("undetermined").value_counts().to_string())


if __name__ == "__main__":
    run_q4()
