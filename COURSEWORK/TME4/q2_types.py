"""q2_types.py — Q2: do the four types differ in structure? (spec: QnA_what_to_do.md, section Q2)

WHAT THIS FILE PRODUCES
For every structure feature of features.INSTANCE_FEATURES (the 12 features defined for all four types),
on results/features.csv (one row per distinct input set, D4.1):
  1. the pipeline check of the forced cells (forced_table.pipeline_check), BEFORE anything else: if it is
     not empty the run stops (a forced value that differs from its derivation is a bug);
  2. the HGAM fit of mixed_models.fit_type_model, giving ICC, effective sample size, each type at
     n = 8, 10, 12, level test, shape test, marginal R², pairwise differences;
     run twice: on all rows (labelled 'all rows, includes forced cells') and on finding rows only
     (forced_table.finding_mask), which is the version reported as findings (D2.2);
  3. Holm correction over the features, separately for the level tests and the shape tests;
  4. the group permutation test of the level statistic (mixed_models.group_permutation_test; 5 000
     permutations in a full run), on finding rows;
  5. the R6 sensitivity run: the same models on results/features_noflags.csv, side by side;
  6. figures: per feature, input sets as points against n coloured by type with each type's fitted curve
     and 95 % band; ICC and effective sample size per feature; marginal R² by feature.
Type-specific features are NOT tested here (they do not exist for every type).
Outputs: results/q2_models_{all,findings}.*, q2_type_at_n.*, q2_pairwise.*, q2_permutation.*,
q2_sensitivity.*, q2_pipeline_check.*, figures/fig_q2_*.
"""
from __future__ import annotations

import analysis_common as ac
import forced_table as ftb
import mixed_models as mm
from features import INSTANCE_FEATURES


def load_features(noflags: bool = False) -> "pandas.DataFrame":
    """WHAT: read results/features.csv (or features_noflags.csv), with the DIAGONAL constraint ratio added
    (forced_table.add_ratio). Raise with a clear message telling to run features.build_feature_table first
    if the file is missing."""
    import pandas as pd
    path = ac.paths().results / ("features_noflags.csv" if noflags else "features.csv")
    if not path.exists():
        raise FileNotFoundError(f"{path} missing: run features.build_feature_table() first")
    ac.import_harvest()
    return ftb.add_ratio(pd.read_csv(path), ac.load_index())


def results_table(results: list) -> "pandas.DataFrame":
    """WHAT: one row per feature from a list of mixed_models.TypeModelResult: feature, rows, groups, icc,
    n_eff, spline_df, level_lr, level_df, level_p, level_p_holm, shape_lr, shape_df, shape_p, shape_p_holm,
    marginal_r2, converged, warnings.
    HOW: analysis_common.holm on the level p-values and, separately, on the shape p-values."""
    import pandas as pd
    table = pd.DataFrame([{"feature": r.feature, "rows": r.n_rows, "groups": r.n_groups, "icc": r.icc,
                           "n_eff": r.n_eff, "spline_df": r.spline_df, "level_lr": r.level_lr,
                           "level_df": r.level_df, "level_p": r.level_p, "shape_lr": r.shape_lr,
                           "shape_df": r.shape_df, "shape_p": r.shape_p, "marginal_r2": r.marginal_r2,
                           "converged": r.converged, "warnings": " | ".join(r.warnings)} for r in results])
    table.insert(table.columns.get_loc("level_p") + 1, "level_p_holm", ac.holm(list(table["level_p"])))
    table.insert(table.columns.get_loc("shape_p") + 1, "shape_p_holm", ac.holm(list(table["shape_p"])))
    return table


def run_models(features: "pandas.DataFrame", findings_only: bool, transforms: dict = None) -> list:
    """WHAT: fit_type_model for every feature of features.INSTANCE_FEATURES.
    HOW: rows_mask = forced_table.finding_mask(features, f) if findings_only else None; transform per
    feature from `transforms` (OPEN POINT: default "none" for every feature until the distributions have
    been looked at with Charbel)."""
    transforms = transforms or {}
    results = []
    for feature in INSTANCE_FEATURES:
        mask = ftb.finding_mask(features, feature) if findings_only else None
        results.append(mm.fit_type_model(features, feature, rows_mask=mask, transform=transforms.get(feature, "none")))
    return results


def detail_tables(results: list, label: str) -> tuple:
    """WHAT: (type_at_n, pairwise) tables of every feature stacked, with columns feature and rows (label)."""
    import pandas as pd
    at_n, pairs = [], []
    for r in results:
        if r.type_at_n is not None:
            at_n.append(r.type_at_n.assign(feature=r.feature, rows=label))
        if r.pairwise is not None and len(r.pairwise):
            pairs.append(r.pairwise.assign(feature=r.feature, rows=label))
    return (pd.concat(at_n, ignore_index=True) if at_n else pd.DataFrame(),
            pd.concat(pairs, ignore_index=True) if pairs else pd.DataFrame())


def run_permutations(features: "pandas.DataFrame", n_permutations: int = 5000, n_jobs: int = 1) -> "pandas.DataFrame":
    """WHAT: per feature (finding rows only): observed level LR, permutation p-value, number of failed fits.
    HOW: mixed_models.group_permutation_test with statistic = the level LR (mixed_models.level_statistic)."""
    import pandas as pd
    rows = []
    for feature in INSTANCE_FEATURES:
        mask = ftb.finding_mask(features, feature)
        observed, p, _, failed = mm.group_permutation_test(features, feature, n_permutations=n_permutations,
                                                           rows_mask=mask, n_jobs=n_jobs)
        rows.append({"feature": feature, "observed_level_lr": observed, "permutation_p": p,
                     "permutations": n_permutations, "failed_fits": failed})
        print(f"permutations {feature}: p = {p:.4f}", flush=True)
    table = pd.DataFrame(rows)
    table["permutation_p_holm"] = ac.holm(list(table["permutation_p"]))
    return table


def sensitivity_flags(with_flags: list = None) -> "pandas.DataFrame":
    """WHAT (spec 2.7): the headline numbers (level_p_holm, shape_p_holm, marginal_r2) with and without the
    R6-flagged samples, side by side, for every feature (finding rows).
    HOW: run_models on load_features(noflags=False) (or reuse `with_flags`) and on
    load_features(noflags=True); merge on feature."""
    with_table = results_table(with_flags or run_models(load_features(False), findings_only=True))
    without = results_table(run_models(load_features(True), findings_only=True))
    cols = ["feature", "level_p_holm", "shape_p_holm", "marginal_r2", "icc"]
    return with_table[cols].merge(without[cols], on="feature", suffixes=("_with_flagged", "_without_flagged"))


def figures(features: "pandas.DataFrame", results: list, folder=None) -> list:
    """WHAT (spec 2.8): fig_q2_<feature> for each feature (input sets as points against n coloured by type,
    each type's fitted curve with its 95 % band from mixed_models.predict_at over the type's observed range of
    n, finding rows only); fig_q2_icc (ICC per feature, bars, with the effective sample size written);
    fig_q2_r2 (marginal R² per feature: all rows vs finding rows)."""
    import matplotlib.pyplot as plt
    import numpy as np
    ac.figure_style()
    saved = []
    for r in results:
        mask = ftb.finding_mask(features, r.feature)
        frame = mm.model_frame(features, r.feature, rows_mask=mask)
        fig, ax = plt.subplots(figsize=(5.5, 3.6))
        rng = np.random.default_rng(0)
        fit = None
        if r.converged and frame["problem_type"].nunique() >= 2:
            _, full_f, _, _ = mm._type_formulas(frame)
            fit = mm.fit_mixed(frame, full_f, reml=True)
        for t in ac.TYPES:
            part = frame[frame["problem_type"] == t]
            if not len(part):
                continue
            ax.scatter(part["n"] + rng.uniform(-0.15, 0.15, len(part)), part["y"], s=10, alpha=0.45,
                       color=ac.TYPE_COLOURS[t], edgecolors="none")
            if fit is not None:
                grid = np.linspace(part["n"].min(), part["n"].max(), 40)
                est = np.array([mm.predict_at(fit, frame, t, g) for g in grid])
                ax.plot(grid, est[:, 0], color=ac.TYPE_COLOURS[t], label=t)
                ax.fill_between(grid, est[:, 1], est[:, 2], color=ac.TYPE_COLOURS[t], alpha=0.15, linewidth=0)
        ax.set_xlabel("n (sites)")
        ax.set_ylabel(r.feature)
        ax.set_title(f"{r.feature}: types at equal n (finding rows)")
        ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5))
        saved.append(ac.save_figure(fig, f"fig_q2_{r.feature}", folder))
    table = results_table(results)
    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.bar(table["feature"], table["icc"], color="#5f5e5a", width=0.7)
    for x, (icc, neff) in enumerate(zip(table["icc"], table["n_eff"])):
        if icc == icc:
            ax.text(x, icc, f"{neff:.0f}", ha="center", va="bottom", fontsize=7, color=ac.INK["secondary"])
    ax.set_ylabel("ICC (share of variance between groups)")
    ax.set_title("ICC per feature; number above bar = effective sample size (Kish)")
    ax.tick_params(axis="x", rotation=45)
    ax.grid(axis="x", visible=False)
    saved.append(ac.save_figure(fig, "fig_q2_icc", folder))
    return saved


def run_q2(n_permutations: int = 5000, n_jobs: int = 1, folder=None) -> None:
    """WHAT: the whole of Q2 in order: pipeline check (stop and report if it is not empty), models on all
    rows and on finding rows, results tables, type-at-n and pairwise tables, permutations, sensitivity,
    figures. Save everything with analysis_common.save_table / save_figure (folder overrides results/).
    n_permutations = 0 skips the permutation test."""
    features = load_features()
    check = ftb.pipeline_check(features)
    ac.save_table(check, "q2_pipeline_check", folder)
    ac.save_table(ftb.prediction_coverage(features), "q2_prediction_coverage", folder)
    if len(check):
        raise RuntimeError(f"pipeline check: {len(check)} forced values differ from their derivation; "
                           f"see q2_pipeline_check.csv before reading any result")
    all_rows = run_models(features, findings_only=False)
    findings = run_models(features, findings_only=True)
    ac.save_table(results_table(all_rows), "q2_models_all_rows", folder)
    ac.save_table(results_table(findings), "q2_models_findings", folder)
    import pandas as pd
    at_a, pair_a = detail_tables(all_rows, "all rows")
    at_f, pair_f = detail_tables(findings, "finding rows")
    ac.save_table(pd.concat([at_f, at_a], ignore_index=True), "q2_type_at_n", folder)
    ac.save_table(pd.concat([pair_f, pair_a], ignore_index=True), "q2_pairwise", folder)
    if n_permutations:
        ac.save_table(run_permutations(features, n_permutations, n_jobs), "q2_permutation", folder)
    ac.save_table(sensitivity_flags(findings), "q2_sensitivity", folder)
    figures(features, findings, folder)
    print(results_table(findings)[["feature", "rows", "icc", "level_p_holm", "shape_p_holm", "marginal_r2"]].to_string(index=False))


if __name__ == "__main__":
    run_q2()
