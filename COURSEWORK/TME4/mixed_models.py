"""mixed_models.py — the statistical models shared by Q2 and Q4 (spec 2.5, 2.6, 4.2-4.4; decisions D2.3,
D2.5, D4.2, D4.3).

WHAT THIS FILE PROVIDES
1. The hierarchical generalized additive mixed model (Wood 2017; Pedersen et al. 2019):
       feature = baseline of the type + smooth curve in n for that type + offset of the group + noise
   - smooth curve: a B-spline basis in n (patsy `bs`) with at most 3 degrees of freedom, one curve per type
     (`C(problem_type):bs(n, ...)`); the degrees of freedom drop automatically to (smallest number of
     distinct n over the types) - 1 when a type has few sizes;
   - offset of the group: random intercept per group_id (statsmodels MixedLM, groups=group_id);
   - one row = one distinct input set (D4.1).
   Fitted by REML for estimates, by ML (reml=False) for likelihood-ratio tests.
2. What is read out of a fit (spec 2.5): ICC; effective sample size (Kish); each type's value at equal n
   (n = 8, 10, 12) with 95 % intervals; the level test (do types differ at equal n?); the shape test (do
   types scale differently?); marginal R² (Nakagawa & Schielzeth 2013); pairwise type differences at
   equal n with intervals.
3. The group permutation test (spec 2.6): shuffle type labels between whole groups.
4. The same machinery restricted to one line (one sub-family across n) for Q4: a smooth curve plus a
   random offset per unit, the slope against log n, the comparison with a theoretical law, and the BIC
   comparison of candidate laws with the ΔBIC > 10 rule (Kass & Raftery 1995). A line with a single unit
   has no random effect to estimate: it is fitted by ordinary least squares (OLS) and says so.
References to cite in the chapter: Aarts et al. 2014 (Nat. Neurosci.); Lazic 2010; Hurlbert 1984; Kish
1965 (design effect); Wood 2017; Pedersen et al. 2019 (PeerJ); Nakagawa & Schielzeth 2013; Kass & Raftery
1995.
"""
from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field

import numpy as np

EQUAL_N = (8, 10, 12)
"""Sizes at which types are compared (all four types are present there)."""

NAN = float("nan")


@dataclass
class TypeModelResult:
    """FIELDS (one feature, spec 2.5):
    feature; n_rows (input sets); n_groups; icc; n_eff (Kish); type_at_n (DataFrame: type, n, estimate,
    ci_low, ci_high); level_lr, level_df, level_p; shape_lr, shape_df, shape_p; marginal_r2;
    pairwise (DataFrame: type_a, type_b, n, difference, ci_low, ci_high, p); spline_df; converged (bool);
    warnings (list of str, e.g. singular fit, boundary variance)."""
    feature: str
    n_rows: int = 0
    n_groups: int = 0
    icc: float = NAN
    n_eff: float = NAN
    type_at_n: object = None
    level_lr: float = NAN
    level_df: int = 0
    level_p: float = NAN
    shape_lr: float = NAN
    shape_df: int = 0
    shape_p: float = NAN
    marginal_r2: float = NAN
    pairwise: object = None
    spline_df: int = 0
    converged: bool = False
    warnings: list = field(default_factory=list)


def model_frame(features: "pandas.DataFrame", feature: str, rows_mask=None, transform: str = "none"):
    """WHAT: the data frame a model is fitted on: columns y, problem_type, n, group_id, unit, subfamily.
    HOW: y = features[feature + "_median"]; apply rows_mask (e.g. forced_table.finding_mask); drop NaN;
    transform "log" -> log(y) (only if all y > 0; raise otherwise). The number of dropped rows is stored in
    frame.attrs["dropped"].
    OPEN POINT (spec 2.5): whether a skewed feature is log-transformed is decided when the distributions
    are seen; the choice is an argument, never hard-coded."""
    column = f"{feature}_median" if f"{feature}_median" in features.columns else feature
    data = features if rows_mask is None else features[rows_mask]
    keep = ["problem_type", "n", "group_id", "subfamily"] + (["unit"] if "unit" in data.columns else [])
    frame = data[keep].copy()
    frame["y"] = data[column].astype(float)
    before = len(frame)
    frame = frame[np.isfinite(frame["y"])]
    if transform == "log":
        if (frame["y"] <= 0).any():
            raise ValueError(f"{feature}: log transform needs y > 0")
        frame["y"] = np.log(frame["y"])
    elif transform != "none":
        raise ValueError(f"unknown transform {transform!r}")
    frame["n"] = frame["n"].astype(float)
    frame = frame.reset_index(drop=True)
    frame.attrs["dropped"] = before - len(frame)
    return frame


def spline_df(n_values) -> int:
    """WHAT: degrees of freedom of the B-spline in n: min(3, number of distinct n - 1), at least 1.
    WHY: periodic chains and EEG have only even n (5 sizes); a curve cannot have more freedom than points."""
    return max(1, min(3, len(set(n_values)) - 1))


def _bs(k: int, lower: float, upper: float) -> str:
    """WHAT: the patsy term of a B-spline in n with k degrees of freedom, degree min(3, k), and boundary
    knots fixed at the given range (so every type's curve uses the same basis)."""
    return f"bs(n, df={k}, degree={min(3, k)}, lower_bound={lower}, upper_bound={upper})"


def fit_mixed(frame: "pandas.DataFrame", formula: str, groups: str = "group_id", reml: bool = True):
    """WHAT: fit statsmodels MixedLM.from_formula(formula, frame, groups=frame[groups]); OLS instead when the
    frame holds a single group (no random effect can be estimated; recorded in fit.notes).
    HOW: try method "lbfgs", then "powell" and "cg" if it does not converge; record convergence and
    boundary warnings (group variance estimated at 0) in fit.notes instead of hiding them.
    WHY: one wrapper so every model is fitted and checked the same way."""
    import statsmodels.formula.api as smf
    if frame[groups].nunique() < 2:
        fit = smf.ols(formula, frame).fit()
        fit.notes = ["single group: OLS, no random effect"]
        fit.is_mixed = False
        return fit
    model = smf.mixedlm(formula, frame, groups=frame[groups])
    notes, fit = [], None
    for method in ("lbfgs", "powell", "cg"):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                fit = model.fit(reml=reml, method=method)
            except Exception as error:                     # noqa: BLE001  (try the next optimiser)
                notes.append(f"{method}: {type(error).__name__}")
                continue
        messages = [str(w.message) for w in caught]
        if getattr(fit, "converged", True):
            notes += [m for m in messages if "boundary" in m.lower() or "singular" in m.lower()]
            break
        notes.append(f"{method}: did not converge")
    if fit is None:
        raise RuntimeError(f"mixed model failed with every optimiser: {notes}")
    fit.notes = notes
    fit.is_mixed = True
    return fit


def _variances(fit) -> tuple:
    """WHAT: (group variance, residual variance) of a fit; group variance 0 for OLS fits."""
    if getattr(fit, "is_mixed", False):
        return float(fit.cov_re.iloc[0, 0]), float(fit.scale)
    return 0.0, float(fit.scale)


def icc(fit) -> float:
    """WHAT: intraclass correlation = group variance / (group variance + residual variance).
    HOW: fit.cov_re and fit.scale. WHY: how alike problems of one source are (Aarts et al. 2014)."""
    g, r = _variances(fit)
    return g / (g + r) if g + r > 0 else NAN


def effective_sample_size(n_rows: int, mean_group_size: float, icc_value: float) -> float:
    """WHAT: Kish's effective sample size N / (1 + (m - 1)·ICC).
    TEST: N = 100, m = 10, ICC = 0 -> 100; ICC = 1 -> 10."""
    return n_rows / (1 + (mean_group_size - 1) * icc_value)


def _fixed(fit):
    """WHAT: (fixed-effect parameter vector, its covariance matrix) as pandas objects."""
    params = fit.fe_params if getattr(fit, "is_mixed", False) else fit.params
    cov = fit.cov_params().loc[params.index, params.index]
    return params, cov


def _n_fixed(fit) -> int:
    """WHAT: number of fixed-effect parameters of a fit."""
    return len(_fixed(fit)[0])


def likelihood_ratio_test(full_ml, reduced_ml) -> tuple:
    """WHAT: (LR statistic, degrees of freedom, p-value) for two nested models fitted by ML (reml=False).
    HOW: LR = 2 (llf_full - llf_reduced); df = difference in number of fixed-effect parameters;
    p = scipy.stats.chi2.sf(LR, df). Raise if either mixed fit used REML (REML likelihoods are not comparable
    when the fixed effects differ)."""
    from scipy.stats import chi2
    for fit in (full_ml, reduced_ml):
        if getattr(fit, "is_mixed", False) and fit.reml:
            raise ValueError("likelihood-ratio tests need ML fits (reml=False)")
    lr = max(0.0, 2 * (float(full_ml.llf) - float(reduced_ml.llf)))
    df = _n_fixed(full_ml) - _n_fixed(reduced_ml)
    return lr, df, float(chi2.sf(lr, df)) if df > 0 else NAN


def marginal_r2(fit) -> float:
    """WHAT: Nakagawa & Schielzeth (2013) marginal R²: variance of the fixed-effect predictions /
    (that variance + group variance + residual variance).
    HOW: var(X @ fe_params) from the fit's design matrix; variances from cov_re and scale."""
    params, _ = _fixed(fit)
    X = np.asarray(fit.model.exog)
    var_f = float(np.var(X @ np.asarray(params)))
    g, r = _variances(fit)
    return var_f / (var_f + g + r) if var_f + g + r > 0 else NAN


def _design_row(fit, values: dict) -> np.ndarray:
    """WHAT: the fixed-effect design row of one new point (e.g. {"problem_type": "GRAPH", "n": 10.0}).
    HOW: patsy.dmatrix of the right-hand side of the fit's formula on a one-row frame, columns reordered to
    the fit's parameters. This rebuilds exactly the fitted columns because (1) the type enters as
    C(problem_type, levels=[...]) with its levels written in the formula, and (2) every spline has
    df == degree (no interior knots) and fixed boundary knots (_bs), so its basis depends on nothing
    learned from the data. (statsmodels stores the formula's design information differently across
    versions, so it is not read from the fit.)"""
    import pandas as pd
    from patsy import dmatrix
    rhs = fit.model.formula.split("~", 1)[1]
    row = dmatrix(rhs, pd.DataFrame([values]), return_type="dataframe")
    params, _ = _fixed(fit)
    return row[list(params.index)].to_numpy()[0]


def predict_at(fit, frame, problem_type: str, n: int) -> tuple:
    """WHAT: (estimate, ci_low, ci_high) of the fixed-effect prediction for one type at one n (group
    offset 0 = an average source).
    HOW: design row x (patsy, with the fit's design_info); estimate = x @ beta; se = sqrt(x Σ xᵀ) with
    Σ = covariance of the fixed effects; 95 % = ±1.96 se. NaN if n is outside the type's observed range
    (no extrapolation)."""
    seen = frame.loc[frame["problem_type"] == problem_type, "n"] if "problem_type" in frame else frame["n"]
    if len(seen) == 0 or not (seen.min() <= n <= seen.max()):
        return NAN, NAN, NAN
    params, cov = _fixed(fit)
    x = _design_row(fit, {"problem_type": problem_type, "n": float(n)})
    estimate = float(x @ params.values)
    se = float(math.sqrt(max(0.0, x @ cov.values @ x)))
    return estimate, estimate - 1.96 * se, estimate + 1.96 * se


def _type_formulas(frame) -> tuple:
    """WHAT: (k, full, shared, null) formulas of fit_type_model, with the spline basis fixed on the range
    of n present in the frame."""
    k = min(spline_df(part["n"]) for _, part in frame.groupby("problem_type"))
    term = _bs(k, frame["n"].min(), frame["n"].max())
    t = f"C(problem_type, levels={sorted(frame['problem_type'].unique())!r})"
    return (k, f"y ~ {t} + {t}:{term}", f"y ~ {t} + {term}", f"y ~ {term}")


def fit_type_model(features: "pandas.DataFrame", feature: str, rows_mask=None,
                   transform: str = "none") -> TypeModelResult:
    """WHAT (spec 2.5): the full Q2 model for one feature and everything read out of it.
    HOW:
      frame = model_frame(...); k = spline_df (smallest over types);
      full  : y ~ C(problem_type) + C(problem_type):bs(n, df=k)     random intercept per group_id
      shared: y ~ C(problem_type) + bs(n, df=k)                      (one curve for all types)
      null  : y ~ bs(n, df=k)                                        (no type at all)
      REML fit of `full` -> icc, n_eff (mean group size = rows / groups), type_at_n for EQUAL_N,
      marginal_r2, pairwise differences (contrast between two design rows at the same n, same Σ; p from the
      normal distribution). ML fits -> level test = LR(shared vs null); shape test = LR(full vs shared).
    WHY: types compared at equal n, using every input set, with the honest error rate (D2.3, D2.5)."""
    import pandas as pd
    from scipy.stats import norm
    frame = model_frame(features, feature, rows_mask, transform)
    result = TypeModelResult(feature=feature, n_rows=len(frame), n_groups=frame["group_id"].nunique())
    if frame["problem_type"].nunique() < 2 or len(frame) < 10:
        result.warnings.append("fewer than 2 types or 10 rows: not fitted")
        return result
    k, full_f, shared_f, null_f = _type_formulas(frame)
    result.spline_df = k
    full = fit_mixed(frame, full_f, reml=True)
    result.converged = bool(getattr(full, "converged", True))
    result.warnings += [f"full REML: {m}" for m in full.notes]
    result.icc = icc(full)
    result.n_eff = effective_sample_size(len(frame), len(frame) / result.n_groups, result.icc)
    result.marginal_r2 = marginal_r2(full)
    types = sorted(frame["problem_type"].unique())
    rows, pairs = [], []
    params, cov = _fixed(full)
    for n in EQUAL_N:
        for t in types:
            est, lo, hi = predict_at(full, frame, t, n)
            rows.append({"type": t, "n": n, "estimate": est, "ci_low": lo, "ci_high": hi})
        for a in range(len(types)):
            for b in range(a + 1, len(types)):
                ta, tb = types[a], types[b]
                if math.isnan(predict_at(full, frame, ta, n)[0]) or math.isnan(predict_at(full, frame, tb, n)[0]):
                    continue
                c = _design_row(full, {"problem_type": ta, "n": float(n)}) - _design_row(full, {"problem_type": tb, "n": float(n)})
                diff = float(c @ params.values)
                se = float(math.sqrt(max(0.0, c @ cov.values @ c)))
                pairs.append({"type_a": ta, "type_b": tb, "n": n, "difference": diff, "ci_low": diff - 1.96 * se,
                              "ci_high": diff + 1.96 * se, "p": float(2 * norm.sf(abs(diff) / se)) if se > 0 else NAN})
    result.type_at_n = pd.DataFrame(rows)
    result.pairwise = pd.DataFrame(pairs)
    full_ml, shared_ml, null_ml = (fit_mixed(frame, f, reml=False) for f in (full_f, shared_f, null_f))
    result.level_lr, result.level_df, result.level_p = likelihood_ratio_test(shared_ml, null_ml)
    result.shape_lr, result.shape_df, result.shape_p = likelihood_ratio_test(full_ml, shared_ml)
    return result


def level_statistic(frame) -> float:
    """WHAT: the level-test LR statistic of a frame (shared vs null, ML): the statistic the group permutation
    test recomputes. -inf when a fit fails."""
    try:
        _, _, shared_f, null_f = _type_formulas(frame)
        return likelihood_ratio_test(fit_mixed(frame, shared_f, reml=False), fit_mixed(frame, null_f, reml=False))[0]
    except Exception:                                       # noqa: BLE001
        return -math.inf


def _permuted_statistic(args) -> float:
    """WHAT: one permutation of group_permutation_test (top-level so multiprocessing can call it)."""
    frame, statistic, seed = args
    rng = np.random.default_rng(seed)
    groups = frame.groupby("group_id")["problem_type"].first()
    shuffled = dict(zip(groups.index, rng.permutation(groups.values)))
    permuted = frame.copy()
    permuted["problem_type"] = permuted["group_id"].map(shuffled)
    return statistic(permuted)


def group_permutation_test(features: "pandas.DataFrame", feature: str, statistic=level_statistic,
                           n_permutations: int = 5000, seed: int = 0, rows_mask=None, n_jobs: int = 1) -> tuple:
    """WHAT (spec 2.6): (observed statistic, permutation p-value, array of permuted statistics, failed fits).
    HOW: statistic(frame) -> float (default: the level LR of fit_type_model's ML fits).
      1. observed = statistic(real frame);
      2. per permutation: shuffle the TYPES among the groups (numpy Generator(seed + k).permutation), so
         each group keeps all its rows and its n values and the number of groups per type is unchanged;
         rewrite problem_type of every row from its group; recompute the statistic;
      3. p = (1 + #{permuted >= observed}) / (1 + n_permutations).
    Fits that fail in a permutation count as -inf and are counted. n_jobs > 1 runs permutations in parallel
    (multiprocessing).
    WHY: a check with no assumption on the shape of the noise."""
    frame = model_frame(features, feature, rows_mask)
    observed = statistic(frame)
    jobs = [(frame, statistic, seed + k + 1) for k in range(n_permutations)]
    if n_jobs > 1:
        from multiprocessing import Pool
        with Pool(n_jobs) as pool:
            null = np.array(pool.map(_permuted_statistic, jobs))
    else:
        null = np.array([_permuted_statistic(j) for j in jobs])
    p = (1 + int(np.sum(null >= observed))) / (1 + n_permutations)
    return observed, p, null, int(np.sum(np.isinf(null)))


# ---------------- Q4: one line (sub-family across n) ----------------

def fit_line(frame: "pandas.DataFrame", unit_column: str = "unit", reml: bool = True):
    """WHAT (spec 4.2): y ~ bs(n, df=spline_df(n)) with a random intercept per unit_column (OLS when the line
    has a single unit). unit: the problem's task id with its n removed (q4_scaling.unit_of): the same
    spin cut and window, the same SAT label, the same EEG subject-session followed across n (EEG channels
    are NESTED: MOTOR_STRIP[:n] in harvest/sources/moabb_eeg.py)."""
    k = spline_df(frame["n"])
    return fit_mixed(frame, f"y ~ {_bs(k, frame['n'].min(), frame['n'].max())}", groups=unit_column, reml=reml)


def slope_vs_log_n(frame: "pandas.DataFrame", unit_column: str = "unit") -> tuple:
    """WHAT (spec 4.2, tier 1): (slope, ci_low, ci_high) of y ~ log(n) with the same random intercept.
    Reading: 'increases' if ci_low > 0, 'decreases' if ci_high < 0, else 'flat'."""
    fit = fit_mixed(frame, "y ~ np.log(n)", groups=unit_column, reml=True)
    params, cov = _fixed(fit)
    slope = float(params["np.log(n)"])
    se = float(math.sqrt(cov.loc["np.log(n)", "np.log(n)"]))
    return slope, slope - 1.96 * se, slope + 1.96 * se


def reading(ci_low: float, ci_high: float) -> str:
    """WHAT: 'increases' / 'decreases' / 'flat' from a 95 % interval of a slope."""
    return "increases" if ci_low > 0 else "decreases" if ci_high < 0 else "flat"


def test_against_law(frame: "pandas.DataFrame", law, unit_column: str = "unit") -> dict:
    """WHAT (spec 4.3, tier 2): does the line deviate from a theoretical law?
    INPUT: law = a callable n -> predicted value, or a sequence of predicted values, one per row of frame
    (used when the prediction depends on more than n, e.g. the constraint ratio of a SAT input set).
    HOW (ML fits): residual r = y - law(n) (or y - law[row]). Imposed model: r ~ 1 (random intercept per unit); free model:
    fit_line on the same rows. (1) shape: LR test free vs imposed (does the curve differ from the law's
    shape?); (2) offset: z-test of the imposed model's intercept = 0 (is the law's level right?).
    OUTPUT {"lr", "df", "p_shape", "offset", "offset_ci_low", "offset_ci_high", "p_offset", "mean_deviation"}."""
    from scipy.stats import norm
    residual = frame.copy()
    predicted = np.array([law(n) for n in frame["n"]]) if callable(law) else np.asarray(law, dtype=float)
    residual["y"] = frame["y"].to_numpy() - predicted
    imposed = fit_mixed(residual, "y ~ 1", groups=unit_column, reml=False)
    free = fit_line(residual, unit_column, reml=False)
    lr, df, p = likelihood_ratio_test(free, imposed)
    params, cov = _fixed(imposed)
    variance = float(cov.iloc[0, 0])
    offset = float(params.iloc[0])
    se = math.sqrt(variance) if variance > 0 else NAN       # an ill-conditioned fit can return a negative variance
    return {"lr": lr, "df": df, "p_shape": p, "offset": offset, "offset_ci_low": offset - 1.96 * se,
            "offset_ci_high": offset + 1.96 * se, "p_offset": float(2 * norm.sf(abs(offset) / se)) if se == se and se > 0 else NAN,
            "mean_deviation": float(residual["y"].mean())}


CANDIDATE_LAWS = ("constant", "a+b/n", "a+b*log(n)", "power", "exponential")
"""The laws compared in spec 4.4."""


def fit_law(frame: "pandas.DataFrame", law: str, unit_column: str = "unit") -> tuple:
    """WHAT (spec 4.4): (log-likelihood on the ORIGINAL scale of y, number of parameters, fit) for one
    candidate law, inside the mixed model (same random intercept per unit, ML fit); None if the law cannot
    be fitted (power / exponential need y > 0).
    HOW: constant: y ~ 1; a+b/n: y ~ I(1/n); a+b*log(n): y ~ np.log(n) — linear in their parameters.
    power a·n^b: log y ~ np.log(n); exponential a·e^{bn}: log y ~ n.
    CAUTION: a model of log y has a likelihood for log y; to compare it with models of y, the Jacobian
    term -Σ log y_i is added to its log-likelihood. Without it the BIC comparison is wrong.
    Number of parameters k = fixed effects + 2 (unit variance, residual variance) for mixed fits, + 1
    (residual variance) for OLS fits."""
    data = frame.copy()
    jacobian = 0.0
    if law in ("power", "exponential"):
        if (data["y"] <= 0).any():
            return None
        jacobian = -float(np.log(data["y"]).sum())
        data["y"] = np.log(data["y"])
    formula = {"constant": "y ~ 1", "a+b/n": "y ~ I(1/n)", "a+b*log(n)": "y ~ np.log(n)",
               "power": "y ~ np.log(n)", "exponential": "y ~ n"}[law]
    fit = fit_mixed(data, formula, groups=unit_column, reml=False)
    k = _n_fixed(fit) + (2 if getattr(fit, "is_mixed", False) else 1)
    return float(fit.llf) + jacobian, k, fit


def choose_law(frame: "pandas.DataFrame", unit_column: str = "unit", threshold: float = 10.0) -> dict:
    """WHAT (spec 4.4, tier 3): BIC of every candidate law and the verdict.
    HOW: BIC = -2 llf + k log(N), N = rows; a law whose fit is degenerate (infinite log-likelihood, seen with
    2-unit random effects on 11 Oct) is left out; best = smallest BIC; named only if BIC(second best) - BIC(best)
    > threshold (10 = 'very strong', Kass & Raftery 1995), else verdict "undetermined on n = <min>-<max>".
    OUTPUT {"bic": {law: value}, "best": law, "delta_bic": float, "verdict": str}."""
    bic = {}
    for law in CANDIDATE_LAWS:
        try:
            fitted = fit_law(frame, law, unit_column)
        except Exception:                                   # noqa: BLE001  (a failed law is left out)
            fitted = None
        if fitted is not None:
            llf, k, _ = fitted
            if math.isfinite(llf):                          # a degenerate fit (infinite likelihood) is left out
                bic[law] = -2 * llf + k * math.log(len(frame))
    ranked = sorted(bic, key=bic.get)
    if not ranked:
        return {"bic": bic, "best": None, "delta_bic": NAN, "verdict": "no law could be fitted"}
    span = f"n = {int(frame['n'].min())}-{int(frame['n'].max())}"
    if len(ranked) < 2:                                     # nothing to compare against: no law can be named
        return {"bic": bic, "best": ranked[0], "delta_bic": NAN,
                "verdict": f"undetermined on {span} (only one law could be fitted)"}
    delta = bic[ranked[1]] - bic[ranked[0]]
    verdict = ranked[0] if delta > threshold else f"undetermined on {span}"
    return {"bic": bic, "best": ranked[0], "delta_bic": delta, "verdict": verdict}
