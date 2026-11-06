"""q3_classify.py — Q3: can a problem's kind be recovered from its structure alone?
(spec: QnA_what_to_do.md, section Q3; decisions D3.1-D3.5)

WHAT THIS FILE PRODUCES
Two layers (D3.1):
  Q3a  type level (GRAPH / COVARIANCE / DIAGONAL / QUANTUM), ALL instance features: a PIPELINE CHECK,
       expected ~100 %; every misclassified problem is listed for inspection (results/q3a_errors.csv).
  Q3b  within each type, recover the DOMAIN, using only the features that are predictable or found in at
       least one cell of the type (forced_table): THE FINDING. Domain table (fixed before results; group
       lists in DOMAINS below):
         GRAPH       biology (6 networks), social/collaboration (5), infrastructure (6), technology (3),
                     molecules (2 datasets)
         COVARIANCE  EEG (2) vs other sensors (9) — indicative only (2 EEG groups)
         DIAGONAL    random k-SAT (2-, 3-, 4-, 5-, 2+p-SAT), structured (community, planted, SR),
                     other constraints (XOR, NAE, 1-in-3)
         QUANTUM     spin chains (8) vs molecules (6)
       Domains with fewer than 3 groups are reported as indicative only.
       Two corrections to the 9 Oct table, from the network catalogue in harvest/sources/real_networks.py:
       biology has 6 networks, not 7 (the dataset holds 20 networks); polblogs is labelled "information" in
       the catalogue and is kept in social/collaboration (a one-network class cannot be held out).
For each layer and each classification task:
  - unit = problem (each problem reads its features from its input set, results/features.csv);
  - multinomial logistic regression, L2 penalty of fixed strength C = 1 (PROPOSED, to confirm), features
    standardised on the training part only (scaler inside the pipeline), training weights 1/(size of the
    problem's group) (D3.2);
  - leave-one-group-out: train on all groups but one, predict the held-out group (Roberts et al. 2017);
  - scores: balanced accuracy, weighted confusion matrix, score per held-out group (D3.2);
  - size control (D3.3): (a) n-only reference classifier, n as a category; (b) post-hoc test in the spirit
    of Dinga et al. (2020): true class ~ n vs ~ n + held-out predicted probabilities, likelihood-ratio test;
    (c) size-balanced subset re-scored (Chyzhyk et al. 2022). n is never regressed out of the features
    (Snoek et al. 2019; Hamdan et al. 2023);
  - permutation importance per feature and per block of correlated features (D3.4);
  - classifier permutation test, labels shuffled between whole groups (Ojala & Garriga 2010);
  - PCA (group medians overlaid) and k-means + adjusted Rand index as description (D3.5);
  - R6 sensitivity run (spec 3.8).
Outputs: results/q3_*.csv/.md, figures/fig_q3_*.
"""
from __future__ import annotations

import math

import numpy as np

import analysis_common as ac
import forced_table as ftb
from features import INSTANCE_FEATURES, TYPE_SPECIFIC_FEATURES

MIN_GROUPS_PER_CLASS = 3
"""Below this number of groups a class (domain) is reported as indicative only."""

C_REGULARISATION = 1.0
"""Inverse strength of the L2 penalty of the logistic regression (scikit-learn's default; fixed, never tuned
on test groups). PROPOSED, to confirm."""

BLOCK_THRESHOLD = 0.8
"""|Spearman| above which two features are shuffled together as a block (PROPOSED, to confirm)."""

DOMAINS = {
    # GRAPH (catalogue descriptions in harvest/sources/real_networks.py)
    "NET:celegans_2019": "biology", "NET:celegans_interactomes": "biology", "NET:celegans_metabolic": "biology",
    "NET:collins_yeast": "biology", "NET:drosophila_flybi": "biology", "NET:ecoli_transcription": "biology",
    "NET:advogato": "social", "NET:anybeat": "social", "NET:arxiv_authors": "social", "NET:netscience": "social",
    "NET:polblogs": "social",
    "NET:chicago_road": "infrastructure", "NET:euroroad": "infrastructure", "NET:london_transport": "infrastructure",
    "NET:openflights": "infrastructure", "NET:power": "infrastructure", "NET:us_air_traffic": "infrastructure",
    "NET:internet_as": "technology", "NET:linux": "technology", "NET:python_dependency": "technology",
    "MOL:alchemy_full": "molecules", "MOL:ZINC_full": "molecules",
    # COVARIANCE
    "MOABB:BNCI2014_001": "EEG", "MOABB:Cho2017": "EEG",
    "TS:ERing": "other sensors", "TS:EigenWorms": "other sensors", "TS:HandMovementDirection": "other sensors",
    "TS:JapaneseVowels": "other sensors", "TS:LSST": "other sensors", "TS:PhonemeSpectra": "other sensors",
    "TS:RacketSports": "other sensors", "TS:SelfRegulationSCP1": "other sensors", "TS:SelfRegulationSCP2": "other sensors",
    # DIAGONAL
    "SAT:2-SAT": "random k-SAT", "SAT:3-SAT": "random k-SAT", "SAT:4-SAT": "random k-SAT", "SAT:5-SAT": "random k-SAT",
    "SAT:2+P-SAT": "random k-SAT", "SAT:CA-3": "structured", "SAT:PLANTED-3": "structured", "SAT:SR": "structured",
    "SAT:XOR-3": "other constraints", "SAT:NAE-3": "other constraints", "SAT:1-IN-3": "other constraints",
    # QUANTUM
    "SPIN:ANNNI": "spin chains", "SPIN:BOND_ALT": "spin chains", "SPIN:CLUSTER": "spin chains",
    "SPIN:DISORDERED": "spin chains", "SPIN:J1J2": "spin chains", "SPIN:TFIM": "spin chains", "SPIN:XXZ": "spin chains",
    "SPIN:XY": "spin chains",
    "CHEM:BeH2": "molecules", "CHEM:H2O": "molecules", "CHEM:H4-chain": "molecules", "CHEM:H4-rectangle": "molecules",
    "CHEM:H6-chain": "molecules", "CHEM:LiH": "molecules",
}
"""The 58 groups and their Q3b domain, written out in full (no pattern matching)."""


def domain_labels(index: "pandas.DataFrame") -> "pandas.Series":
    """WHAT: the domain of every problem (Q3b), from its group_id, per DOMAINS.
    HOW: map group_id through DOMAINS. Raise if a group is missing (every group must be decided).
    WHY: the classes of Q3b, fixed before any result."""
    missing = sorted(set(index["group_id"]) - set(DOMAINS))
    if missing:
        raise KeyError(f"groups without a domain: {missing}")
    return index["group_id"].map(DOMAINS)


def problem_table(features: "pandas.DataFrame" = None) -> "pandas.DataFrame":
    """WHAT: one row per problem: identity (task_id, problem_type, n, group_id, subfamily, domain) and the
    features of its input set.
    HOW: attach_input_sets(load_index()) gives each problem its input set; join the feature table on it."""
    import pandas as pd
    index, _ = ac.attach_input_sets(ac.load_index())
    index["domain"] = domain_labels(index)
    if features is None:
        features = pd.read_csv(ac.paths().results / "features.csv")
    keep = ["input_set"] + [c for c in features.columns if c.endswith("_median")]
    table = index[["task_id", "problem_type", "n", "group_id", "subfamily", "domain", "tier", "input_set"]].merge(
        features[keep], on="input_set", how="left", validate="many_to_one")
    return table


def feature_columns(layer: str, problem_type: str = None, problems: "pandas.DataFrame" = None) -> list:
    """WHAT: the feature columns used. layer "Q3a": all INSTANCE_FEATURES. layer "Q3b": the INSTANCE_FEATURES
    and TYPE_SPECIFIC_FEATURES whose status is predictable or found in at least one cell of problem_type
    (a feature forced in every cell of the type is dropped); then any column with a missing value in the
    task's rows is dropped (e.g. in_sector_gap, which molecules do not have), and the drop is reported."""
    if layer == "Q3a":
        names = list(INSTANCE_FEATURES)
    else:
        cells = [c for c in ftb.CELLS if (problem_type == "QUANTUM" and (c.startswith("SPIN") or c == "MOLECULE"))
                 or c == problem_type]
        names = []
        for f in INSTANCE_FEATURES + TYPE_SPECIFIC_FEATURES:
            statuses = [ftb.status(f, c) for c in cells]
            if any(s in (ftb.Status.PREDICTABLE, ftb.Status.FOUND) for s in statuses):
                names.append(f)
    columns = [f"{f}_median" for f in names]
    if problems is not None:
        dropped = [c for c in columns if problems[c].isna().any()]
        if dropped:
            print(f"{layer} {problem_type or ''}: dropped (missing values) {dropped}", flush=True)
        columns = [c for c in columns if c not in dropped]
    return columns


def group_weights(groups) -> np.ndarray:
    """WHAT: weight 1 / (number of problems of the same group) for every row, so each group totals 1.
    TEST: groups [a, a, b] -> [0.5, 0.5, 1]."""
    groups = np.asarray(groups)
    _, inverse, counts = np.unique(groups, return_inverse=True, return_counts=True)
    return 1.0 / counts[inverse]


def make_classifier(seed: int = 0):
    """WHAT: a scikit-learn Pipeline(StandardScaler(), LogisticRegression(L2, C = C_REGULARISATION,
    max_iter 5000, random_state=seed)); with more than two classes scikit-learn fits the multinomial model.
    The scaler is inside the pipeline so it is fitted on the training part only (no leakage)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    return make_pipeline(StandardScaler(), LogisticRegression(C=C_REGULARISATION, max_iter=5000, random_state=seed))


def leave_one_group_out(X, y, groups, weights, model_factory=make_classifier, keep_models: bool = False) -> dict:
    """WHAT: held-out predictions for every row.
    HOW: for each group g: fit model_factory() on rows not in g with sample weights
    (logisticregression__sample_weight); predict class and class probabilities on rows of g, the
    probabilities placed in the columns of the full class list (a class absent from a training fold gets
    probability 0 there, and the fold is recorded).
    OUTPUT {"pred", "proba" (rows x classes), "classes", "missing_class_folds", "models" (if keep_models:
    {group: fitted model})}."""
    X, y, groups, weights = np.asarray(X, float), np.asarray(y), np.asarray(groups), np.asarray(weights, float)
    classes = sorted(set(y))
    pred = np.empty(len(y), dtype=object)
    proba = np.zeros((len(y), len(classes)))
    missing, models = [], {}
    for g in sorted(set(groups)):
        test = groups == g
        train = ~test
        model = model_factory()
        if len(set(y[train])) < 2:
            raise ValueError(f"fold {g}: training part has a single class")
        model.fit(X[train], y[train], logisticregression__sample_weight=weights[train])
        fold_classes = list(model.classes_)
        if len(fold_classes) < len(classes):
            missing.append(g)
        p = model.predict_proba(X[test])
        for k, c in enumerate(fold_classes):
            proba[test, classes.index(c)] = p[:, k]
        pred[test] = np.array(fold_classes, dtype=object)[p.argmax(axis=1)]
        if keep_models:
            models[g] = model
    return {"pred": pred, "proba": proba, "classes": classes, "missing_class_folds": missing, "models": models}


def scores(y_true, y_pred, groups, weights, classes) -> dict:
    """WHAT (spec 3.3): balanced accuracy (sklearn balanced_accuracy_score with sample_weight=weights);
    weighted confusion matrix (rows true, columns predicted); per held-out group, the fraction of its
    problems predicted correctly; chance = 1 / len(classes)."""
    import pandas as pd
    from sklearn.metrics import balanced_accuracy_score, confusion_matrix
    y_true, y_pred, groups = np.asarray(y_true, dtype=object), np.asarray(y_pred, dtype=object), np.asarray(groups)
    bacc = float(balanced_accuracy_score(y_true, y_pred, sample_weight=weights))
    cm = pd.DataFrame(confusion_matrix(y_true, y_pred, labels=classes, sample_weight=weights), index=classes,
                      columns=classes)
    per_group = (pd.DataFrame({"group_id": groups, "true": y_true, "correct": y_true == y_pred})
                 .groupby(["group_id", "true"])["correct"].agg(["mean", "size"]).reset_index()
                 .rename(columns={"mean": "accuracy", "size": "problems"}))
    return {"balanced_accuracy": bacc, "chance": 1 / len(classes), "confusion": cm, "per_group": per_group}


def n_onehot(n) -> np.ndarray:
    """WHAT: n as a category: one column per size present, 1 where the row has that size."""
    n = np.asarray(n)
    sizes = sorted(set(n))
    return (n[:, None] == np.array(sizes)[None, :]).astype(float)


def n_only_reference(n, y, groups, weights) -> dict:
    """WHAT (spec 3.4a): the same leave_one_group_out + scores with ONE input: n as a category (one-hot)."""
    out = leave_one_group_out(n_onehot(n), y, groups, weights)
    return {**out, **scores(y, out["pred"], groups, weights, out["classes"])}


def _multinomial_loglik(X, y) -> tuple:
    """WHAT: (maximised log-likelihood, number of parameters) of an unpenalised multinomial logistic
    regression of y on X (with intercept)."""
    from sklearn.linear_model import LogisticRegression
    classes = sorted(set(y))
    model = LogisticRegression(penalty=None, max_iter=10000).fit(X, y)
    p = model.predict_proba(X)
    idx = np.array([list(model.classes_).index(c) for c in y])
    loglik = float(np.log(np.clip(p[np.arange(len(y)), idx], 1e-300, None)).sum())
    k = (X.shape[1] + 1) * (len(classes) - 1 if len(classes) > 2 else 1)
    return loglik, k


def dinga_posthoc_test(y, n, proba) -> dict:
    """WHAT (spec 3.4b, after Dinga et al. 2020): predictive power of structure that n does not explain.
    HOW: model A: unpenalised multinomial logit of y on one-hot(n); model B: on one-hot(n) + log of the
    held-out predicted probabilities (one class column dropped to avoid collinearity); LR = 2 (llf_B - llf_A),
    df = added parameters, p from chi2; McFadden pseudo-R² of A and B (1 - llf / llf of the intercept-only
    model). Unweighted (a likelihood-ratio test needs counts, not the 1/group-size weights).
    NOTE: only the abstract of Dinga et al. was checked (11 Oct); read the full paper and adjust if its
    procedure differs."""
    from scipy.stats import chi2
    y = np.asarray(y, dtype=object)
    A = n_onehot(n)[:, 1:]                                  # drop one size: the intercept carries it
    logp = np.log(np.clip(np.asarray(proba), 1e-12, None))[:, 1:]
    llf_a, k_a = _multinomial_loglik(A, y)
    llf_b, k_b = _multinomial_loglik(np.hstack([A, logp]), y)
    freq = np.array([np.mean(y == c) for c in sorted(set(y))])
    llf_0 = float(sum(np.log(freq[sorted(set(y)).index(c)]) for c in y))
    lr = max(0.0, 2 * (llf_b - llf_a))
    df = k_b - k_a
    return {"lr": lr, "df": df, "p": float(chi2.sf(lr, df)), "pseudo_r2_n_only": 1 - llf_a / llf_0,
            "pseudo_r2_n_plus_structure": 1 - llf_b / llf_0}


def size_balanced_subset(classes_of_rows, n, seed: int = 0):
    """WHAT (spec 3.4c, after Chyzhyk et al. 2022): row indices of a subset in which every class has the
    same distribution of n. HOW: keep only n values present in every class; for each such n, subsample
    every class to the smallest class count at that n (fixed seed). None if no n is shared by all classes."""
    rng = np.random.default_rng(seed)
    c, n = np.asarray(classes_of_rows, dtype=object), np.asarray(n)
    labels = sorted(set(c))
    shared = [v for v in sorted(set(n)) if all(np.any((n == v) & (c == lab)) for lab in labels)]
    if not shared:
        return None
    rows = []
    for v in shared:
        smallest = min(int(np.sum((n == v) & (c == lab))) for lab in labels)
        for lab in labels:
            candidates = np.flatnonzero((n == v) & (c == lab))
            rows.extend(rng.choice(candidates, size=smallest, replace=False).tolist())
    return np.array(sorted(rows))


def correlated_blocks(X, columns, threshold: float = BLOCK_THRESHOLD) -> list:
    """WHAT: groups of features whose |Spearman correlation| >= threshold (connected components of the graph
    'correlated above threshold'), as lists of column names; blocks of size >= 2 only.
    HOW: computed once on all rows of the task (the blocks only decide which columns are shuffled together
    in permutation_importance; they never enter a prediction)."""
    import pandas as pd
    rho = pd.DataFrame(np.asarray(X, float), columns=columns).corr(method="spearman").abs().fillna(0).to_numpy()
    parent = list(range(len(columns)))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a
    for i in range(len(columns)):
        for j in range(i + 1, len(columns)):
            if rho[i, j] >= threshold:
                parent[find(i)] = find(j)
    blocks = {}
    for i, c in enumerate(columns):
        blocks.setdefault(find(i), []).append(c)
    return [b for b in blocks.values() if len(b) >= 2]


def permutation_importance(X, y, groups, weights, columns, blocks, models, n_repeats: int = 30, seed: int = 0):
    """WHAT (spec 3.5, D3.4): for each feature and each block: mean and standard deviation of the drop in
    balanced accuracy when its values are shuffled.
    HOW: baseline = held-out predictions of the leave-one-group-out models (`models`, one per held-out group).
    For each feature (or block, its columns shuffled with one common row permutation) and repeat: permute the
    values across ALL rows (the pooled held-out rows: a group with a single problem cannot be shuffled within
    itself), predict every row with the model that did not see its group, re-score; importance = baseline -
    permuted. (Breiman 2001; Fisher, Rudin & Dominici 2019.)"""
    import pandas as pd
    X, y, groups = np.asarray(X, float), np.asarray(y, dtype=object), np.asarray(groups)
    rng = np.random.default_rng(seed)

    def predict_all(matrix):
        pred = np.empty(len(y), dtype=object)
        for g, model in models.items():
            rows = groups == g
            pred[rows] = model.predict(matrix[rows])
        return pred
    from sklearn.metrics import balanced_accuracy_score
    base = balanced_accuracy_score(y, predict_all(X), sample_weight=weights)
    targets = [[c] for c in columns] + blocks
    rows = []
    for target in targets:
        idx = [columns.index(c) for c in target]
        drops = []
        for _ in range(n_repeats):
            permuted = X.copy()
            order = rng.permutation(len(y))
            permuted[:, idx] = X[order][:, idx]
            drops.append(base - balanced_accuracy_score(y, predict_all(permuted), sample_weight=weights))
        rows.append({"features": " + ".join(c.replace("_median", "") for c in target), "block": len(target) > 1,
                     "importance": float(np.mean(drops)), "sd": float(np.std(drops))})
    return pd.DataFrame(rows).sort_values("importance", ascending=False).reset_index(drop=True)


def _null_score(args) -> float:
    """WHAT: one shuffled run of classifier_permutation_test (top-level for multiprocessing)."""
    X, y, groups, weights, seed = args
    rng = np.random.default_rng(seed)
    g_unique = np.array(sorted(set(groups)))
    g_class = {g: y[groups == g][0] for g in g_unique}
    shuffled = dict(zip(g_unique, rng.permutation([g_class[g] for g in g_unique])))
    y_perm = np.array([shuffled[g] for g in groups], dtype=object)
    try:
        out = leave_one_group_out(X, y_perm, groups, weights)
    except ValueError:
        return float("nan")
    from sklearn.metrics import balanced_accuracy_score
    return float(balanced_accuracy_score(y_perm, out["pred"], sample_weight=weights))


def classifier_permutation_test(X, y, groups, weights, observed: float, n_permutations: int = 1000,
                                seed: int = 0, n_jobs: int = 1) -> dict:
    """WHAT (spec 3.6, Ojala & Garriga 2010): permutation p-value of the observed balanced accuracy, and the
    null scores. HOW: each group gets the class of another group (classes shuffled between whole groups;
    number of groups per class unchanged — every group holds a single class, checked), rerun
    leave_one_group_out + balanced accuracy, n_permutations times; p = (1 + #{null >= observed}) /
    (1 + valid permutations). Parallel when n_jobs > 1."""
    y, groups = np.asarray(y, dtype=object), np.asarray(groups)
    for g in set(groups):
        if len(set(y[groups == g])) != 1:
            raise ValueError(f"group {g} holds several classes; group-level shuffling is undefined")
    jobs = [(np.asarray(X, float), y, groups, np.asarray(weights, float), seed + k + 1) for k in range(n_permutations)]
    if n_jobs > 1:
        from multiprocessing import Pool
        with Pool(n_jobs) as pool:
            null = np.array(pool.map(_null_score, jobs))
    else:
        null = np.array([_null_score(j) for j in jobs])
    valid = null[~np.isnan(null)]
    return {"observed": observed, "p": (1 + int(np.sum(valid >= observed))) / (1 + len(valid)),
            "null": null, "invalid": int(np.isnan(null).sum())}


def pca_description(X, groups) -> dict:
    """WHAT (spec 3.7): standardise X; PCA; the first two components of every row; explained variance ratios;
    per group the median position."""
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    Z = StandardScaler().fit_transform(np.asarray(X, float))
    pca = PCA(n_components=min(2, Z.shape[1])).fit(Z)
    coords = pca.transform(Z)
    groups = np.asarray(groups)
    medians = {g: np.median(coords[groups == g], axis=0) for g in sorted(set(groups))}
    return {"coords": coords, "explained": pca.explained_variance_ratio_, "group_medians": medians}


def kmeans_ari(X, labels, seed: int = 0) -> float:
    """WHAT (spec 3.7, secondary): KMeans(k = number of classes, n_init=20, random_state=seed) on
    standardised X; adjusted_rand_score(labels, clusters)."""
    from sklearn.cluster import KMeans
    from sklearn.metrics import adjusted_rand_score
    from sklearn.preprocessing import StandardScaler
    Z = StandardScaler().fit_transform(np.asarray(X, float))
    k = len(set(labels))
    clusters = KMeans(n_clusters=k, n_init=20, random_state=seed).fit_predict(Z)
    return float(adjusted_rand_score(labels, clusters))


def run_task(name: str, problems: "pandas.DataFrame", layer: str, problem_type: str = None,
             n_permutations: int = 1000, n_jobs: int = 1, folder=None) -> dict:
    """WHAT: one classification task end to end (Q3a, or Q3b for one type): matrix, classes, weights,
    leave-one-group-out predictions, scores, size control (a, b, c), importance, permutation test, PCA,
    k-means. Classes with fewer than MIN_GROUPS_PER_CLASS groups are marked 'indicative'. Saves
    results/q3_<name>_*.csv and returns a dict of results (summary row in "summary")."""
    import pandas as pd
    rows = problems if problem_type is None else problems[problems["problem_type"] == problem_type]
    rows = rows.reset_index(drop=True)
    columns = feature_columns(layer, problem_type, rows)
    target = "problem_type" if layer == "Q3a" else "domain"
    X, y, groups, n = rows[columns].to_numpy(float), rows[target].to_numpy(object), rows["group_id"].to_numpy(), rows["n"].to_numpy()
    w = group_weights(groups)
    out = leave_one_group_out(X, y, groups, w, keep_models=True)
    sc = scores(y, out["pred"], groups, w, out["classes"])
    ref = n_only_reference(n, y, groups, w)
    dinga = dinga_posthoc_test(y, n, out["proba"])
    subset = size_balanced_subset(y, n)
    balanced_sub = math.nan
    if subset is not None and len(set(y[subset])) > 1:
        from sklearn.metrics import balanced_accuracy_score
        balanced_sub = float(balanced_accuracy_score(y[subset], out["pred"][subset], sample_weight=w[subset]))
    blocks = correlated_blocks(X, columns)
    importance = permutation_importance(X, y, groups, w, columns, blocks, out["models"])
    perm = classifier_permutation_test(X, y, groups, w, sc["balanced_accuracy"], n_permutations, n_jobs=n_jobs) \
        if n_permutations else {"p": math.nan, "null": np.array([]), "invalid": 0}
    groups_per_class = rows.groupby(target)["group_id"].nunique()
    summary = {"task": name, "classes": len(out["classes"]), "problems": len(rows), "groups": len(set(groups)),
               "features": len(columns), "balanced_accuracy": sc["balanced_accuracy"], "chance": sc["chance"],
               "n_only_balanced_accuracy": ref["balanced_accuracy"], "dinga_lr": dinga["lr"], "dinga_df": dinga["df"],
               "dinga_p": dinga["p"], "pseudo_r2_n_only": dinga["pseudo_r2_n_only"],
               "pseudo_r2_n_plus_structure": dinga["pseudo_r2_n_plus_structure"],
               "size_balanced_rows": 0 if subset is None else len(subset), "size_balanced_accuracy": balanced_sub,
               "permutation_p": perm["p"], "permutations": n_permutations, "kmeans_ari": kmeans_ari(X, y),
               "indicative_classes": ", ".join(c for c, k in groups_per_class.items() if k < MIN_GROUPS_PER_CLASS),
               "missing_class_folds": len(out["missing_class_folds"])}
    ac.save_table(sc["confusion"].reset_index().rename(columns={"index": "true"}), f"q3_{name}_confusion", folder)
    ac.save_table(sc["per_group"], f"q3_{name}_per_group", folder)
    ac.save_table(importance, f"q3_{name}_importance", folder)
    if layer == "Q3a":
        errors = rows.loc[out["pred"] != y, ["task_id", "group_id", "n", "problem_type"]].assign(predicted=out["pred"][out["pred"] != y])
        ac.save_table(errors, "q3a_errors", folder)
    return {"summary": summary, "scores": sc, "importance": importance, "null": perm["null"],
            "pca": pca_description(X, groups), "y": y, "groups": groups, "pred": out["pred"]}


def sensitivity_flags(n_jobs: int = 1, folder=None) -> "pandas.DataFrame":
    """WHAT (spec 3.8): balanced accuracy, Dinga p and n-only reference of every task with features.csv and
    with features_noflags.csv, side by side (no permutations: they do not change with the flags question)."""
    import pandas as pd
    out = []
    for label, file in (("with_flagged", "features.csv"), ("without_flagged", "features_noflags.csv")):
        problems = problem_table(pd.read_csv(ac.paths().results / file))
        for name, layer, t in tasks():
            s = run_task(name, problems, layer, t, n_permutations=0, folder=(folder or ac.paths().results) / "sensitivity" / label)["summary"]
            out.append({"task": name, "run": label, "balanced_accuracy": s["balanced_accuracy"], "dinga_p": s["dinga_p"],
                        "n_only_balanced_accuracy": s["n_only_balanced_accuracy"]})
    table = pd.DataFrame(out)
    return table.pivot(index="task", columns="run").reset_index().pipe(
        lambda t: t.set_axis(["task"] + [f"{a}_{b}" for a, b in t.columns[1:]], axis=1))


def tasks() -> list:
    """WHAT: the classification tasks: (name, layer, type) for Q3a and the four Q3b tasks."""
    return [("q3a_types", "Q3a", None)] + [(f"q3b_{t.lower()}", "Q3b", t) for t in ac.TYPES]


def figures(results: dict, folder=None) -> list:
    """WHAT (spec 3.9): per task: the confusion matrix (counts weighted 1/group size, values written in the
    cells), the PCA plot (problems small, group medians large, colours = classes with a legend), and the
    permutation null distribution with the observed score marked; one figure of permutation importance per
    task (bars with standard deviation)."""
    import matplotlib.pyplot as plt
    ac.figure_style()
    saved = []
    palette = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
    for name, r in results.items():
        cm = r["scores"]["confusion"]
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
        ax = axes[0]
        share = cm.to_numpy() / cm.to_numpy().sum(axis=1, keepdims=True).clip(1e-12)
        ax.imshow(share, cmap="Blues", vmin=0, vmax=1)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, f"{cm.iat[i, j]:.1f}", ha="center", va="center", fontsize=8,
                        color="white" if share[i, j] > 0.55 else ac.INK["primary"])
        ax.set_xticks(range(cm.shape[1]), cm.columns, rotation=30, ha="right")
        ax.set_yticks(range(cm.shape[0]), cm.index)
        ax.set_xlabel("predicted")
        ax.set_ylabel("true")
        ax.grid(False)
        ax.set_title(f"balanced accuracy {r['summary']['balanced_accuracy']:.2f} (chance {r['summary']['chance']:.2f})")
        ax = axes[1]
        coords, y, groups = r["pca"]["coords"], r["y"], r["groups"]
        classes = sorted(set(y))
        for k, c in enumerate(classes):
            colour = ac.TYPE_COLOURS.get(c, palette[k % len(palette)])
            mask = y == c
            ax.scatter(coords[mask, 0], coords[mask, 1] if coords.shape[1] > 1 else 0 * coords[mask, 0], s=6,
                       alpha=0.35, color=colour, edgecolors="none")
            for g, m in r["pca"]["group_medians"].items():
                if y[groups == g][0] == c:
                    ax.scatter(m[0], m[1] if len(m) > 1 else 0, s=45, color=colour, edgecolors="white", linewidths=1.5)
            ax.scatter([], [], s=45, color=colour, label=c)
        ex = r["pca"]["explained"]
        ax.set_xlabel(f"PC1 ({ex[0]:.0%})")
        ax.set_ylabel(f"PC2 ({ex[1]:.0%})" if len(ex) > 1 else "")
        ax.legend(loc="best", fontsize=7)
        ax.set_title("PCA: problems (small), group medians (large)")
        ax = axes[2]
        null = r["null"][~np.isnan(r["null"])] if len(r["null"]) else []
        if len(null):
            ax.hist(null, bins=30, color="#c3c2b7")
            ax.axvline(r["summary"]["balanced_accuracy"], color="#1f1f1e", linewidth=2, label="observed")
            ax.legend(loc="upper left")
        else:
            ax.text(0.5, 0.5, "permutation test not run\n(run_analysis.py runs it)", ha="center", va="center",
                    transform=ax.transAxes, color=ac.INK["secondary"])
            ax.set_xticks([])
            ax.set_yticks([])
        ax.set_xlabel("balanced accuracy under group-level label shuffling")
        ax.set_title(f"permutation p = {r['summary']['permutation_p']:.3g}")
        fig.suptitle(name)
        fig.tight_layout()
        saved.append(ac.save_figure(fig, f"fig_{name}", folder))
        imp = r["importance"].head(15).iloc[::-1]
        fig, ax = plt.subplots(figsize=(6, 0.3 * len(imp) + 1.2))
        ax.barh(imp["features"], imp["importance"], xerr=imp["sd"], color="#5f5e5a", height=0.6)
        ax.set_xlabel("drop in balanced accuracy when shuffled")
        ax.set_title(f"{name}: permutation importance")
        ax.grid(axis="y", visible=False)
        saved.append(ac.save_figure(fig, f"fig_{name}_importance", folder))
    return saved


def run_q3(n_permutations: int = 1000, n_jobs: int = 1, folder=None) -> None:
    """WHAT: Q3a (misclassified problems in q3a_errors), then Q3b for each type, the summary table
    (q3_summary), the sensitivity run (q3_sensitivity), the figures."""
    import pandas as pd
    problems = problem_table()
    results = {}
    for name, layer, t in tasks():
        results[name] = run_task(name, problems, layer, t, n_permutations, n_jobs, folder)
        print(f"{name}: balanced accuracy {results[name]['summary']['balanced_accuracy']:.3f}", flush=True)
    ac.save_table(pd.DataFrame([r["summary"] for r in results.values()]), "q3_summary", folder)
    ac.save_table(sensitivity_flags(n_jobs, folder), "q3_sensitivity", folder)
    figures(results, folder)


if __name__ == "__main__":
    run_q3()
