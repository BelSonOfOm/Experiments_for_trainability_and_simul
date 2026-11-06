"""layer0_concentration.py — layer zero of the TFIM problems: overlaps, known-answer test, concentration.

WHAT THIS FILE DOES (plan approved by Charbel, 1 Nov 2026, 21:46; see LAYER0/LOG.md)
  1. load_tfim            the 160 states, h values, labels and TRAIN/TEST split of one TFIM problem;
  2. fidelity_matrix      F_ij = |<psi_i|psi_j>|^2 for every pair of inputs (the raw material of layer zero);
  3. exact_fidelity       the same quantity from the closed form of Damski & Rams (arXiv:1308.5917, Eqs. 9-10),
                          valid for periodic chains of even n with h > 0;
  4. known_answer_test    functions 2 and 3 must agree on every pair of every periodic problem, max
                          |difference| <= 1e-9 (tolerance chosen by Charbel); nothing else runs otherwise;
  5. offdiag_stats        mean and variance of the off-diagonal entries of F, on ALL, TRAIN or TEST inputs;
  6. concentration_table  one row per (problem, subset);
  7. decay_law            per line (boundary x window) and subset: does the variance follow an exponential
                          law in n? Same rule as Q4 (mixed_models.choose_law: BIC, named only if dBIC > 10,
                          Kass & Raftery 1995);
  8. figure               variance against n, one panel per line.
Run from anywhere:  python layer0_concentration.py      (outputs in LAYER0/results/)
"""
from __future__ import annotations

import os
import sqlite3
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
LAYER0 = HERE.parent
TME4 = LAYER0.parent / "COURSEWORK" / "TME4"
sys.path.insert(0, str(TME4))
import analysis_common as ac          # noqa: E402  (the project's reader of data_final, unchanged)
import mixed_models as mm             # noqa: E402  (Q4's law-choice rule, unchanged)

RESULTS = LAYER0 / "results"
TOLERANCE = 1e-9                      # Charbel, 21:46
SUBSETS = ("ALL", "TRAIN", "TEST")


def tfim_task_ids() -> list:
    """WHAT: the 28 TFIM task ids (open and periodic, near and far), in a fixed order.
    HOW: read from data_final/harvest.sqlite, ORDER BY task_id."""
    db = sqlite3.connect(TME4 / "data_final" / "harvest.sqlite")
    try:
        return [r[0] for r in db.execute("SELECT task_id FROM task WHERE task_id LIKE 'spin-generator-2:TFIM-%' "
                                         "ORDER BY task_id")]
    finally:
        db.close()


def describe(task_id: str) -> dict:
    """WHAT: boundary, n and window of a task id such as 'spin-generator-2:TFIM-periodic:n=4:h:far'."""
    _, model, size, _, window = task_id.split(":")
    return {"boundary": model.split("-")[1], "n": int(size.split("=")[1]), "window": window}


def load_tfim(task_id: str) -> dict:
    """WHAT: states (rows, real, length 2^n), h, labels y, split of one TFIM problem.
    HOW: ac.load_problem; x["state"] and info["h"] of every instance.
    CHECKS (stop if one fails): every state has norm 1 (to 1e-12); every state is in parity sector +1;
    no state carries residual_degeneracy. WHY: a bad input must not slip through silently."""
    p = ac.load_problem(task_id, with_operators=False)
    states = np.array([inst.x["state"] for inst in p.instances], dtype=float)
    norms = np.linalg.norm(states, axis=1)
    if np.max(np.abs(norms - 1)) > 1e-12:
        raise ValueError(f"{task_id}: a state does not have norm 1 (worst {np.max(np.abs(norms - 1)):.2e})")
    if any(inst.info["sector"] != 1 for inst in p.instances):
        raise ValueError(f"{task_id}: a state is not in the even parity sector")
    if any(inst.info["residual_degeneracy"] for inst in p.instances):
        raise ValueError(f"{task_id}: a state carries residual_degeneracy")
    return {"task_id": task_id, **describe(task_id), "states": states,
            "h": np.array([inst.info["h"] for inst in p.instances]),
            "y": np.array([inst.y for inst in p.instances]), "split": np.array(p.split)}


def fidelity_matrix(states: np.ndarray) -> np.ndarray:
    """WHAT: F_ij = |<psi_i|psi_j>|^2 for all pairs. HOW: the states are real, so the overlap is a dot
    product: F = (S S^T)^2 elementwise."""
    overlaps = states @ states.T
    return overlaps ** 2


def exact_fidelity(h1: float, h2: float, n: int) -> float:
    """WHAT: |<psi(h1)|psi(h2)>|^2 for the periodic TFIM of even n (Damski & Rams, arXiv:1308.5917).
    HOW: momenta k = pi/n, 3pi/n, ..., pi - pi/n (positive-parity sector, n/2 values); angle theta_k(h) with
    cos = (h - cos k)/eps_k, sin = sin k/eps_k (their Eq. 9), i.e. theta_k = atan2(sin k, h - cos k);
    overlap = prod_k cos((theta_k(h1) - theta_k(h2))/2) (from their Eq. 10, approved 16:42); squared."""
    if n % 2:
        raise ValueError("the formula used here is for even n")
    k = np.pi * (2 * np.arange(n // 2) + 1) / n
    t1 = np.arctan2(np.sin(k), h1 - np.cos(k))
    t2 = np.arctan2(np.sin(k), h2 - np.cos(k))
    return float(np.prod(np.cos((t1 - t2) / 2)) ** 2)


def known_answer_test() -> "pandas.DataFrame":
    """WHAT: for every periodic TFIM problem, compare fidelity_matrix with exact_fidelity on ALL pairs.
    OUTPUT one row per problem: n, window, pairs, max |difference|, pass (max <= TOLERANCE)."""
    import pandas as pd
    rows = []
    for tid in tfim_task_ids():
        if describe(tid)["boundary"] != "periodic":
            continue
        d = load_tfim(tid)
        F = fidelity_matrix(d["states"])
        E = np.array([[exact_fidelity(a, b, d["n"]) for b in d["h"]] for a in d["h"]])
        worst = float(np.max(np.abs(F - E)))
        rows.append({"task_id": tid, "n": d["n"], "window": d["window"], "pairs": F.size,
                     "max_abs_difference": worst, "pass": worst <= TOLERANCE})
    return pd.DataFrame(rows)


def offdiag_stats(F: np.ndarray, mask: np.ndarray) -> dict:
    """WHAT: mean and variance of the off-diagonal entries of F restricted to the inputs in mask.
    HOW: take the sub-matrix, drop the diagonal (always 1), population variance (numpy default)."""
    sub = F[np.ix_(mask, mask)]
    off = sub[~np.eye(len(sub), dtype=bool)]
    return {"inputs": int(mask.sum()), "mean": float(off.mean()), "variance": float(off.var())}


def concentration_table() -> "pandas.DataFrame":
    """WHAT: one row per (TFIM problem, subset in ALL / TRAIN / TEST) with offdiag_stats."""
    import pandas as pd
    rows = []
    for tid in tfim_task_ids():
        d = load_tfim(tid)
        F = fidelity_matrix(d["states"])
        for subset in SUBSETS:
            mask = np.ones(len(F), bool) if subset == "ALL" else d["split"] == subset
            rows.append({"task_id": tid, "boundary": d["boundary"], "window": d["window"], "n": d["n"],
                         "subset": subset, **offdiag_stats(F, mask)})
    return pd.DataFrame(rows)


def decay_law(table) -> "pandas.DataFrame":
    """WHAT: per line (boundary x window) and subset, the law of the variance against n.
    HOW: mixed_models.choose_law (Q4's rule) on y = variance, one unit per line (a single unit: OLS)."""
    import pandas as pd
    rows = []
    for (b, w, s), part in table.groupby(["boundary", "window", "subset"]):
        frame = pd.DataFrame({"y": part["variance"].to_numpy(), "n": part["n"].to_numpy(),
                              "unit": f"{b}-{w}"})
        verdict = mm.choose_law(frame, "unit")
        rows.append({"boundary": b, "window": w, "subset": s, "sizes": len(part),
                     "best_law": verdict["best"], "delta_bic": verdict["delta_bic"], "verdict": verdict["verdict"]})
    return pd.DataFrame(rows)


def figure(table) -> Path:
    """WHAT: log10 of the variance against n, one panel per line, one curve per subset."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    lines = sorted(set(zip(table["boundary"], table["window"])))
    fig, axes = plt.subplots(1, len(lines), figsize=(4 * len(lines), 3.4), sharey=True)
    for ax, (b, w) in zip(axes, lines):
        for s in SUBSETS:
            part = table[(table.boundary == b) & (table.window == w) & (table.subset == s)].sort_values("n")
            ax.plot(part["n"], np.log10(part["variance"]), marker="o", label=s)
        ax.set_title(f"{b}, {w}")
        ax.set_xlabel("n")
    axes[0].set_ylabel("log10 variance of off-diagonal F")
    axes[0].legend()
    return ac.save_figure(fig, "layer0_concentration", RESULTS / "figures")


def main() -> None:
    kat = known_answer_test()
    ac.save_table(kat, "known_answer_test", RESULTS)
    print(kat.to_string(index=False))
    if not kat["pass"].all():
        print("KNOWN-ANSWER TEST FAILED: nothing else is run.")
        return
    table = concentration_table()
    ac.save_table(table, "concentration_table", RESULTS)
    laws = decay_law(table)
    ac.save_table(laws, "concentration_laws", RESULTS)
    figure(table)
    print(laws.to_string(index=False))


if __name__ == "__main__":
    main()
