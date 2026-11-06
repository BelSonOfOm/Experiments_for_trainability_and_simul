"""layer0_build.py — layer zero of the periodic TFIM problems: FGW distances and intrinsic Čech barcodes.

WHAT THIS FILE DOES (plan approved by Charbel, 2 Nov 2026, 17:53; see LAYER0/LOG.md, 2 Nov entry)
  1. load_tfim          reused unchanged from layer0_concentration.py; only the TRAIN inputs are used;
  2. qubit_states       Bloch vector (<X>, <Y>, <Z>) of each qubit's reduced density matrix (partial trace);
  3. bures_angle        arccos sqrt(F) between two qubit states, F = Tr(rho sigma) + 2 sqrt(det rho det sigma)
                        (Hübner 1992); in Bloch form F = (1 + r.s + sqrt((1-|r|^2)(1-|s|^2))) / 2;
  4. structure_matrix   C(k, l) = number of steps between qubits k and l along the ring, min(|k-l|, n-|k-l|);
  5. content_scales     ESL §14.3.3: average Bures angle and average |h - h'| over all ordered pairs of qubits
                        of the training inputs; each part of the content cost is divided by its average;
  6. content_cost       M(k, l) = 1/2 * Bures/scale_B + 1/2 * |h - h'|/scale_h       (beta = 1/2, Charbel 17:38);
  7. fgw                FGW with q = 1, alpha = 1/2, weight 1/n per qubit; conditional gradient with exact line
                        search (Vayer et al., ICML 2019, Alg. 1); several starting matchings, smallest value kept;
  8. distance_matrix    FGW between every pair of training inputs of one problem;
  9. intrinsic_cech     vertices at 0; an edge or triangle sigma enters at min over training inputs w of
                        max over v in sigma of D(w, v) (Chazal, de Silva & Oudot 2014); persistence by gudhi
                        over F2, dimensions 0 and 1;
 10. barcodes           CSV of all bars, one summary table, one figure.
Known-answer checks run first (known_answer_checks); a problem's barcodes are computed only if its FGW matrix
matches the closed form. Run:  python3 layer0_build.py          (outputs in LAYER0/results/layer0_fgw/)
"""
from __future__ import annotations

import math
import sys
import time
from pathlib import Path

import numpy as np
import ot
import gudhi

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from layer0_concentration import load_tfim, tfim_task_ids, describe   # noqa: E402  (reused unchanged)

OUT = HERE.parent / "results" / "layer0_fgw"
ALPHA = 0.5          # Charbel, 1 Nov 23:05: wiring and content count equally
BETA = 0.5           # Charbel, 2 Nov 17:38-17:39: state part and term part count equally, after rescaling
TOLERANCE = 1e-9     # Charbel's tolerance for known-answer tests (1 Nov 21:46)


# ---------------------------------------------------------------- 2. local states
def qubit_states(state: np.ndarray, n: int) -> np.ndarray:
    """WHAT: the n Bloch vectors (<X>, <Y>, <Z>) of the one-qubit reduced density matrices of a pure state.
    HOW: psi reshaped to n axes of size 2; qubit k's axis moved first; rho_k = A A^dagger with A of shape
    (2, 2^(n-1)) (partial trace over the other qubits); <X> = 2 Re rho_01, <Y> = -2 Im rho_01, <Z> = rho_00 - rho_11.
    WHY: each qubit's local state is the state part of its content (Charbel, 1 Nov 23:08)."""
    psi = np.asarray(state, dtype=complex).reshape([2] * n)
    out = np.empty((n, 3))
    for k in range(n):
        A = np.moveaxis(psi, k, 0).reshape(2, -1)
        rho = A @ A.conj().T
        out[k] = (2 * rho[0, 1].real, -2 * rho[0, 1].imag, (rho[0, 0] - rho[1, 1]).real)
    return out


# ---------------------------------------------------------------- 3. Bures angle
def bures_angle(r: np.ndarray, s: np.ndarray) -> np.ndarray:
    """WHAT: Bures angle arccos sqrt(F) between qubit states with Bloch vectors r and s (broadcasts on leading axes).
    HOW: F = (1 + r.s + sqrt((1-|r|^2)(1-|s|^2))) / 2 (Hübner 1992 in Bloch form). For accuracy when r ~ s,
    1 - F is computed as N / (2 Dn) with N = |r - s|^2 - |r x s|^2 = (1 - r.s)^2 - (1-|r|^2)(1-|s|^2) and
    Dn = 1 - r.s + sqrt((1-|r|^2)(1-|s|^2)); the angle is atan2(sqrt(1-F), sqrt(F)).
    WHY: Charbel chose the Bures angle for the intrinsic geometry (2 Nov 17:36)."""
    r = np.asarray(r, float)
    s = np.asarray(s, float)
    rs = np.sum(r * s, axis=-1)
    pr = np.clip(1.0 - np.sum(r * r, axis=-1), 0.0, None)
    ps = np.clip(1.0 - np.sum(s * s, axis=-1), 0.0, None)
    root = np.sqrt(pr * ps)
    diff2 = np.sum((r - s) ** 2, axis=-1)
    cross2 = np.sum(np.cross(r, s) ** 2, axis=-1)
    num = np.clip(diff2 - cross2, 0.0, None)
    den = 1.0 - rs + root
    with np.errstate(divide="ignore", invalid="ignore"):
        one_minus_F = np.where(den > 0, num / (2.0 * den), 1.0)
    one_minus_F = np.clip(one_minus_F, 0.0, 1.0)
    return np.arctan2(np.sqrt(one_minus_F), np.sqrt(1.0 - one_minus_F))


# ---------------------------------------------------------------- 4. wiring
def structure_matrix(n: int) -> np.ndarray:
    """WHAT: C(k, l) = number of steps between qubits k and l along the ring of n qubits.
    WHY: FGW needs a true distance between the qubits of an input (Vayer et al. 2020, Def. 2.1); steps along the
    graph joined by the ZZ terms (approved by Charbel for this run, 2 Nov 17:53)."""
    k = np.arange(n)
    d = np.abs(k[:, None] - k[None, :])
    return np.minimum(d, n - d).astype(float)


# ---------------------------------------------------------------- 5-6. content
def content_scales(blochs: list, hs: np.ndarray) -> tuple:
    """WHAT: (scale_B, scale_h): the average Bures angle and the average |h - h'| over all ordered pairs of qubits
    of the training inputs (self-pairs included, as in ESL §14.3.3, d_j = (1/N^2) sum_i sum_i' d_j).
    WHY: rescaling so that the state part and the term part count equally (Charbel, 2 Nov 17:39)."""
    R = np.concatenate(blochs, axis=0)
    H = np.concatenate([np.full(len(b), h) for b, h in zip(blochs, hs)])
    sB = float(np.mean(bures_angle(R[:, None, :], R[None, :, :])))
    sh = float(np.mean(np.abs(H[:, None] - H[None, :])))
    if sB <= 0 or sh <= 0:
        raise ValueError("a rescaling divisor is zero")
    return sB, sh


def content_cost(bA: np.ndarray, hA: float, bB: np.ndarray, hB: float, scales: tuple) -> np.ndarray:
    """WHAT: M(k, l) between qubit k of input A and qubit l of input B:
    BETA * Bures(rho_k, rho_l)/scale_B + (1 - BETA) * |h_A - h_B|/scale_h.
    WHY: on periodic TFIM the only term coefficient that differs between inputs is h (every ZZ coefficient is 1)."""
    sB, sh = scales
    return BETA * bures_angle(bA[:, None, :], bB[None, :, :]) / sB + (1 - BETA) * abs(hA - hB) / sh


# ---------------------------------------------------------------- 7. FGW
def fgw(M: np.ndarray, C1: np.ndarray, C2: np.ndarray, alpha: float = ALPHA,
        max_iter: int = 500, tol: float = 1e-14) -> float:
    """WHAT: FGW_{q=1, alpha} = min over couplings P (marginals 1/n1, 1/n2) of
    (1 - alpha) sum_kl M_kl P_kl + alpha sum_{k,l,k',l'} |C1(k,k') - C2(l,l')| P_kl P_k'l'.
    HOW: conditional gradient (Vayer et al. 2019, Alg. 1): gradient G = (1-alpha) M + 2 alpha L P; direction from
    the linear transport problem with cost G (POT ot.emd); exact line search on the quadratic in the step; stop when
    the Frank-Wolfe gap is below tol. Started from the independent coupling, the identity matching and the mirror
    matching (k -> n-1-k); the smallest final value is kept, since only a local optimum is guaranteed.
    WHY: POT's built-in FGW only does q = 2 (square loss); Charbel chose q = 1 (2 Nov 17:27)."""
    n1, n2 = M.shape
    p, q = np.full(n1, 1.0 / n1), np.full(n2, 1.0 / n2)
    L = np.abs(C1[:, None, :, None] - C2[None, :, None, :]).reshape(n1 * n2, n1 * n2)

    def energy(P):
        v = P.ravel()
        return (1 - alpha) * float(np.sum(M * P)) + alpha * float(v @ L @ v)

    starts = [np.outer(p, q)]
    if n1 == n2:
        starts += [np.eye(n1) / n1, np.fliplr(np.eye(n1)) / n1]
    best = math.inf
    for P in starts:
        P = P.copy()
        for _ in range(max_iter):
            G = (1 - alpha) * M + 2 * alpha * (L @ P.ravel()).reshape(n1, n2)
            D = ot.emd(p, q, G) - P
            b = float(np.sum(G * D))                       # derivative of the energy along D at step 0
            if -b <= tol:                                  # Frank-Wolfe gap
                break
            a = alpha * float(D.ravel() @ L @ D.ravel())   # curvature along D
            t = min(1.0, max(0.0, -b / (2 * a))) if a > 0 else 1.0
            P = P + t * D
        best = min(best, energy(P))
    return best


# ---------------------------------------------------------------- 8. distances
def distance_matrix(d: dict) -> dict:
    """WHAT: FGW distances between all pairs of TRAIN inputs of one periodic problem, with what was used.
    HOW: Bloch vectors of every qubit of every training input; the two rescaling divisors; FGW for each pair i < j
    (the matrix is filled symmetrically, zeros on the diagonal)."""
    if d["boundary"] != "periodic":
        raise ValueError("only periodic chains are built (the rule for end qubits is open)")
    idx = np.flatnonzero(d["split"] == "TRAIN")
    n = d["n"]
    blochs = [qubit_states(d["states"][i], n) for i in idx]
    hs, ys = d["h"][idx], d["y"][idx]
    scales = content_scales(blochs, hs)
    C = structure_matrix(n)
    N = len(idx)
    D = np.zeros((N, N))
    for i in range(N):
        for j in range(i + 1, N):
            D[i, j] = D[j, i] = fgw(content_cost(blochs[i], hs[i], blochs[j], hs[j], scales), C, C)
    return {"D": D, "h": hs, "y": ys, "blochs": np.array(blochs), "scales": scales}


def closed_form_matrix(res: dict) -> np.ndarray:
    """WHAT: the known answer on periodic TFIM: FGW = (1 - alpha) * [BETA * (1/2)|arcsin a - arcsin b| / scale_B
    + (1 - BETA) |h - h'| / scale_h], with a, b the <X> of any qubit of each input (2 Nov entry of LOG.md)."""
    a = res["blochs"][:, 0, 0]
    sB, sh = res["scales"]
    bures = 0.5 * np.abs(np.arcsin(a)[:, None] - np.arcsin(a)[None, :])
    m = BETA * bures / sB + (1 - BETA) * np.abs(res["h"][:, None] - res["h"][None, :]) / sh
    return (1 - ALPHA) * m


# ---------------------------------------------------------------- 9. Čech
def cech_values(D: np.ndarray) -> tuple:
    """WHAT: entry values of edges E[a, b] and, per vertex a, of triangles {a, b, c}:
    min over w of max over the vertices v of D(w, v), w over all training inputs."""
    E = np.min(np.maximum(D[:, :, None], D[:, None, :]), axis=0)
    return E


def intrinsic_cech(D: np.ndarray) -> "gudhi.SimplexTree":
    """WHAT: the intrinsic Čech filtration up to triangles, persistence over F2 computed.
    HOW: vertices at 0; edges at E[a, b]; triangles at min_w max(D[w,a], D[w,b], D[w,c]); inserted in that order,
    so every face is present with its own value before its cofaces (gudhi keeps the lower value)."""
    N = len(D)
    st = gudhi.SimplexTree()
    for v in range(N):
        st.insert([v], filtration=0.0)
    E = cech_values(D)
    for a in range(N):
        for b in range(a + 1, N):
            st.insert([a, b], filtration=float(E[a, b]))
    for a in range(N):
        Ma = np.maximum(D[:, a][:, None], D)                       # (w, b): max(D[w,a], D[w,b])
        T = np.min(np.maximum(Ma[:, :, None], D[:, None, :]), axis=0)   # (b, c)
        for b in range(a + 1, N):
            for c in range(b + 1, N):
                st.insert([a, b, c], filtration=float(T[b, c]))
    if st.make_filtration_non_decreasing():
        raise RuntimeError("a face entered after one of its cofaces: the Čech values are inconsistent")
    st.compute_persistence(homology_coeff_field=2, min_persistence=0.0)
    return st


def last_two_groups(D: np.ndarray, y: np.ndarray) -> dict:
    """WHAT: the two groups that remain just before the last merge in dimension 0, compared with the labels.
    HOW: union-find over the edges in order of their Čech value; stop when two groups are left."""
    N = len(D)
    E = cech_values(D)
    parent = list(range(N))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    pairs = sorted(((E[a, b], a, b) for a in range(N) for b in range(a + 1, N)))
    groups = N
    for val, a, b in pairs:
        ra, rb = find(a), find(b)
        if ra != rb:
            if groups == 2:
                roots = np.array([find(v) for v in range(N)])
                g = roots == roots[0]
                agree = float(max(np.sum(y[g] == 1) + np.sum(y[~g] == -1), np.sum(y[g] == -1) + np.sum(y[~g] == 1))) / N
                return {"merge_radius": float(val), "sizes": (int(g.sum()), int((~g).sum())), "label_agreement": agree}
            parent[ra] = rb
            groups -= 1
    return {}


# ---------------------------------------------------------------- checks
def known_answer_checks() -> None:
    """Bures: 0 for identical states, pi/2 for orthogonal pure states, (1/2)|arcsin a - arcsin b| for commuting
    states. Čech: four points on a line at 0, 1, 2, 3 (distance |i - j|): edge {0, 2} enters at 1, edge {0, 3} at 2,
    dimension 0 has three bars [0, 1) and one infinite bar, dimension 1 has none. Raises on any failure."""
    r = np.array([[0.3, -0.2, 0.5], [0.0, 0.0, 0.9], [0.7, 0.1, -0.1]])
    assert np.max(bures_angle(r, r)) <= TOLERANCE, "Bures: identical states"
    for u in ([0, 0, 1], [1, 0, 0], [0, 1, 0]):
        u = np.array(u, float)
        assert abs(bures_angle(u, -u) - math.pi / 2) <= TOLERANCE, "Bures: orthogonal pure states"
    a = np.linspace(-0.99, 0.99, 41)
    A, B = np.meshgrid(a, a)
    ra = np.stack([A, 0 * A, 0 * A], -1)
    rb = np.stack([B, 0 * B, 0 * B], -1)
    err = np.max(np.abs(bures_angle(ra, rb) - 0.5 * np.abs(np.arcsin(A) - np.arcsin(B))))
    assert err <= TOLERANCE, f"Bures: commuting states, worst {err:.1e}"
    x = np.arange(4.0)
    D = np.abs(x[:, None] - x[None, :])
    E = cech_values(D)
    assert E[0, 2] == 1 and E[0, 3] == 2, "Čech: edge values on the line"
    st = intrinsic_cech(D)
    h0 = sorted(map(tuple, st.persistence_intervals_in_dimension(0)))
    h1 = st.persistence_intervals_in_dimension(1)
    assert h0 == [(0.0, 1.0)] * 3 + [(0.0, math.inf)] and len(h1) == 0, f"Čech: barcode {h0}, {h1}"
    print(f"known-answer checks passed (Bures commuting worst {err:.1e})")


def check_problem(res: dict) -> dict:
    """On one periodic problem: <Y> = <Z> = 0, <X> equal on every qubit, FGW = closed form, all within TOLERANCE."""
    bl = res["blochs"]
    yz = float(np.max(np.abs(bl[:, :, 1:])))
    xspread = float(np.max(bl[:, :, 0].max(axis=1) - bl[:, :, 0].min(axis=1)))
    fgw_err = float(np.max(np.abs(res["D"] - closed_form_matrix(res))))
    ok = yz <= TOLERANCE and xspread <= TOLERANCE and fgw_err <= TOLERANCE
    return {"max_|<Y>|,|<Z>|": yz, "max_<X>_spread": xspread, "max_|FGW-closed_form|": fgw_err, "passed": ok}


# ---------------------------------------------------------------- 10. run
def main() -> None:
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    known_answer_checks()
    OUT.mkdir(parents=True, exist_ok=True)
    tasks = [t for t in tfim_task_ids() if describe(t)["boundary"] == "periodic"]
    bars, summary, checks = [], [], []
    for t in tasks:
        t0 = time.time()
        d = load_tfim(t)
        res = distance_matrix(d)
        chk = check_problem(res)
        checks.append({"task_id": t, **chk})
        np.savez(OUT / (t.replace(":", "_") + ".npz"), D=res["D"], h=res["h"], y=res["y"],
                 scales=np.array(res["scales"]))
        if not chk["passed"]:
            print(f"{t}: known-answer check FAILED {chk}; no barcode")
            continue
        st = intrinsic_cech(res["D"])
        info = describe(t)
        for dim in (0, 1):
            for b, e in st.persistence_intervals_in_dimension(dim):
                bars.append({"task_id": t, **info, "dim": dim, "birth": b, "death": e})
        h1 = st.persistence_intervals_in_dimension(1)
        lt = last_two_groups(res["D"], res["y"])
        summary.append({"task_id": t, **info, "N_train": len(res["D"]), "scale_B": res["scales"][0],
                        "scale_h": res["scales"][1], "max_distance": float(res["D"].max()),
                        "last_merge_radius": lt.get("merge_radius"), "last_two_sizes": lt.get("sizes"),
                        "last_two_vs_labels": lt.get("label_agreement"), "n_H1": len(h1),
                        "max_H1_length": float(np.max(h1[:, 1] - h1[:, 0])) if len(h1) else 0.0,
                        "seconds": round(time.time() - t0, 1)})
        print(f"{t}: done in {time.time() - t0:.0f}s, check {chk}")
    pd.DataFrame(checks).to_csv(OUT / "known_answer_checks.csv", index=False)
    pd.DataFrame(bars).to_csv(OUT / "barcodes.csv", index=False)
    S = pd.DataFrame(summary)
    S.to_csv(OUT / "summary.csv", index=False)

    fig, axes = plt.subplots(2, 5, figsize=(20, 7), sharex=False)
    for r, window in enumerate(("far", "near")):
        for c, n in enumerate((4, 6, 8, 10, 12)):
            ax = axes[r, c]
            sel = [b for b in bars if b["window"] == window and b["n"] == n]
            fin = [b for b in sel if math.isfinite(b["death"])]
            top = max([b["death"] for b in fin] + [1e-9]) * 1.1
            rows = sorted(sel, key=lambda b: (b["dim"], -(b["death"] if math.isfinite(b["death"]) else top)))
            for i, b in enumerate(rows):
                e = b["death"] if math.isfinite(b["death"]) else top
                ax.plot([b["birth"], e], [i, i], color="tab:blue" if b["dim"] == 0 else "tab:red", lw=1)
            ax.set_title(f"periodic, n = {n}, {window}")
            ax.set_yticks([])
            ax.set_xlabel("radius (rescaled FGW)")
    fig.suptitle("Layer zero, periodic TFIM: intrinsic Čech barcodes of the training set "
                 "(blue: dimension 0, red: dimension 1; infinite bar cut at the right edge)")
    fig.tight_layout()
    fig.savefig(OUT / "barcodes_periodic.png", dpi=130)
    print(S.to_string())


if __name__ == "__main__":
    main()
