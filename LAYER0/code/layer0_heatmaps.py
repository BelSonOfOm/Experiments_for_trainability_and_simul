"""layer0_heatmaps.py — pictures of layer zero's raw material: F_ij = |<psi_i|psi_j>|^2 as a heat map.

WHAT: for the periodic TFIM, windows far and near, n = 4, 8, 12: the 160 x 160 table F with the inputs
sorted by h (small h at the top-left), on a common colour scale from 0 (orthogonal) to 1 (identical);
a dashed line marks h = 1, where the label changes. Uses load_tfim and fidelity_matrix of
layer0_concentration.py (checked by the known-answer test). Requested by Charbel, 1 Nov 2026, 22:00.
"""
import numpy as np
import layer0_concentration as L


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    for r, window in enumerate(("far", "near")):
        for c, n in enumerate((4, 8, 12)):
            d = L.load_tfim(f"spin-generator-2:TFIM-periodic:n={n}:h:{window}")
            order = np.argsort(d["h"])
            F = L.fidelity_matrix(d["states"][order])
            h = d["h"][order]
            ax = axes[r, c]
            im = ax.imshow(F, vmin=0, vmax=1, cmap="viridis", origin="upper")
            cut = np.searchsorted(h, 1.0) - 0.5
            ax.axhline(cut, color="white", ls="--", lw=0.8)
            ax.axvline(cut, color="white", ls="--", lw=0.8)
            ticks = [0, 79, 80, 159]
            ax.set_xticks(ticks, [f"{h[t]:.2f}" for t in ticks], fontsize=7)
            ax.set_yticks(ticks, [f"{h[t]:.2f}" for t in ticks], fontsize=7)
            ax.set_title(f"periodic, {window}, n = {n}", fontsize=10)
    fig.colorbar(im, ax=axes, shrink=0.8, label="F = |<psi_i|psi_j>|^2")
    fig.suptitle("Layer zero, raw material: overlaps between inputs, sorted by h (dashed line: h = 1)")
    return L.ac.save_figure(fig, "layer0_heatmaps_periodic", L.RESULTS / "figures")


if __name__ == "__main__":
    print(main())
