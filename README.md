# Experiments for trainability and simulability

Charbel A. Dargham, M1 Quantum Information, Sorbonne Université (DALAS course, autumn 2026).

The question behind this work is whether the underlying structure of a learning problem (its symmetry, locality, spectrum and data complexity) can predict, before any training, whether a structure-preserving quantum model of it will be trainable and will resist dequantization. The repository holds the code of two stages of that work.

## `COURSEWORK/TME4/`: the dataset and its first analysis

The analysis code of the TME4 deliverable (October 2026): a dataset of 1 368 binary learning problems in four types (GRAPH, COVARIANCE, DIAGONAL, QUANTUM), built from 58 independent sources, and four study questions on whether the structure extracted from them is correct, varied and lawful in the problem size. Its own `README.md` describes every file. The report, the slides, every result table and figure, and the dataset index live in the submission repository, [Bel-The-Wizard/TME4](https://github.com/Bel-The-Wizard/TME4); only the code is copied here. The problem folders themselves (7.5 GB) are in neither repository and are available from the author.

## `LAYER0/`: layer zero of the Logical State Space on these problems

Started after the TME4 submission, this stage applies layer zero of the [Logical State Space](https://github.com/BelSonOfOm/Logical-State-Space) to the learning problems, beginning with the transverse-field Ising (TFIM) chains. Layer zero is how an algorithm's inputs are related to one another before the algorithm does anything; for a learning problem the inputs are its samples, here the ground states of the chain at different values of the field h.

`code/layer0_concentration.py` computes the fidelity matrix F_ij = |⟨ψ_i|ψ_j⟩|² of each problem's input states and checks it against the closed form of the ground-state fidelity for periodic chains of even length (Damski & Rams, arXiv:1308.5917): on all ten periodic problems, 25 600 pairs each, the largest difference is 1.3 × 10⁻¹⁴, against a tolerance of 10⁻⁹. It then measures whether the off-diagonal entries concentrate as the chain grows, the diagnostic of Thanasilp et al. (Nat. Commun. 15, 2024, arXiv:2208.11060), choosing the decay law by the BIC rule with a margin of 10 (Kass & Raftery 1995). None of the four lines (open or periodic, far or near window) is chosen as exponential: the far window follows a + b log n, the near window of open chains a power law, and the near window of periodic chains stays undetermined on n = 4 to 12. `code/layer0_heatmaps.py` draws the fidelity matrices.

`code/layer0_build.py` builds layer zero itself. Each input is a ring of qubits whose content is the Bures angle between their reduced states (Hübner 1992) and the field h; two inputs are compared by the Fused Gromov–Wasserstein distance (Vayer et al., ICML 2019, arXiv:1805.09114), and the distances give an intrinsic Čech filtration (Chazal, de Silva & Oudot 2014) whose barcodes are computed with gudhi. Every distance matrix is checked against its closed form before any barcode is computed. On all ten periodic problems the last two clusters to merge are exactly the two label classes; since the sampling leaves a gap in h around the transition at h = 1, this is a check of the construction rather than a finding.

`code/expA_circuits.py` is the circuit side of the first experiment: the TFIM circuit built from the problem's two generators, its depth set from the dimension of their dynamical Lie algebra (computed by closure in Pauli strings, 3n − 1 for n = 4 to 12), training by Adam with exact adjoint gradients, and the variance of the gradient and of the output over 500 random starts (McClean et al. 2018; Ragone et al. 2024). It runs exactly in the subspace of states invariant under shift, mirror and global flip. The script is written and checked but has not been run yet; it writes to `results/expA_circuits/`. Its first check compares against a dense Lie-algebra script, `code/tools/dla_check.py`, that is not part of this repository.

`results/` holds the tables, figures, distance matrices and barcodes the first three scripts produced.

## Running

The scripts need Python 3 with numpy, pandas, matplotlib, POT (`ot`) and gudhi. They read the dataset from `COURSEWORK/TME4/data_final/`: its index, `harvest.sqlite`, is in [Bel-The-Wizard/TME4](https://github.com/Bel-The-Wizard/TME4), and the problem folders must be obtained from the author. The working logs and private notes of the project are not included.
