# Next steps

1. **Complete the statistics of this study.**
   - Run the permutation tests: `python run_analysis.py --only q2 q3 --jobs <cores>`. Q2 needs about 3 h on one core.
   - Check the post-hoc size-control test against the full paper of Dinga et al. (2020).
2. **Settle the open choices and re-run** (minutes):
   - COVARIANCE site weights;
   - classifier penalty;
   - correlation threshold for blocks;
   - minimum sizes per line;
   - transformations of skewed features;
   - central charges for J1-J2, ANNNI, cluster-Ising and XY (to look up and cite).
3. **Decide how a problem's inputs are summarised** (added 17 Oct).
   - The analysis summarises each feature over a problem's inputs by the median; this is now under review.
   - Plot each problem's per-input distributions and check whether they form one group or several, which a single summary would hide; then choose the summary.
4. **Richer structure: a first instance of the Logical State Space L = (C; F; Φ)** (framing of 17 Oct).
   - The aim: one common medium for all problems in which the correlation between structure and outcome can be explained in mathematical terms, and from which circuits are generated with each of the four structures as a separate ingredient, so that circuits built with and without each one measure its effect.
   - C: build each problem as a simplicial complex from its factor graph (each term's sites, closed under subsets); F: the constant sheaf; Φ: diffusion under the Hodge Laplacians.
   - Read its Hodge Laplacians and Betti numbers.
   - Test whether this recovers what the factor graph missed (Q3b). For example, the three-body term of XOR-3 becomes a filled triangle, while three pairwise terms leave a hollow one.
   - Quantum algorithms for these quantities come with both hardness and dequantization results (Lloyd, Garnerone & Zanardi 2016; Gyurik, Cade & Dunjko 2022; Berry et al. 2024).
5. **Build circuits from the extracted structure** (step 3 of the chain).
6. **Simulate them and measure:**
   - trainability, through gradient variance and the dimension of the dynamical Lie algebra (Ragone et al. 2024);
   - classical simulability (Cerezo et al. 2023).
7. **Learn the predictor** from structure to those measurements, validated by holding out whole sources (the lesson of Q2 and Q3).
8. **Extend n** where possible, so that scaling laws can be named rather than left undetermined.
