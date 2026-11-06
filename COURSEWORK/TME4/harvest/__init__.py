"""The harvest package: turns learning problems into stored problem records 𝒫ₙ.

Read the files in this order (each one only uses the ones above it):

    config.py           the numbers that decide what is kept (thresholds, sizes)
    records.py          the objects the harvest produces (Instance, Operator, Equation, HarvestedProblem, ...)
    pauli.py            Pauli strings and Pauli sums: how Hamiltonians are written down
    operators.py        operator_of: the operator M(x) of one input x (Laplacian, precision, Hamiltonian, clause penalty)
    checks.py           equation_check: does M(x) satisfy what its equation requires?
    dedupe.py           how many *different* inputs a problem has, and whether equal inputs carry different labels
    labels.py           turning labels into {-1, +1} and balancing the two classes
    split.py            the fixed TRAIN / TEST split
    screen.py           the screening rules S1 to S7 (keep or reject, with a reason)
    task_operator.py    M̄: the one operator of the whole task, built from TRAIN only
    near_duplicates.py  merging two problems that are really the same problem
    store.py            writing problems to disk and to the SQLite database; the harvest log
    pipeline.py         build_problem and run_harvest: the steps above, in order
    sources/            where problems come from (generators that run offline, adapters to public datasets)

Design rule for this first version: readability first, speed never. Every function says
WHAT it does, HOW, and WHY it exists. Speed is a later job.
"""
