"""run_analysis.py — runs the whole study in the right order.

WHAT THIS FILE DOES
A command-line entry point, run from COURSEWORK/TME4/:
    python run_analysis.py                     # everything, full permutation counts
    python run_analysis.py --only q2 q3        # selected steps
    python run_analysis.py --quick             # reduced permutations, outputs in results/quick/
    python run_analysis.py --jobs 4            # parallel permutations (number of processor cores)
    python run_analysis.py --rebuild           # recompute the feature tables even if they exist
Order (each step needs the outputs of the previous ones):
  0. analysis_common.import_harvest(); check data_final exists and has been audited
     (audit/data_final/TME4__data_final.json with 0 FAIL); stop otherwise.
  1. Q1  q1_describe.run_q1()
  2. features.build_feature_table() and build_feature_table(exclude_flagged=True)
     (skipped if both tables exist, unless --rebuild)
  3. forced_table.pipeline_check(): must be empty; stop and report if not
  4. Q2  q2_types.run_q2()
  5. Q3  q3_classify.run_q3()
  6. Q4  q4_scaling.run_q4()
  (Q6, robustness, was terminated by Charbel on 11 Oct 2026: no time before the deadline.)
--quick: Q2 200 permutations instead of 5 000, Q3 100 instead of 1 000, outputs written under
results/quick/ so they are never mistaken for the real ones.
COST of a full run (measured 11 Oct on a 2-core machine): Q2's 5 000 permutations x 12 features each refit
two mixed models, which takes hours on one core: use --jobs with every core you have, or --quick first.
Every step prints its duration; the run writes results/RUN_LOG.md (or results/quick/RUN_LOG.md): date,
quick or full, package versions, each step's duration.
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime

import analysis_common as ac

STEPS = ("q1", "features", "check", "q2", "q3", "q4")


def check_ready() -> None:
    """WHAT: step 0: import the harvest package, check data_final and its audit report exist and that the
    audit has 0 FAIL; raise with a clear message otherwise."""
    ac.import_harvest()
    p = ac.paths()
    if not (p.data_final / "harvest.sqlite").exists():
        raise FileNotFoundError(f"{p.data_final} has no harvest.sqlite: build data_final first")
    if not p.audit.exists():
        raise FileNotFoundError(f"{p.audit} missing: run code/scripts/audit_dataset.py on data_final first")
    report = json.loads(p.audit.read_text())
    if report["fail"]:
        raise RuntimeError(f"the audit of data_final has {len(report['fail'])} FAIL: fix them first")


def write_run_log(durations: dict, quick: bool, folder) -> None:
    """WHAT: RUN_LOG.md with the date, quick or full, the versions of numpy, pandas, statsmodels,
    scikit-learn, scipy, matplotlib, and each step's duration in seconds."""
    import importlib
    versions = {}
    for name in ("numpy", "pandas", "statsmodels", "sklearn", "scipy", "matplotlib"):
        try:
            versions[name] = importlib.import_module(name).__version__
        except ImportError:
            versions[name] = "missing"
    lines = [f"# Run log", "", f"- date: {datetime.now().isoformat(timespec='seconds')}",
             f"- mode: {'quick' if quick else 'full'}", "- versions: " + ", ".join(f"{k} {v}" for k, v in versions.items()),
             "", "| step | seconds |", "|---|---|", *[f"| {k} | {v:.0f} |" for k, v in durations.items()]]
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "RUN_LOG.md").write_text("\n".join(lines) + "\n")


def main(argv: list = None) -> None:
    """WHAT: parse --only / --quick / --jobs / --rebuild (argparse) and run the steps in order, timing each."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", nargs="*", choices=STEPS, default=list(STEPS))
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--rebuild", action="store_true")
    args = parser.parse_args(argv)
    folder = ac.paths().results / "quick" if args.quick else None
    q2_perm, q3_perm = (200, 100) if args.quick else (5000, 1000)
    durations = {}

    def step(name, function):
        if name not in args.only:
            return
        start = time.time()
        print(f"--- {name}", flush=True)
        function()
        durations[name] = time.time() - start
        print(f"--- {name} done in {durations[name]:.0f} s", flush=True)

    check_ready()

    def features_step():
        import features
        results = ac.paths().results
        for noflags, file in ((False, "features.csv"), (True, "features_noflags.csv")):
            if args.rebuild or not (results / file).exists():
                features.build_feature_table(exclude_flagged=noflags)

    def check_step():
        import q2_types
        import forced_table
        bad = forced_table.pipeline_check(q2_types.load_features())
        if len(bad):
            raise RuntimeError(f"pipeline check: {len(bad)} forced values differ from their derivation")
        print("pipeline check: every forced value matches its derivation", flush=True)

    def q1_step():
        import q1_describe
        q1_describe.run_q1()

    def q2_step():
        import q2_types
        q2_types.run_q2(n_permutations=q2_perm, n_jobs=args.jobs, folder=folder)

    def q3_step():
        import q3_classify
        q3_classify.run_q3(n_permutations=q3_perm, n_jobs=args.jobs, folder=folder)

    def q4_step():
        import q4_scaling
        q4_scaling.run_q4(folder=folder)

    for name, function in (("q1", q1_step), ("features", features_step), ("check", check_step), ("q2", q2_step),
                           ("q3", q3_step), ("q4", q4_step)):
        step(name, function)
    write_run_log(durations, args.quick, folder or ac.paths().results)


if __name__ == "__main__":
    main()
