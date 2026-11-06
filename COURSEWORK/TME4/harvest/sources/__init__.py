"""Where problems come from.

Tested and runnable offline (no internet needed):
    sat_generator.SatGenerator           type DIAGONAL: random 3-SAT and SR(n) formulas, labels by brute force
    spin_generator.SpinChainGenerator    type QUANTUM: ground states of 4 spin-chain models by exact diagonalisation
    tudataset.TUDatasetSource            type GRAPH: reads TUDataset files already on disk
                                         (downloading needs a machine that can reach chrsmrrs.com)

Written but NOT tested (their hosts are blocked from Claude's sandbox; run them on a machine
with internet and expect small fixes):
    moabb_eeg.MoabbSource                type COVARIANCE: EEG trials from MOABB
    timeseries_archive.TimeSeriesArchiveSource   type COVARIANCE: UEA / Multiverse archives via aeon
    pennylane_qspin.QspinSource          type QUANTUM: PennyLane qspin datasets
"""
