# Legacy artefacts from before the v3 million-scale track

These files were carried over from an earlier repository snapshot so that nothing is lost, but
**no number in the current paper comes from them**. They belong to the 50,000-launch canonical
simulation that preceded the frozen v3 mixture, and the manuscript replaced those tables when it
moved to 1,225,000 launches per sampling strategy.

| Path | What it is |
| --- | --- |
| `p1/` | The frozen P1 network: `best_config.json`, `protocol.json`, three-seed summary and robustness tables, paired Bayesian comparison, the seed-11 checkpoint (`phys_reduced_k3_11.pt`) and `plot_case.py`, the script that drew the P1 case figure |
| `v1_fresh/` | The later "v1 fresh" retrain: config, model, learning curve, test scores, Asai replay and its data manifest |
| `source_holdout/` | A leave-one-Asai-trial-out audit (`LOTO.md`) with its evaluation matrix |

`plot_case.py` needs `--output`; it imports the current canonical physics and MDN builder from
`code/model/` and writes into the directory you pass it.

Where these numbers still appear, they do so only as history:

- the paper's model comparison table keeps one “legacy coordinate MDN” row, retrained on v3 rows;
- `paper/archive/main_P1_archive.tex` is the earlier draft written against these results;
- `paper/archive/figures/case_regions_p1.png` and `coverage_fresh_p1.png` are the P1-era
  versions of the case and calibration figures, superseded by `paper/figures/*_v3.png`.

Kept because the earlier snapshot was a teammate's distribution of the work; the simulator,
model and diagnostics in `code/` and the runs in `experiments/main_v3_all/` are the current
source of truth. The commit that introduced these files is preserved in this repository's
history.
