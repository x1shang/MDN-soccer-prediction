# Anchored-prior audit: leave-one-cohort-out transfer matrix

Median Euclidean crossing error at 5 mm (cm), independent synthetic hold-outs
generated from fitted anchor parameters (no hold-out point enters any training run).

| hold-out | all anchors | no Asai | no DFL penalties | no ISSIA | prior only | frozen P1 |
|---|---|---|---|---|---|---|
| canonical | 30.73 | 30.68 | 31.05 | 31.14 | 29.11 | 28.99 |
| asai | 27.13 | 47.30 | 25.58 | 25.37 | 40.93 | 41.87 |
| dfl_penalty | 15.14 | 13.90 | 30.60 | 13.37 | 24.01 | 24.29 |
| dfl_extra | 30.34 | 29.91 | 30.35 | 30.74 | 31.01 | 30.68 |
| issia | 27.79 | 26.39 | 26.49 | 29.60 | 27.78 | 27.95 |
| barber | 26.88 | 25.04 | 24.34 | 27.03 | 30.64 | 30.48 |

50% HDR coverage on the same hold-outs (fraction):

| hold-out | all anchors | no Asai | no DFL penalties | no ISSIA | prior only | frozen P1 |
|---|---|---|---|---|---|---|
| canonical | 0.418 | 0.410 | 0.406 | 0.405 | 0.484 | 0.469 |
| asai | 0.630 | 0.159 | 0.616 | 0.630 | 0.291 | 0.274 |
| dfl_penalty | 0.642 | 0.622 | 0.314 | 0.627 | 0.492 | 0.484 |
| dfl_extra | 0.364 | 0.409 | 0.332 | 0.400 | 0.344 | 0.349 |
| issia | 0.535 | 0.575 | 0.573 | 0.444 | 0.465 | 0.475 |
| barber | 0.591 | 0.640 | 0.648 | 0.596 | 0.380 | 0.372 |

Real Asai replay (retrospective for models whose training prior contains Asai anchors):

| case | all anchors | no Asai | no DFL penalties | no ISSIA | prior only | frozen P1 |
|---|---|---|---|---|---|---|
| Fig3 | 24.04 | 59.17 | 22.26 | 22.58 | 52.26 | 54.49 |
| Sample05 | 29.89 | 57.58 | 24.46 | 29.66 | 46.47 | 49.87 |
| Sample12 | 30.87 | 69.08 | 25.18 | 30.19 | 60.57 | 59.57 |

| model | dataset excluded cohorts | train n | selected epoch | train s |
|---|---|---:|---:|---:|
| all anchors | none | 33810 | 93 | 98 |
| no Asai | asai | 33150 | 100 | 95 |
| no DFL penalties | dfl_penalty | 34185 | 87 | 96 |
| no ISSIA | issia | 33635 | 93 | 94 |
| prior only | asai, dfl_penalty, dfl_extra, issia, barber | 33356 | 97 | 104 |
| frozen P1 | -- | 33,367 (canonical P1) | 150 | -- |

The numeric matrix is stored in `eval_matrix.csv`. Replay rows on the three real trials are retrospective for every model whose fitted prior contains those trials.

## Leave-one-trial-out supplement

See [LOTO.md](LOTO.md): excluding one Asai trial from the fitted prior and replaying it
gives 28.9/38.0/33.3 cm (frozen P1: 54.5/49.9/59.6 cm), i.e. same-source transfer works
for an unseen trial, while cross-source transfer (no-Asai model) does not.
