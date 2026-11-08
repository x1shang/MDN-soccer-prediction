# Data

Two kinds of data live here:

* **Anchors** (`anchors/`) — the measured trajectories that seed the simulation prior. Small, committed.
* **Simulated datasets** (`datasets/`) — the million-scale launch banks. Large, published as GitHub
  Release assets; this directory only carries the manifests and the split indices.

## 1. Anchors committed to the repository

| Path | Content | Used for |
| --- | --- | --- |
| `anchors/asai/*.xlsx` | Three Asai et al. (2020) tracked free kicks at 100 Hz: `Fig3`, `Sample 05`, `Sample 12` | Asai prior anchors and the source holdout |
| `anchors/dfl/flight_trajectory_validation_2026-11-07/` | DFL penalties `J03WOH` / `J03WR9`: input and held-out positions, case config, source audit | DFL prior anchors, Table "Prefix Forecasts" |
| `anchors/dfl/additional_flight_data_2026-11-07/` | Eleven screened DFL flight segments (`positions.csv`), screening audit | Aerodynamic prior anchors |
| `anchors/issia/seg_0608_0627.csv`, `anchors/issia/seg_1144_1172.csv` | ISSIA-3D arcs, 25 fps, 3D positions | ISSIA prior anchors, Table "Model-Form Check" |
| `anchors/issia/ISSIA-3D.csv` | Raw ISSIA-3D export | Provenance for the two arcs |
| `manifests/frozen_design.json` | Frozen design of the v3 simulation (seeds, cohort shares, aiming rule) | Regenerating datasets |

The SHA-256 of every anchor file is recorded inside each dataset's `data_manifest.json`
(`anchors[].sources`), so provenance can be checked without the original workbooks.
Upstream licences and the role each source plays in the prior are in [`SOURCES.md`](SOURCES.md).

## 2. Simulated datasets (Release assets)

Five `.npz` files are published as assets of the
**[`data-v1.0`](https://github.com/x1shang/MDN-soccer-prediction/releases/tag/data-v1.0)** release.
They are not stored in Git because GitHub rejects files above 100 MB and the three main banks are
~140 MB each.

| Release asset | Rows (`n`) | Bytes | SHA-256 | Lands at |
| --- | ---: | ---: | --- | --- |
| `v3_all_1.225M_data.npz` | 1,225,000 | 140.65 MB | `8774a9fa746bcd1e326cff606fbcdf62fad8517f8038d1ef48299c7c44a9eba7` | `data/datasets/v3_all/data.npz` |
| `v2_no_asai_1.225M_data.npz` | 1,225,000 | 139.17 MB | `e9b2662f85010383538d2fe2b0148e898b6dbf433bdb315d6f2d107310b15a16` | `data/datasets/v2_no_asai/data.npz` |
| `v3_no_asai_1.225M_data.npz` | 1,225,000 | 140.41 MB | `93033ad5276b2b1f433b8ad03be00978481a0a261027acc093981bfa23c734c7` | `data/datasets/v3_no_asai/data.npz` |
| `v3_all_100k_data.npz` | 100,000 | 11.48 MB | `e16935d3d8cb004042e11cfd0669ba7b5ac853f4df8dc6936a98c008315b0b9f` | `experiments/ground_contact/data.npz` |
| `bounce_100k_data.npz` | 100,000 | 11.99 MB | `46cf86a83d6749079f343a2b52440dc8cea7de0ee3c317f0ed7e6384c0fd5e10` | `experiments/ground_contact/bounce/data_bounce.npz` |

Download everything into place with:

```powershell
powershell -ExecutionPolicy Bypass -File code\scripts\fetch_data.ps1
```

or manually:

```powershell
gh release download data-v1.0 --repo x1shang/MDN-soccer-prediction --dir _tmp
# then move each asset to the path listed above
```

`fetch_data.ps1` also verifies the SHA-256 of every file it places.

A row-level split is stored next to each manifest as `indices.npz`
(`train` / `selection` / `calibration` / `test`), so a dataset and the artefacts trained on it stay
matched: `data_manifest.json` records `sha256` (data) and `indices_sha256` (split), and every
`experiments/.../seed_*/config.json` repeats both.

## 3. Dataset contents

Each `data.npz` holds the launch parameters and the simulated outcome per row:

| Key | Shape | Meaning |
| --- | --- | --- |
| `theta` | `(n, 7)` | launch parameters `(vx, vy, vz, wx, wy, wz, C_D)` |
| `anchor` | `(n,)` | source anchor index, `-1` for independent uniform draws |
| `split` | `(n,)` | split code per row; cross-checked against `indices.npz` |
| `observations` | `(n, 2, 3)` | observed positions at 0.05 s and 0.10 s |
| `endpoint` | `(n, 2)` | `(Y, Z)` where the ball centre crosses `x = 11 m` |
| `arrival` | `(n,)` | crossing time in seconds |
| `status` | `(n,)` | `1` airborne crossing, `2` ground contact first, `0` neither within the horizon |

Only `status == 1` rows are kept for training, calibration and scoring; the writer asserts the
endpoint is above `z = R` and the arrival time is later than both observations.

`indices.npz` holds four `int64` arrays named after the splits. Split counts are in
`data_manifest.json["index_counts"]`.

## 4. Regenerating from scratch

The simulator is deterministic given the frozen design and prior:

```powershell
python code\simulator\generate.py out --n 1225000 --arm all
```

`--n` must be a multiple of 2500. Generation was validated by re-integration: 1,024 sampled launches
per dataset at a 1 ms step (versus the production 2 ms) keep the crossing status unchanged and move
endpoints by at most ~0.012 mm.
