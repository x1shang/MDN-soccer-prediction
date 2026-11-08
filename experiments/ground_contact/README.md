# Ground-contact control experiment

Self-contained reproduction package for the paired comparison in the paper's
“Rebound and Rolling Crossings” section. It keeps its own simulator copy, training script and
result folders so the numbers can be re-run without touching the million-scale track.

## What is here

| Path | Content |
| --- | --- |
| `train_bounce_control_mdn.py` | Wiener-residual MDN with `--arm airborne` / `--arm bounce` |
| `simulation/` | contact-aware simulator (`ground_contact.py`), v3 generator, prior |
| `bounce/` | manifest, sensitivity sweep, case audit, bounce dataset (download) |
| `results/` | three fits (`airborne`, `bounce`, `airborne_e200`), paired scores, bootstrap |
| `verify_package.py` | re-checks hashes, counts and split agreement |
| `make_report.py`, `check_report_*.py` | regenerate and audit `results/report_tables.md` |
| `回传结果.md` | the team's Chinese write-up of this run (desktop reproduction) |

## Data

Two arrays come from the `data-v1.0` release (see `data/README.md`):

| File | Release asset | SHA-256 |
| --- | --- | --- |
| `data.npz` | `v3_all_100k_data.npz` | `e16935d3…` |
| `bounce/data_bounce.npz` | `bounce_100k_data.npz` | `46cf86a8…` |

```powershell
powershell -ExecutionPolicy Bypass -File ..\..\code\scripts\fetch_data.ps1
```

## Re-run

```powershell
cd experiments\ground_contact
python -m venv .venv
.\.venv\Scripts\pip install -r ..\..\requirements.txt
.\run_all.ps1            # both arms at 100 epochs, then merges the paired scores
python verify_package.py # hash and count audit -> results/verify_package.json
python make_report.py    # -> results/report_tables.md
```

`data.npz` holds 100,000 launches: 88,999 airborne crossings plus 11,001 additional crossings
found after a rebound (6,636) or while rolling (4,365). The contact-inclusive arm trains on the
same rows plus 8,798 post-contact rows; both arms are scored on the identical 10,000-row test
split.
