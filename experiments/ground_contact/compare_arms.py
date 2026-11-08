import csv
import json
from pathlib import Path

R = Path("results")
ARMS = ["airborne", "airborne_e200", "bounce"]


def load(a):
    rows = csv.DictReader((R / a / "test_scores.csv").open(encoding="utf-8-sig"))
    return {(r["subset"], int(r["sigma_mm"])): r for r in rows}


d = {a: load(a) for a in ARMS}
HEAD = f"{'median':>7}{'p90':>7}{'NLL':>9}{'cov50':>7}{'cov95':>7}{'below':>7}"

for subset in ("post_contact", "airborne", "combined"):
    print(f"\n### {subset}")
    print(f"{'sigma':>5} | " + " | ".join(f"{a:>37}" for a in ARMS))
    print(f"{'':>5} | " + " | ".join(f"{HEAD:>37}" for a in ARMS))
    for s in (2, 5, 10, 15):
        cells = []
        for a in ARMS:
            r = d[a][(subset, s)]
            cells.append(
                f"{float(r['median_m']):7.3f}{float(r['p90_m']):7.3f}{float(r['nll']):9.3f}"
                f"{float(r['cov50']):7.3f}{float(r['cov95']):7.3f}"
                f"{float(r['sample_below_ground_fraction']):7.3f}")
        print(f"{s:5d} | " + " | ".join(f"{c:>37}" for c in cells))

print("\n### run metadata")
for a in ARMS:
    c = json.loads((R / a / "config.json").read_text(encoding="utf-8"))
    print(f"{a:14s} selected_epoch={c['selected_epoch']:3d}  seconds={c['seconds']:6.1f}  "
          f"params={c['parameters']}  sizes={c['sizes']}")
