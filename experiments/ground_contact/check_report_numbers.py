"""Diff the markdown tables in 回传结果.md against paired_test_scores.csv."""
import csv
import re
from pathlib import Path

R = Path(__file__).resolve().parent
md = (R / "回传结果.md").read_text(encoding="utf-8")

rows = list(csv.DictReader((R / "results/paired_test_scores.csv").open(encoding="utf-8-sig")))
d = {(r["model"], r["subset"], int(r["sigma_mm"])): r for r in rows}

FIELDS = ["median_m", "p90_m", "nll", "cov50", "cov95", "area50_m2",
          "projected_median_m", "sample_below_ground_fraction"]
SUBSETS = {"3.1": "airborne", "3.2": "post_contact", "3.3": "combined"}

errors = []
checked = 0
for sec, subset in SUBSETS.items():
    start = md.index(f"### {sec}")
    end = md.index("###", start + 5) if md.find("###", start + 5) != -1 else len(md)
    block = md[start:end]
    for line in block.splitlines():
        if not line.startswith("| ") or line.startswith("| σ") or set(line) <= set("|-: "):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) != 9:
            errors.append(f"{sec}: unexpected column count {len(cells)}: {line[:60]}")
            continue
        sigma = int(cells[0])
        for j, field in enumerate(FIELDS):
            a, b = [float(x.replace("\u2212", "-").replace("\u2013", "-").replace("*", "").strip())
                    for x in cells[1 + j].split("/")]
            va = float(d[("airborne", subset, sigma)][field])
            vb = float(d[("bounce", subset, sigma)][field])
            checked += 1
            if abs(a - round(va, 3)) > 1.01e-3 or abs(b - round(vb, 3)) > 1.01e-3:
                errors.append(f"{sec} sigma={sigma} {field}: md={a}/{b} csv={va:.4f}/{vb:.4f}")

# Case study table against the JSON files
import json
case = {arm: json.loads((R / f"results/{arm}/case_predictions.json").read_text(encoding="utf-8"))
        for arm in ("airborne", "bounce")}
start = md.index("## 6.")
end = md.index("## 7.")
for line in md[start:end].splitlines():
    if line.startswith("| ") and "sigma" not in line and not set(line) <= set("|-: "):
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) < 5 or not cells[0].isdigit():
            continue
        s = int(cells[0])
        key = f"sigma_{s}mm"
        got = [float(x) for x in cells[1].split("/")]
        for j, arm in enumerate(("airborne", "bounce")):
            checked += 1
            want = case[arm][key]["goal_probability_given_crossing"]
            if abs(got[j] - round(want, 4)) > 1.01e-4:
                errors.append(f"case sigma={s} {arm} P(goal): md={got[j]} json={want:.5f}")

print(f"checked {checked} numbers; {len(errors)} mismatches")
for e in errors:
    print("  ", e)
