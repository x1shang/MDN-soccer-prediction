"""Check the §3.4 mode/goal table and the §5 epoch-control table against the CSVs."""
import csv
import json
import re
from pathlib import Path

R = Path(__file__).resolve().parent
md = (R / "回传结果.md").read_text(encoding="utf-8")
rows = list(csv.DictReader((R / "results/paired_test_scores.csv").open(encoding="utf-8-sig")))
d = {(r["model"], r["subset"], int(r["sigma_mm"])): r for r in rows}
d200 = {(r["subset"], int(r["sigma_mm"])): r
        for r in csv.DictReader((R / "results/airborne_e200/test_scores.csv").open(encoding="utf-8-sig"))}

errs, checked = [], 0
num = r"[-+\u2212]?\d+\.\d+"


def f(x):
    return float(x.replace("\u2212", "-").replace("*", "").strip())


# §3.4 -------------------------------------------------------------
s = md.index("### 3.4")
e = md.index("## 4.")
COLS = [("mode_median_m", 3), ("mode_p90_m", 3), ("goal_brier", 4), ("goal_ece", 4)]
for line in md[s:e].splitlines():
    if not line.startswith("| ") or set(line) <= set("|-: "):
        continue
    cells = [c.strip() for c in line.strip().strip("|").split("|")]
    if len(cells) != 6 or not cells[1].startswith("2 /"):
        continue
    subset = "airborne" if cells[0] == "直接过门" else "post_contact"
    for ci, (field, nd) in enumerate(COLS):
        pairs = [p.strip().split("→") for p in cells[2 + ci].split("/")]
        if len(pairs) != 4:
            errs.append(f"3.4 {cells[0]} {field}: expected 4 pairs, got {len(pairs)}")
            continue
        for k, (a, b) in enumerate(pairs):
            sig = 2 if k == 0 else (5 if k == 1 else (10 if k == 2 else 15))
            for arm, val in (("airborne", a), ("bounce", b)):
                checked += 1
                want = round(float(d[(arm, subset, sig)][field]), nd)
                if abs(f(val) - want) > 1.01 * 10 ** -nd:
                    errs.append(f"3.4 {subset} sigma={sig} {field} {arm}: md={f(val)} csv={want}")

# §5 ---------------------------------------------------------------
s = md.index("## 5.")
e = md.index("## 6.")
row_map = {"空中组 100 ep（正式）": ("airborne", None),
           "空中组 200 ep（对照）": (None, "e200"),
           "触地组 100 ep（正式）": ("bounce", None)}
for line in md[s:e].splitlines():
    if not line.startswith("| ") or set(line) <= set("|-: "):
        continue
    cells = [c.strip() for c in line.strip().strip("|").split("|")]
    if len(cells) != 5 or cells[0] not in row_map:
        continue
    arm, ctrl = row_map[cells[0]]
    for ci, cell in enumerate(cells[1:]):
        sig = (2, 5, 10, 15)[ci]
        med, nll, cov = [f(x) for x in re.findall(num, cell)]
        ref = d[(arm, "post_contact", sig)] if ctrl is None else d200[("post_contact", sig)]
        checked += 3
        for got, field, nd in ((med, "median_m", 3), (nll, "nll", 2), (cov, "cov95", 3)):
            want = round(float(ref[field]), nd)
            if abs(got - want) > 1.01 * 10 ** -nd:
                errs.append(f"5 {cells[0]} sigma={sig} {field}: md={got} csv={want}")

print(f"checked {checked} numbers; {len(errs)} mismatches")
for x in errs:
    print("  ", x)
