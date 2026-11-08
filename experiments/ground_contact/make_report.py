"""Turn the two arms' test_scores.csv into report-ready markdown tables."""
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RES = ROOT / "results"

rows = list(csv.DictReader((RES / "paired_test_scores.csv").open(encoding="utf-8-sig")))
for r in rows:
    for k, v in list(r.items()):
        if k in ("model", "subset"):
            continue
        r[k] = float(v)
        if k in ("n",):
            r[k] = int(v)

ORDER = ("airborne", "bounce")
SUBSETS = ("combined", "airborne", "post_contact")


def get(model, subset, sigma):
    for r in rows:
        if r["model"] == model and r["subset"] == subset and int(r["sigma_mm"]) == sigma:
            return r
    raise KeyError((model, subset, sigma))


def fmt(v, nd=3):
    return f"{v:.{nd}f}"


lines = []
for subset in SUBSETS:
    lines.append(f"\n### {subset}（n={get('airborne', subset, 2)['n']}）\n")
    head = ("| σ (mm) | 中位误差 m (空中/触地) | P90 m (空中/触地) | NLL (空中/触地) "
            "| 50% 覆盖 (空中/触地) | 95% 覆盖 (空中/触地) | 面积50 m² (空中/触地) "
            "| 投影中位 m (空中/触地) | 地下采样 (空中/触地) |")
    lines.append(head)
    lines.append("|" + "---|" * 10)
    for s in (2, 5, 10, 15):
        a, b = get("airborne", subset, s), get("bounce", subset, s)
        lines.append(
            f"| {s} | {fmt(a['median_m'])} / {fmt(b['median_m'])} "
            f"| {fmt(a['p90_m'])} / {fmt(b['p90_m'])} "
            f"| {fmt(a['nll'])} / {fmt(b['nll'])} "
            f"| {fmt(a['cov50'])} / {fmt(b['cov50'])} "
            f"| {fmt(a['cov95'])} / {fmt(b['cov95'])} "
            f"| {fmt(a['area50_m2'])} / {fmt(b['area50_m2'])} "
            f"| {fmt(a['projected_median_m'])} / {fmt(b['projected_median_m'])} "
            f"| {fmt(a['sample_below_ground_fraction'])} / {fmt(b['sample_below_ground_fraction'])} |")

# Mode-based point error + goal calibration
lines.append("\n### 众数点误差与进球概率校准\n")
lines.append("| 子集 | σ | 众数中位 m (空中/触地) | 众数 P90 m (空中/触地) | P(goal) Brier (空中/触地) | P(goal) ECE (空中/触地) | goal_rate |")
lines.append("|" + "---|" * 7)
for subset in SUBSETS:
    for s in (2, 5, 10, 15):
        a, b = get("airborne", subset, s), get("bounce", subset, s)
        lines.append(
            f"| {subset} | {s} | {fmt(a['mode_median_m'])} / {fmt(b['mode_median_m'])} "
            f"| {fmt(a['mode_p90_m'])} / {fmt(b['mode_p90_m'])} "
            f"| {fmt(a['goal_brier'], 4)} / {fmt(b['goal_brier'], 4)} "
            f"| {fmt(a['goal_ece'], 4)} / {fmt(b['goal_ece'], 4)} | {fmt(a['goal_rate'], 4)} |")

text = "\n".join(lines)
(RES / "report_tables.md").write_text(text, encoding="utf-8")
print(text)

print("\n=== run metadata ===")
for arm in ORDER:
    cfg = json.loads((RES / arm / "config.json").read_text(encoding="utf-8"))
    print(arm, json.dumps({k: cfg[k] for k in ("arm", "device", "parameters", "epochs", "seed",
                                               "selected_epoch", "seconds", "sizes")}, ensure_ascii=False))
    print("   calibration:", json.dumps(cfg["calibration"], ensure_ascii=False))
