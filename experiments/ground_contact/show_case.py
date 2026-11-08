import json
from pathlib import Path

R = Path("results")
for arm in ("airborne", "bounce"):
    c = json.loads((R / arm / "case_predictions.json").read_text(encoding="utf-8"))
    print("==", arm)
    print(f"{'sigma':>8} {'P(goal)':>9} {'mode y,z':>16} {'mean y,z':>16} "
          f"{'wiener y,z':>16} {'hdr50':>8} {'hdr95':>8}")
    for k, v in c.items():
        print(f"{k:>8} {v['goal_probability_given_crossing']:9.4f} "
              f"({v['mode_yz_m'][0]:6.3f},{v['mode_yz_m'][1]:6.3f}) "
              f"({v['mean_yz_m'][0]:6.3f},{v['mean_yz_m'][1]:6.3f}) "
              f"({v['wiener_anchor_yz_m'][0]:6.3f},{v['wiener_anchor_yz_m'][1]:6.3f}) "
              f"{v['hdr_log_density_thresholds']['0.5']:8.3f} "
              f"{v['hdr_log_density_thresholds']['0.95']:8.3f}")
    print()
