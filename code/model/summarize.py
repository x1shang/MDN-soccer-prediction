"""Summarize the three arms and paired seed results after all runs complete."""
import csv
import hashlib
import json
from pathlib import Path
from statistics import mean, stdev

HERE = Path(__file__).resolve().parent          # code/model
ROOT = HERE.parents[1]                          # repository root
DATASETS = ROOT / "data" / "datasets"
EXPERIMENTS = ROOT / "experiments"
ARMS = ("v3_all", "v2_no_asai", "v3_no_asai")
SEEDS = (11, 42, 155)
NOISE_LEVELS = (2, 5, 10, 15)
CASES = ("Fig3", "Sample05", "Sample12")


def run_directory(arm):
    """Run artifacts live apart from the datasets they were trained on."""
    return EXPERIMENTS / "main_v3_all" if arm == "v3_all" else EXPERIMENTS / "asai_holdout" / arm


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def fmt(values, digits=2):
    return f"{mean(values):.{digits}f} ± {stdev(values):.{digits}f}"


def metrics(row):
    return tuple(float(row[key]) * scale for key, scale in
                 (("median_m", 100), ("p90_m", 100), ("nll", 1),
                  ("cov50", 100), ("cov95", 100)))


def aggregate(rows):
    return " | ".join(fmt(values, 4 if i == 2 else 2)
                      for i, values in enumerate(zip(*rows)))


def main():
    lines = [
        "# UPC155 百万有效轨迹与多种子训练",
        "",
        "使用交接包冻结的 RK4、锚点和瞄准规则。每组仿真 1,225,000 条候选，保留所有空中过 11 m 截面的轨迹供训练或评估。三组分别为全锚点 v3，以及留出 Asai 来源的 v2 和 v3。",
        "",
        "| 数据组 | 有效数 | 空中过截面率 | 题例 10 cm 邻域 | 高速入框 | 6×3 最稀疏格 | 训练数 | 测试数 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    scores, shared, replay, checkpoints, case_preds = {}, {}, {}, {}, {}
    trainer_hash = hashlib.sha256((HERE / "train_wiener_mdn.py").read_bytes()).hexdigest()
    for arm in ARMS:
        directory = run_directory(arm)
        manifest = read_json(DATASETS / arm / "data_manifest.json")
        q, g, sizes = (manifest[k] for k in ("quality", "generic", "index_counts"))
        lines.append(f"| {arm} | {sum(sizes.values()):,} | {q['crossing_rate']:.2%} | "
                     f"{q['near_case_10cm']:,} | {g['fast_goal_count']:,} | "
                     f"{g['fast_goal_min_cell']:,} | {sizes['train']:,} | {sizes['test']:,} |")
        for seed in SEEDS:
            run = directory / f"seed_{seed}"
            cfg = read_json(run / "config.json")
            curve = read_csv(run / "curve.csv")
            assert len(curve) == cfg["epochs"] == 100 and cfg["seed"] == seed
            assert cfg["selected_epoch"] == int(min(curve, key=lambda row: float(row["val_nll"]))["epoch"])
            assert cfg["dataset_sha256"] == manifest["sha256"]
            assert cfg["indices_sha256"] == manifest["indices_sha256"]
            assert cfg["training_script_sha256"] == trainer_hash
            assert cfg["checkpoint_sha256"] == hashlib.sha256((run / "model.pt").read_bytes()).hexdigest()
            checkpoints[arm, seed] = cfg["selected_epoch"]
            for row in read_csv(run / "test_scores.csv"):
                if row["calibration"] == "scaled_hdr":
                    scores[arm, seed, int(float(row["sigma_mm"]))] = metrics(row)
            case_preds[arm, seed] = read_json(run / "case_predictions.json")["sigma_5mm"]
            if arm != "v3_all":
                row, = read_csv(run / "common_canonical_scores.csv")
                assert int(row["n"]) == 53_126
                shared[arm, seed] = metrics(row)
                replay[arm, seed] = {r["case"]: float(r["error_cm"])
                                     for r in read_csv(run / "asai_replay.csv")}

    lines += [
        "",
        "v2_no_asai 的整体空中过截面率低于 85%，保留其原始分布作为对照。三组均达到 150 条题例邻域门槛。该门槛检查观测空间的样本量。题例没有可用于评分的终点标签。",
        "",
        "每组抽查 1,024 条轨迹，以冻结 RK4 配置重新积分并核对标签。步长由 2 ms 减至 1 ms 时，抽查中的过截面状态一致，终点差异最大约 0.012 mm。",
        "",
        "每组以种子 11、42、155 从头训练，每次 100 轮、批大小 512。仿真及数据划分固定，训练种子控制初始化、训练噪声和批次顺序。selection 以 NLL 选权重，calibration 独立校准，test 负责评分。下表汇总使用三个训练种子的均值 ± 样本标准差。",
        "",
        "## 共享 canonical 测试集",
        "",
        "53,126 条轨迹的索引、观测和标签已核对一致。两组使用相同的 5 mm 测试噪声，HDR 取各自校准结果。NLL 越低越好，覆盖率以接近标称值为准。采样策略同时改变了训练分布和有效样本数，配对差反映这两项变化的合并效应。",
        "",
        "| 数据组 | 种子 | 中位误差 (cm) | P90 (cm) | NLL | 50% HDR (%) | 95% HDR (%) |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for arm in ARMS[1:]:
        for seed in SEEDS:
            med, p90, nll, cov50, cov95 = shared[arm, seed]
            lines.append(f"| {arm} | {seed} | {med:.2f} | {p90:.2f} | {nll:.4f} | {cov50:.2f} | {cov95:.2f} |")
        lines.append(f"| {arm} | 汇总 | {aggregate([shared[arm, seed] for seed in SEEDS])} |")
    differences = [[b - a for a, b in zip(shared["v2_no_asai", seed], shared["v3_no_asai", seed])]
                   for seed in SEEDS]
    lines.append(f"| v3 − v2 | 配对差 | {aggregate(differences)} |")

    lines += ["", "## Asai 留来源回放", "",
              "用实测 0.05 s 和 0.10 s 观测回放至 11 m，模型的观测噪声参数设为 10 mm。单位为 cm。实测证据来自三个球次，种子间变动反映训练随机性。",
              "", "| 数据组 | 种子 | Fig3 | Sample05 | Sample12 | 三球平均 |",
              "| --- | --- | ---: | ---: | ---: | ---: |"]
    for arm in ARMS[1:]:
        by_seed = []
        for seed in SEEDS:
            errors = [replay[arm, seed][case] for case in CASES]
            values = errors + [mean(errors)]
            by_seed.append(values)
            lines.append(f"| {arm} | {seed} | " + " | ".join(f"{v:.2f}" for v in values) + " |")
        lines.append(f"| {arm} | 汇总 | " + " | ".join(fmt(v) for v in zip(*by_seed)) + " |")

    lines += ["", "## 全锚点 v3 的独立仿真测试", "",
              "| 观测噪声 | 中位误差 (cm) | P90 (cm) | NLL | 50% HDR (%) | 95% HDR (%) |",
              "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for sigma in NOISE_LEVELS:
        values = [scores["v3_all", seed, sigma] for seed in SEEDS]
        lines.append(f"| {sigma} mm | {aggregate(values)} |")

    lines += ["", "题例 5 mm 下的终点预测为球心坐标 (y, z)，单位为 m。",
              "", "| v3_all 种子 | Selection 最优轮次 | 分布均值 | 近似众数 |",
              "| --- | ---: | --- | --- |"]
    for seed in SEEDS:
        pred = case_preds["v3_all", seed]
        lines.append(f"| {seed} | {checkpoints['v3_all', seed]} | "
                     f"{tuple(pred['mean_yz_m'])} | {tuple(pred['mode_yz_m'])} |")
    lines += [
        "",
        "完整指标、模型权重、训练曲线和校准参数见各组的种子目录。首次 GPU 加速路径出现批次更新不一致，修复后通过逐批数值对照。受影响的两次训练已重跑，汇总使用修复后的结果。",
    ]
    (ROOT / "docs" / "summary_million_sim.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
