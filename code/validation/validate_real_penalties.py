"""Prefix-only point forecasts on the two DFL airborne penalty cases.

The fit accepts only inputs_only.csv. Predictions are saved before held_out.csv
is opened. All fixed models are reported, without selecting by held-out error.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares

from physics import Config, derivative


MODELS = {'quadratic': None, 'gravity': 0.0, 'drag_022': 0.22, 'drag_045': 0.45}
CASES = ('J03WOH', 'J03WR9')


def read_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def coordinates(rows):
    return np.array([float(row['t_s']) for row in rows]), np.array([
        [float(row[key]) for key in ('x_m', 'y_m', 'z_m')] for row in rows])


def save_json(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def fit_prefix(times, positions, model):
    if len(times) != 3 or times[0] != 0 or np.any(np.diff(times) <= 0):
        raise ValueError('Require exactly three distinct increasing times starting at zero')
    if not np.isfinite(positions).all():
        raise ValueError('Non-finite prefix')
    slope, intercept = np.polyfit(times, positions, 1)
    if model == 'quadratic':
        coefficients = np.linalg.solve(np.column_stack([np.ones(3), times, .5*times**2]), positions)
        return {'state': np.r_[coefficients[0], coefficients[1]].tolist(),
                'acceleration': coefficients[2].tolist(), 'prefix_rmse_m': 0.0}
    cfg = Config(fixed_cd=MODELS[model])
    theta = np.zeros((1, 7))  # Fixed zero spin for the drag/gravity comparison.

    def residual(state):
        solution = solve_ivp(lambda t, y: derivative(y[None], theta, t, cfg)[0],
            (0, times[-1]), state, t_eval=times, rtol=1e-9, atol=1e-10)
        if not solution.success:
            raise RuntimeError(solution.message)
        return (solution.y[:3].T - positions).ravel()

    fit = least_squares(residual, np.r_[intercept, slope], xtol=1e-10, ftol=1e-10, gtol=1e-10)
    if not fit.success:
        raise RuntimeError(fit.message)
    return {'state': fit.x.tolist(), 'acceleration': None,
            'prefix_rmse_m': float(np.sqrt(np.mean(np.sum(fit.fun.reshape(-1, 3)**2, axis=1))))}


def forecast(fit, model, target, rtol=1e-9, atol=1e-10, max_step=.002):
    cfg = Config(fixed_cd=MODELS[model])
    state = np.array(fit['state'])
    if state[2] <= cfg.radius:
        raise ValueError('Fitted initial centre is at/below ground contact height')
    theta = np.zeros((1, 7))

    def rhs(t, y):
        if model == 'quadratic':
            return np.r_[y[3:], fit['acceleration']]
        return derivative(y[None], theta, t, cfg)[0]

    def ground(t, y):
        return y[2] - cfg.radius

    def goal(t, y):
        return y[0] - target

    ground.terminal = True
    ground.direction = -1
    goal.terminal = False
    goal.direction = 1
    solution = solve_ivp(rhs, (0, 1.5), state, events=(ground, goal), dense_output=True,
                         rtol=rtol, atol=atol, max_step=max_step)
    if not solution.success:
        raise RuntimeError(solution.message)
    return solution


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[2]
    parser.add_argument('--data', type=Path, default=root/'data/anchors/dfl/flight_trajectory_validation_2026-11-07')
    parser.add_argument('--out', type=Path, default=root/'experiments/validation/real_penalties')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    config = json.loads((args.data/'case_config.json').read_text(encoding='utf-8-sig'))
    inputs = read_csv(args.data/'inputs_only.csv')
    protocol = {'cases': CASES, 'models_fixed_cd': MODELS, 'spin': 'zero for physical baselines',
                'fit': 'three prefix frames; six initial-state coordinates; unweighted position least squares',
                'quadratic': 'exact interpolation of three prefix positions, constant acceleration thereafter',
                'density_kg_m3': 1.20, 'mass_kg': .430, 'radius_m': .110,
                'primary_score': 'Euclidean position RMSE at held-out frames before observed goal-plane crossing',
                'failure': 'report predictions stopped by ground and absent forward goal crossings',
                'goal_reference': 'linear interpolation of the held-out tracking bracket',
                'horizon_s': 1.5, 'selection': 'report every fixed model; no held-out selection or calibration',
                'source_sha256': {name: hashlib.sha256((args.data/name).read_bytes()).hexdigest()
                                  for name in ('inputs_only.csv', 'case_config.json')},
                'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    save_json(args.out/'protocol.json', protocol)
    forecasts, fitted = {}, {}
    for case in CASES:
        times, positions = coordinates([row for row in inputs if row['case_id'] == case])
        fitted[case] = {}
        for model in MODELS:
            fit = fit_prefix(times, positions, model)
            solution = forecast(fit, model, config[case]['target_local_x_m'])
            forecasts[case, model] = solution
            grid = np.linspace(0, solution.t[-1], 301)
            fit['forecast_time_s'] = grid.tolist()
            fit['forecast_xyz_m'] = solution.sol(grid)[:3].T.tolist()
            fitted[case][model] = fit
    save_json(args.out/'prefix_predictions.json', fitted)
    # Future observations are loaded only after predictions have been fixed.
    held = read_csv(args.data/'held_out.csv')
    scores = {}
    point_rows = []
    for case in CASES:
        times, truth = coordinates([r for r in held if r['case_id'] == case])
        target = config[case]['target_local_x_m']
        bracket = np.flatnonzero((truth[:-1, 0] < target) & (truth[1:, 0] >= target))
        if len(bracket) != 1:
            raise ValueError(f'{case}: require one observed goal bracket in held-out data')
        index = bracket[0]
        fraction = (target - truth[index, 0]) / (truth[index+1, 0] - truth[index, 0])
        reference = truth[index] + fraction * (truth[index+1] - truth[index])
        reference_time = times[index] + fraction * (times[index+1] - times[index])
        eligible = times < reference_time
        common = eligible & (times <= min(forecasts[case, m].t[-1] for m in MODELS) + 1e-12)
        scores[case] = {'reference_time_s': float(reference_time), 'reference_local_yz_m': reference[1:].tolist(),
                        'goal_center_y_m': config[case]['target_local_center_y_m'], 'models': {}}
        for model in MODELS:
            solution = forecasts[case, model]
            available = times <= solution.t[-1] + 1e-12
            valid = eligible & available
            prediction = solution.sol(times[valid])[:3].T
            residual = prediction - truth[valid]
            cross = solution.y_events[1]
            record = {'prefix_rmse_m': fitted[case][model]['prefix_rmse_m'],
                      'eligible_frames': int(eligible.sum()), 'scored_frames': int(valid.sum()),
                      'missing_frames': int((eligible & ~available).sum()),
                      'post_goal_frames_excluded': int((~eligible).sum()),
                      'position_rmse_m': float(np.sqrt(np.mean(np.sum(residual**2, axis=1)))) if valid.any() else None,
                      'axis_rmse_m': np.sqrt(np.mean(residual**2, axis=0)).tolist() if valid.any() else None,
                      'goal_reached': bool(len(cross)),
                      'ground_time_s': float(solution.t_events[0][0]) if len(solution.t_events[0]) else None}
            common_residual = solution.sol(times[common])[:3].T - truth[common]
            record['common_frames'] = int(common.sum())
            record['common_rmse_m'] = float(np.sqrt(np.mean(np.sum(common_residual**2, axis=1)))) if common.any() else None
            if len(cross):
                record.update(goal_time_s=float(solution.t_events[1][0]),
                              goal_local_yz_m=cross[0, 1:3].tolist(),
                              goal_error_m=float(np.linalg.norm(cross[0, 1:3] - reference[1:])),
                              goal_time_error_s=float(solution.t_events[1][0]-reference_time))
            scores[case]['models'][model] = record
            for t, actual, predicted in zip(times[valid], truth[valid], prediction):
                point_rows.append(dict(case=case, model=model, t_s=float(t),
                    **{f'true_{a}': float(actual[j]) for j, a in enumerate('xyz')},
                    **{f'pred_{a}': float(predicted[j]) for j, a in enumerate('xyz')}))
    save_json(args.out/'scores.json', scores)
    with (args.out/'point_predictions.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(point_rows[0]))
        writer.writeheader()
        writer.writerows(point_rows)
    lines = ['# 两次真实点球的前段拟合与留出外推', '',
        '固定前三帧（0、0.04、0.08秒）拟合初始状态，仅评分参考过门时刻之前的留出帧。',
        '物理对照固定零旋，Cd=0、0.22、0.45；恒加速度由三点插值得到。所有模型预先指定，逐项报告。', '',
        '| 案例 | 模型 | 留出评分/应评分 | 三维RMSE m | 共同帧RMSE m | 过门点误差 m |',
        '| --- | --- | --- | ---: | ---: | ---: |']
    for case, data in scores.items():
        for model, row in data['models'].items():
            error = f"{row['goal_error_m']:.3f}" if row['goal_reached'] else '未到达'
            rmse = f"{row['position_rmse_m']:.3f}" if row['position_rmse_m'] is not None else '无预测'
            common_error = f"{row['common_rmse_m']:.3f}" if row['common_rmse_m'] is not None else '无预测'
            lines.append(f"| {case} | {model} | {row['scored_frames']}/{row['eligible_frames']} | {rmse} | {common_error} | {error} |")
    lines += ['', '共同帧列对每个案例使用所有模型均能预测的相同帧集；数量分别为 ' +
              '、'.join(f"{case} {scores[case]['models']['gravity']['common_frames']}帧" for case in CASES) + '。提前触地造成的缺失仍作为失败报告。', '',
              '![逐坐标外推](trajectory_comparison.png)']
    lines += ['', '## 解释范围', '',
        '- 参考轨迹为摄像追踪估计。过门标签由相邻留出帧插值，绝对三维精度和离脚时刻未独立标定。',
        '- 空气密度1.20 kg/m³，质量0.430 kg，半径0.110 m。预测在球心下降到半径高度时停止。',
        '- 两次观测之外的起始位置也有追踪误差，因此用三帧联合拟合六维初始状态。无参数使用留出帧拟合。',
        '- 三点恒加速度恰好插值噪声，误差可能随外推放大。零旋物理模型也可能因缺少升力而失配。',
        '- 本轮是点预测基线评测，尚未评估旋转后验或神经网络，也没有给出实测概率覆盖率。',
        '- 在参考过门后的追踪点单独排除，避免把可能的球网接触纳入自由飞行误差。完整预测保留以便复核。',
        '- 两个进球案例不足以确定总体方法排名。后续若据此调整模型，应报告为开发对照并使用新案例验证。', '',
        '- 前期数据完整性审计已查看完整轨迹，本轮定位为开发外推对照。代码确保留出帧不进入初值拟合，不能将本轮描述为盲评。', '',
        '## 复现', '', '```powershell', 'python student_b/validate_real_penalties.py', '```', '',
        '配置见 protocol.json，冻结前段预测见 prefix_predictions.json，逐点结果见 point_predictions.csv。', '']
    (args.out/'实测点球外推结果.md').write_text('\n'.join(lines), encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(2, 3, figsize=(12, 7), constrained_layout=True)
    for case_index, case in enumerate(CASES):
        prefix_times, prefix_xyz = coordinates([r for r in inputs if r['case_id'] == case])
        future_times, future_xyz = coordinates([r for r in held if r['case_id'] == case])
        cutoff = scores[case]['reference_time_s']
        for coordinate, axis in enumerate(axes[case_index]):
            axis.scatter(prefix_times, prefix_xyz[:, coordinate], color='black', marker='s', label='Prefix', zorder=5)
            before_goal = future_times < cutoff
            axis.scatter(future_times[before_goal], future_xyz[before_goal, coordinate], color='black', marker='x', label='Held out', zorder=5)
            for model in MODELS:
                solution = forecasts[case, model]
                grid = np.linspace(0, min(cutoff, solution.t[-1]), 150)
                axis.plot(grid, solution.sol(grid)[coordinate], label=model)
            axis.set(title=f'{case}: {"xyz"[coordinate]}', xlabel='Time from first input (s)', ylabel='Position (m)')
            axis.grid(alpha=.2)
    axes[0, 0].legend(fontsize=8)
    figure.savefig(args.out/'trajectory_comparison.png', dpi=180)
    plt.close(figure)
    print('\n'.join(lines[:16]))


if __name__ == '__main__':
    main()
