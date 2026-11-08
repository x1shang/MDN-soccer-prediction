# DFL 比赛足球空中轨迹：保守筛选结果

筛选日期：2026-11-07。原来的 60 段经过运动质量与事件日志筛查后，当前目录保留 **11 段、254 个位置点**（每段前三点共 33 个输入点，221 个后续对照点）。其余 49 段已从当前目录的轨迹数据中移除；[逐段审计](screening_audit.csv)保留所有原始 case ID、指标、排除理由和邻近事件，便于复核。Git 历史中仍存在筛选前版本。

**11 段仅适合标记为探索性轨迹对照，不能确认是无接触飞行，也不能称为独立点球实测验证集。** 比赛事件存在漏标、时间偏差，且未取得这些片段的同步视频；无邻近事件和运动平滑都不能证明没有头球、触球或追踪异常。事件类型、准确离脚时刻、旋转、风况未知。通过筛选只代表未触发下面的检查，并不是人工核验结论。请勿由这些数据声称模型已被全面验证。

## 筛选方法及结果

1. 核查连续帧、UTC 时间间隔 0.04 s、有限坐标、离地高度至少 0.5 m、速度 3–40 m/s。
2. 对完整轨迹的三个坐标分别拟合二次曲线，要求三维位置残差 RMS ≤0.08 m、最大残差 ≤0.18 m，拟合竖直加速度在 −20 至 −1 m/s²、水平加速度模 ≤15 m/s²。这些是保守工程筛查阈值，不是测量精度标准或足球全部真实运动的边界。拟合用到了后续点，因此不是预测结果。
3. 核对 DFL 同场事件 XML。以片段开始前 1 s 至结束后 1 s 为保守窗口，若存在 Play、OtherBallAction、TacklingGame、ThrowIn、BallClaiming、FreeKick、ShotAtGoal、GoalKick、CornerKick、KickOff、BallDeflection、Penalty 或 RefereeBall，则排除。此规则可能把启动动作、只是附近发生的事件误排除，故排除不代表确定发生中途接触；也无法排除窗口外实际发生的接触。

49 段触发邻近球事件规则；其中一段还触发速度下限。其余 11 段没有触发以上规则。每段均已核对逐帧坐标和时间与源 CSV 一致。筛选会偏向平滑、近似抛物线的轨迹，可能误排真实弧线，且用了整段后续观测；**在此筛选集上的良好成绩不能无偏地代表全部真实射球的表现**。不要再用这些轨迹调参后称它们是独立测试。

## 文件

- [positions.csv](positions.csv)：仅含保留的 11 段，原始球场坐标、UTC 时间戳、帧号、相对片段起点时间与 input/held_out。t=0 为所选追踪帧，不是已知离脚时刻；单位米、秒。
- [segments.csv](segments.csv)：保留片段的帧范围、高度、位移和未知事件及接触状态。
- [screening_audit.csv](screening_audit.csv)：全部 60 段的筛选结果及邻近事件；不含被删除片段的逐帧轨迹。
- [extract.py](extract.py)、[screen_quality.py](screen_quality.py)：复现原候选筛选及二次筛查的脚本，使用 Python 标准库。需要本地 `B_standard_ball_trajectories/processed/` 三个完整下半场球 CSV 和 `raw/dfl/` 三个比赛事件 XML；它们不包含在本目录。运行 `extract.py` 会在本地生成原始候选文件，再运行 `screen_quality.py` 生成筛选结果；不应把中间候选文件当作已验证数据。
- [source_hashes.json](source_hashes.json)：源下半场中间 CSV 的 SHA-256；[event_hashes.json](event_hashes.json)：用于二次筛选的事件 XML SHA-256。

## 来源、许可和限制

Bassek, M., Rein, R., Weber, H. & Memmert, D., “An integrated dataset of spatiotemporal and event data in elite soccer”, *Scientific Data* 12, 195 (2025), [论文 DOI](https://doi.org/10.1038/s41597-025-04505-y)；[原始数据集 DOI](https://doi.org/10.6084/m9.figshare.28196177.v1)，[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)。本包从原比赛球追踪截取、筛选并添加相对时间及数据用途标记。

原始位置文件：[J03WMX](https://ndownloader.figshare.com/files/51643514)、[J03WOH](https://ndownloader.figshare.com/files/51643520)、[J03WR9](https://ndownloader.figshare.com/files/51643532)。此前本地只下载 XML 尾部字节并提取完整下半场 BALL FrameSet，不是下载完整 XML。原始事件 XML 来自同一数据集，具体文件名和哈希见 event_hashes.json。坐标是多摄像机追踪系统的估计，球高度约定和绝对精度没有独立标定。

初选规则为有效状态、连续帧、Z≥0.5 m、速度≤40 m/s、相邻速度变化≤1.6 m/s、至少 16 帧、首尾位移≥4 m，每段最多 25 帧，段间至少间隔 125 帧，每场按时间最多选 20 段。间隔不证明独立踢球。该数据与此前 DFL 点球案例同源，有些来自同一场比赛。25 Hz 不是题目的 20 Hz，不能当作题目观测时刻的精确复现；无球门交点标签。

