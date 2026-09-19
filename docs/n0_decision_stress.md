# N0-TWAM 决策时刻故障诊断

`optical_decision_stress_v1` 是独立的工程压力测试配置；不改变旧版本结果、
官方 checkpoint、动作 chunk、成功判据或标准 S1–S5 路径。

- F1 在故障启动后 6 个 source observations 达到极端配置的目标增益，随后
  保持到 episode 结束。原 `optical_marker_extreme_v1` 仍在整段窗口内渐变。
- T1/T3 延迟 48 个 source observations，等于当前官方 N0 24-step chunk 的
  两段长度。秒数由 backend 的真实采样周期计算，不使用模型名字中的 60 Hz。
- T2 固定保留启动前最后一帧直到结束。初始即存在接触的任务中，这不是
  “无接触帧”；报告须结合该帧实际 contact phase 解释。
- F3 复用既有极端表面伪影。其他算子保留 extreme 的实现；A1/A2 的 native
  N0 unsupported 状态不变。

启动时刻按 task/seed 在早期窗口随机确定，同一个 task/seed 的算子共享启动
时刻。首轮固定 `grasp_classify`、`lift_can`；seed 5、3、7；每组 Clean、
F1、F3、T2、T3，共 30 个 episode，使用 preview_v1 保存最多 64 个关键帧。

在已经部署好的 A800 repo-local N0 Python 环境中，从仓库目录执行：

```bash
python -m scripts.n0_twam.run_decision_stress prepare \
  --prior /absolute/path/to/completed-official-early-campaign \
  --output /absolute/path/to/new-decision-pilot \
  --seeds 5 3 7 --tasks grasp_classify lift_can --port 29695

python -m scripts.n0_twam.run_decision_stress run \
  --output /absolute/path/to/new-decision-pilot \
  --package-root /absolute/path/to/versioned-wheel-extraction

python -m scripts.n0_twam.run_decision_stress report \
  --output /absolute/path/to/new-decision-pilot
```

`--prior` 提供当前机器的模型配置、官方动作协议、Clean 请求及 rest reference。
runner 等待 prior 的完成标记，并对自身使用进程锁；每个 task/seed 的五个
条件复用一个 Isaac session。已完成 group 跳过；存在部分工件的 group 不会
无声重跑，须明确读取失败原因后只补缺失实验。

每组结束生成 `decision_report.json`；全组完成生成 `pilot_results.json`。
报告区分 valid / invalid / pending，给出毫米级 EE 位置和夹爪差异、消除
四元数符号歧义后的姿态差异、首次故障后重规划和后续轨迹差异。捕获的回填
关键帧记录原始 tactile hash，并检查同一时刻 RGB/proprio 未被算子修改。
这些 hash 是观测交付证据，不是内部 VAE 或 attention 使用触觉的证明。

本轮属于探索：保留全部成功、失败和改善结果。后续若根据本轮选择更有效
的噪声参数，应在独立 seed 上冻结验证；不能只展示导致失败的样例，也不能
把基础设施失败或 action-bound crash 混同于普通任务 timeout。
