# train759 Vision + tactile 重训练权重评测

本入口适用于 `checkpoint_step_10000` 的 **MoT、20D absolute-EE、10 Hz、4 actions/latent** 权重，不是公开 delta checkpoint 的路径替换。原 `prepare_official.py` / `serve_univtac.sh` 和默认 60 Hz 协议不变。

## 本次执行范围

先做八任务各一次 Clean diagnostic，使用 UniVTAC 原始成功判据。每任务保留独立 `request.json`、`invocation.json`、`summary.json` 与标准 `live/` 工件，可通过现有视频工具查看。基础设施错误不计作模型失败，成功率不得按未执行任务补零。

这不是官方权重的复现结果，也不是论文统计结果。随后才能在同一重训练权重、初始 seed、时钟和预算下配对比较 Clean/Faulted。模型失败不为提高 SR 而重跑。

## 必须配套的文件

- `checkpoint_step_10000/{train_meta.json,transformer/config.json,transformer/diffusion_pytorch_model.safetensors}`。
- 本次训练的 `norm_stat_absee_per_repo.json`，键是八个实际 task ID；不得用旧 checkpoint 的 norm 或默认 ±1 占位 norm。
- `train759/<task>/meta/{info.json,tasks.jsonl}`，用于读取真实 FPS、输入特征与逐任务 prompt。
- 同次训练的 N0 源码，以及已有 base 的 VAE、text encoder、tokenizer、`empty_emb.pt`。不复制 base，不修改上游源码。
- 真实数据划分 manifest。其 hash 被记录为数据来源，**并不意味着在线仿真逐一重放 frozen40 的初始化**；本阶段是在线 reset 的八任务诊断。

新输入采用训练转换器的 numerical RGB、不翻转通道；这来自 `decode_legacy_rgb` 的代码契约，仍须在模型部署后检查真实观测对比。每个动作推进 12 个 120 Hz physics ticks，观测/动作间隔 0.1 s。冷启动仅执行第二个 latent 的 4 个动作；后续 chunk 执行 8 个，每动作采集一帧，并提交实际执行后的 history。

预算保留 task registry 的动作数上限（300/500/600），本入口以 **10 Hz** 解释，并在 invocation 中保存动作数与频率；不要直接把旧 60 Hz 运行的墙钟/轨迹时长作为公平对比。墙钟超时是 infrastructure watchdog，不用于改善或降低 SR。

## 准备与启动

以下命令从 RoboTactile 根目录执行。所有变量由调用方提供真实绝对路径；`python` 指当前已激活的 repo-local N0 runtime。

```bash
python scripts/n0_twam/prepare_retrained.py \
  --checkpoint "$RT_CHECKPOINT" --base "$RT_BASE" \
  --source-root "$RT_N0_SOURCE" --per-repo-norm "$RT_PER_REPO_NORM" \
  --metadata-root "$RT_TRAIN759" --output "$RT_PREPARED" \
  --source-commit cdd87b6a141667123ad2c25f452478afdb71e287

python -m torch.distributed.run --standalone --nproc-per-node=1 \
  scripts/n0_twam/serve_retrained.py \
  --artifact "$RT_PREPARED/artifact.json" --task lift_bottle \
  --output "$RT_SERVER_OUTPUT" --port 29601
```

准备阶段一次 hash 大权重，后续验证文件状态与小型元数据；输出目录不得已存在。只有匹配实际 Git 根或源快照 receipt 时才声明源码 commit 已验证；未带 `.git` 的源拷贝记录为 `declared_unverified`，允许诊断但不升级为论文证据。

服务实际监听后，在另一个终端使用已配置 Isaac runtime（`python` 必须指该 runtime，并具备本仓库版本的 package、IsaacLab、cuRobo 导入路径）：

```bash
python scripts/n0_twam/run_retrained_clean.py \
  --artifact "$RT_PREPARED/artifact.json" --task lift_bottle \
  --dataset-manifest "$RT_SPLIT_MANIFEST" \
  --upstream "$RT_UNIVTAC" --runtime "$RT_ISAAC_TASK_RUNTIME" \
  --server-receipt "$RT_SERVER_OUTPUT/server_receipt.json" \
  --output "$RT_EPISODE_OUTPUT" --seed 42 --port 29601 \
  --capture-profile preview_v1

robotactile visualize-live-artifact \
  --artifact "$RT_EPISODE_OUTPUT/live" --output "$RT_VIDEO_OUTPUT" \
  --video --fps 10
```

视频命令参数请以已安装版本 `robotactile visualize-live-artifact --help` 为准。`preview_v1` 最多保存 64 帧，抽样视频不能当作完整 episode 时长；需完整时序时明确选择 `paper_full_v1`。

八个任务是 `grasp_classify`、`insert_HDMI`、`insert_hole`、`insert_tube`、`lift_bottle`、`lift_can`、`pull_out_key`、`put_bottle_in_shelf`。服务按 task 绑定 normalizer，因此换 task 时必须匹配新的服务配置与 receipt；不得仅更换 client prompt。该最小入口暂不支持跨 task 热切换，已有同 task worker 优化不在本次适配中改动。

## 结果边界

`summary.json` 保留 `score_eligible`、`score_success`、`terminal_status`、`failure_stage/code`、control cycles 和观测数。只有真实执行且可计分的结果进入 SR 分母。报告同时列出已完成、可计分、成功、基础设施错误和未执行数量；全部完成前不宣称 all-8 SR。`failure.json` 只是运行错误记录，不能替代有效终止工件。

当前代码验证不等于该 checkpoint 已完成 GPU 推理或闭环测试。具体部署进度记录在 `plan/n0_retrained_step10000_evaluation.md`。

## 单任务故障窗口协议

`scripts.n0_twam.single_task_robustness run` 默认采用下文的
`early_random_onset_v1`。如需复现旧的全程协议，显式指定
`--fault-window full_episode_v1`：从策略首次看到的 observation 0 开始，
持续到真实 episode 终止；manifest 的
`stop_index` 为 `max_observation_steps`（右开区间），不预先根据 Clean 成功时刻截断。
Clean 使用相同任务、seed、初始快照和预算，但完全不注入故障。

```bash
python -m scripts.n0_twam.single_task_robustness run \
  --binding "$RT_BINDING" --task grasp_classify \
  --campaign "$RT_NEW_CAMPAIGN" --code "$RT_CODE" --package "$RT_PACKAGE" \
  --fault-window full_episode_v1
```

该显式全程协议不能重命名或覆盖旧结果。旧版中途故障仍可通过
`--fault-window delayed_onset_v1` 复现：N0 从 observation 20 开始，T2 保留
registry 的有限冻结时长。其他模型的共享 group 入口不改变默认值；如显式使用
全程模式，在 binding 的 `evaluation` 中设置 `fault_window_mode: full_episode_v1`，
不能同时指定非零 `fault_start_index`。

- T1：无 episode 前史时因果复用首帧，`source=max(0, t-lag)`；历史够长后才
  达到完整固定延迟。工件记录 `startup_policy=hold_first` 和稳态起点，
  不使用未来帧、不伪造负时间前史。
- T2：整个 episode 保持首帧，不再在固定 15 帧后恢复。它是明确标注的
  persistent-freeze 压力条件，不应与原 1.5 s 的 S5 T2 混作同一剂量；
  工件同时保存基准 `requested_duration_s` 与 `effective_hold_duration_s`。
- T3：两路按现有错时规则工作，历史不足时使用首帧；启动期真实 skew 单独保留。
- F6：从当前 episode 的真实观测累积历史，不拼接其他 episode。
  全程激活不保证出现释放样本；无卸载/释放证据仍为 invalid/N/A，不放宽验证。
- A1/A2：N0 原生接口仍不支持缺失 payload，保持 unsupported/N/A。

报告同时给出 scheduled-window、fault-active、payload-change 和 action-query
暴露比例。全程指故障机制持续启用，不要求每帧像素不同：首帧时间延迟/冻结、
无接触时的形变类失效等都可能有 active 标记但 payload 不变。更长暴露也不保证
SR 一定下降，必须保留真实结果。`paper_full_v1` 视频包含全部观测；汇总报告
及指标面板标记窗口协议。

当前 T2 的 source 缓存上限随 episode horizon 线性增长；全程冻结不宣称常数内存。
旧中途注入结果与新全程协议应分别报告，每条件一次仍仅为 diagnostic。

## 八任务、三 seed 早期随机起点实验

统一入口复用上述权重与 10 Hz 时钟。新 campaign 默认使用
`early_random_onset_v1`：同一 task/seed 的全部故障共用一个由 SHA256 确定的
起点，不随算子改变；不同 seed 会重新抽取。起点不能为第 0 帧，最大为
`min(8, floor((max_observation_steps-1)/3))`。T1/T3 若要求的历史长度超过
起点已有历史，启动期因果复用第 0 帧；历史足够后达到指定的固定延迟/错时。
故障窗口从该帧持续至 `max_observation_steps`（右开区间），Clean 完全不注入。
因此旧 `full_episode_v1` 工件与新协议不能混合计算 SR。

```bash
python -m scripts.n0_twam.multi_task_robustness \
  --binding "$RT_BINDING" --campaign "$RT_NEW_CAMPAIGN" \
  --code "$RT_CODE" --package "$RT_PACKAGE" \
  --tasks grasp_classify insert_HDMI insert_hole insert_tube \
    lift_bottle lift_can pull_out_key put_bottle_in_shelf \
  --seeds 0 1 2
```

campaign 路径必须尚不存在；binding 和启动日志应放在独立部署 bundle 中。
每个 task 启动一次 N0 server，复用于三个 seed；每个 seed 独立启动 Isaac，
避免跨 seed 残留快照或策略历史。每组包含同快照的 Clean 与 12 项非缺失故障，
因此计划执行 `8 × 3 × 13 = 312` 个 episode；A1/A2 在账目中保留为 unsupported。
有任务匹配的 certified rest 时复用；缺少参考的 task 只标定一次，不将标定计分。
每个 group 的 `group.json` 和 fault manifest 记录真实起点；报告必须同时核查
`active_fault_action_queries` 与 `changed_frames_at_action_query`。极短 episode
若在随机起点前终止，按真实无暴露结果记录，不为凑出扰动效果而重跑。

逐 seed 输出位于 `tasks/TASK/seeds/seed-NNN/`，包含 `groups/n0_twam/TASK/`
和 `visualizations/`。真实失败不重跑；视频导出失败单独记录，不触发 episode 重跑。
汇总按 task 和 condition 分别列出执行、有效、成功、无效、未执行及不支持数量。
每条件仅三个 seed，是小样本诊断，不承诺接近论文 SR，也不把不同故障合并为独立
随机重复。`paper_full_v1` 仅表示完整数据保存，不代表自动达到论文统计要求。
