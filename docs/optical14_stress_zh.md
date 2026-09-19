# Optical 14 高强度 Stress：参数、对照图与复现

新增独立 `optical_marker_stress_v1`，仅接受 `severity_level=5`。原有
`optical_marker_v1` S1–S5 不变；两套 registry 的分数不得混入同一条 severity
曲线。本配置是更强的观测层工程压力测试，不是实测传感器损伤等级。

## 14 种算子如何增强

半径均按原生图像短边归一化；像素数对应本次 320×240 触觉输入。
这张表描述同一算子的增量，不表示不同算子之间强度等价。

| 算子 | 标准 optical S5 | 新增 Stress |
|---|---|---|
| A1 连续缺失 | 窗口占比 80% | 100%，窗口内全部缺失 |
| A2 间歇丢帧 | 80% | 请求 90%，至少保留一个恢复帧 |
| F1 全局光度漂移 | RGB 目标增益 0.52/0.736/1.072 | 0.12/0.35/0.60，明显变暗并偏色 |
| F2 局部灵敏度衰减 | 核心保留 10%，核心/支撑半径 0.16/0.28 | 保留 2%，半径 0.30/0.42，中心不移动 |
| F3 表面外观伪影 | 条纹半宽 6 px | 半宽 14 px，同时提高颜色扰动幅度 |
| F4 局部失响应 | 核心半径 0.225 | 0.34，仍使用边缘区域中心 (0.74, 0.50) |
| F5 接触图像形变 | 最大位移 14.4 px | 28.8 px，扩大支撑范围并保持无折叠 |
| F6 历史残影 | 混合 0.68，恢复时间常数 2.4 s | 混合 0.90，恢复时间常数 6 s |
| F7 高响应压缩 | 光学残差 soft-knee 0.035 | 0.008，更早压缩接触光学响应 |
| T1 共同源延迟 | 两侧延迟 0.8 s | 两侧延迟 1.2 s |
| T2 末帧冻结 | 冻结 0.8 s | 冻结 1.2 s，然后恢复当前输入 |
| T3 左右时间差 | 右侧延迟 0.8 s | 右侧延迟 1.2 s，左侧保持当前输入 |
| C1 来源交换 | 窗口占比 80% | 100%；完整交换的单帧幅度不能再增大 |
| C2 图像注册错位 | 向右 30 px | 向右 60 px |

A2 是真实 payload 缺失，不是黑图。短窗口会按整数帧数取整并保留恢复帧；
小于两帧的窗口拒绝执行。F2/F3/F4/F6/F7 保留当前 marker 核心及 2 px 保护邻域，
F2/F4 不做坐标移动；F5 将当前图像与 marker 一致变形，不叠加另一套点阵。
外部相机 RGB、proprio 不在这些触觉扰动的修改范围内。

## 接入与复现

在已有、绑定 rest reference 的 `FaultManifest` 构造中显式设置：

```python
severity_registry = "optical_marker_stress_v1"
severity_level = 5
```

其余参数沿用 [production 接入说明](optical14.md)，包括真实 `sample_period_s`、
所需 `rest_reference_sha256`，以及 C2 的 `realization="registered_pixels"`。
`apply_fault` 与 `StreamingFaultSession` 共享正式注入实现；图像展示不另行增强像素。

完成上述文档的数据准备后，在已激活的项目环境、仓库根目录执行：

```bash
PYTHONPATH=src python -m scripts.render_optical14_stress \
  --source-root deployment/artifacts/optical14-sources/calibrated_standalone_v1 \
  --output-root deployment/artifacts/optical14-stress-reproduction-v1 \
  --font '/System/Library/Fonts/STHeiti Light.ttc'
```

Linux 改用已安装的 Noto Sans CJK 字体路径。输出目录必须不存在。
图版脚本明确绑定原始 `pull_out_key/raw55`、`insert_HDMI/raw90` 和 60 Hz 观测时钟，
不是任意数据集的通用自动排版器。运行算子本身不依赖这些特定 episode。

时间算子需要足够的真实历史：本例延迟由 48 帧变为 72 帧，故障窗口为
`[90, 177)`，共 87 帧。T2 在观测 90–161 持有源帧 89，162 恢复。
不能只增大请求秒数、却仍用不足以展示冻结和恢复的短窗口。

## 已生成结果与验证

本次输出在 `deployment/artifacts/optical14-stress-v2/`：

- `01_F1-F4.png`：相同 Clean、标准 S5、新增 Stress 三列。
- `02_F5-F7-C2.png`：形变、残影、饱和、错位三列。
- `03_A1-A2-C1.png`：实际缺失与来源交换时间线。
- `04_T2_freeze.png`：冻结、持续冻结、恢复的源帧对照。
- `05_T1-T3_delays.png`：左右触觉及真实源时间差。
- `stress_receipt.json`：28 个 production 实例的 manifest/trace/validation、
  14 个逐算子增强检查，以及源文件、代码、字体、输出 SHA256。

固定展示帧上的平均绝对 RGB 变化量（0–255 单位）均增大：F1 19.102→62.132、
F2 0.831→2.576、F3 1.335→6.239、F4 0.891→1.495、F5 3.372→8.342、
F6 2.503→3.774、F7 1.703→3.526、C2 17.114→26.664。
这些数值只检查展示输出的变化，不是接触力、物理灵敏度或 success rate。
A1/C1 实际作用帧数 70→87，A2 70→78；时间算子检查实际源时间/持有帧数，
不以单张 RGB 差异代替时延。运动可能往返，更老的图像不一定逐像素更不同。

代码验证包括逐算子 streaming/batch 一致性、标准配置不变、时间映射、短窗口
A2、资源配置一致性，以及图版禁止覆盖、源时钟约束和增强检查的负例。
最终 Stress 功能及图版聚焦回归 34 tests 通过，图版 Ruff 与 strict mypy 通过；
更广的相关算子/schema 回归也已通过。全仓库发布 inventory 存在已有工作树漂移，
本次没有重新生成整仓 `release/source_manifest`，没有宣称全仓发布检查通过。

## 视觉判断与论文边界

F2 现在能明显看见中心接触边缘被压低；F3/F5/F7/C2 也较原 S5 更易辨认。
F4 的原 S5 已经抹去关键局部轮廓，本次主要增强面积，而不是凭空再制造另一种故障。
A1/C1 用持续范围表达增强；T1/T2/T3 用时间序列表达，避免单帧歧义。

F6 更强但也更显露 marker 保护邻域附近的合成色边：可作为明确标注的工程 Stress
示例，不能用这一张图证明真实材料恢复机理。它来自同一 episode 的因果历史和
局部卸载，**不是完全脱离接触**；6 s 时间常数不是该短片段测得的材料参数。
若要将其作为物理失效的主证据，仍需长时真实卸载/恢复序列与拟合，而不是继续加重残影。

全部触觉图均由录制的 UniVTAC 仿真数据经确定性 production 算子得到；没有大模型
生成的传感器像素，也不是实物传感器失效照片。零压入参考来自绑定校准的 Taxim
渲染，不冒充数据中观测到的无接触帧。本次新增闭环 episode 数为 0，未安装到远端、
未修改模型配置，也未产生新的鲁棒性 SR。显著扰动不保证策略一定失败。

## 再严格一档：Extreme

`optical_marker_extreme_v1` 是独立的 level-5-only 工程极限配置，不修改前版
Stress 的定义。它保留 14 类失效，但不虚称所有参数都能无限增强：

| 算子 | 前版 Stress → Extreme |
|---|---|
| A1 / C1 | 100% → 100%，已到上限，明确记录 unchanged-at-cap |
| A2 | 请求丢帧 90% → 98%，至少保留一帧，仍为间歇缺失 |
| F1 | RGB 目标增益 [0.12, 0.35, 0.60] → [0.04, 0.12, 0.30] |
| F2 | 核心响应保留 2% → 0.5%；核心半径 0.30 → 0.36，支撑半径 0.42 → 0.48 |
| F3 | 条纹半宽 14 → 20 px，并增强颜色扰动 |
| F4 | 失响应核心半径 0.34 → 0.42；窄过渡带 0.03 → 0.04 |
| F5 | 最大位移 28.8 → 48 px；支撑半径 0.42 → 0.48，仍无折叠 |
| F6 | 混合 0.90 → 0.97；恢复时间常数 6 → 10 s |
| F7 | soft-knee 0.008 → 0.002 |
| T1 / T2 / T3 | 1.2 → 1.5 s，即 60 Hz 下 72 → 90 帧 |
| C2 | 向右 60 → 96 px，即本次图像宽度的 30% |

图版入口不复制实现；新增 `--extreme` 选项，将三列改为
**相同 Clean / 前版 Stress / 新 Extreme**。剂量标签直接读取 materialized
manifest，源帧号和时间差来自实际 delivery，不靠手写图注冒充效果。

```bash
PYTHONPATH=src python -m scripts.render_optical14_stress --extreme \
  --source-root deployment/artifacts/optical14-sources/calibrated_standalone_v1 \
  --output-root deployment/artifacts/optical14-extreme-reproduction-v1 \
  --font '/System/Library/Fonts/STHeiti Light.ttc'
```

两组 T2 都使用 `[80, 178)` 窗口，源帧 79 被持有；Stress 在观测 152 恢复，
Extreme 在 170 恢复。T1/T3 仍使用 `[90, 177)` 窗口，有足够 90 帧真实历史。
没有延长、循环复制或伪造源 episode。A1/C1 的作用上限检查单独保存，
不能当作“进一步增强”计数；其他算子相同或减弱则拒绝完成图版。

Extreme 更接近破坏性输入消融，不是更可信的材料损伤标定。F3 的强色带、F6
的强残影尤其不应被当作实物失效照片。F2 保持低而非零响应，F4 才是局部零响应；
时序算子仍分别保留共同延迟、持有末帧、单侧延迟的区别。论文中应分开报告
标准 S1–S5、Stress 和 Extreme，不能用 Extreme 的性能降幅替换标准结论。

C2 的增强以正式输出签名校验通过后的位移像素数为准，不用整图 MAE 判断。
规则 marker 点阵会在更大平移后再次重合，因而更大位移可能产生更小 MAE；
图版同时保留 MAE 作为描述量，但不会为了让 MAE 单调而挑选位移参数。

本次完整输出：`deployment/artifacts/optical14-extreme-v1/`，沿用上述五张 PNG
及 `stress_receipt.json` 文件名。28 个正式注入实例通过，12 项 increased、
2 项 unchanged-at-cap。A2 实际丢失 85/87 帧（前版 78/87）；F1–F7 固定帧
MAE 均增大。C2 位移 60→96 px，但 MAE 26.6641→26.5162，按上述正确单位验收。
所有输出 SHA256、当前生产代码与图版脚本 hash 已核验。

Extreme/Stress/标准光学算子聚焦回归 84 tests 通过；图版与 schema 联合回归
14 tests、156 subtests 通过；修改源文件与图版 Ruff、strict mypy 通过。
旧两套 registry 的全部 14 算子 fixture manifest/trace 聚合 hash 与改动前一致。
本次仍未运行新闭环 episode，未部署到任何模型 runtime。
