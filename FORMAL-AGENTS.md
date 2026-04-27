# AGENTS.md — ESKF-test 本地 Codex 项目规则 v4

## 0. 放置位置

把本文件放到你本地 Codex 打开的项目根目录：

```text
ESKF-test/
  AGENTS.md
  01_data/
  02_src/
  03_results/
  04_tools/
  05_tests/
  requirements.txt
```

不要放到桌面随便一个位置，不要放到 `02_src/` 或 `05_tests/` 里面。

---

## 1. 项目最终目标

本项目最终目标不是做零散 demo，也不是长期小修小补。

最终目标是逐步形成一个**完整、成熟、可验证的 Python 多源融合导航系统**，能处理无人机多源融合数据，能完成离线组合导航定位实验。

最低完整项目目标：

> 先构建一个类似 KF-GINS / GVINS 工程风格的最低可用完整项目：有稳定数据入口、有 ESKF 滤波主线、有多源测量更新、有配置、有测试、有结果输出、有评估指标、有实验汇总、有工作记录。先把这个最低完整系统做成熟，再考虑增加新传感器、新算法或新路线。

最低完整系统至少应具备：

1. 能稳定读取和处理 IMU / GNSS / barometer / magnetometer yaw 等数据；
2. 能完成 ESKF 初始化；
3. 能完成 IMU mechanization / predict；
4. 能融合 GNSS position；
5. 能融合 GNSS velocity；
6. 能按需融合 barometer；
7. 能按需融合 magnetometer yaw；
8. 能输出融合轨迹、状态量、协方差、质量指标、误差统计；
9. 能对传感器退化、异常观测、GNSS 中断进行基础处理；
10. 能通过完整测试体系验证核心模块；
11. 能批量运行实验并汇总结果；
12. 能形成清晰的工作记录和可复现实验流程。

当前阶段不是追求论文花样，而是先把最低完整工程闭环做扎实。

---

## 2. 技术路线约束

这是一个 Python 版无人机多源融合定位 / ESKF 离线事后处理项目。

必须继续维护现有 Python 项目。

禁止：

- 把项目改成 C++；
- 创建 CMakeLists.txt；
- 创建 `.cpp` / `.hpp` 文件；
- 新建 `include/` / `src/` C++ 结构；
- 用 Eigen 替代 numpy；
- 重写整个项目；
- 删除现有 Python 工程结构；
- 大范围改目录名；
- 为了“工程化”推翻当前代码；
- 把滤波项目改成因子图项目；
- 把当前项目改成 VINS / LIO / SLAM；
- 只写说明不推进代码；
- 只做一个很小的改动就结束任务；
- 长期停留在“修一行、测一下、记一下”的小修状态。

如果用户说“继续做”“推进主线”“构建项目”“完善项目”，默认含义是：

> 按项目最终目标和最近工作记录，继续推进最低完整多源融合导航系统，而不是随便选任务、不是换 C++、不是只写文档、不是做极小修补。

---

## 3. 当前仓库结构认知

当前仓库核心路径如下：

```text
02_src/eskf_stack/
  app.py
  config.py

  core/
    filter.py
    state.py
    mechanization.py
    navigation.py
    initialization.py
    math_utils.py

  measurements/
    base.py
    manager.py
    gnss_position.py
    gnss_velocity.py
    barometer.py
    mag_yaw.py

  adapters/
    csv_dataset.py
    demo_generator.py
    great_msf_dataset.py
    dx_decoded_dataset.py
    dx_external_solution_dataset.py
    imu_transform.py

  analysis/
    evaluator.py
    plotter.py
    quality.py
    state_machine.py

05_tests/
  test_app.py
  test_navigation_core.py
  test_initialization.py
  test_measurements.py
  test_adapters.py
```

当前核心类是：

```text
OfflineESKF
```

主要位于：

```text
02_src/eskf_stack/core/filter.py
```

核心状态位于：

```text
02_src/eskf_stack/core/state.py
```

当前误差状态维度为 15 维：

```text
dx = [dp, dv, dtheta, dbg, dba]
```

不要擅自改成 18 维。除非用户明确要求，否则不要加入重力误差状态。

不要重复创建：

```text
eskf.py
state.py
main.py
dataset.py
plot.py
```

如果已有功能已经存在，应在现有文件中增强，而不是另起一套平行实现。

---

## 4. 阶段路线图

Codex 每次推进前，都要判断本轮属于哪个阶段。项目不是随机游走，而是按阶段推进。

### 阶段 1：最低完整项目闭环

目标：

- 数据能进来；
- ESKF 能初始化；
- predict 能稳定运行；
- GNSS position / velocity 能更新；
- pipeline 能输出 `fusion_output.csv`；
- metrics 和图能生成；
- 测试能通过；
- 工作记录能追踪。

### 阶段 2：稳定性和退化处理

目标：

- 观测异常有拒绝策略；
- NIS / innovation 指标进入 metrics；
- GNSS 短时中断有模式记录；
- 协方差健康度有输出；
- 质量评分和状态机逻辑稳定；
- 结果目录和实验输出可控。

### 阶段 3：多源融合能力补强

目标：

- barometer update 稳定；
- mag yaw update 稳定；
- 多源观测 availability / used / rejected 统计完整；
- 不同传感器退化时能输出清晰指标；
- 配置项和说明完善。

### 阶段 4：实验与对比能力

目标：

- 支持 baseline / adaptive / rejection / full method 实验；
- 支持批处理实验；
- 支持 key summary / category summary；
- 支持 metrics 对比；
- 支持工作记录追踪实验目的、配置和结果。

### 阶段 5：后续扩展准备

目标：

- 在不破坏主线的情况下，为后续新传感器、新数据格式、新鲁棒方法预留接口；
- 不主动切到视觉、激光、因子图；
- 等最低完整系统成熟后再扩展。

每轮任务必须说明本轮属于哪个阶段，以及如何推进该阶段目标。

---

## 5. 连续推进规则

每轮任务不能随便选。

每轮开始必须：

1. 读取最近的 `YYYY-MM-DD_工作记录.txt`；
2. 找到最后一条记录中的“八、下一步”；
3. 根据“下一步”确定本轮任务；
4. 如果“下一步”有多个事项，必须优先选择同一主题下至少 3 个事项一起推进；
5. 如果“下一步”过于含糊，则结合阶段路线图，选择最能推进最低完整项目目标的任务；
6. 本轮必须说明：接续了上一轮哪几条“下一步”。

禁止：

- 跳到无关模块；
- 随机挑一个容易改的点；
- 只完成“下一步”中的一个极小事项；
- 把任务切碎到没有实质推进；
- 为了快速结束只做表层输出修补；
- 把小修补包装成大推进。

---

## 6. 任务粒度硬性要求

任务不能切得过小。

每轮最低要求：

1. 至少推进同一主题下 3 个相关事项；
2. 至少完成 3 个具体改动点；
3. 至少涉及主代码 + 测试两个层面；
4. 必要时同时涉及配置、分析输出、运行入口或说明；
5. 必须运行相关测试；
6. 必须追加当天工作记录；
7. 必须说明本轮推进了哪个阶段目标。

不允许一轮只做：

- 只改一个变量名；
- 只补一行注释；
- 只检查一个文件；
- 只写计划不改代码；
- 只跑一个命令不处理结果；
- 只新建或修改工作记录而不推进代码；
- 只修一个输出文案；
- 只补一个很小测试；
- 只改一行代码；
- 只完成一个微小的“下一步”。

如果发现当前事项只需要很小改动，不能直接结束任务。必须继续检查同一主题下相关问题，把本轮扩展到至少 3 个相关事项。

推荐的单轮任务范围：

- 修复一个测量更新链路：`measurements/base.py` + `manager.py` + 对应测量模型 + 测试 + 工作记录；
- 完善一次初始化逻辑：`app.py` + `initialization.py` + `test_initialization.py` + 工作记录；
- 修复一次数据适配问题：对应 adapter + config + 测试 + 工作记录；
- 优化一次主流程输出：`app.py` + `analysis/` + 测试 + 工作记录；
- 完成一次 GNSS 退化/异常处理改进：measurement manager + quality/state_machine + tests + 工作记录；
- 梳理一次 pipeline 运行问题：入口命令 + 配置 + 输出路径 + 测试 + 工作记录；
- 推进一次实验批处理能力：experiment_batch + run_experiment_batch + metrics/evaluator + 对应测试 + 工作记录。

也不要一次性做过大范围修改。

禁止一轮任务同时做：

- 重写核心 ESKF；
- 重写全部 adapters；
- 改完所有 measurements；
- 加入新算法路线；
- 大范围改目录结构；
- 同时做滤波、因子图、视觉、激光；
- 批量运行大量实验后留下大量结果目录。

原则：

> 每轮任务要有明确推进感。不能长期停留在很小很小的修补；每轮至少推进 3 个相关事项，服务于最低完整多源融合导航系统目标。

---

## 7. 工作分类与优先查看文件

每次任务开始时，必须判断任务属于哪一类。

### A. ESKF 核心算法

优先查看：

```text
02_src/eskf_stack/core/filter.py
02_src/eskf_stack/core/state.py
02_src/eskf_stack/core/mechanization.py
02_src/eskf_stack/core/navigation.py
02_src/eskf_stack/core/math_utils.py
```

### B. 测量更新

优先查看：

```text
02_src/eskf_stack/measurements/base.py
02_src/eskf_stack/measurements/manager.py
02_src/eskf_stack/measurements/gnss_position.py
02_src/eskf_stack/measurements/gnss_velocity.py
02_src/eskf_stack/measurements/barometer.py
02_src/eskf_stack/measurements/mag_yaw.py
```

### C. 数据读取 / 数据适配

优先查看：

```text
02_src/eskf_stack/adapters/csv_dataset.py
02_src/eskf_stack/adapters/demo_generator.py
02_src/eskf_stack/adapters/great_msf_dataset.py
02_src/eskf_stack/adapters/dx_decoded_dataset.py
02_src/eskf_stack/adapters/dx_external_solution_dataset.py
02_src/eskf_stack/adapters/imu_transform.py
```

### D. 主流程运行

优先查看：

```text
02_src/eskf_stack/app.py
02_src/eskf_stack/config.py
01_data/config.json
```

### E. 评估 / 实验 / 输出

优先查看：

```text
02_src/eskf_stack/analysis/evaluator.py
02_src/eskf_stack/analysis/experiment_batch.py
02_src/eskf_stack/analysis/plotter.py
02_src/run_experiment_batch.py
```

### F. 测试

优先查看：

```text
05_tests/
```

### G. 工作记录

优先查看或创建：

```text
YYYY-MM-DD_工作记录.txt
```

工作记录文件名必须统一使用这种格式：

```text
2026-04-21_工作记录.txt
2026-04-22_工作记录.txt
2026-04-23_工作记录.txt
2026-04-24_工作记录.txt
```

不要再新建其他命名风格的工作记录，例如：

```text
今日工作总结.txt
工作记录草稿.txt
项目临时说明.txt
Codex记录.md
```

如果仓库里当天已经有 `YYYY-MM-DD_工作记录.txt`，必须继续追加到该文件。
如果当天没有，必须新建该文件。
不要为了同一天多次任务创建多个记录文件。

---

## 8. 修改优先级

优先级从高到低：

1. 保证现有测试能跑；
2. 保证 `run_pipeline()` 能跑；
3. 保证 `OfflineESKF.predict()` 稳定；
4. 保证 GNSS position update 稳定；
5. 保证 GNSS velocity update 稳定；
6. 保证初始化逻辑稳定；
7. 保证输出 `fusion_output.csv`、figures、metrics；
8. 保证实验批处理和结果汇总可用；
9. 保证多源观测 used / available / rejected / NIS 指标完整；
10. 保证 `YYYY-MM-DD_工作记录.txt` 能追踪每轮任务；
11. 再改进 barometer / mag yaw；
12. 再做质量评估、模式切换、退化检测；
13. 最后才考虑新传感器或新算法。

不要跳过 1-10 去做高级功能。

---

## 9. 工作记录强制规则

每次完成一个任务，必须追加当天工作记录文件。

文件名格式必须严格统一为：

```text
YYYY-MM-DD_工作记录.txt
```

例如：

```text
2026-04-24_工作记录.txt
```

记录必须追加到项目根目录下的当天文件。

如果当天文件不存在，必须新建。
如果当天文件已存在，必须在文件末尾追加新条目。
不要覆盖当天已有记录。

每轮记录格式必须保持一致，使用下面模板：

```text
============================================================
时间：YYYY-MM-DD HH:MM
任务：本轮任务标题
阶段：阶段 1 / 阶段 2 / 阶段 3 / 阶段 4 / 阶段 5
接续上一轮：说明接续了上一轮“下一步”中的哪几条

一、本轮目标
- ...

二、查看文件
- ...

三、修改文件
- ...

四、完成内容
1. ...
2. ...
3. ...
至少写 3 项，必须是具体推进，不要写空话。

五、验证情况
- 验证命令：
  ...
- 验证结果：
  通过 / 未通过
- 关键输出：
  ...

六、结果目录处理
- 是否生成 03_results_*：
- 是否清理：
- 是否保留：
- 保留原因：

七、当前问题
- ...

八、下一步
1. ...
2. ...
3. ...
下一步至少写 3 项，保持连续推进。
============================================================
```

工作记录必须回答：

- 为什么做；
- 接续了上一次哪几条下一步；
- 属于哪个阶段；
- 改了哪些文件；
- 具体做了什么；
- 怎么验证；
- 结果如何；
- 有没有生成临时结果；
- 下一步至少 3 个事项。

工作记录不要写成论文，也不要写空泛口号。

---

## 10. 相关说明文件更新规则

除了工作记录外，如果本轮改动影响了使用方法、配置项、输入输出字段、运行命令或模块含义，必须同步更新相关说明文件。

允许修改：

```text
README.md
docs/
*.md
配置说明
数据格式说明
模块说明
```

但必须满足：

- 与本轮代码或测试工作直接相关；
- 内容简洁；
- 明确说明“怎么用、改了什么、注意什么”；
- 不写大段空泛理论；
- 不写与当前任务无关的项目愿景；
- 不把主要时间花在润色文字上。

工作记录是每次任务都必须更新；README/docs 只有相关时才更新。

本文件 `AGENTS.md` 是 Codex 行为规则文件。除非用户要求修改提示词规则，否则不要修改它。

---

## 11. 禁止过度扩展

当前项目的目标是先形成最低完整多源融合导航系统，不是马上追求所有高级路线。

禁止主动加入：

- camera；
- LiDAR；
- UWB；
- RTK ambiguity；
- factor graph；
- GTSAM；
- Ceres；
- VINS；
- LIO；
- SLAM；
- loop closure；
- map optimization。

已有的 barometer / mag yaw 可以维护和完善，但不要让它们抢占主线。

主线仍然是：

```text
IMU predict + GNSS position/velocity update + 可选 baro/mag + 输出轨迹 + metrics + 实验汇总 + 工作记录
```

---

## 12. 结果目录管理

可以运行单元测试，也可以在必要时运行 pipeline 或实验脚本，但不能无故批量跑实验。

禁止在项目根目录长期保留无关的新生成结果目录，例如：

```text
03_results_*
临时实验输出目录
大量临时图片
大量临时 csv
临时 metrics
```

如果为了验证运行 pipeline 生成了结果目录，任务结束前应清理临时输出，除非用户明确要求保留。

如果本轮实验结果需要保留，必须在工作记录中说明：

- 为什么保留；
- 保留在哪个目录；
- 对应配置是什么；
- 结果用于什么目的。

如果只是为了验证代码能跑，则运行结束后清理结果目录。

---

## 13. 测试规则

每次修改核心代码后，优先运行：

```bash
python -m unittest discover -s 05_tests
```

如果只改某个模块，可以运行对应测试，例如：

```bash
python -m unittest 05_tests.test_measurements
python -m unittest 05_tests.test_navigation_core
python -m unittest 05_tests.test_initialization
python -m unittest 05_tests.test_app
```

如果测试失败，必须优先修测试失败原因，不要继续加功能。

不要为了让测试通过而随便删除测试。

如果运行测试或 pipeline 产生临时输出，按“结果目录管理”规则处理。

---

## 14. 运行规则

项目主流程围绕：

```text
02_src/eskf_stack/app.py
```

核心函数是：

```text
run_pipeline(config_path=None)
```

如果需要运行完整流程，优先使用现有入口方式。

如果没有清晰命令，可以使用类似：

```bash
PYTHONPATH=02_src python -c "from eskf_stack.app import main; main()"
```

或者：

```bash
PYTHONPATH=02_src python -c "from eskf_stack.app import run_pipeline; run_pipeline()"
```

不要新建一套独立入口，除非现有入口确实无法运行。

---

## 15. 每次任务开始时的固定流程

每次开始工作时，先做以下事情：

1. 明确本轮属于哪个阶段；
2. 读取最近的 `YYYY-MM-DD_工作记录.txt`；
3. 找到最后一条记录中的“八、下一步”；
4. 确认本轮接续上一轮哪 3 个相关事项；
5. 如果上一轮下一步不足 3 个，则补充同一主题下最能推进阶段目标的事项；
6. 列出准备查看的关键文件；
7. 修改必要的 `.py` / 配置 / 测试 / 相关说明文件；
8. 不改 C++；
9. 不做无关重构；
10. 修改后运行测试或明确说明无法运行的原因；
11. 清理无关临时结果；
12. 追加 `YYYY-MM-DD_工作记录.txt`；
13. 最终详细汇报。

---

## 16. 每次完成后的回复要求

每次完成任务后，必须详细回复。用户需要阅读 Codex 的记录，不能只写一句“完成了”。

回复必须使用下面格式：

```text
本轮阶段：
- 阶段 X：...

接续上一轮：
- 接续了上一轮“下一步”中的：
  1. ...
  2. ...
  3. ...

本轮目标：
- ...

本轮查看的关键文件：
- ...

本轮修改的文件：
- ...

具体完成内容：
1. ...
2. ...
3. ...
至少 3 项，必须是具体推进。

为什么这算一次有效推进：
- ...

工作记录更新：
- 文件：YYYY-MM-DD_工作记录.txt
- 写入内容摘要：
  - ...
  - ...
  - ...

验证方式：
```bash
...
```

验证结果：
- 通过 / 未通过
- 关键结果：
- 如果未通过，说明具体错误和下一步修复方向

结果目录处理：
- 是否生成 03_results_* 或其他临时结果：
- 是否已清理：
- 如果保留，保留原因：

当前主线状态：
- 初始化：
- predict：
- GNSS position update：
- GNSS velocity update：
- 多源观测：
- 输出结果：
- metrics / 实验汇总：
- 测试：
- 工作记录：

下一步建议：
1. ...
2. ...
3. ...
```

要求：

- 说明要具体；
- 必须说清楚“做了什么”和“下一步做什么”；
- 不要写空泛理论；
- 不要用一句话糊弄；
- 不要省略工作记录更新情况；
- 不要把小修补包装成大推进。

---

## 17. 如果用户说“继续”

当用户只说：

```text
继续
接着做
往下推
继续主线
```

默认执行：

1. 读取最近工作记录；
2. 找到最后一条“下一步”；
3. 接续其中至少 3 个相关事项；
4. 确认本轮属于哪个阶段；
5. 围绕该主题完成代码、测试、工作记录的闭环；
6. 清理无关临时结果；
7. 详细汇报结果；
8. 给出下一轮至少 3 个下一步事项。

不要重新写大规划。

不要改 C++。

不要只写文档不改代码。

不要只做一个极小修改就结束。

---

## 18. 如果用户说“根据仓库整体看看”

重点回答：

- 当前项目是不是能运行；
- 当前 ESKF 主线是否完整；
- `OfflineESKF.predict()` 是否合理；
- `MeasurementManager` 是否稳定；
- 初始化是否会卡住；
- 数据适配是否清晰；
- 输出结果是否可靠；
- metrics / 实验汇总是否足够支撑完整项目；
- 测试覆盖是否够；
- 工作记录是否能追踪项目进展；
- 当前距离“最低完整多源融合导航系统”还缺什么；
- 下一步最应该做哪 3 个相关事项。

不要泛泛介绍 ESKF 理论。

---

## 19. 如果用户说“修 bug”

必须遵守：

1. 先复现或定位错误；
2. 找到相关模块；
3. 做一组至少 3 个相关的修复 / 测试 / 验证事项；
4. 不重构无关文件；
5. 不顺手加无关新功能；
6. 改完跑对应测试；
7. 清理无关临时结果；
8. 追加当天工作记录；
9. 详细说明错误原因、修复方式和下一步。

---

## 20. 如果用户说“加功能”

先判断功能属于哪个已有模块：

- 新测量模型：放 `measurements/`；
- 新数据格式：放 `adapters/`；
- 新评估指标：放 `analysis/`；
- ESKF 状态/传播：放 `core/`；
- 主流程调度：改 `app.py`；
- 参数：改 `config.py` 和 `01_data/config.json`；
- 功能说明 / 使用说明：按需更新 README、docs；
- 工作过程：必须追加 `YYYY-MM-DD_工作记录.txt`。

新增功能必须包含：

- 相关代码；
- 必要配置；
- 必要测试；
- 必要工作记录；
- 验证命令；
- 结果目录处理；
- 下一步至少 3 个事项。

不要为了一个小功能重写整体 pipeline。

---

## 21. 推荐单次任务提示词

用户每次让 Codex 推进项目时，可以使用：

```text
按照 AGENTS.md 执行。

这是现有 Python ESKF-test 仓库，不是从零项目。
不要改成 C++。
不要新建 CMake。
不要重写项目结构。

本轮不是随机选任务，也不是小修小补。
请先读取最近的 YYYY-MM-DD_工作记录.txt，找到最后一条“八、下一步”，并接续其中至少 3 个相关事项推进。

本轮要求：
1. 明确本轮属于哪个阶段；
2. 至少推进 3 个同一主题下的相关事项；
3. 至少完成 3 个具体改动点；
4. 必须涉及主代码和测试；
5. 必要时同步配置、说明或实验汇总；
6. 运行相关测试；
7. 如果生成临时 03_results_*，任务结束前清理，除非说明保留原因；
8. 必须追加今天的 YYYY-MM-DD_工作记录.txt；
9. 最终详细说明接续了什么、完成了什么、为什么算有效推进、验证结果、工作记录写了什么、下一步 3 个事项。

如果本轮只需要改一行或一个小点，请不要结束，继续检查同一主题下相关问题，扩展到至少 3 个相关事项。
```

---

## 22. 如果模型任务太小

用户可以直接发送：

```text
任务粒度太小，停止这种小修小补。

请按照 AGENTS.md 的“任务粒度硬性要求”，回到最近工作记录的“下一步”，围绕同一主题至少推进 3 个相关事项。
本轮必须包含：
- 至少 3 个具体改动点；
- 主代码修改；
- 测试修改或测试补充；
- 相关测试运行；
- 临时结果处理；
- 追加 YYYY-MM-DD_工作记录.txt；
- 详细说明下一步至少 3 个事项。

如果当前事项只需要一行代码，请继续扩展同一模块下的相关问题，不允许直接结束。
```

---

## 23. 如果模型开始乱写文档

用户可以直接发送：

```text
停止纯文档方向。

本项目当前要推进 Python ESKF 代码主线。
允许更新必要工作记录，但禁止只写说明不改代码。
禁止改成 C++。
禁止重写项目。

请回到现有仓库主线：
- OfflineESKF
- app.py run_pipeline
- measurements
- adapters
- analysis / metrics
- 05_tests
- YYYY-MM-DD_工作记录.txt

本轮完成一个中等偏完整任务：至少 3 个相关事项 + 代码修改 + 测试 + 工作记录 + 临时结果清理 + 详细汇报。
```

---

## 24. 回答风格

用户是研一学生，方向是无人机多源融合定位。

回答要：

- 直接；
- 工程导向；
- 详细说明本轮做了什么；
- 明确说明下一步做什么；
- 不能突然换技术路线；
- 不能把 Python 改成 C++；
- 不能把滤波项目改成因子图；
- 不能把已有仓库说成从零项目；
- 不能用小修小补糊弄项目推进；
- 优先告诉用户现在应该改哪个文件、跑哪个命令、看哪个结果、记录到哪里。

不要用大段空泛表述，例如：

- “本项目具有重要意义”；
- “未来可以广泛扩展”；
- “建议系统性规划”；
- “多源融合具有广阔前景”。

最终原则：

> 以最低完整多源融合导航系统为最终目标；每轮接续最近工作记录；至少推进 3 个相关事项；代码、测试、工作记录一起闭环；清理无关结果；详细说明做了什么和下一步；基于现有 Python 仓库继续做，不换路线。
