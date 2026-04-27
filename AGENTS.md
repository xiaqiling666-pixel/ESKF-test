# AGENTS.md — ESKF-test 主线推进版

## 0. 放置位置

把本文件放到本地 Codex 打开的项目根目录，并命名为：

```text
AGENTS.md
```

推荐目录结构：

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

不要放到 `02_src/`、`05_tests/` 或桌面其他位置。

---

## 1. 项目最终目标

本项目最终目标是形成一个**完整、成熟、可验证的 Python 离线多源融合导航系统**。

最低完整系统应具备：

- 能稳定读取 IMU / GNSS / barometer / magnetometer yaw 等数据；
- 能完成 ESKF 初始化、IMU predict、误差状态传播和测量更新；
- 能融合 GNSS position、GNSS velocity，并按需融合 barometer / magnetometer yaw；
- 能输出融合轨迹、状态量、协方差、质量指标、误差统计；
- 能对 GNSS 中断、异常观测、退化场景做基础处理；
- 能通过测试验证核心链路；
- 能进行实验对比和结果汇总；
- 能留下简洁工作记录，方便后续复盘。

当前重点：

> 先把最低完整多源融合导航系统跑通、跑稳、能评估。不要陷入 CLI、manifest、preview、字段展示、记录格式等边角细节。

---

## 2. 技术路线约束

这是 Python 版 ESKF 离线融合项目。

禁止：

- 改成 C++；
- 新建 CMake；
- 新建 `.cpp` / `.hpp`；
- 重写整个项目结构；
- 把项目改成因子图 / VINS / LIO / SLAM；
- 删除现有 Python 工程主线；
- 为了格式、文档、输出展示反复小修。

允许：

- 在现有 `02_src/eskf_stack/` 结构内增强功能；
- 小范围重构当前主线模块；
- 增加必要测试；
- 简洁更新工作记录；
- 必要时更新 README / docs，但文档不能成为主任务。

---

## 3. 当前工程主线

核心代码路径：

```text
02_src/eskf_stack/
  app.py
  config.py
  core/
  measurements/
  adapters/
  analysis/
05_tests/
```

核心类：

```text
OfflineESKF
```

核心文件：

```text
02_src/eskf_stack/core/filter.py
02_src/eskf_stack/core/state.py
```

当前误差状态维度为 15 维：

```text
dx = [dp, dv, dtheta, dbg, dba]
```

不要擅自改成 18 维。不要主动加入重力误差状态，除非用户明确要求。

不要重复创建：

```text
eskf.py
state.py
main.py
dataset.py
plot.py
```

已有功能应在现有模块中增强，不要另起一套平行实现。

---

## 4. 阶段路线图

### 阶段 1：最低完整闭环

- 数据能进入 pipeline；
- ESKF 能初始化；
- IMU predict 稳定；
- GNSS position / velocity 更新稳定；
- 能输出 `fusion_output.csv`；
- metrics 和图能生成；
- 基础测试通过。

### 阶段 2：稳定性和退化处理

- 有 NIS / innovation / rejected / used 统计；
- GNSS 中断、异常观测、退化场景有基础记录；
- quality / covariance health / state machine 稳定。

### 阶段 3：多源融合补强

- barometer update 稳定；
- mag yaw update 稳定；
- 多源观测 availability / used / rejected 统计完整；
- 配置项和输入输出字段清晰。

### 阶段 4：实验与对比

- baseline / adaptive / rejection / full method 可以对比；
- batch 实验能跑；
- metrics 汇总能支撑结果分析。

### 阶段 5：后续扩展

- 为新传感器、新数据格式、新鲁棒方法预留接口；
- 不主动切到视觉、激光、因子图。

---

## 5. 主线推进原则

每轮任务只推进**一个主线模块**，不要同时散开太多方向。

主线模块包括：

- 数据读取 / adapter / 时间戳处理；
- 初始化 / bootstrap；
- ESKF predict / covariance / error-state；
- GNSS position / velocity update；
- barometer / mag yaw update；
- MeasurementManager / NIS / rejection / recovery；
- app.py pipeline；
- metrics / quality / state machine；
- 实验对比能力。

一轮任务是否合格，看的是：

- 是否让系统能力更完整；
- 是否能运行或测试；
- 是否减少主线风险；
- 是否更接近最低完整多源融合导航系统。

不以“改了多少文件、多少行、多少测试点”作为主要标准。

如果只改 1-2 个文件但实质推进了主线，可以接受。

如果改了很多文件但只是 CLI、manifest、preview、格式、工作记录，不算主线推进。

---

## 6. 禁止用边角任务冒充主线

以下内容可以顺手做，但不能作为本轮主任务：

- CLI 输出多打印几行；
- manifest / preview / 字段名小修；
- 工作记录格式修补；
- README / 文档润色；
- 测试断言位置调整；
- 常量名整理；
- 空白、换行、格式化；
- 只增加一个展示字段；
- 只修一个输出文案。

如果最近工作记录的“下一步”只是这些边角内容，应上升到同阶段的主线模块目标。

---

## 7. 每轮任务流程

每轮开始时：

1. 快速读取最近的 `YYYY-MM-DD_工作记录.txt`；
2. 看最后的“下一步”；
3. 判断它是否是主线模块；
4. 如果偏小或偏边角，就提升为同阶段主线目标；
5. 只查看和本轮模块直接相关的文件；
6. 不全仓库扫描；
7. 不写长篇计划；
8. 直接修改代码；
9. 跑相关测试；
10. 简洁追加工作记录；
11. 简洁汇报。

---

## 8. 测试规则

日常任务只运行相关测试，减少消耗。

常用测试：

```bash
python -m unittest 05_tests.test_measurements
python -m unittest 05_tests.test_navigation_core
python -m unittest 05_tests.test_initialization
python -m unittest 05_tests.test_app
python -m unittest 05_tests.test_experiment_batch
```

阶段收口、提交前、涉及大范围修改时，再运行完整测试：

```bash
python -m unittest discover -s 05_tests
```

不要为了让测试通过而删除测试。

---

## 9. 运行规则

主流程围绕：

```text
02_src/eskf_stack/app.py
```

核心函数：

```text
run_pipeline(config_path=None)
```

常用运行方式：

```bash
PYTHONPATH=02_src python -c "from eskf_stack.app import main; main()"
```

或：

```bash
PYTHONPATH=02_src python -c "from eskf_stack.app import run_pipeline; run_pipeline()"
```

不要新建一套独立入口，除非现有入口无法满足任务。

---

## 10. 结果目录管理

可以运行测试或必要 pipeline，但不要无故批量跑实验。

禁止长期保留无关临时结果目录：

```text
03_results_*
临时实验输出目录
大量临时图片
大量临时 csv
临时 metrics
```

如果只是验证代码能跑，任务结束前清理临时结果。

如果实验结果需要保留，工作记录里说明原因、目录、配置和用途。

---

## 11. 工作记录规则

每次主线推进任务完成后，简洁追加当天工作记录。

文件名：

```text
YYYY-MM-DD_工作记录.txt
```

记录模板：

```text
============================================================
时间：YYYY-MM-DD HH:MM
任务：
阶段：
主线模块：

一、本轮目标
- ...

二、修改文件
- ...

三、完成内容
1. ...
2. ...
3. ...

四、验证情况
- 命令：
- 结果：

五、结果目录处理
- 是否生成 03_results_*：
- 是否清理：
- 是否保留及原因：

六、系统能力提升
- ...

七、下一步
1. ...
2. ...
3. ...
============================================================
```

工作记录要简洁，不要写成长报告。

小 bug、小字段、小路径修复可以不写工作记录，除非用户明确要求。

---

## 12. 三种工作模式

### A. 轻量修复模式

用于小 bug、小报错、小字段问题。

规则：

- 不读取长工作记录；
- 不做模块闭环；
- 不跑全量测试；
- 不追加工作记录，除非用户要求；
- 只修当前问题；
- 只运行最相关测试。

### B. 省额度主线推进模式

日常最常用。

规则：

- 快速读最近工作记录；
- 选一个主线模块；
- 只看相关文件；
- 做一个可运行、可测试的模块改进；
- 运行相关测试；
- 简洁追加工作记录；
- 简洁说明系统能力提升和下一步。

### C. 阶段收口模式

用于准备提交、阶段总结、实验整理。

规则：

- 可以运行完整测试；
- 可以检查 pipeline / metrics / 输出；
- 可以整理文档；
- 可以补齐工作记录；
- 可以做阶段性总结。

---

## 13. 用户常用命令

### 日常主线推进

```text
按照 AGENTS.md 执行，本轮使用省额度主线推进模式。

不要凑文件数量和改动数量。本轮只围绕一个主线模块完成可运行、可测试的实质推进。
先快速读取最近的 YYYY-MM-DD_工作记录.txt；如果最后“下一步”偏小或偏边角，就上升到同阶段的主线模块目标。

优先围绕：数据读取、ESKF predict、GNSS update、baro/mag 融合、metrics 评估、退化处理、pipeline 跑通或实验对比能力。

要求：只查看相关文件，不全仓库扫描；不做 CLI/manifest/preview/格式/工作记录这种边角任务作为主任务；只运行相关测试；简洁追加工作记录；最后说明系统能力提升和下一步。

不要改 C++，不要重写结构。
```

### 轻量修复

```text
按照 AGENTS.md 执行，本轮使用轻量修复模式。

只修当前问题，只查看相关文件，只运行最相关测试。
不要读长工作记录，不做模块闭环，不追加工作记录，不跑全量测试。
最后简短说明改了什么和测试结果。
```

### 阶段收口

```text
按照 AGENTS.md 执行，本轮使用阶段收口模式。

检查当前阶段的代码、测试、pipeline、metrics、输出和工作记录。
可以运行完整测试，可以补齐必要说明，但不要引入新路线。
最后说明当前阶段完成度、遗留问题和下一阶段任务。
```

---

## 14. 如果模型任务太小

用户可以直接发送：

```text
任务量还是太小，停止这种小修小补。

不要围绕 CLI、manifest、preview、格式、字段展示或工作记录打转。
请回到最低完整多源融合导航系统目标，选择一个主线模块做可运行、可测试的推进。

只看相关文件，直接改主线代码，运行相关测试，简洁追加工作记录，并说明系统能力提升。
```

---

## 15. 回答风格

用户是研一学生，方向是无人机多源融合定位。

回答要：

- 直接；
- 工程导向；
- 说明本轮做了什么；
- 说明下一步做什么；
- 不突然换技术路线；
- 不把 Python 改成 C++；
- 不把滤波项目改成因子图；
- 不把已有仓库说成从零项目；
- 不用小修小补糊弄项目推进。

不要用大段空泛表述，例如：

- “本项目具有重要意义”；
- “未来可以广泛扩展”；
- “建议系统性规划”；
- “多源融合具有广阔前景”。

最终原则：

> 以最低完整多源融合导航系统为最终目标；日常使用省额度主线推进模式；每轮围绕一个主线模块做可运行、可测试的实质推进；不按文件数量和改动数量凑任务；少读无关文件，少写废话，少跑无关测试；基于现有 Python 仓库继续做，不换路线。
