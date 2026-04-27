# AGENTS.md — ESKF-test 本地 Codex 项目规则

## 0. 放置位置

把本文件放到你本地 Codex 打开的项目根目录，例如：

```text
ESKF-test/
  AGENTS.md
  01_data/
  02_src/
  05_tests/
```

不要放到桌面随便一个位置，不要放到 `02_src/` 或 `05_tests/` 里面。

它的作用是约束 Codex 在当前本地项目里的工作方式。

---

## 1. 仓库定位

这是一个 Python 版无人机多源融合定位 / ESKF 离线事后处理项目。

当前仓库已经有完整工程雏形，不是空项目，不要从零重写。

当前主线是：

- Python ESKF；
- 离线飞行数据 / 模拟数据处理；
- IMU mechanization / predict；
- GNSS position update；
- GNSS velocity update；
- 可选 barometer update；
- 可选 magnetometer yaw update；
- 输出融合轨迹、误差图、质量评估和 metrics；
- 通过 `05_tests/` 中的单元测试保证核心逻辑稳定；
- 每轮工作结束后，保留必要的文字记录，说明本轮做了什么、改了哪些文件、验证结果如何、下一步是什么。

当前阶段不是：

- C++ 重写；
- CMake 工程；
- 飞控底层移植；
- 因子图优化；
- VINS / LIO / SLAM；
- 纯论文文档整理阶段。

最重要原则：

> 当前仓库的方向是 Python 离线 ESKF 工程验证。不要改变路线，只沿着现有代码主线持续推进。每轮任务要形成一个相对完整的小闭环：代码修改 + 必要测试 + 相关文字记录。

---

## 2. 最高优先级规则

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
- 把当前项目改成 VINS / LIO / SLAM。

如果用户说“继续做”“推进主线”“构建项目”“完善项目”，默认含义是：

> 在现有 Python 仓库基础上继续推进 ESKF 可运行主线，同时更新相关文字记录；不是重写项目，不是换 C++，也不是只写说明不改代码。

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

修改代码前必须优先理解这些已有文件的职责。

不要重复创建：

```text
eskf.py
state.py
main.py
dataset.py
plot.py
```

如果已有功能已经存在，应在现有文件中小步增强，而不是另起一套平行实现。

---

## 4. 当前 ESKF 主线

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

当前状态包括 position、velocity、quaternion、gyro_bias、accel_bias、covariance P。

后续修改必须保持这些状态定义和维度一致。

---

## 5. 工作分类与优先查看文件

每次任务开始时，必须先判断任务属于哪一类。

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

### E. 测试

优先查看：

```text
05_tests/
```

### F. 文字记录 / 说明更新

优先查看：

```text
README.md
docs/
工作记录.txt
YYYY-MM-DD_工作记录.txt
YYYY-MM-DD_今日工作总结.txt
项目总结.txt
```

如果仓库已有当天工作记录文件，优先更新当天文件。
如果没有当天工作记录文件，可以新建 `YYYY-MM-DD_工作记录.txt`。
不要为了记录工作新建一堆零散说明文件。

不要全仓库无目的扫描。

---

## 6. 本项目不是“从零实现 ESKF”

禁止做这些事：

- 新建一个完全独立的 `eskf.py` 来替代现有 `OfflineESKF`；
- 新建一个完全独立的 `main.py` 来绕过 `app.py`；
- 新建一套新的状态定义来绕过 `NavState`；
- 新建一套新的 measurement manager；
- 把现有 pipeline 全部删掉重来；
- 把现有 baro / mag / gnss velocity 删除；
- 把当前项目降级成简单 demo；
- 把当前 Python 工程改成 C++ 工程。

正确做法是：

> 在现有 `OfflineESKF`、`MeasurementManager`、`app.py run_pipeline()` 的基础上，按阶段做中等粒度的功能推进，并同步更新必要文字记录。

---

## 7. 任务粒度要求

任务不能切得过小。

不要一轮只做：

- 只改一个变量名；
- 只补一行注释；
- 只检查一个文件；
- 只写计划不改代码；
- 只跑一个命令不处理结果。

每轮任务应尽量形成一个中等粒度闭环，通常包括：

1. 明确一个小主题；
2. 查看相关 3-8 个文件；
3. 完成 1-3 个相关代码修改点；
4. 必要时同步配置或测试；
5. 运行相关测试或完整测试；
6. 清理临时输出；
7. 更新相关文字记录；
8. 给出下一步任务。

推荐的单轮任务范围示例：

- 修复一个测量更新链路：`measurements/base.py` + `manager.py` + 对应测量模型 + 测试；
- 完善一次初始化逻辑：`app.py` + `initialization.py` + `test_initialization.py` + 工作记录；
- 修复一次数据适配问题：对应 adapter + config + 测试 + 记录；
- 优化一次主流程输出：`app.py` + `analysis/` + 测试 + 记录；
- 完成一次 GNSS 退化/异常处理改进：measurement manager + quality/state_machine + tests + 记录。

但也不要一次性做过大范围修改。

禁止一轮任务同时做：

- 重写核心 ESKF；
- 重写全部 adapters；
- 改完所有 measurements；
- 加入新算法路线；
- 大范围改目录结构；
- 同时做滤波、因子图、视觉、激光。

原则：

> 每轮工作量要比“只改一个小点”更完整，但仍然控制在一个清晰主题内。

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
8. 保证相关文字记录能反映当前项目进展；
9. 再改进 barometer / mag yaw；
10. 再做质量评估、模式切换、退化检测；
11. 最后才考虑新传感器或新算法。

不要跳过 1-8 去做高级功能。

---

## 9. 文字说明与工作记录规则

每次完成一个任务，必须同步更新相关文字说明。

这不是要求写长篇理论文档，而是要求保留可追踪的工程记录。

每轮结束至少更新以下一种：

### A. 工作记录

优先更新当天工作记录文件，例如：

```text
YYYY-MM-DD_工作记录.txt
YYYY-MM-DD_今日工作总结.txt
```

如果当天没有记录文件，可以新建：

```text
YYYY-MM-DD_工作记录.txt
```

记录内容建议包含：

```text
## 本轮工作：简短标题

时间：
目标：
修改文件：
完成内容：
验证命令：
验证结果：
生成/清理的结果目录：
当前问题：
下一步：
```

### B. 功能说明

如果本轮改动改变了某个模块的用法、配置项、输入输出字段、运行方式，则必须更新相关说明：

- README.md；
- docs/；
- 配置说明；
- 数据格式说明；
- 实验说明；
- 模块说明。

### C. 代码注释

如果本轮修复了容易误解的数学逻辑、坐标系逻辑、噪声参数逻辑、初始化逻辑，可以补充必要代码注释。

注意：

- 可以改相关文字说明；
- 必须避免空泛长篇；
- 不要只写文档不改代码；
- 不要为了记录工作制造大量零散文档；
- 不要把工作记录写成论文；
- 不要把说明文字替代代码推进。

原则：

> 代码推进是主线，文字记录是留痕和复盘；二者都要有，但不能本末倒置。

---

## 10. 文档修改边界

允许修改与本轮任务直接相关的文字说明。

允许修改：

```text
README.md
docs/
*.md
YYYY-MM-DD_工作记录.txt
YYYY-MM-DD_今日工作总结.txt
项目总结.txt
配置说明
数据格式说明
```

但必须满足：

- 与本轮代码或测试工作直接相关；
- 内容简洁；
- 明确记录“做了什么、为什么、如何验证、下一步”；
- 不写大段空泛理论；
- 不写与当前任务无关的项目愿景；
- 不把主要时间花在润色文字上。

注意：本文件 `AGENTS.md` 是 Codex 行为规则文件。除非用户要求修改提示词规则，否则不要修改它。

---

## 11. 禁止过度扩展

仓库里虽然已经有 barometer 和 mag yaw，但当前任务不能主动扩展到更多传感器。

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

已有的 barometer / mag yaw 可以维护，但不要让它们抢占主线。

主线仍然是：

```text
IMU predict + GNSS position/velocity update + 输出轨迹 + 评估结果 + 工作记录
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

1. 明确本轮目标；
2. 判断任务属于核心算法、测量更新、数据适配、主流程、测试、文字记录中的哪一类；
3. 选择一个中等粒度任务，不要切得过碎；
4. 列出准备查看的关键文件；
5. 只读取和当前任务相关的文件；
6. 找出当前主题下的主要缺口；
7. 修改必要的 `.py` / 配置 / 测试 / 相关说明文件；
8. 不改 C++；
9. 不做无关重构；
10. 修改后尽量运行测试或给出测试命令；
11. 清理无关临时结果；
12. 更新相关文字记录。

---

## 16. 每次输出格式

每次完成任务后，必须按这个格式回复：

```text
本轮目标：
- ...

本轮查看的关键文件：
- ...

本轮修改的文件：
- ...

完成内容：
- ...

文字记录更新：
- 更新了哪个记录/说明文件
- 记录了哪些关键信息

验证方式：
```bash
...
```

验证结果：
- 通过 / 未通过
- 如果未通过，说明具体错误和下一步

结果目录处理：
- 是否生成 03_results_* 或其他临时结果
- 是否已清理
- 如保留，说明原因

当前主线状态：
- 初始化：
- predict：
- GNSS position update：
- GNSS velocity update：
- 输出结果：
- 测试：
- 工作记录：

下一步中等粒度任务：
- ...
```

不要写长篇理论说明。

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

1. 查看最近修改；
2. 检查测试；
3. 选择一个中等粒度主题；
4. 围绕该主题完成代码、测试、说明记录的闭环；
5. 清理无关临时结果；
6. 汇报结果；
7. 给出下一步中等粒度任务。

不要重新写大规划。

不要改 C++。

不要只写文档不改代码。

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
- 测试覆盖是否够；
- 文字记录是否能追踪项目进展；
- 下一步最应该做哪个中等粒度任务。

不要泛泛介绍 ESKF 理论。

---

## 19. 如果用户说“修 bug”

必须遵守：

1. 先复现或定位错误；
2. 找到相关模块；
3. 做一组中等粒度的关联修复；
4. 不重构无关文件；
5. 不顺手加无关新功能；
6. 改完跑对应测试；
7. 清理无关临时结果；
8. 更新工作记录或相关说明；
9. 明确说明错误原因。

---

## 20. 如果用户说“加功能”

先判断功能属于哪个已有模块：

- 新测量模型：放 `measurements/`；
- 新数据格式：放 `adapters/`；
- 新评估指标：放 `analysis/`；
- ESKF 状态/传播：放 `core/`；
- 主流程调度：改 `app.py`；
- 参数：改 `config.py` 和 `01_data/config.json`；
- 功能说明 / 使用说明：更新 README、docs 或工作记录。

不要为了一个小功能重写整体 pipeline。

新增功能必须包含：

- 相关代码；
- 必要配置；
- 必要测试；
- 必要文字说明；
- 验证命令；
- 结果目录处理。

---

## 21. 推荐单次任务提示词

用户每次让 Codex 推进项目时，可以使用：

```text
按照 AGENTS.md 执行。

这是现有 Python ESKF-test 仓库，不是从零项目。
不要改成 C++。
不要新建 CMake。
不要重写项目结构。

本轮目标：
在现有 02_src/eskf_stack 结构下，选择一个中等粒度任务推进 ESKF 主线。

要求：
1. 先说明本轮准备查看哪些文件；
2. 不要把任务切得太小，本轮至少完成一个相对完整的小闭环；
3. 只改必要的 .py / 配置 / 测试 / 相关说明文件；
4. 不重构无关模块；
5. 改完运行相关测试；
6. 如果生成临时 03_results_*，任务结束前清理，除非说明保留原因；
7. 更新当天工作记录或相关说明；
8. 输出本轮完成、修改文件、文字记录、验证结果、结果目录处理、下一步任务。

不要写空泛理论，不要写大规划，只做当前仓库主线推进。
```

---

## 22. 如果模型开始乱写文档

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
- 05_tests
- 工作记录

本轮完成一个中等粒度任务：代码修改 + 测试 + 必要文字记录 + 临时结果清理。
```

---

## 23. 回答风格

用户是研一学生，方向是无人机多源融合定位。

回答要：

- 直接；
- 工程导向；
- 少绕；
- 不吓人；
- 不突然换技术路线；
- 不把 Python 改成 C++；
- 不把滤波项目改成因子图；
- 不把已有仓库说成从零项目；
- 优先告诉用户现在应该改哪个文件、跑哪个命令、看哪个结果、记录到哪里。

不要用大段空泛表述，例如：

- “本项目具有重要意义”；
- “未来可以广泛扩展”；
- “建议系统性规划”；
- “多源融合具有广阔前景”。

最终原则：

> 中等粒度推进；代码、测试、说明记录一起闭环；清理无关结果；基于现有 Python 仓库继续做，不换路线。
