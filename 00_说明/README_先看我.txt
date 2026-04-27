项目名称
离线 ESKF 多传感器融合项目

当前项目状态
截至 2026-04-23，这个项目已经不是最初的教学 demo，但它当前仍处于“原型框架搭建阶段”，不是已经定型完成的项目。
当前已经具备：
1. `core / measurements / adapters / analysis / tests` 分层。
2. 一套可执行的离线融合原型主流程。
3. GNSS 创新管理，包括 `NIS`、门限拒绝和自适应 `R`。
4. 观测管理层当前已统一覆盖：
   `GNSS 位置`
   `GNSS 速度`
   `气压计高度`
   `磁罗盘航向`
   并已支持按 `fusion_policy`
   打开或关闭：
   `NIS reject`
   `adaptive R`
   `recovery scale`
5. 局部 `ENU` 导航环境、`earth rate / transport rate`、重力梯度、科里奥利相关诊断。
6. 质量评分、状态机、滞回、防抖、恢复桥接和摘要统计。
7. 自动化回归测试和结果图表输出。

当前需要特别注意
1. “能在样例数据上跑起来”
   不等于“项目已经完整搭好”。
2. `00000422` 相关配置和结果
   当前只能看成样例验证线，
   不能看成项目主线已经完成。
3. 当前真正的主任务仍然是：
   把项目本体搭对，
   而不是继续围着某一组数据做专用优化。

项目目的
这个项目的目标不是只画一条轨迹，而是搭一套：
1. 能看懂。
2. 能运行。
3. 能继续扩展。
4. 能接真实日志。
5. 能做实验对比和方法迭代。

当前版本已经做的关键升级
1. 顶层入口和说明分开，减少“文件全堆在一起”的问题。
2. 代码已拆成 `core`、`measurements`、`adapters`、`analysis`、`tests` 五层。
3. `app.py` 正在从“大总管”收口为主流程编排：
   结果导出已交给 `analysis.exporter`，
   初始化流程控制已交给 `pipeline.initialization_controller`。
4. `core` 已完成第一轮 KF-GINS 风格重构，状态布局、噪声布局、传播步骤和协方差传播已经拆清楚。
5. `navigation.py` 和 `mechanization.py` 已从 `filter` 主体里拆出。
6. 当前传播已接入局部曲率半径、重力梯度、`earth rate`、`transport rate` 和二阶离散化。
7. 当前结果文件已输出导航环境诊断、协方差健康诊断和状态机诊断。
8. 当前初始化已新增：
   静止粗对准入口
   roll/pitch 粗估
   gyro bias 粗估
   accel bias 粗估
9. 当前初始化过程已新增：
   阶段状态
   失败原因
   heading 来源诊断
   static alignment 诊断
10. 当前初始化等待已新增：
   超时控制
   零航向回退开关
11. 当前传播已新增：
   `dt` 下限检查
   `dt` 上限检查
   大步长跳过策略
   `dt` 诊断输出
12. 状态机已经不只看有没有 GNSS，还会看：
   `reject streak`
   `GNSS outage`
   `quality score`
   `covariance health`
   `covariance duration`
13. 当前已经加入：
   `GNSS_DEGRADED`
   `RECOVERING`
   滞回切换
   模式统计
   原因统计
   状态机摘要图
14. 当前已新增 `fusion_policy`：
   用于支持基础 ablation，
   当前可分别控制：
   `use_nis_rejection`
   `use_adaptive_r`
   `use_recovery_scale`
   这一步只是实验开关雏形，
   不代表质量评分和状态机已经真正反作用到滤波器。
15. 当前 adapter 统一入口已新增输入质量报告：
   字段完整性
   可选观测覆盖率
   诊断真值覆盖率
   时间戳单调性
   重复时间戳
   非正时间步
   大时间间隔
   都会进入 source summary 和 metrics。

建议你先看哪里
1. 第一次接手项目，先看 `00_说明\运行环境与验收说明.txt`
2. 先双击 `04_tools\检查运行环境.bat`
3. 如果你要先做检查，再双击 `04_tools\运行核心检查.bat`
4. 明确要看图和结果时，再双击 `04_tools\一键运行示例.bat`
5. 再看 `03_results` 里面生成的图和指标文件
6. 看结果时先看 `00_说明\运行结果阅读说明.txt`
7. 再看 `00_说明\项目主线与样例边界.txt`
8. 再看 `00_说明\统一输入契约.txt`
9. 想判断当前整体进度时看 `00_说明\项目完成度评估.txt`
10. 最后再看 `02_src` 里面的代码

目录说明
00_说明
放项目说明、数据格式说明和后续扩展建议。

01_data
放输入数据。
当前默认用 `generated\demo_sensor_log.csv` 作为示例数据。

02_src
放核心代码。

03_results
放运行结果。
`figures` 里面是图。
`metrics` 里面是 csv 和 txt 指标。

04_tools
放一键运行和检查脚本。

05_tests
放回归测试。
当前已经不只是最小核心检查，也覆盖了：
健康状态
状态机
指标统计
静止粗对准初始化
姿态误差注入方向
纯旋转传播方向

最常用入口
方式零
双击 `04_tools\检查运行环境.bat`
这个入口只检查 Python 和依赖包版本，
不会运行主流程，
也不会生成结果文件。

方式一
双击 `04_tools\一键运行示例.bat`
这个入口会更新 `03_results`，适合你明确要看图和结果时再用。

方式二
在项目根目录打开终端后运行：
`python 02_src\main.py`

方式三
在项目根目录打开终端后运行：
`python -m unittest discover 05_tests`
这是当前更推荐的“日常开发检查”路径，
不会额外生成新的结果文件。

方式四
如果你要直接跑 `00000422` 的当前推荐真实数据配置，
双击 `04_tools\运行00000422_推荐配置.bat`
或运行：
`python 02_src\main.py 01_data\config_00000422_decoded.json`

方式五
如果你要跑当前 baseline / ablation 实验模板，
双击 `04_tools\运行实验对比模板.bat`
或运行：
`python 02_src\run_experiment_batch.py`
这个入口会顺序运行当前实验配置模板，
并在 `03_results_experiment_batch`
下生成：
`experiment_metrics_summary.csv`
和：
`experiment_metrics_key_summary.csv`

配置使用原则
1. `01_data\config.json`
   继续保留为通用默认配置。
2. 真实数据的专用配置单独建文件，不覆盖通用配置。
3. 专用配置当前只用于样例验证，不代表项目主线已经按该配置定型。
4. 配置角色现在已经进入代码约束：
   `config.json` 只能是 `default_general`
   其他单独配置不能冒充主配置。
5. 如果要做 baseline / ablation，
   优先新建单独实验配置，
   修改其中的 `fusion_policy`，
   不要直接覆盖通用默认配置。
6. 当前已提供第一组实验配置模板：
   `config_experiment_baseline_eskf.json`
   `config_experiment_nis_reject.json`
   `config_experiment_adaptive_r.json`
   `config_experiment_adaptive_r_recovery.json`
   `config_experiment_full_method.json`
   它们用于固定策略开关组合，
   先服务于对比实验链路。

当前代码结构
02_src\main.py
保留单一入口，负责调用完整管线。

02_src\eskf_stack\core
放滤波内核。
这里管理状态、传播、协方差和通用更新框架。
当前已经拆出误差状态布局、过程噪声布局、名义状态传播、离散协方差传播、局部导航环境、机械编排步骤和静止粗对准初始化模块。

02_src\eskf_stack\measurements
放各类传感器测量模型。
当前已经拆出：
`GNSS 位置`
`GNSS 速度`
`气压计高度`
`磁罗盘航向`
当前这层已经进一步分成：
量测模块
观测管理层
量测模块负责 `residual / H / base_R` 建模，
观测管理层负责：
`NIS`
门限拒绝
自适应 `R`
恢复阶段渐进重接入
当前这层只应该消费观测层字段，不应该直接读取 `truth_*`。

02_src\eskf_stack\adapters
放数据适配层。
当前已经支持：
`standard_csv`
`great_msf_imu_ins`
`dx_decoded_flight_csv`
后面接 `PX4 / ROS2 / Gazebo / 新 GNSS 解` 时，优先从这里扩展。
所有 adapter 当前都应该输出统一输入契约定义的标准 DataFrame。

当前真实数据接入进展
1. 已能直接读取：
   `DX\05_data\dx_data\decoded\00000422`
   这一类 decoded 目录。
2. 当前会读取：
   `IMU_Full.csv`
   `GPS_Raw.csv`
   `POS_Global_Truth.csv`
   `XKF_Local_Truth.csv`
3. `GPS_Raw.csv` 当前按“位置观测”为主处理。
4. 对 decoded 数据，当前已经支持：
   `gps_velocity_mode = derived_from_pos`
   `gps_velocity_mode = none`
   `gps_velocity_mode = from_xkf`
5. 其中更符合你当前数据语义的默认建议是：
   先用 `position-only`
   不要把位置差分硬当高可信速度。
6. 每次运行后，`metrics\dataset_source_summary.txt`
   会明确写出本次到底用了哪些源文件和参考点。
7. 每次运行后，`metrics\metrics_summary.txt`
   现在会先写一段人类可读的初始化摘要，
   再写完整指标键值。
   初始化摘要当前包括：
   初始化阶段
   初始化方式
   初始化原因
   heading 来源
   是否用了 static coarse alignment
   是否触发了 zero yaw fallback
   初始化前等待时长
8. 对当前 `00000422` 这组数据，
   `IMU_Full.csv` 需要做轴系变换。
   当前推荐：
   `imu_transform_mode = ardupilot_frd_to_flu`
9. 在这个变换下，
   `dx_decoded_flight_csv + gps_velocity_mode = none`
   已经能作为样例验证链工作，
   比当前外部 TCPPP 解算入口更稳。
10. 以上内容当前的定位是：
   adapter 设计验证和输入语义核对，
   不是项目最终能力定义。

02_src\eskf_stack\analysis
放图表、指标、质量评分和状态机。
当前已经包含：
结果导出
质量评分
状态机
滞回
恢复桥接
协方差健康分类
状态机摘要图
模式统计和原因统计
初始化指标统计
初始化摘要文本输出
实验批处理指标汇总
当前这层才允许消费 `truth_*` 这类诊断/真值字段。
当前结果文件保存也已经从 `app.py`
收口到 `analysis.exporter`，
避免主流程继续承担过多输出细节。

你现在最值得看的结果文件
1. `03_results\figures\trajectory.png`
2. `03_results\figures\error_summary.png`
3. `03_results\figures\quality_mode.png`
4. `03_results\figures\covariance_diagnostics.png`
5. `03_results\figures\state_machine_summary.png`
6. `03_results\metrics\metrics_summary.txt`
7. `03_results\metrics\fusion_output.csv`

其中第 6 项当前建议优先看顶部的 `Initialization Summary`，
再看下面的 `Metric Values`。
这样可以先快速判断这次初始化到底是：
`direct`
还是 `bootstrap_position_pair`，
有没有用 `static coarse alignment`，
有没有触发 `zero yaw fallback`。

你后面怎么继续用
1. 先把项目主线和样例验证线区分开，不要把样例结论直接当成项目路线。
2. 再看懂示例数据字段和 adapter 入口，再替换成你自己的日志。
3. 如果你当前接的是 decoded 飞控目录，优先走 `adapters`，不要先改 `core`。
4. 如果你接下来要接 `PX4 / ROS2 / Gazebo` 数据，优先在 `adapters` 里做适配，不要先改 `core`。
5. 日常构建项目时，优先先跑：
   `04_tools\运行核心检查.bat`
   或
   `python -m unittest discover 05_tests`
   少跑完整主入口，避免把结果文件和源码改动混在一起。

当前版本边界
1. 这是研究原型，不是直接面向实机的高鲁棒工程版。
2. 当前坐标系主线仍然是局部 `ENU`，还不是完整经纬高 `KF-GINS` 主线。
3. 当前磁罗盘只用于更新 `yaw`，不估计完整磁场模型。
4. 当前还没有杆臂、外参、时间偏移、尺度因子等更完整状态。
5. 当前虽然已经有若干真实数据样例接入，但这些仍主要用于验证 adapter 和输入语义。
6. 当前还没有把 raw `obs / nav` 直接整理成正式统一观测适配器。
7. 当前的 `core` 已经明显比最初 demo 更规整，但还不是完整 `KF-GINS` 工程实现。
8. 当前项目不是“只差接数据”，而是仍在“框架搭建和边界固化”阶段。
9. 当前静止初始化对 `gyro bias`、`roll/pitch` 比较可靠，
   但纯静止段对横向 `accel bias`
   只能给粗估，
   不能把它理解成完整可观测解。
10. 当前仓库里仍跟踪了：
    `__pycache__ / .pyc`
    `03_results`
    这会让测试和运行很容易污染工作区，
    属于后续需要专门处理的工程卫生问题，
    但这次没有擅自改动。
