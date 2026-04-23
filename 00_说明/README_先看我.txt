项目名称
离线 ESKF 多传感器融合项目

当前项目状态
截至 2026-04-23，这个项目已经不是最初的教学 demo，但它当前仍处于“原型框架搭建阶段”，不是已经定型完成的项目。
当前已经具备：
1. `core / measurements / adapters / analysis / tests` 分层。
2. 一套可执行的离线融合原型主流程。
3. GNSS 创新管理，包括 `NIS`、门限拒绝和自适应 `R`。
4. 局部 `ENU` 导航环境、`earth rate / transport rate`、重力梯度、科里奥利相关诊断。
5. 质量评分、状态机、滞回、防抖、恢复桥接和摘要统计。
6. 自动化回归测试和结果图表输出。

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
3. `core` 已完成第一轮 KF-GINS 风格重构，状态布局、噪声布局、传播步骤和协方差传播已经拆清楚。
4. `navigation.py` 和 `mechanization.py` 已从 `filter` 主体里拆出。
5. 当前传播已接入局部曲率半径、重力梯度、`earth rate`、`transport rate` 和二阶离散化。
6. 当前结果文件已输出导航环境诊断、协方差健康诊断和状态机诊断。
7. 当前初始化已新增：
   静止粗对准入口
   roll/pitch 粗估
   gyro bias 粗估
   accel bias 粗估
8. 当前传播已新增：
   `dt` 下限检查
   `dt` 上限检查
   大步长跳过策略
   `dt` 诊断输出
9. 状态机已经不只看有没有 GNSS，还会看：
   `reject streak`
   `GNSS outage`
   `quality score`
   `covariance health`
   `covariance duration`
10. 当前已经加入：
   `GNSS_DEGRADED`
   `RECOVERING`
   滞回切换
   模式统计
   原因统计
   状态机摘要图

建议你先看哪里
1. 先双击 `04_tools\一键运行示例.bat`
2. 如果你要先做检查，再双击 `04_tools\运行核心检查.bat`
3. 再看 `03_results` 里面生成的图和指标文件
4. 再看 `00_说明\项目主线与样例边界.txt`
5. 再看 `00_说明\统一输入契约.txt`
6. 最后再看 `02_src` 里面的代码

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

配置使用原则
1. `01_data\config.json`
   继续保留为通用默认配置。
2. 真实数据的专用配置单独建文件，不覆盖通用配置。
3. 专用配置当前只用于样例验证，不代表项目主线已经按该配置定型。
4. 配置角色现在已经进入代码约束：
   `config.json` 只能是 `default_general`
   其他单独配置不能冒充主配置。

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
7. 对当前 `00000422` 这组数据，
   `IMU_Full.csv` 需要做轴系变换。
   当前推荐：
   `imu_transform_mode = ardupilot_frd_to_flu`
8. 在这个变换下，
   `dx_decoded_flight_csv + gps_velocity_mode = none`
   已经能作为样例验证链工作，
   比当前外部 TCPPP 解算入口更稳。
9. 以上内容当前的定位是：
   adapter 设计验证和输入语义核对，
   不是项目最终能力定义。

02_src\eskf_stack\analysis
放图表、指标、质量评分和状态机。
当前已经包含：
质量评分
状态机
滞回
恢复桥接
协方差健康分类
状态机摘要图
模式统计和原因统计
当前这层才允许消费 `truth_*` 这类诊断/真值字段。

你现在最值得看的结果文件
1. `03_results\figures\trajectory.png`
2. `03_results\figures\error_summary.png`
3. `03_results\figures\quality_mode.png`
4. `03_results\figures\covariance_diagnostics.png`
5. `03_results\figures\state_machine_summary.png`
6. `03_results\metrics\metrics_summary.txt`
7. `03_results\metrics\fusion_output.csv`

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
