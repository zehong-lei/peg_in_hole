# 本项目的技术方法（按机器人臂框架组织）

> 项目：MuJoCo Franka Panda 多形状装配 / Peg-in-Hole 系统。
> 本文只描述**主线流水线**实际使用的技术方法，按 3.1–3.10 归类。
> 主线 = 感知位姿 → 末端预插入 OCP → 操作空间惯量整形控制 + 接触隐式 LCS-MPC 插入力前馈。
> 括号内的术语表仅为参考，本项目只涉及其中一部分。

---

## 3.1 机器人与任务表示

- **关节空间表示**：7-DoF Panda，`ARM_JOINT_NAMES = joint1..7`（`scene_builder.py`）。
- **任务空间表示**：末端 6D 笛卡尔表示，控制与规划均在任务空间进行（`operational_space_controller.py`）。
- **坐标帧**：base 帧、end-effector site、工具点 `peg_tip`/`peg_top`、object 帧（peg/board/hole）、camera 帧（`camera_top`/`camera_oblique`）、contact 帧。
- **旋转表示**：旋转矩阵为主（`ee_rot ∈ R^{3×3}`）；姿态噪声用轴角/四元数扰动（`noise_model.orientation_noise`）。
- **齐次变换 / SE(3)**：抓取偏移 `T_grasp`、孔位推断 `hole_pos = R(yaw)·offset + center`。
- **力旋量 (wrench)**：接触力以世界系 6D 力旋量 `[fx,fy,fz,tx,ty,tz]` 表示（`_get_body_contact_wrench`）。
- **物体 / 抓取位姿表示**：`PoseEstimate`、`SceneObservation`（`perception/types.py`）；抓取以 weld + `T_grasp` 表示。
- **接触几何**：圆孔以 4 墙盒近似建模（`lcs_mpc.py`）。

## 3.2 感知与传感

- **编码器感知**：关节位置/速度 `q, qdot`（`Observation`）。
- **力/力矩感知**：peg↔board 接触 6D 力旋量（排除手指接触），带阈值的接触检测（`sensor_wrapper.py`）。
- **相机 / RGB-D 感知**：两路固定 RGB-D 相机，`get_rgb` / `get_depth`（`camera_module.py`）。
- **点云感知**：深度图反投影为世界系点云（`pointcloud_utils.py`）。
- **物体检测**：HSV 颜色分割 + 形态学清理，定位各 peg（`color_segmenter.py`）。
- **物体位姿估计**：
  - peg：centroid / bbox2d / pca_obb 三种取心法 + PCA 求 yaw（`pointcloud_pose_estimator.py`）。
  - board：深度 z-band 分割 + 粗 XY 滤波 + PCA-OBB 求中心与 yaw（`board_pose_estimator.py`）。
  - 由 board 位姿 + CAD 偏移推断各孔位。
- **接触感知**：`contact_detected` 与侧向力 `external_force[:2]` 用于插入卡阻判定（`scripted_planner.py`）。

## 3.3 状态估计

- **关节 / 速度估计**：一阶低通滤波平滑 `q, qdot`（`StateEstimator`）。
- **末端位姿估计**：`ee_pos, ee_rot` 经噪声 + 滤波进入 `BeliefState`。
- **物体状态估计**：peg/hole 位姿带来源标签（`noisy_ground_truth` / `rgbd_pointcloud` / `grasp_propagation`）；抓取后由 EE 位姿经 `T_grasp` 传播 peg 位姿。
- **接触 / 力旋量估计**：接触力旋量 + 阈值接触判定。

## 3.4 运动学

- **正运动学**：MuJoCo `mj_kinematics` / `mj_comPos`（`assembly_env.py`）。
- **逆运动学**：阻尼最小二乘 Jacobian IK（`operational_space_controller.py`）。
- **微分 / 速度运动学**：几何 Jacobian `J`，末端速度 `v_ee = J·qdot`。
- **Jacobian 转置映射**：`tau = Jᵀ·F_task`。
- **逆速度运动学**：OCP 输出末端速度 → IK 折算为关节命令。

## 3.5 动力学与接触建模

- **刚体 / 逆动力学**：MuJoCo 全身动力学；`qfrc_bias` 重力 / 科氏前馈补偿。
- **操作空间动力学**：任务空间惯量 `Λ = (J M⁻¹ Jᵀ)⁻¹` 惯量整形。
- **接触动力学 / 互补约束模型**：接触隐式降阶 LCS
  `x⁺ = A x + B u + D λ`，`0 ≤ λ ⊥ φ(x) = E x + c ≥ 0`（`lcs_mpc.py`）。
- **接触力模型**：圆孔 4 墙盒近似，接触力旋量求和。
- **抓取模型**：刚性 weld 抓取（`grasp_weld`，`solref=0.004` 防 peg 倾斜）。
- **物体动力学**：peg 自由关节刚体。
- **执行器 / 力矩限制**：位置伺服执行器模型；`_TAU_MAX = [87×4, 12×3]` 力矩饱和。

## 3.6 运动与操作规划

- **轨迹优化 / 最优控制**：末端空间预插入 OCP（SLSQP，解析梯度；终端 + 速度 + 平滑 + 间隙 + 工作空间代价，`preinsert_ocp.py`）。
- **轨迹生成 / 执行**：`TrajectoryTracker` 开环跟踪 OCP 轨迹（`trajectory_utils.py`）。
- **模型预测控制 (MPC)**：接触隐式 LCS-MPC，插入阶段力前馈，有限时域 LQR 求解（后向 Riccati，`lcs_mpc.py`）。
- **装配 / pick-and-place 规划**：8 阶段状态机
  PREGRASP→GRASP→LIFT→MOVE_TO_PREINSERT→ALIGN→INSERT→RELEASE→RETREAT（`scripted_planner.py`）。
- **接触规划**：插入接触感知 + 卡阻后外扩螺旋搜索恢复。
- **多任务编排**：按 (peg, hole) 序列逐段规划执行（`multi_task_assembly.py`）。

## 3.7 控制

- **操作空间控制 (OSC)**：惯量整形 `Λ`；自由段刚性跟踪，插入段柔顺、恢复段柔和（`operational_space_controller.py`）。
- **笛卡尔位置跟踪（自由段）**：resolved-motion-rate 控制——阻尼 Jacobian IK 求关节目标 + 内环关节 PD 力矩（`position_controller.py`）。
- **逆动力学 / 计算力矩控制**：`tau = Jᵀ·F_task + qfrc_bias`（重力 / 偏置补偿）。
- **笛卡尔阻抗（插入段）**：OSC 插入模式下低 x/y 刚度 + roll/pitch 柔顺，实现接触柔顺。
- **混合位置/力控制**：插入阶段 x/y 柔顺位置 + z 力前馈（LCS-MPC）。
- **力控制**：恒定插入力 + MPC 力前馈。
- **LQR**：LCS-MPC 的求解内核（有限时域，后向 Riccati）。
- **模型预测控制**：见 3.6。

## 3.8 操作学习

（本项目未涉及。）

## 3.9 系统集成

- **仿真环境 / MuJoCo**：全栈基于 MuJoCo（`assembly_env.py`, `scene_builder.py`, menagerie Panda）。
- **控制器接口**：统一 `Command` dict；planner 只读 `BeliefState`，控制器只读 `Observation`（严格分层）。
- **感知-控制流水线**：感知位姿 → `set_peg_pos_estimate` → sensor 来源切换 → 规划 / 控制。
- **硬件抽象 / 传感封装**：`SensorWrapper` 禁止控制器访问 `get_true_state()` / `_d`。
- **实时控制环**：仿真步进控制环，MPC 频率比 `mpc_freq_ratio` 可配。
- **轨迹执行**：`TrajectoryTracker` 开环执行 OCP 轨迹。
- **数据记录 / 指标汇总**：`run_assembly.py` 按 seed 跑多回合,打印每任务 + 回合级指标(成功率、插入深度、峰值力、恢复次数、用时)。

## 3.10 验证与安全

- **几何 / 位姿验证**：perception 测试套件(`tests/test_board_perception.py` 等)校验孔位推断与装配几何（误差 ~1.2mm）。
- **力限制 / 接触力监测**：`max_force_abort` 中止阈值、插入侧向力卡阻阈值、峰值力记录 `max_peak_force`。
- **关节 / 速度 / 力矩限制**：`_TAU_MAX`、`_VEL_MAX`、`_JVEL_MAX` 饱和限幅。
- **稳定性安全手段**：LQR 理论支撑、阻抗无源性设计依据、软约束 + 力中止(`max_force_abort`)作为安全过滤层。
- **鲁棒性检查**：`run_assembly.py --seeds N --offset-mm X` 在多 seed 与横向偏移下统计成功率 / 峰值力。

---
---

# Technical Methods in This Project (Organized by the Robot-Arm Framework)

> Project: MuJoCo Franka Panda multi-shape assembly / peg-in-hole system.
> This document describes only the techniques used by the **mainline pipeline**, grouped under 3.1–3.10.
> Mainline = perceived pose → end-effector pre-insertion OCP → operational-space inertia-shaping control + contact-implicit LCS-MPC insertion force feedforward.
> The parenthetical term lists in the framework are reference vocabulary only; this project covers a subset of them.

---

## 3.1 Robot and Task Representation

- **Joint-space representation**: 7-DoF Panda, `ARM_JOINT_NAMES = joint1..7` (`scene_builder.py`).
- **Task-space representation**: 6D Cartesian end-effector representation; control and planning operate in task space (`operational_space_controller.py`).
- **Coordinate frames**: base frame, end-effector site, tool points `peg_tip`/`peg_top`, object frames (peg/board/hole), camera frames (`camera_top`/`camera_oblique`), contact frames.
- **Rotation representation**: rotation matrices primarily (`ee_rot ∈ R^{3×3}`); orientation noise via axis-angle / quaternion perturbation (`noise_model.orientation_noise`).
- **Homogeneous transform / SE(3)**: grasp offset `T_grasp`, hole inference `hole_pos = R(yaw)·offset + center`.
- **Wrench**: contact force represented as a world-frame 6D wrench `[fx,fy,fz,tx,ty,tz]` (`_get_body_contact_wrench`).
- **Object / grasp pose representation**: `PoseEstimate`, `SceneObservation` (`perception/types.py`); grasp represented by weld + `T_grasp`.
- **Contact geometry**: circular hole modeled as a 4-wall box approximation (`lcs_mpc.py`).

## 3.2 Perception and Sensing

- **Encoder sensing**: joint positions/velocities `q, qdot` (`Observation`).
- **Force/torque sensing**: peg↔board 6D contact wrench (gripper contacts excluded), with thresholded contact detection (`sensor_wrapper.py`).
- **Camera / RGB-D sensing**: two fixed RGB-D cameras, `get_rgb` / `get_depth` (`camera_module.py`).
- **Point-cloud sensing**: depth back-projected into a world-frame point cloud (`pointcloud_utils.py`).
- **Object detection**: HSV color segmentation + morphological cleaning to localize each peg (`color_segmenter.py`).
- **Object pose estimation**:
  - peg: centroid / bbox2d / pca_obb center estimates + PCA yaw (`pointcloud_pose_estimator.py`).
  - board: depth z-band segmentation + coarse XY filter + PCA-OBB center and yaw (`board_pose_estimator.py`).
  - hole positions inferred from board pose + CAD offsets.
- **Contact perception**: `contact_detected` and lateral force `external_force[:2]` used for insertion-jam detection (`scripted_planner.py`).

## 3.3 State Estimation

- **Joint / velocity estimation**: first-order low-pass filtering of `q, qdot` (`StateEstimator`).
- **End-effector pose estimation**: `ee_pos, ee_rot` pass through noise + filtering into `BeliefState`.
- **Object state estimation**: peg/hole poses carry a source tag (`noisy_ground_truth` / `rgbd_pointcloud` / `grasp_propagation`); after grasp, peg pose is propagated from the EE pose via `T_grasp`.
- **Contact / wrench estimation**: contact wrench + thresholded contact decision.

## 3.4 Kinematics

- **Forward kinematics**: MuJoCo `mj_kinematics` / `mj_comPos` (`assembly_env.py`).
- **Inverse kinematics**: damped least-squares Jacobian IK (`operational_space_controller.py`).
- **Differential / velocity kinematics**: geometric Jacobian `J`, end-effector velocity `v_ee = J·qdot`.
- **Jacobian-transpose mapping**: `tau = Jᵀ·F_task`.
- **Inverse velocity kinematics**: OCP outputs EE velocity → IK resolves it to joint commands.

## 3.5 Dynamics and Contact Modeling

- **Rigid-body / inverse dynamics**: MuJoCo full-body dynamics; `qfrc_bias` gravity/Coriolis feedforward compensation.
- **Operational-space dynamics**: task-space inertia `Λ = (J M⁻¹ Jᵀ)⁻¹` inertia shaping.
- **Contact dynamics / complementarity model**: contact-implicit reduced-order LCS
  `x⁺ = A x + B u + D λ`, `0 ≤ λ ⊥ φ(x) = E x + c ≥ 0` (`lcs_mpc.py`).
- **Contact-force model**: 4-wall box approximation of the circular hole, contact wrench summation.
- **Grasp model**: rigid weld grasp (`grasp_weld`, `solref=0.004` to prevent peg tilting).
- **Object dynamics**: peg as a free-joint rigid body.
- **Actuator / torque limits**: position-servo actuator model; `_TAU_MAX = [87×4, 12×3]` torque saturation.

## 3.6 Motion and Manipulation Planning

- **Trajectory optimization / optimal control**: end-effector-space pre-insertion OCP (SLSQP with analytic gradients; terminal + velocity + smoothness + clearance + workspace costs, `preinsert_ocp.py`).
- **Trajectory generation / execution**: `TrajectoryTracker` open-loop tracking of the OCP trajectory (`trajectory_utils.py`).
- **Model predictive control (MPC)**: contact-implicit LCS-MPC with insertion-phase force feedforward, solved by finite-horizon LQR (backward Riccati, `lcs_mpc.py`).
- **Assembly / pick-and-place planning**: 8-stage state machine
  PREGRASP→GRASP→LIFT→MOVE_TO_PREINSERT→ALIGN→INSERT→RELEASE→RETREAT (`scripted_planner.py`).
- **Contact planning**: insertion contact sensing + outward spiral search recovery after a jam.
- **Multi-task orchestration**: per-(peg, hole) sequential plan-and-execute (`multi_task_assembly.py`).

## 3.7 Control

- **Operational-space control (OSC)**: inertia shaping `Λ`; stiff tracking in free space, compliant during insertion, gentle during recovery (`operational_space_controller.py`).
- **Cartesian position tracking (free space)**: resolved-motion-rate control — damped Jacobian IK to a joint target + inner joint-PD torque (`position_controller.py`).
- **Inverse-dynamics / computed-torque control**: `tau = Jᵀ·F_task + qfrc_bias` (gravity/bias compensation).
- **Cartesian impedance (insertion)**: OSC insertion mode with low x/y stiffness + roll/pitch compliance for contact compliance.
- **Hybrid position/force control**: insertion-phase compliant x/y position + z force feedforward (LCS-MPC).
- **Force control**: constant insertion force + MPC force feedforward.
- **LQR**: solver core of the LCS-MPC (finite-horizon, backward Riccati).
- **Model predictive control**: see 3.6.

## 3.8 Learning for Manipulation

(Not covered in this project.)

## 3.9 System Integration

- **Simulation environment / MuJoCo**: full stack on MuJoCo (`assembly_env.py`, `scene_builder.py`, menagerie Panda).
- **Controller interface**: unified `Command` dict; planner reads only `BeliefState`, controller reads only `Observation` (strict layering).
- **Perception-control pipeline**: perceived pose → `set_peg_pos_estimate` → sensor source switch → planning / control.
- **Hardware abstraction / sensor wrapping**: `SensorWrapper` forbids the controller from accessing `get_true_state()` / `_d`.
- **Real-time control loop**: stepped simulation control loop, configurable MPC frequency ratio `mpc_freq_ratio`.
- **Trajectory execution**: `TrajectoryTracker` open-loop execution of the OCP trajectory.
- **Data logging / metric aggregation**: `run_assembly.py` runs multiple seeds and prints per-task + episode-level metrics (success rate, insertion depth, peak force, recovery count, time).

## 3.10 Verification and Safety

- **Geometry / pose verification**: the perception test suite (`tests/test_board_perception.py`, etc.) validates hole inference and assembly geometry (~1.2 mm error).
- **Force limits / contact-force monitoring**: `max_force_abort` abort threshold, insertion lateral-force jam threshold, peak-force logging `max_peak_force`.
- **Joint / velocity / torque limits**: `_TAU_MAX`, `_VEL_MAX`, `_JVEL_MAX` saturation clamping.
- **Stability safeguards**: LQR theoretical grounding, impedance passivity as design rationale, soft constraints + force abort (`max_force_abort`) as a safety-filter layer.
- **Robustness check**: `run_assembly.py --seeds N --offset-mm X` measures success rate / peak force across seeds and lateral offsets.
