# 钉孔装配系统：动力学与控制数学原理

**系统：** Franka Panda 7-DOF · MuJoCo 仿真
**架构：** FSM → OCP（预插入轨迹）/ LCS-MPC（插入力前馈）→ OSC（任务空间力矩）

---

## 1. 系统动力学

关节空间动力学：

$$
M(q)\ddot{q} + C(q,\dot{q})\dot{q} + g(q) = \tau + J(q)^\top F_\mathrm{ext}
$$

实际关节力矩为两路叠加：MuJoCo 仿射伺服（`ctrl` 设为关节目标 `q_des`）+ 控制器写入的 `qfrc_applied`（即 OSC 的 `Jᵀ·F_task` 加偏置补偿）。伺服执行器（`gainprm/biasprm`）令 `q̇_des = 0`，为带饱和的纯 PD：

$$
\tau_{\mathrm{servo},j} = K_{p,j}(q_{\mathrm{des},j} - q_j) - K_{d,j}\dot{q}_j, \qquad |\tau_{\mathrm{servo},j}| \leq \tau_{\mathrm{max},j}
$$

增益（Kp / Kd）：关节 1–2 = 4500 / 450；关节 3–4 = 3500 / 350；关节 5–7 = 2000 / 200。轨迹跟踪由上层 OSC 在任务空间完成。末端速度 `v_ee = J·q̇ ∈ R^6`，平移雅可比 `J₃ = J[:3,:] ∈ R^{3×7}`。

---

## 2. 预插入轨迹优化（OCP）

进入 MOVE_TO_PREINSERT 时一次性求解、开环跟踪。决策变量 `z = [v_0,…,v_{N-1}] ∈ R^{3N}`，动力学 `p_{k+1} = p_k + Δt·v_k`（cumsum 解析重构）。

$$
J = Q_T\|p_N-p_g\|^2 + Q_v\sum\|v_k\|^2 + Q_s\sum\|v_k-v_{k-1}\|^2 + w_\mathrm{cl}\sum w_k\max(0,\,z_\mathrm{cl}-p_k^z)^2 + w_\mathrm{ws}(\cdot)
$$

参数：`N=30`, `Δt=0.10 s`, `Q_terminal=2000`, `Q_vel=0.5`, `Q_smooth=5`, `w_clearance=500`, `w_workspace=100`；间隙高度 `z_cl = z_board + 2·ℓ_peg = 0.310 + 0.140 = 0.450 m`；间隙权 `w_k = min(1, d_lat(k)/0.05)` 由直线初值预计算以保持梯度解析。

解析梯度（cumsum 仿射结构，复杂度 O(N)），SLSQP 求解，盒约束 `‖v_k‖∞ ≤ [0.15, 0.15, 0.10] m/s`，暖启动后约 1 ms。

---

## 3. LCS 建模与 MPC

### 3.1 降阶 LCS

状态 `x = [eₓ, e_y, z, ėₓ, ė_y, ż]`，控制 `u = [Fₓ, F_y, F_z]`：

$$
x_{k+1} = A x_k + B u_k + D \lambda_k, \qquad 0 \leq \lambda_k \perp \phi(x_k)=E x_k + c \geq 0
$$

$$
A = \begin{bmatrix} I_3 & \Delta t\,I_3 \\ 0 & \mathrm{diag}(\alpha_x,\alpha_x,\alpha_z) \end{bmatrix}, \quad B = \begin{bmatrix} 0 \\ \frac{\Delta t}{m} I_3 \end{bmatrix}, \quad \alpha = 1 - \frac{b\,\Delta t}{m}
$$

参数：`m=0.5 kg`, `b_xy=8`, `b_z=12`, `Δt=0.02 s`（仿真步长 0.002 s × MPC 频率比 10）, `c=0.004 m`（孔-peg 间隙）。`D` 传递 4 面碰撞侧向力；规划时令 `λ=0`。

### 3.2 有限时域 LQR-MPC

$$
\min_{u_{0:N-1}} \sum_{k=0}^{N-1}\left[(x_k-x^\star)^\top Q\,(x_k-x^\star)+u_k^\top R\,u_k\right] + (x_N-x^\star)^\top Q_N (x_N-x^\star)
$$

参数：`x* = [0, 0, z_g, 0, 0, 0]`（`z_g=0.075`）, `Q=diag(500,500,200,1,1,1)`, `R=0.01·I`, `Q_N=5Q`, `N=8`。后向 Riccati 递推（离线，< 1 ms）：

$$
K_k = (R+B^\top P_{k+1}B)^{-1}B^\top P_{k+1}A, \qquad P_k = Q + A^\top P_{k+1}A - A^\top P_{k+1}B K_k
$$

在线律 `u_k* = clip(−K_k·(x_k − x*), u_min, u_max)`。`P_N ≻ 0` ⟹ `A − BK₀` Schur 稳定。

---

## 4. 操作空间控制器（OSC）

默认 `dynamics_aware: true`（惯量整形模式）。三模式增益（对角，平移 xyz）：

| 模式 | Kp(xyz) | Kd(xyz) | Kp(rot) | Kd(rot) |
|------|---------|---------|---------|---------|
| free_space | 300,300,300 | 30,30,30 | 30,30,30 | 3,3,3 |
| insertion  | 80,80,50    | 15,15,10 | 10,10,20 | 1,1,2 |
| recovery   | 150,150,150 | 20,20,20 | 20,20,20 | 2,2,2 |

### 4.1 阻抗模式（`dynamics_aware=False`）

$$
F_\mathrm{task} = K_p e + K_d(v_\mathrm{des}-v_\mathrm{ee}) + F_\mathrm{des}, \qquad \tau = J^\top F_\mathrm{task} + \tau_\mathrm{bias}
$$

`Kp, Kd ≻ 0` → 阻抗端口无源性（Hogan 1985）→ 与任意无源接触环境稳定。

### 4.2 惯量整形模式（默认）

$$
\Lambda = (J_3 M^{-1} J_3^\top + \varepsilon I)^{-1}, \quad \varepsilon = 10^{-3}
$$

$$
a_\mathrm{cmd} = a_\mathrm{des} + K_p^\mathrm{pos} e_p + K_d^\mathrm{pos}(v_\mathrm{des}-v_\mathrm{ee}^{xyz}), \qquad F_\mathrm{motion} = \Lambda\,a_\mathrm{cmd}
$$

旋转通道为阻抗：`F_rot = Kp_rot·e_r − Kd_rot·v_ee_rot`。合成力矩：

$$
\tau = J^\top[F_\mathrm{motion};\,F_\mathrm{rot}] + \tau_\mathrm{bias} + J^\top F_\mathrm{des}, \qquad \tau \leftarrow \mathrm{clip}(\tau,\,\pm\tau_{\mathrm{max}})
$$

（insertion 模式若无 `F_des`，施加恒定下压 `F_z=8 N`。）标称精确线性化（忽略 `J̇·q̇`）得 `p̈ = a_cmd`，即：

$$
\ddot{e}_p + K_d^\mathrm{pos}\dot{e}_p + K_p^\mathrm{pos}e_p = 0
$$

各轴解耦，阻尼比 `ζ = Kd/(2·√Kp) ∈ [0.71, 0.87]`（最低为 insertion z 轴）。加速度前馈 `a_des(t) = (v_des(t+Δt) − v_des(t))/Δt` 消除稳态跟踪误差。

---

## 5. 稳定性与安全

**稳定性：** 阻抗无源性（严格，Hogan 1985）；Riccati `P_N ≻ 0` → LCS 标称 Schur（严格）；惯量整形线性化 → `ζ ∈ [0.71, 0.87]` 各轴解耦（严格，标称）；`Λ` 正则化 + 力矩饱和 → 奇异处局部保证。

**硬约束：**

$$
\tau_{\mathrm{max}}=[87,87,87,87,12,12,12]\ \mathrm{Nm}, \quad \|v_\mathrm{cmd}^{xyz}\| \leq 0.25\ \mathrm{m/s}, \quad u_k \in [-5,5]^2 \times [0.5,10]\ \mathrm{N}
$$

**监督层：** `|F_z| > 40 N` 触发退缩；滑窗侧向力 `> 1.5 N`（50 步）触发螺旋搜索（最多 8 次，退缩高度 3 mm）。

---

## 符号表

| 符号 | 含义 |
|------|------|
| `q, q̇ ∈ R^7` | 关节角 / 速度 |
| `M ∈ R^{7×7}` | 关节惯量矩阵 |
| `J ∈ R^{6×7}`, `J₃ ∈ R^{3×7}` | EE 雅可比 / 平移子矩阵 |
| `Λ ∈ R^{3×3}` | 操作空间惯量矩阵 |
| `e = [e_p, e_r] ∈ R^6` | 任务空间误差 |
| `F_des` | MPC 力前馈（3D 插入力嵌入任务力旋量） |
| `x ∈ R^6`, `λ ∈ R^4`, `φ(x)` | LCS 状态 / 互补力 / 间隙 |
| `P_k, K_k` | Riccati 矩阵 / LQR 增益 |
| `ζ` | 阻尼比 |
