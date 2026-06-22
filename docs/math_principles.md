# 钉孔装配系统：动力学与控制数学原理

> **系统**：Franka Panda 7-DOF · MuJoCo 仿真  
> **架构**：FSM → OCP（预插入轨迹）/ LCS-MPC（插入力前馈）→ OSC（任务空间力矩）

---

## 目录

- [钉孔装配系统：动力学与控制数学原理](#钉孔装配系统动力学与控制数学原理)
  - [目录](#目录)
  - [1. 系统动力学](#1-系统动力学)
  - [2. 预插入轨迹优化 OCP](#2-预插入轨迹优化-ocp)
  - [3. LCS 建模与 MPC](#3-lcs-建模与-mpc)
    - [3.1 降阶 LCS](#31-降阶-lcs)
    - [3.2 有限时域 LQR-MPC](#32-有限时域-lqr-mpc)
  - [4. 操作空间控制器 OSC](#4-操作空间控制器-osc)
    - [4.1 阻抗模式（`dynamics_aware=False`）](#41-阻抗模式dynamics_awarefalse)
    - [4.2 惯量整形模式（默认）](#42-惯量整形模式默认)
  - [5. 稳定性与安全](#5-稳定性与安全)
    - [稳定性](#稳定性)
    - [硬约束](#硬约束)
    - [监督层](#监督层)
  - [符号表](#符号表)

---

## 1. 系统动力学

关节空间动力学：

$$
M(q)\ddot{q} + C(q,\dot{q})\dot{q} + g(q) = \tau + J(q)^\top F_\mathrm{ext}
$$

实际关节力矩为两路叠加：

1. **MuJoCo 仿射伺服**：`ctrl` 设为关节目标 `q_des`。
2. **控制器外加力矩**：写入 `qfrc_applied`，即 OSC 的 `Jᵀ·F_task` 加偏置补偿。

伺服执行器（`gainprm/biasprm`）令 `q̇_des = 0`，为带饱和的纯 PD：

$$
\tau_{\mathrm{servo},j}
= K_{p,j}(q_{\mathrm{des},j} - q_j) - K_{d,j}\dot{q}_j,
\qquad
|\tau_{\mathrm{servo},j}| \leq \tau_{\mathrm{max},j}
$$

**伺服增益：**

| 关节 | Kp | Kd |
|---|---:|---:|
| 1–2 | 4500 | 450 |
| 3–4 | 3500 | 350 |
| 5–7 | 2000 | 200 |

轨迹跟踪由上层 OSC 在任务空间完成：

- 末端速度：`v_ee = J·q̇ ∈ R^6`
- 平移雅可比：`J₃ = J[:3, :] ∈ R^{3×7}`

---

## 2. 预插入轨迹优化 OCP

进入 `MOVE_TO_PREINSERT` 时一次性求解、开环跟踪。

**决策变量：**

$$
z = [v_0, \dots, v_{N-1}] \in \mathbb{R}^{3N}
$$

**离散动力学：**

$$
p_{k+1} = p_k + \Delta t\,v_k
$$

其中位置序列由速度序列通过 `cumsum` 解析重构。

**目标函数：**

$$
J
= Q_T\|p_N-p_g\|^2
+ Q_v\sum\|v_k\|^2
+ Q_s\sum\|v_k-v_{k-1}\|^2
+ w_\mathrm{cl}\sum w_k\max(0,\,z_\mathrm{cl}-p_k^z)^2
+ w_\mathrm{ws}(\cdot)
$$

**参数：**

| 参数 | 数值 |
|---|---:|
| `N` | `30` |
| `Δt` | `0.10 s` |
| `Q_terminal` | `2000` |
| `Q_vel` | `0.5` |
| `Q_smooth` | `5` |
| `w_clearance` | `500` |
| `w_workspace` | `100` |

**间隙高度：**

$$
z_\mathrm{cl}
= z_\mathrm{board} + 2\ell_\mathrm{peg}
= 0.310 + 0.140
= 0.450\ \mathrm{m}
$$

**间隙权重：**

$$
w_k = \min(1, d_\mathrm{lat}(k)/0.05)
$$

`w_k` 由直线初值预计算，以保持梯度解析。

**求解设置：**

- 梯度：解析梯度，利用 `cumsum` 仿射结构，复杂度 `O(N)`
- 求解器：`SLSQP`
- 盒约束：

$$
\|v_k\|_\infty \leq [0.15,\ 0.15,\ 0.10]\ \mathrm{m/s}
$$

- 暖启动后求解时间：约 `1 ms`

---

## 3. LCS 建模与 MPC

### 3.1 降阶 LCS

**状态与控制：**

$$
x = [e_x, e_y, z, \dot{e}_x, \dot{e}_y, \dot{z}],
\qquad
u = [F_x, F_y, F_z]
$$

**LCS 动力学：**

$$
x_{k+1} = A x_k + B u_k + D \lambda_k,
\qquad
0 \leq \lambda_k \perp \phi(x_k)=E x_k + c \geq 0
$$

**系统矩阵：**

$$
A =
\begin{bmatrix}
I_3 & \Delta t\,I_3 \\
0 & \mathrm{diag}(\alpha_x,\alpha_x,\alpha_z)
\end{bmatrix},
\qquad
B =
\begin{bmatrix}
0 \\
\frac{\Delta t}{m} I_3
\end{bmatrix},
\qquad
\alpha = 1 - \frac{b\,\Delta t}{m}
$$

**参数：**

| 参数 | 数值 | 含义 |
|---|---:|---|
| `m` | `0.5 kg` | 降阶等效质量 |
| `b_xy` | `8` | 横向阻尼 |
| `b_z` | `12` | 竖直阻尼 |
| `Δt` | `0.02 s` | 仿真步长 `0.002 s` × MPC 频率比 `10` |
| `c` | `0.004 m` | 孔-peg 间隙 |

`D` 传递 4 面碰撞侧向力；规划时令 `λ = 0`。

### 3.2 有限时域 LQR-MPC

**优化问题：**

$$
\min_{u_{0:N-1}}
\sum_{k=0}^{N-1}
\left[
(x_k-x^\star)^\top Q\,(x_k-x^\star)
+ u_k^\top R\,u_k
\right]
+ (x_N-x^\star)^\top Q_N (x_N-x^\star)
$$

**目标状态：**

$$
x^\star = [0, 0, z_g, 0, 0, 0],
\qquad
z_g = 0.075
$$

**权重与时域：**

| 参数 | 数值 |
|---|---|
| `Q` | `diag(500, 500, 200, 1, 1, 1)` |
| `R` | `0.01·I` |
| `Q_N` | `5Q` |
| `N` | `8` |

**后向 Riccati 递推**（离线，`< 1 ms`）：

$$
K_k = (R+B^\top P_{k+1}B)^{-1}B^\top P_{k+1}A
$$

$$
P_k = Q + A^\top P_{k+1}A - A^\top P_{k+1}B K_k
$$

**在线控制律：**

$$
u_k^* = \mathrm{clip}(-K_k(x_k - x^\star),\ u_\mathrm{min},\ u_\mathrm{max})
$$

若 `P_N ≻ 0`，则 `A − BK₀` Schur 稳定。

---

## 4. 操作空间控制器 OSC

默认 `dynamics_aware: true`，即惯量整形模式。

**三模式增益**（对角，平移 `xyz`）：

| 模式 | Kp(xyz) | Kd(xyz) | Kp(rot) | Kd(rot) |
|---|---|---|---|---|
| `free_space` | `300, 300, 300` | `30, 30, 30` | `30, 30, 30` | `3, 3, 3` |
| `insertion` | `80, 80, 50` | `15, 15, 10` | `10, 10, 20` | `1, 1, 2` |
| `recovery` | `150, 150, 150` | `20, 20, 20` | `20, 20, 20` | `2, 2, 2` |

### 4.1 阻抗模式（`dynamics_aware=False`）

$$
F_\mathrm{task}
= K_p e + K_d(v_\mathrm{des}-v_\mathrm{ee}) + F_\mathrm{des},
\qquad
\tau = J^\top F_\mathrm{task} + \tau_\mathrm{bias}
$$

当 `Kp, Kd ≻ 0` 时：

```text
正定刚度/阻尼 → 阻抗端口无源性（Hogan 1985）→ 与任意无源接触环境稳定
```

### 4.2 惯量整形模式（默认）

**操作空间惯量：**

$$
\Lambda = (J_3 M^{-1} J_3^\top + \varepsilon I)^{-1},
\qquad
\varepsilon = 10^{-3}
$$

**任务空间加速度命令：**

$$
a_\mathrm{cmd}
= a_\mathrm{des}
+ K_p^\mathrm{pos} e_p
+ K_d^\mathrm{pos}(v_\mathrm{des}-v_\mathrm{ee}^{xyz})
$$

$$
F_\mathrm{motion} = \Lambda\,a_\mathrm{cmd}
$$

旋转通道为阻抗：

```text
F_rot = Kp_rot·e_r − Kd_rot·v_ee_rot
```

**合成力矩：**

$$
\tau
= J^\top[F_\mathrm{motion};\,F_\mathrm{rot}]
+ \tau_\mathrm{bias}
+ J^\top F_\mathrm{des},
\qquad
\tau \leftarrow \mathrm{clip}(\tau,\,\pm\tau_\mathrm{max})
$$

> 在 `insertion` 模式下，若无 `F_des`，则施加恒定下压 `F_z = 8 N`。

标称精确线性化（忽略 `J̇·q̇`）得到：

$$
\ddot{p} = a_\mathrm{cmd}
$$

即误差动力学为：

$$
\ddot{e}_p + K_d^\mathrm{pos}\dot{e}_p + K_p^\mathrm{pos}e_p = 0
$$

因此各轴解耦，阻尼比为：

$$
\zeta = \frac{K_d}{2\sqrt{K_p}} \in [0.71, 0.87]
$$

最低阻尼比出现在 `insertion` 模式的 `z` 轴。

加速度前馈：

$$
a_\mathrm{des}(t)
= \frac{v_\mathrm{des}(t+\Delta t) - v_\mathrm{des}(t)}{\Delta t}
$$

该前馈项用于消除稳态跟踪误差。

---

## 5. 稳定性与安全

### 稳定性

| 模块 | 结论 | 说明 |
|---|---|---|
| 阻抗控制 | 严格无源 | Hogan 1985；可与任意无源接触环境稳定交互 |
| LQR-MPC | 标称 Schur 稳定 | `P_N ≻ 0` → `A − BK₀` Schur |
| 惯量整形 OSC | 标称线性化稳定 | `ζ ∈ [0.71, 0.87]`，各轴解耦 |
| 奇异处处理 | 局部保证 | `Λ` 正则化 + 力矩饱和 |

### 硬约束

$$
\tau_\mathrm{max}
= [87,87,87,87,12,12,12]\ \mathrm{Nm}
$$

$$
\|v_\mathrm{cmd}^{xyz}\| \leq 0.25\ \mathrm{m/s}
$$

$$
u_k \in [-5,5]^2 \times [0.5,10]\ \mathrm{N}
$$

### 监督层

| 触发条件 | 动作 |
|---|---|
| \|F_z\| > 40 N | 触发退缩 |
| 滑窗侧向力 `> 1.5 N`，窗口 `50` 步 | 触发螺旋搜索 |
| 螺旋搜索失败 | 最多 `8` 次，退缩高度 `3 mm` |

---

## 符号表

| 符号 | 含义 |
|---|---|
| `q, q̇ ∈ R^7` | 关节角 / 速度 |
| `M ∈ R^{7×7}` | 关节惯量矩阵 |
| `J ∈ R^{6×7}` | EE 雅可比 |
| `J₃ ∈ R^{3×7}` | 平移雅可比子矩阵 |
| `Λ ∈ R^{3×3}` | 操作空间惯量矩阵 |
| `e = [e_p, e_r] ∈ R^6` | 任务空间误差 |
| `F_des` | MPC 力前馈，3D 插入力嵌入任务力旋量 |
| `x ∈ R^6` | LCS 状态 |
| `λ ∈ R^4` | 互补接触力 |
| `φ(x)` | 间隙函数 |
| `P_k` | Riccati 矩阵 |
| `K_k` | LQR 增益 |
| `ζ` | 阻尼比 |