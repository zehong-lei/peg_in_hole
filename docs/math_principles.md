# Dynamics and Control of a Robotic Peg-in-Hole Assembly System: Mathematical Principles

**Platform.** Franka Emika Panda (7-DOF) · MuJoCo rigid-body simulation
**Architecture.** Hierarchical control: a supervisory finite-state machine (FSM) sequences manipulation phases; a planning layer generates a global pre-insertion trajectory (OCP) and a contact-aware insertion force (LCS-MPC); a low-level operational-space controller (OSC) realises task-space commands as joint torques on top of MuJoCo's affine joint servo.

> **Scope.** This document is the formal mathematical specification of the control stack. Every model, cost, and gain is stated together with its concrete realisation in the codebase so that the equations are reproducible rather than illustrative. Section cross-references to source files use the form `module.py:line`.

---

## Contents

1. [System overview and notation](#1-system-overview-and-notation)
2. [Robot dynamics and actuation model](#2-robot-dynamics-and-actuation-model)
3. [Operational-space control](#3-operational-space-control)
4. [Pre-insertion trajectory optimisation (OCP)](#4-pre-insertion-trajectory-optimisation-ocp)
5. [Contact-implicit modelling and predictive control (LCS-MPC)](#5-contact-implicit-modelling-and-predictive-control-lcs-mpc)
6. [Supervisory hybrid automaton](#6-supervisory-hybrid-automaton)
7. [Stability and safety analysis](#7-stability-and-safety-analysis)
8. [Parameter summary](#8-parameter-summary)
9. [Assumptions and limitations](#9-assumptions-and-limitations)
10. [Nomenclature](#10-nomenclature)
11. [References](#11-references)

---

## 1. System overview and notation

### 1.1 Control hierarchy

The system is organised as a three-layer hierarchy, each layer running at a distinct rate and abstraction level:

```
┌────────────────────────────────────────────────────────────────────┐
│  Supervisory layer  —  finite-state machine (§6)                     │
│  8 manipulation stages + contact-recovery sub-automaton              │
│  Selects reference, control mode, and feedforward source per stage   │
└───────────────┬────────────────────────────────────────────────────┘
                │  reference (p_des, R_des, v_des, a_des), mode, F_des
┌───────────────▼────────────────────────────────────────────────────┐
│  Planning layer                                                      │
│   • Pre-insertion OCP (§4): one-shot SLSQP trajectory, open-loop     │
│   • LCS-MPC (§5): contact-aware insertion wrench, receding horizon   │
└───────────────┬────────────────────────────────────────────────────┘
                │  task-space wrench / acceleration command
┌───────────────▼────────────────────────────────────────────────────┐
│  Low-level control                                                   │
│   • Operational-space controller (§3): τ = Jᵀ F_task + bias          │
│   • Resolved-motion-rate position controller (§3.5) for free space   │
└───────────────┬────────────────────────────────────────────────────┘
                │  τ → qfrc_applied,  q_des → ctrl
┌───────────────▼────────────────────────────────────────────────────┐
│  Plant — MuJoCo                                                      │
│  Rigid-body dynamics + affine joint servo + soft-constraint contacts │
└─────────────────────────────────────────────────────────────────────┘
```

The simulation integrates at $\Delta t_\mathrm{sim} = 2\,\mathrm{ms}$ (`task.yaml:41`) with the `implicitfast` integrator (`scene_builder.py:314`). Control and planning run once per simulation step except the LCS-MPC, which re-solves every tenth step (Section 5.6).

### 1.2 Notation

| Symbol | Meaning |
|---|---|
| $q,\dot q,\ddot q \in \mathbb{R}^7$ | joint position, velocity, acceleration |
| $M(q) \in \mathbb{R}^{7\times 7}$ | joint-space inertia (mass) matrix, symmetric positive definite |
| $C(q,\dot q)\dot q,\; g(q)$ | Coriolis/centrifugal and gravity torques |
| $h(q,\dot q) = C(q,\dot q)\dot q + g(q)$ | combined bias torque (`qfrc_bias`) |
| $\tau \in \mathbb{R}^7$ | actuated joint torque |
| $J(q)\in\mathbb{R}^{6\times7}$ | end-effector (EE) geometric Jacobian, $J=[J_v^\top\;J_\omega^\top]^\top$ |
| $J_3 = J_v \in\mathbb{R}^{3\times7}$ | translational sub-Jacobian |
| $p\in\mathbb{R}^3,\;R\in SO(3)$ | EE position and orientation |
| $v_\mathrm{ee} = J\dot q \in\mathbb{R}^6$ | EE spatial velocity |
| $\Lambda\in\mathbb{R}^{3\times3}$ | operational-space (task) inertia matrix |
| $e = [e_p;\,e_r]\in\mathbb{R}^6$ | task-space pose error |
| $F_\mathrm{task}\in\mathbb{R}^6$ | commanded task wrench |
| $F_\mathrm{des}$ | force/wrench feedforward from LCS-MPC |
| $x\in\mathbb{R}^6,\;\lambda\in\mathbb{R}^4,\;\phi(x)$ | LCS state, contact impulses, gap function |
| $P_k,K_k$ | Riccati cost-to-go matrix and LQR gain |
| $\zeta$ | closed-loop damping ratio |

Throughout, $\|\cdot\|$ denotes the Euclidean norm, $\mathrm{diag}(\cdot)$ a diagonal matrix, and $I_n$ the $n\times n$ identity. Subscript $k$ indexes discrete time within a planning horizon.

---

## 2. Robot dynamics and actuation model

### 2.1 Manipulator equations of motion

The arm obeys the standard rigid-body manipulator dynamics [Siciliano et al. 2009; Featherstone 2008]:

$$
M(q)\,\ddot q + C(q,\dot q)\,\dot q + g(q) \;=\; \tau \;+\; J(q)^\top F_\mathrm{ext},
\tag{2.1}
$$

where $F_\mathrm{ext}\in\mathbb{R}^6$ is the external wrench applied at the EE (here, contact between peg and hole). MuJoCo evaluates $M$, $h = C\dot q + g$, and $J$ analytically each step; the controller reads them via `get_mass_matrix()`, `qfrc_bias`, and `get_ee_jacobian()` (`assembly_env.py:345,359`).

### 2.2 Actuation: superposition of servo and applied torque

The net joint torque applied to the plant is the sum of two channels:

$$
\tau \;=\; \underbrace{\tau_\mathrm{servo}(q,\dot q,\,q_\mathrm{des})}_{\text{MuJoCo affine actuator, } \texttt{ctrl}}
\;+\;
\underbrace{\tau_\mathrm{applied}}_{\texttt{qfrc\_applied}} .
\tag{2.2}
$$

Both channels are written every step by the task loop: `d.ctrl[:7] = q_des` and `d.qfrc_applied[:7] = τ` (`multi_task_assembly.py:264,270`).

**Affine servo.** Each Panda joint is driven by a MuJoCo `general` actuator with `biastype="affine"` (`panda.xml:10,265–274`). For a joint actuator whose transmission length equals the joint angle, the produced force is

$$
\tau_{\mathrm{servo},j}
= \underbrace{k_{p,j}}_{\texttt{gainprm}}\,\texttt{ctrl}_j
+ \underbrace{\big(0 - k_{p,j}\,q_j - k_{d,j}\,\dot q_j\big)}_{\texttt{biasprm}=[0,\,-k_p,\,-k_d]},
$$

so with $\texttt{ctrl}_j = q_{\mathrm{des},j}$ the servo is a pure PD law with implicit velocity set-point $\dot q_\mathrm{des}=0$:

$$
\boxed{\;\tau_{\mathrm{servo},j} = k_{p,j}\,(q_{\mathrm{des},j}-q_j) - k_{d,j}\,\dot q_j,\qquad
|\tau_{\mathrm{servo},j}| \le \tau_{\max,j}.\;}
\tag{2.3}
$$

The gains, read directly from the model, are

$$
(k_p,k_d) =
\begin{cases}
(4500,\,450) & j\in\{1,2\}\\
(3500,\,350) & j\in\{3,4\}\\
(2000,\,200) & j\in\{5,6,7\}
\end{cases},
\qquad
\tau_{\max} = [87,87,87,87,12,12,12]\ \mathrm{N\,m}.
$$

**Design rationale.** The applied channel $\tau_\mathrm{applied}$ carries the model-based control (operational-space torque or gravity-compensated PD), while the stiff servo (2.3) provides a high-bandwidth inner damping/position loop and torque saturation matching the hardware. Writing a position target $q_\mathrm{des}$ to `ctrl` rather than holding the joints at the current angle lets the servo contribute useful tracking effort instead of pure damping; the position controller exploits this explicitly (`position_controller.py:56–62`).

### 2.3 Kinematics and Jacobians

The EE spatial velocity is $v_\mathrm{ee} = J(q)\,\dot q$. We partition $J = \big[J_3;\,J_\omega\big]$ with translational block $J_3\in\mathbb{R}^{3\times7}$ used for inertia shaping (Section 3.2). The orientation error between desired $R_\mathrm{des}$ and current $R$ is taken as the rotation-vector (axis–angle) of the relative rotation $R_\mathrm{err}=R_\mathrm{des}R^\top$ (`position_controller.py:92–104`):

$$
e_r = \theta\,\hat n,\qquad
\theta = \arccos\!\Big(\tfrac{\operatorname{tr}(R_\mathrm{err})-1}{2}\Big),\qquad
\hat n = \frac{1}{2\sin\theta}
\begin{bmatrix} R_{err,32}-R_{err,23}\\ R_{err,13}-R_{err,31}\\ R_{err,21}-R_{err,12}\end{bmatrix}.
\tag{2.4}
$$

This is the standard logarithmic map $\mathrm{Log}:SO(3)\to\mathbb{R}^3$, valid away from $\theta=\pi$; the small-angle branch returns $0$ for $\theta<10^{-7}$.

---

## 3. Operational-space control

The operational-space controller (OSC) [Khatib 1987] converts a task-space objective into joint torques through the Jacobian transpose. Two modes are implemented (`operational_space_controller.py`): a passive **impedance** law and, by default (`controller.yaml: dynamics_aware: true`), a **dynamically-consistent inertia-shaping** law. All gains are diagonal and mode-dependent:

| Mode | $K_p^\mathrm{pos}$ (xyz) | $K_d^\mathrm{pos}$ (xyz) | $K_p^\mathrm{rot}$ | $K_d^\mathrm{rot}$ |
|---|---|---|---|---|
| `free_space` | $300,300,300$ | $30,30,30$ | $30,30,30$ | $3,3,3$ |
| `insertion`  | $80,80,50$ | $15,15,10$ | $10,10,20$ | $1,1,2$ |
| `recovery`   | $150,150,150$ | $20,20,20$ | $20,20,20$ | $2,2,2$ |

### 3.1 Impedance mode (`dynamics_aware = False`)

$$
F_\mathrm{task} = K_p\,e + K_d\,(v_\mathrm{des}-v_\mathrm{ee}) + F_\mathrm{des},
\qquad
\tau = J^\top F_\mathrm{task} + h .
\tag{3.1}
$$

Because the bias $h=$`qfrc_bias` is added back, the closed loop renders a Cartesian mechanical impedance $(K_p,K_d)$ at the EE. With $K_p,K_d\succ0$ this is a passive spring–damper port (Section 7.1), which is the classical guarantee of stable interaction with any passive environment [Hogan 1985].

### 3.2 Inertia-shaping mode (default)

The translational channel is rendered as a **decoupled unit-mass system** by pre-multiplying the desired acceleration with the operational-space inertia matrix. Define

$$
\Lambda = \big(J_3 M^{-1} J_3^\top + \varepsilon I_3\big)^{-1},\qquad \varepsilon = 10^{-3},
\tag{3.2}
$$

computed without explicit inversion of $M$ by solving $M X = J_3^\top$ and forming $\Lambda^{-1}=J_3X+\varepsilon I$ (`operational_space_controller.py:207–218`). The commanded task acceleration and motion force are

$$
a_\mathrm{cmd} = a_\mathrm{des} + K_p^\mathrm{pos} e_p + K_d^\mathrm{pos}(v_\mathrm{des}-v_\mathrm{ee}^{xyz}),
\qquad
F_\mathrm{motion} = \Lambda\,a_\mathrm{cmd}.
\tag{3.3}
$$

The rotational channel remains an impedance, $F_\mathrm{rot}=K_p^\mathrm{rot}e_r - K_d^\mathrm{rot}\,\omega_\mathrm{ee}$, and the wrench is assembled, feedforward added, and projected to torque with saturation:

$$
\tau = J^\top\!\big[F_\mathrm{motion};\,F_\mathrm{rot}\big] + h + J^\top F_\mathrm{des},
\qquad
\tau \leftarrow \operatorname{clip}(\tau,\,\pm\tau_{\max}).
\tag{3.4}
$$

(In `insertion` mode with no MPC feedforward, a constant downward bias $F_z=-8\,\mathrm{N}$ is applied instead, `operational_space_controller.py:184–185`.)

**Exact-linearisation property.** Substituting (3.4) without saturation into the dynamics (2.1) with $F_\mathrm{ext}=0$, the bias cancels exactly and

$$
M\ddot q = J^\top F_\mathrm{task}
\;\Rightarrow\;
\ddot q = M^{-1}\big(J_3^\top F_\mathrm{motion} + J_\omega^\top F_\mathrm{rot}\big).
$$

The translational task acceleration is $\ddot p = J_3\ddot q + \dot J_3\dot q$. Neglecting the velocity-product term $\dot J_3\dot q$ and the translation–rotation coupling $J_3M^{-1}J_\omega^\top F_\mathrm{rot}$ (both small at the low task speeds used here) and ignoring the regulariser $\varepsilon$,

$$
\ddot p \approx J_3 M^{-1} J_3^\top\,\Lambda\,a_\mathrm{cmd} = a_\mathrm{cmd}.
\tag{3.5}
$$

With $e_p=p_\mathrm{des}-p$, $\dot e_p=v_\mathrm{des}-v_\mathrm{ee}^{xyz}$, and $a_\mathrm{des}=\ddot p_\mathrm{des}$, the error obeys a **decoupled second-order system per axis**:

$$
\boxed{\;\ddot e_p + K_d^\mathrm{pos}\,\dot e_p + K_p^\mathrm{pos}\,e_p = 0.\;}
\tag{3.6}
$$

Because $K_p^\mathrm{pos},K_d^\mathrm{pos}$ are diagonal, each Cartesian axis is an independent linear oscillator with natural frequency $\omega_n=\sqrt{K_p}$ and damping ratio

$$
\zeta = \frac{K_d}{2\sqrt{K_p}} .
\tag{3.7}
$$

Evaluating per mode and axis:

| Mode / axis | $K_p$ | $K_d$ | $\omega_n$ [rad/s] | $\zeta$ |
|---|---|---|---|---|
| free_space (x,y,z) | 300 | 30 | 17.3 | **0.87** |
| insertion (x,y) | 80 | 15 | 8.94 | 0.84 |
| insertion (z) | 50 | 10 | 7.07 | **0.71** |
| recovery (x,y,z) | 150 | 20 | 12.2 | 0.82 |

All axes are sub-critically but well damped, $\zeta\in[0.71,\,0.87]$; the most compliant case is the insertion $z$-axis, deliberately the softest to limit insertion contact force.

**Acceleration feedforward.** $a_\mathrm{des}$ is supplied by the trajectory tracker as the finite difference of the planned velocity profile, $a_\mathrm{des}(t)=\big(v_\mathrm{des}(t+\Delta t)-v_\mathrm{des}(t)\big)/\Delta t$ (`trajectory_utils.py:90–105`). It removes the steady-state lag that pure PD tracking would otherwise incur on a moving reference.

### 3.3 Null-space and servo target

The OSC torque (3.4) does not by itself specify the redundant null-space motion of the 7-DOF arm. The redundancy is resolved implicitly by the inner servo: the controller additionally returns a joint target $q_\mathrm{des}$ obtained from one step of damped resolved-motion-rate inverse kinematics (`operational_space_controller.py:220–239`),

$$
v_\mathrm{cmd}=[k_p^\mathrm{ik}e_p;\,k_r^\mathrm{ik}e_r],\quad
\Delta q = J^\top\big(JJ^\top+\lambda I\big)^{-1}v_\mathrm{cmd},\quad
q_\mathrm{des}=q+\Delta q\,t_\mathrm{look},
\tag{3.8}
$$

with $\lambda=0.01$ (Tikhonov damping for robustness near kinematic singularities) and look-ahead $t_\mathrm{look}=0.3\,\mathrm{s}$. This $q_\mathrm{des}$ feeds the servo (2.3), which regularises the null space and adds damping.

### 3.4 Free-space position controller

For pure free-space tracking (e.g. pregrasp, lift, retreat) the system uses a resolved-motion-rate Cartesian controller (`position_controller.py`) rather than the OSC: a damped pseudo-inverse maps a clamped Cartesian velocity command to joint velocities, integrated over a look-ahead to a target $q_\mathrm{des}$, and tracked by a gravity-compensated joint PD,

$$
\tau = h + K_p^\mathrm{joint}(q_\mathrm{des}-q) - K_d^\mathrm{joint}\dot q,
\tag{3.9}
$$

with $K_p^\mathrm{joint}=[600,600,400,400,200,80,80]$, $K_d^\mathrm{joint}=[60,60,40,40,20,8,8]$. The Cartesian velocity is clamped to $0.25\,\mathrm{m/s}$ and joint velocity to $2\,\mathrm{rad/s}$. The selection between this controller and the OSC is made per-stage by the supervisor (Section 6).

---

## 4. Pre-insertion trajectory optimisation (OCP)

When the FSM enters `MOVE_TO_PREINSERT`, a finite-horizon optimal control problem is solved **once** and tracked open-loop (`preinsert_ocp.py`). The problem plans the EE Cartesian path from the current pose to a waypoint above the hole while keeping the carried peg clear of the board.

### 4.1 Condensed (single-shooting) formulation

The kinematic model is a first-order integrator $p_{k+1}=p_k+\Delta t\,v_k$. Rather than treating positions as decision variables under equality constraints, the velocities are the **only** decision variables, $z=[v_0;\dots;v_{N-1}]\in\mathbb{R}^{3N}$, and positions are reconstructed analytically by prefix sum (single shooting / condensing) [Nocedal & Wright 2006; Betts 2010]:

$$
p_k = p_0 + \Delta t\sum_{j=0}^{k-1} v_j .
\tag{4.1}
$$

This eliminates the $N$ dynamics equality constraints exactly, reducing the problem to a box-constrained nonlinear program — far cheaper for an SLSQP solver. The map $z\mapsto p$ being affine also yields closed-form gradients (Section 4.3).

### 4.2 Objective

$$
J(z) = \underbrace{Q_T\|p_N-p_g\|^2}_{\text{terminal}}
+ \underbrace{Q_v\!\sum_k\|v_k\|^2}_{\text{effort}}
+ \underbrace{Q_s\!\sum_{k\ge1}\|v_k-v_{k-1}\|^2}_{\text{smoothness (accel proxy)}}
+ \underbrace{w_\mathrm{cl}\!\sum_k w_k\,\max(0,\,z_\mathrm{cl}-p_k^z)^2}_{\text{board clearance}}
+ \underbrace{w_\mathrm{ws}\!\sum_{k,d}\big[\cdots\big]}_{\text{workspace box}} .
\tag{4.2}
$$

The clearance term keeps the EE above a height that guarantees the peg tip clears the board:

$$
z_\mathrm{cl} = z_\mathrm{board}^\mathrm{top} + 2\,\ell_\mathrm{peg}
= 0.310 + 2(0.070) = 0.450\ \mathrm{m},
$$

(`preinsert_ocp.py:61–66`). Its per-knot weight is gated by lateral distance to the hole,

$$
w_k = \min\!\Big(1,\ \tfrac{\|p_k^{xy}-p_\mathrm{hole}^{xy}\|}{0.05}\Big),
$$

so the constraint relaxes once the EE is directly above the hole and must descend. **Crucially, $w_k$ is precomputed from the straight-line initial guess and held fixed** during optimisation (`preinsert_ocp.py:92–101`); this keeps the cost — and therefore the gradient — analytic, at the price of a mild conservatism. The workspace term is a quadratic exterior penalty enforcing soft box bounds on $p_k$.

Parameters (`task.yaml:48–67`): $N=30$, $\Delta t = 0.10\,\mathrm{s}$, $Q_T=2000$, $Q_v=0.5$, $Q_s=5$, $w_\mathrm{cl}=500$, $w_\mathrm{ws}=100$.

### 4.3 Analytic gradient

Because $p_k$ is affine in $z$, every cost term differentiates in closed form, and the prefix-sum structure makes the gradient computable in $O(N)$ using **suffix sums** (the adjoint of `cumsum`). Two representative terms:

$$
\frac{\partial}{\partial v_j}\,Q_T\|p_N-p_g\|^2 = 2Q_T\,\Delta t\,(p_N-p_g)\quad\forall j,
$$

(identical for all $j$ because $p_N$ depends on the full sum), and for the clearance term

$$
\frac{\partial J_\mathrm{cl}}{\partial v_{j,z}} = -2\,w_\mathrm{cl}\,\Delta t\!\!\sum_{k\ge j} w_k\,\max(0,z_\mathrm{cl}-p_k^z),
$$

evaluated as a reverse cumulative sum (`preinsert_ocp.py:138–149`). Supplying this exact Jacobian (`jac=True`) is what allows convergence in roughly $1\,\mathrm{ms}$.

### 4.4 Solver and constraints

The only hard constraints are velocity box bounds, enforced feasibly throughout (`keep_feasible=True`):

$$
\|v_k\|_\infty \le [0.15,\,0.15,\,0.10]\ \mathrm{m/s}.
$$

The problem is solved with SLSQP (sequential least-squares quadratic programming, a quasi-Newton SQP method [Kraft 1988]), warm-started from the previous solution or, on the first call, from the straight-line velocity $v_\mathrm{init}=(p_g-p_0)/(N\Delta t)$ (`preinsert_ocp.py:153–168`). The result is wrapped in a time-indexed `TrajectoryTracker` that provides interpolated $p_\mathrm{des}, v_\mathrm{des}, a_\mathrm{des}$ to the OSC (Section 3.2), realising open-loop feedforward tracking.

---

## 5. Contact-implicit modelling and predictive control (LCS-MPC)

During `INSERT`, the lateral guidance and insertion force are computed by a model-predictive controller built on a **reduced-order linear complementarity system (LCS)** that captures peg–hole wall contact (`lcs_mpc.py`). This is a tractable instance of contact-implicit / through-contact control [Stewart & Trinkle 1996; Posa et al. 2014; Aydınoğlu et al. 2022].

### 5.1 Reduced state and the complementarity model

The 7-DOF arm is abstracted to a 3-D point mass at the peg tip, with state and control

$$
x = [\,e_x,\,e_y,\,z,\,\dot e_x,\,\dot e_y,\,\dot z\,]^\top\in\mathbb{R}^6,
\qquad
u = [\,F_x,\,F_y,\,F_z\,]^\top\in\mathbb{R}^3 .
$$

Here $(e_x,e_y)$ are the **lateral position errors relative to the hole centre** (target $0$), $z$ is the **insertion depth measured from the INSERT entry point** (target $z_g$, deeper is positive), and the velocity components follow because the hole is fixed, $\dot e_x=\dot p_x$ etc. (`lcs_mpc.py:19–27`, `scripted_planner.py:522–544`). Expressing the lateral coordinates relative to the hole is what makes the contact model independent of the hole location.

The discrete-time LCS is

$$
x_{k+1} = A x_k + B u_k + D\lambda_k,
\qquad
0 \le \lambda_k \;\perp\; \phi(x_k) = E x_k + c \ge 0 .
\tag{5.1}
$$

The complementarity condition $0\le\lambda\perp\phi\ge0$ encodes the unilateral nature of contact: a wall exerts force ($\lambda_i>0$) only when its gap is closed ($\phi_i=0$), and never pulls.

### 5.2 Model matrices

With effective mass $m$, lateral/axial damping $b_{xy},b_z$, and step $\Delta t$,

$$
A = \begin{bmatrix} I_3 & \Delta t\,I_3\\[2pt] 0 & \operatorname{diag}(\alpha_x,\alpha_x,\alpha_z)\end{bmatrix},
\quad
B = \begin{bmatrix} 0\\[2pt] \tfrac{\Delta t}{m} I_3\end{bmatrix},
\quad
\alpha = 1-\frac{b\,\Delta t}{m}.
\tag{5.2}
$$

The upper block is kinematic integration; the lower block is damped force-to-velocity (semi-implicit Euler on a damped point mass). Numerically (`lcs_mpc.py:35–57`, `task.yaml:72–76`), with $m=0.5\,\mathrm{kg}$, $b_{xy}=8$, $b_z=12$, $\Delta t=0.02\,\mathrm{s}$:

$$
\alpha_x = 1-\frac{8(0.02)}{0.5} = 0.68,\qquad
\alpha_z = 1-\frac{12(0.02)}{0.5} = 0.52,\qquad
\frac{\Delta t}{m} = 0.04 .
$$

The four-wall (box) approximation of the circular hole gives the contact maps

$$
E = \begin{bmatrix} -1&0&0&0&0&0\\ 1&0&0&0&0&0\\ 0&-1&0&0&0&0\\ 0&1&0&0&0&0\end{bmatrix},
\quad c = \begin{bmatrix} c_0\\ c_0\\ c_0\\ c_0\end{bmatrix},
\quad
\phi(x)=\begin{bmatrix} c_0-e_x\\ c_0+e_x\\ c_0-e_y\\ c_0+e_y\end{bmatrix},
\tag{5.3}
$$

so $\phi_i\to0$ exactly when the peg reaches a wall, $|e_\bullet|=c_0$. The radial clearance is the peg–hole gap,

$$
c_0 = r_\mathrm{hole}-r_\mathrm{peg} = 0.014 - 0.010 = 0.004\ \mathrm{m},
$$

(`scripted_planner.py:169`). The impulse map $D$ injects each wall's restoring force into the corresponding lateral velocity with the correct sign (`lcs_mpc.py:59–63`),

$$
D_{4,0}=-\tfrac{\Delta t}{m},\;D_{4,1}=+\tfrac{\Delta t}{m},\;D_{5,2}=-\tfrac{\Delta t}{m},\;D_{5,3}=+\tfrac{\Delta t}{m}
$$

(rows index $\dot e_x,\dot e_y$), so a closed $+x$ wall ($\phi_1=c_0-e_x=0$) drives $\dot e_x$ negative, back toward the centre.

### 5.3 Receding-horizon optimal control

The depth coordinate $z$ does not enter $\phi$ (insertion is decoupled from wall contact), so for planning the contact term is dropped, $\lambda_k\equiv0$, and the controller solves a finite-horizon LQR around the goal $x^\star=[0,0,z_g,0,0,0]$:

$$
\min_{u_{0:N-1}}\;\sum_{k=0}^{N-1}\Big[(x_k-x^\star)^\top Q(x_k-x^\star)+u_k^\top R\,u_k\Big]
+(x_N-x^\star)^\top Q_N(x_N-x^\star),
\tag{5.4}
$$

subject to $x_{k+1}=Ax_k+Bu_k$. Parameters (`lcs_mpc.py:92–109`, `task.yaml:77–86`): $N=8$, $z_g=0.075\,\mathrm{m}$ (= `preinsert_z_offset` $+$ `insertion_depth_goal` $-\ \ell_\mathrm{peg}$ $=0.10+0.045-0.070$), $Q=\operatorname{diag}(500,500,200,1,1,1)$, $R=0.01\,I_3$, $Q_N=5Q$. The lateral weights dominate, encoding "centre first, then descend".

### 5.4 Backward Riccati recursion and the receding-horizon law

The solution is the time-varying LQR gain computed offline by the discrete backward Riccati recursion (`lcs_mpc.py:120–130`), initialised with $P_N=Q_N$:

$$
K_k = \big(R+B^\top P_{k+1}B\big)^{-1}B^\top P_{k+1}A,
\qquad
P_k = Q + A^\top P_{k+1}A - A^\top P_{k+1}B\,K_k .
\tag{5.5}
$$

The online law is a clipped linear feedback applied in receding-horizon fashion (only $u_0$ executed):

$$
u_k^\star = \operatorname{clip}\!\big(-K_k(x_k-x^\star),\;u_{\min},\,u_{\max}\big),
\qquad
u\in[-5,5]^2\times[0.5,10]\ \mathrm{N}.
\tag{5.6}
$$

The asymmetric, strictly positive $F_z$ bound encodes that insertion force is always downward and bounded. Each solve is a single recursion plus a forward roll-out, completing in well under $1\,\mathrm{ms}$.

### 5.5 Stability of the nominal contact-free loop

For the time-invariant infinite-horizon version, the algebraic Riccati equation has a unique stabilising solution $P\succ0$ whenever $(A,B)$ is stabilisable and $(A,Q^{1/2})$ is detectable. Both hold here ($B$ excites all velocity states; $Q\succ0$). The optimal closed loop $A-BK$ is then Schur-stable (all eigenvalues inside the unit disk), with Lyapunov function $V(x)=x^\top Px$ satisfying $V(x_{k+1})-V(x_k)=-x^\top(Q+K^\top RK)x<0$ [Anderson & Moore 1990]. The finite-horizon gain $K_0$ inherits this near the goal. Contact, when it occurs, only adds the *dissipative* restoring impulses of (5.3); it does not destabilise the nominal design, which is the rationale for planning with $\lambda=0$ and handling residual contact through the OSC compliance and the supervisor.

### 5.6 Coupling to the low-level controller

The first control $u_0$ is exposed as `F_des` and embedded into the OSC task wrench, with the LCS insertion axis mapped to world $-z$ (`scripted_planner.py:600–608`):

$$
F_\mathrm{des} = [\,u_{0,x},\;u_{0,y},\;-u_{0,z},\;0,0,0\,]^\top,
$$

added in (3.4). Thus the MPC decides *how hard and which way to push*; the OSC turns that wrench into a dynamically-consistent, saturated joint torque. The MPC re-solves once every `mpc_freq_ratio = 10` simulation steps (i.e. at $\Delta t = 0.02\,\mathrm{s}$, matching the model step), reusing $u_0$ between solves (`scripted_planner.py:573–582`). If the solver fails `mpc_max_failures = 10` times in succession, the layer disables itself and the controller falls back to constant-force compliant insertion (`scripted_planner.py:584–590`).

---

## 6. Supervisory hybrid automaton

The overall task is a hybrid dynamical system: a discrete state machine (`scripted_planner.py`) selects, for each phase, the reference, the active low-level controller, and the feedforward source. Together with the continuous dynamics this forms a hybrid automaton whose guards are geometric/force conditions and whose resets are stage transitions.

### 6.1 Primary stages

| # | Stage | Reference / behaviour | Controller (mode) |
|---|---|---|---|
| 0 | `MOVE_TO_PREGRASP` | EE above peg, gripper open | position |
| 1 | `GRASP` | descend to peg centre, close gripper, activate grasp weld | position |
| 2 | `LIFT` | raise to lift height | position |
| 3 | `MOVE_TO_PREINSERT` | track OCP trajectory (§4) to waypoint above hole | OSC `free_space` + $v_\mathrm{des},a_\mathrm{des}$ |
| 4 | `ALIGN` | fine XY centring over hole | position |
| 5 | `INSERT` | LCS-MPC force feedforward descent (§5) + recovery | OSC `insertion` + $F_\mathrm{des}$ |
| 6 | `RELEASE` | open gripper, deactivate weld | position |
| 7 | `RETREAT` | raise and withdraw | position |

Transitions fire on position/orientation tolerances (`pos_tol = 5\,\mathrm{mm}`, `align_xy_tol = 2\,\mathrm{mm}`) or, terminally, on per-stage timeouts (`stage_timeout`, `task.yaml:31`). The grasp is modelled as a rigid MuJoCo weld equality with stiff soft-constraint parameters `solref=[0.004,1]`, `solimp=[0.999,0.9999,...]` (`scene_builder.py:386–387`), activated at grasp and removed at release.

### 6.2 Contact-recovery sub-automaton

Within `INSERT`, a nested state machine `{descend, retract, search}` provides robustness to jamming (`scripted_planner.py:399–492`):

- **Force abort / retract guard.** If $\|F_\mathrm{ext}^{xyz}\|>F_\mathrm{abort}=40\,\mathrm{N}$, descent halts and the EE retracts by `retract_height = 3\,\mathrm{mm}`.
- **Jam detection.** A jam is declared only when lateral force persists *and* descent stalls: $\|F_\mathrm{ext}^{xy}\|>1.5\,\mathrm{N}$ sustained over `jam_window_steps = 50` consecutive steps **and** the depth gained in that window is below $30\%$ of the commanded descent. This conjunction distinguishes a true jam from benign rim-rubbing while sliding in.
- **Spiral search.** After a retract, the lateral target steps to the next offset of a precomputed outward spiral (centre → 2 mm ring → 4 mm ring → 6 mm ring, cardinal then diagonal; `scripted_planner.py:32–54`), then descent resumes from the new offset. Up to `max_attempts = 8` retract–search cycles are allowed before declaring failure (`search_exhausted` / `jam_max_recovery`).

The spiral template is scaled at construction to a configurable maximum radius. This converts an otherwise open-loop insertion into a simple but effective active search strategy under pose uncertainty.

---

## 7. Stability and safety analysis

This section summarises the guarantees by control component and rates each as **strict** (provable for the nominal model) or **heuristic** (engineering safeguard).

### 7.1 Passivity of impedance interaction — *strict*

With $K_p,K_d\succ0$, the impedance law (3.1) with exact bias compensation renders the storage function $V=\tfrac12 e_p^\top K_p e_p$, whose power balance along the closed loop yields a passive port with respect to the EE force–velocity pair. Coupling a passive controller to any passive environment is $L_2$-stable [Hogan 1985; Colgate & Hogan 1988]. This underpins safe contact during insertion and recovery.

### 7.2 Operational-space error dynamics — *strict (nominal)*

Under exact linearisation (3.5)–(3.6), each Cartesian axis is a Hurwitz second-order system with $\zeta\in[0.71,0.87]$ (Section 3.2). The approximation neglects $\dot J\dot q$ and task coupling; these are bounded and small at the operating speeds, so the guarantee is local and nominal rather than global.

### 7.3 LQR/Riccati optimality — *strict (nominal)*

$P_N\succ0$ and the Riccati recursion (5.5) give a Schur-stable nominal contact-free closed loop with quadratic Lyapunov certificate (Section 5.5).

### 7.4 Inertia regularisation and saturation — *local guarantee*

The term $\varepsilon I$ in (3.2) bounds $\|\Lambda\|$ near kinematic singularities where $J_3M^{-1}J_3^\top$ loses rank, preventing unbounded force commands; torque saturation (3.4) and the damped IK pseudo-inverse (3.8) provide further local robustness. These bound the response but do not by themselves certify global stability.

### 7.5 Hard constraints

$$
\tau_{\max}=[87,87,87,87,12,12,12]\ \mathrm{N\,m},\quad
\|v_\mathrm{cmd}^{xyz}\|\le 0.25\ \mathrm{m/s},\quad
u\in[-5,5]^2\times[0.5,10]\ \mathrm{N}.
$$

### 7.6 Supervisory safeguards — *heuristic*

The force-abort threshold ($40\,\mathrm{N}$), jam detector (sustained lateral force $+$ stall over 50 steps), spiral search (≤ 8 attempts), per-stage timeouts, and MPC self-disable on repeated solver failure together form a monitoring layer that bounds worst-case behaviour and guarantees episode termination, complementing the model-based guarantees above.

---

## 8. Parameter summary

**Simulation.** $\Delta t_\mathrm{sim}=2\,\mathrm{ms}$; integrator `implicitfast`; gravity $9.81\,\mathrm{m/s^2}$.

**Geometry.** board top $z=0.310\,\mathrm{m}$; default peg radius $10\,\mathrm{mm}$, half-length $70\,\mathrm{mm}$; hole radius $14\,\mathrm{mm}$; radial clearance $c_0=4\,\mathrm{mm}$; insertion depth goal $45\,\mathrm{mm}$.

**Servo (per joint group 1–2 / 3–4 / 5–7).** $K_p = 4500/3500/2000$; $K_d = 450/350/200$; $\tau_{\max}=87/87/12\,\mathrm{N\,m}$.

**OSC.** see Section 3 table; $\varepsilon=10^{-3}$; `insert_force_z` $=8\,\mathrm{N}$; $\zeta\in[0.71,0.87]$.

**OCP.** $N=30$, $\Delta t=0.10\,\mathrm{s}$, $Q_T=2000$, $Q_v=0.5$, $Q_s=5$, $w_\mathrm{cl}=500$, $w_\mathrm{ws}=100$; $z_\mathrm{cl}=0.450\,\mathrm{m}$; $\|v\|_\infty\le[0.15,0.15,0.10]$.

**LCS-MPC.** $m=0.5$, $b_{xy}=8$, $b_z=12$, $\Delta t=0.02\,\mathrm{s}$ ($\alpha_x=0.68,\alpha_z=0.52$); $N=8$, $Q=\operatorname{diag}(500,500,200,1,1,1)$, $R=0.01I$, $Q_N=5Q$, $z_g=0.075$; $u\in[-5,5]^2\times[0.5,10]$; re-solve every 10 sim steps.

**Recovery.** $F_\mathrm{abort}=40\,\mathrm{N}$; lateral threshold $1.5\,\mathrm{N}$; jam window 50 steps; retract $3\,\mathrm{mm}$; ≤ 8 attempts; spiral radius up to $6\,\mathrm{mm}$.

---

## 9. Assumptions and limitations

1. **Decoupled rotation.** The OSC inertia shaping is applied to translation only; orientation is impedance-controlled and the translation–rotation inertia coupling (3.5) is neglected. Valid at low task speeds; degrades for fast reorientation.
2. **Single shooting.** The OCP error accumulates along the prefix sum (4.1). Acceptable for the short horizon ($N=30$) and trivial linear dynamics here; not advisable for long-horizon or stiff dynamics.
3. **Frozen clearance weights.** The OCP clearance gating $w_k$ is fixed from the initial guess to preserve analytic gradients (Section 4.2), introducing mild conservatism if the optimal path deviates strongly from the straight line.
4. **Reduced LCS.** The contact model is a 3-D point mass with a four-wall hole and planning-time $\lambda=0$. It captures lateral restoring behaviour and insertion progress but not friction cones, peg tilt, or two-point wedging; those are handled empirically by OSC compliance and the recovery search.
5. **Nominal stability.** The strict guarantees (Sections 7.1–7.3) hold for the linearised/nominal models; the integrated closed loop is validated empirically in simulation rather than certified globally.
6. **Quasi-static regime.** Gains and clearances assume slow, near-quasi-static insertion (descent speed $8\,\mathrm{mm/s}$), consistent with the classical analysis of compliant peg-in-hole [Whitney 1982].

---

## 10. Nomenclature

| Symbol | Meaning | Symbol | Meaning |
|---|---|---|---|
| $q,\dot q,\ddot q$ | joint position/velocity/acceleration | $\Lambda$ | operational-space inertia |
| $M, h$ | mass matrix, bias torque | $e_p,e_r$ | task position/orientation error |
| $J, J_3$ | full / translational Jacobian | $a_\mathrm{cmd},a_\mathrm{des}$ | commanded / desired task accel. |
| $\tau,\tau_\mathrm{servo}$ | torque, servo torque | $F_\mathrm{task},F_\mathrm{des}$ | task wrench, MPC feedforward |
| $k_p,k_d$ | servo gains | $x,u,\lambda,\phi$ | LCS state / control / impulse / gap |
| $K_p,K_d$ | OSC task gains | $A,B,D,E,c$ | LCS model matrices |
| $\zeta,\omega_n$ | damping ratio, natural freq. | $P_k,K_k$ | Riccati matrix, LQR gain |
| $z$ (OCP) | decision velocities $[v_0..v_{N-1}]$ | $c_0$ | peg–hole radial clearance |

---

## 11. References

1. O. Khatib, "A unified approach for motion and force control of robot manipulators: The operational space formulation," *IEEE J. Robotics and Automation*, 3(1):43–53, 1987.
2. N. Hogan, "Impedance control: An approach to manipulation, Parts I–III," *ASME J. Dynamic Systems, Measurement, and Control*, 107:1–24, 1985.
3. J. E. Colgate and N. Hogan, "Robust control of dynamically interacting systems," *Int. J. Control*, 48(1):65–88, 1988.
4. R. Featherstone, *Rigid Body Dynamics Algorithms*, Springer, 2008.
5. B. Siciliano, L. Sciavicco, L. Villani, G. Oriolo, *Robotics: Modelling, Planning and Control*, Springer, 2009.
6. M. W. Spong, S. Hutchinson, M. Vidyasagar, *Robot Modeling and Control*, Wiley, 2006.
7. E. Todorov, T. Erez, Y. Tassa, "MuJoCo: A physics engine for model-based control," *IROS*, 2012.
8. D. E. Stewart and J. C. Trinkle, "An implicit time-stepping scheme for rigid body dynamics with inelastic collisions and Coulomb friction," *Int. J. Numerical Methods in Engineering*, 39:2673–2691, 1996.
9. M. Posa, C. Cantu, R. Tedrake, "A direct method for trajectory optimization of rigid bodies through contact," *Int. J. Robotics Research*, 33(1):69–81, 2014.
10. A. Aydınoğlu, V. M. Preciado, M. Posa, "Contact-implicit model predictive control via linear complementarity systems," *IEEE Trans. Robotics / RA-L*, 2022.
11. B. D. O. Anderson and J. B. Moore, *Optimal Control: Linear Quadratic Methods*, Prentice Hall, 1990.
12. J. Nocedal and S. J. Wright, *Numerical Optimization*, 2nd ed., Springer, 2006.
13. D. Kraft, "A software package for sequential quadratic programming," DFVLR-FB 88-28, 1988.
14. J. T. Betts, *Practical Methods for Optimal Control and Estimation Using Nonlinear Programming*, 2nd ed., SIAM, 2010.
15. D. E. Whitney, "Quasi-static assembly of compliantly supported rigid parts," *ASME J. Dynamic Systems, Measurement, and Control*, 104:65–77, 1982.
16. M. T. Mason, "Compliance and force control for computer controlled manipulators," *IEEE Trans. Systems, Man, and Cybernetics*, 11(6):418–432, 1981.
