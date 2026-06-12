---
name: closedloop-stability-methods
description: "Closed-loop stability safeguards present in the peg-in-hole project, by tier, with code/config evidence and strict-vs-heuristic rating"
metadata: 
  node_type: memory
  type: project
  originSessionId: 26a08650-7813-4064-96a6-381ced3eadf6
---

# 闭环稳定性保障方法(本项目实际机制)

每条标注严格保证 vs 工程启发式。相关:[[taskspace-framework-analysis]] [[phase5b-osc]] [[controller-notes]]

## 一、控制律结构层(核心来源)
1. **被动性/阻抗结构** — `F=Kp·e−Kd·v_ee`,Kp,Kd 对角正定(`controller.yaml`)。EE 处虚拟弹簧-阻尼,对外无源 → 与刚性孔接触仍无源(Hogan 1985)。证据 `impedance_controller.py:118`,`operational_space_controller.py:179`。**✅ 接触稳定性定性严格**;不依赖精确接触模型,是降阶 LCS 与全阶失配仍稳的根因。
2. **惯量整形 → 精确二阶线性(5B)** — `dynamics_aware:true` 时 `F_motion=Λ a_cmd`,闭环化为解耦 `ë+Kd ė+Kp e=0`。阻尼比 ζ=Kd/(2√Kp):free(300,30)→0.87,insert(80,15)→0.84,recovery(150,20)→0.82,全部 ζ≈0.8–0.87(略欠阻尼无振荡)。**✅ 严格(标称)**;5A/纯阻抗下 ζ 是构型相关的,5B 把 ζ 变常数 = 相对 5A 的稳定性升级。

## 二、奇异/数值鲁棒化层
3. **阻尼伪逆 + Λ 正则** — IK `A=JJᵀ+lam_ik·I`(lam_ik=0.01,`:236`);惯量 `Λ⁻¹=J₃M⁻¹J₃ᵀ+ε·I`(lambda_eps=0.001,`:217`)。无界增益钳成有界。**🟡 软封顶**;也是 5B 峰值力 60N 来源,无显式可操作度监控。

## 三、级联/时标分离层
4. **内快外慢双环** — 内环高增益关节 PD(`_KP_JOINT=[600,600,400,...]`,`position_controller.py:19`)+ lookahead;外环任务空间阻抗。奇异摄动意义下时标解耦。**🟡 启发式**(未验证带宽分离比)。`ctrl=q_des` 伺服 + `qfrc_applied` 软 PD 叠加。

## 四、规划层
5. **LQR Riccati** — `_precompute_lqr()`(`lcs_mpc.py:146`),A−BK 渐近稳定(λ=0 规划)。**✅ 严格(线性 LCS)**。
6. **终端代价当 Lyapunov 罚** — `Q_terminal=Q_N_scale·Q`,Q_N_scale=5(`lcs_mpc.py:120,130`)。quasi-infinite-horizon MPC 启发式,N=8 下准稳定。**🟡 启发式**(无终端集)。

## 五、输入饱和 + 监督安全网(BIBO 兜底)
7. **全链路限幅** — τ clip ±87/12Nm(`:188`);max_cart_vel=0.25,max_joint_vel=2.0;MPC u_max/u_min/lam_max。输入有界防 runaway。**🟡** 饱和时反馈线性化失效,保证退化为局部。
8. **监督层故障安全(实践最有效)** — `max_force_abort=40N`(`task.yaml:30`);jam 检测 `jam_window_steps=50`,`lateral_force_threshold=1.5`(`task.yaml:33-36`);`mpc_max_failures=10`→spiral search;recovery 软增益平稳退出。**🟡 监督式 BIBO**(非 Lyapunov),是 ≤11mm 100% 成功的实际兜底。
9. **增益切换稳定性** — 三模式 FSM 切换,靠阶段驻留时间隐式慢切换。**🔴 无显式驻留时间/公共 Lyapunov 证明**。

## 严格 vs 启发式 总表
- ✅ 严格:阻抗无源性、5B 惯量整形 ζ≈0.85、LQR Riccati
- 🟡 有界/启发式:阻尼伪逆+Λ正则、时标分离、MPC 终端罚、限幅+监督退避
- 🔴 缺证明:增益切换驻留时间

## 补强建议(性价比序)
1. 奇异度量监控:`_compute_lambda` 处记 `cond(Λ⁻¹)` 或可操作度 `√det(JJᵀ)`,与 60N 峰值力关联 — 最划算,直接出论文图。
2. 切换稳定性:公共二次 Lyapunov 或最小驻留时间 > 模式时间常数。
3. 饱和域分析:给出力矩不饱和工作区(吸引域估计),明确严格保证边界。
