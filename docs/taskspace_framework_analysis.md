---
name: taskspace-framework-analysis
description: "Dynamics-and-control framework analysis of the operational/task-space line (Phase 3 LCS-MPC + Phase 5A/5B OSC), 8 sections"
metadata: 
  node_type: memory
  type: project
  originSessionId: 26a08650-7813-4064-96a6-381ced3eadf6
---

# 任务空间(操作空间)进展 — Dynamics-and-Control 框架分析

分析对象:任务空间主线 = Phase 3 LCS-MPC + Phase 5A/5B OSC。
落点文件:`src/controllers/operational_space_controller.py`, `src/planners/lcs_mpc.py`。
相关记忆:[[phase3-lcs-mpc]] [[phase5a-os-controller]] [[phase5b-osc]] [[closedloop-stability-methods]]

## 1. System Modeling
- **全阶(仿真真值)**:MuJoCo Panda 7-DOF,`M(q)q̈+C=τ+JᵀF_ext`;软接触 solref;peg 刚性 weld([[phase2-weld-fix]])。
- **降阶 LCS(规划用)**:`lcs_mpc.py:45-91`,`x=[ex,ey,z,vx,vy,vz]`,`u=[Fx,Fy,Fz]`,圆孔=4 墙盒近似;`x_{k+1}=Ax+Bu+Dλ, 0≤λ⊥φ=Ex+c≥0`;A/B 是带阻尼解耦双积分器。
- **操作空间惯量**:`Λ=(J₃M⁻¹J₃ᵀ+εI)⁻¹`,`_compute_lambda()` 用 solve 避免显式求逆。
- 评价:全阶校验 + 降阶实时规划 + Λ 惯量映射,三层完整。

## 2. State and Observation
- OSC 观测:`q,qdot,ee_pos,ee_rot,J,qfrc_bias,M`;`v_ee=J7@qdot`。
- LCS 状态由 EE 相对孔心误差构造;孔位来自 Phase 7 感知([[phase7c2-pointcloud-refinement]] [[phase7d-board-pose]],XY 1.2–3.5mm)。
- 薄弱点:感知误差直接进入 `Kp·e`,等效为初始偏移 → 把第8节鲁棒性与第2节耦合。

## 3. Control Law Design
三层级联:scripted FSM(高层) → OCP/LCS-MPC(中层,出 `F_des` 前馈) → OSC(底层,出力矩)。
- OSC 两态(`operational_space_controller.py:163-185`):纯阻抗 `F=Kp e+Kd(v_des−v_ee)+F_des`;惯量整形 `F_motion=Λ a_cmd`(转动仍阻抗)。统一 `τ=J⊤F+qfrc_bias` 再 clip。
- MPC 两解算器:LQR(Riccati,<1ms)+ SLSQP(伴随梯度,1–5ms),代价含居中/进度/力平滑/软互补/穿透罚。
- 三增益模式 free_space/insertion/recovery 按 FSM 调度。

## 4. Closed-Loop Dynamics
- 自由空间(惯量整形)→ 解耦二阶 `ë+Kd ė+Kp e=0`(5B 相对 5A 的核心收益:5A 各轴经真实惯量耦合)。
- 接触阶段 = LCS 互补动力学 + MPC 前馈;实测恢复尝试 2.40→1.60。

## 5. Stability(详见 [[closedloop-stability-methods]])
正定 Kp,Kd → 渐近稳定 + 无源性;LQR Riccati 保证标称稳定。风险:近奇异 Λ 放大力(PkFcmd=60.1N, peak τ=56.6Nm),τ clip 破坏精确线性化——无形式化裕度。

## 6. Safety
- 硬约束:τ 限幅 87/12Nm;MPC `u_max/u_min/lam_max`;穿透罚 w_pen=1000;IK 限速 0.25/2.0。
- 故障安全:`max_force_abort=40N`、jam 检测(`jam_window_steps=50`,`lateral_force_threshold=1.5`)→ 退避。
- 偏反应式:无前瞻式安全(CBF/可达集),接触力是事后指标。

## 7. Feasibility
- 实时:LQR<1ms 恒可行;SLSQP 冷启170ms/热启5.5ms/p95~9ms → 满足 ~20Hz。
- 优化:box 约束 keep_feasible,success 标志。运动学:阻尼伪逆 IK。几何:clearance 4mm,Phase6 13/13。
- 任务可行边界:≤11mm 可行;13mm 全退化到 spiral search。

## 8. Robustness
- 覆盖:offset 5–13mm × 10–20 seeds × 感知 easy/medium/hard。
- 模型失配靠阻抗顺应 + jam 兜底吸收。5B 惯量整形:恢复 2.40→1.60,F_mot_rms 7.2→5.4N,接触力不变。
- 退化:>11mm 各法靠搜索;感知 hard 噪声成功率 40% → 瓶颈已从控制转移到感知。

## 综合定位与下一步
成熟:建模/控制律/闭环/可行性 ✅;中等:状态观测、稳定性、安全、鲁棒性 🟡。
三个最有价值空白:(1) 稳定性形式化(Λ 条件数/可操作度监控);(2) 安全前瞻化(接触力作 MPC 硬约束/CBF);(3) 感知误差→成功率敏感度,打通第2与第8节。
