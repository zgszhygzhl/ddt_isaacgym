# utils/stair_ik_feedforward.py

import torch


def _cfg_get(cfg, name, default):
    return getattr(cfg, name, default)


def _smoothstep(x: torch.Tensor) -> torch.Tensor:
    x = torch.clamp(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def _bezier5_point(
    s: torch.Tensor,
    rw: float,
    hs: float,
    x0: float,
    x1: float,
    clear_margin: float,
    p1_z_ratio: float,
    p2_x_ratio: float,
    p3_x_ratio: float,
    p4_z_ratio: float,
):
    """Rounded vertical-first 5th-order Bezier wheel-center trajectory.

    World frame:
        x forward, z upward.

    The trajectory is designed for stair contact rescue:
        1. lift mostly upward first;
        2. transfer forward at high clearance;
        3. settle down to upper-step wheel-center height.
    """

    s = torch.clamp(s, 0.0, 1.0)

    z0 = rw
    z1 = hs + rw
    dx = x1 - x0
    z_peak = hs + rw + clear_margin

    p0_x = x0
    p0_z = z0

    p1_x = x0
    p1_z = z0 + p1_z_ratio * (z_peak - z0)

    p2_x = x0 + p2_x_ratio * dx
    p2_z = z_peak

    p3_x = x0 + p3_x_ratio * dx
    p3_z = z_peak

    p4_x = x1
    p4_z = z1 + p4_z_ratio * (z_peak - z1)

    p5_x = x1
    p5_z = z1

    b0 = (1.0 - s) ** 5
    b1 = 5.0 * s * (1.0 - s) ** 4
    b2 = 10.0 * s ** 2 * (1.0 - s) ** 3
    b3 = 10.0 * s ** 3 * (1.0 - s) ** 2
    b4 = 5.0 * s ** 4 * (1.0 - s)
    b5 = s ** 5

    xc = (
        b0 * p0_x
        + b1 * p1_x
        + b2 * p2_x
        + b3 * p3_x
        + b4 * p4_x
        + b5 * p5_x
    )
    zc = (
        b0 * p0_z
        + b1 * p1_z
        + b2 * p2_z
        + b3 * p3_z
        + b4 * p4_z
        + b5 * p5_z
    )

    return xc, zc


def _single_leg_ik_template(
    s: torch.Tensor,
    l1: float,
    l2: float,
    rw: float,
    hs: float,
    x0: float,
    x1: float,
    x_hip: float,
    z_hip: float,
    h_margin: float,
    clear_margin: float,
    p1_z_ratio: float,
    p2_x_ratio: float,
    p3_x_ratio: float,
    p4_z_ratio: float,
):
    """由模板相位 s 计算单腿 IK 绝对角。

    当前版本使用 rounded vertical-first Bezier 轮心轨迹。

    坐标定义：
        世界坐标：x 向前，z 向上；
        髋关节局部坐标：x 向前，z 向下。

    关节角定义：
        q1：大腿角，竖直向下为 0，后摆为正；
        q2：小腿角，相对大腿伸直为 0，前摆为负。
    """

    s = torch.clamp(s, 0.0, 1.0)

    # 兼容旧配置：
    # 如果没写 stair_ff_clear_margin，就用旧 stair_ff_h_margin。
    if clear_margin is None:
        clear_margin = h_margin

    xc, zc = _bezier5_point(
        s=s,
        rw=rw,
        hs=hs,
        x0=x0,
        x1=x1,
        clear_margin=clear_margin,
        p1_z_ratio=p1_z_ratio,
        p2_x_ratio=p2_x_ratio,
        p3_x_ratio=p3_x_ratio,
        p4_z_ratio=p4_z_ratio,
    )

    # 转到髋关节局部坐标
    x_rel = xc - x_hip
    z_rel = z_hip - zc

    # 二连杆 IK
    d = (x_rel ** 2 + z_rel ** 2 - l1 ** 2 - l2 ** 2) / (2.0 * l1 * l2)
    d = torch.clamp(d, -1.0, 1.0)

    q2 = torch.atan2(
        -torch.sqrt(torch.clamp(1.0 - d ** 2, min=0.0)),
        d,
    )

    q1 = torch.atan2(
        -l2 * torch.sin(q2),
        l1 + l2 * torch.cos(q2),
    ) - torch.atan2(x_rel, z_rel)

    return q1, q2


def compute_stair_ik_ff_offsets_b(
    phase_local: torch.Tensor,
    active: torch.Tensor,
    dof_names,
    num_actions: int,
    device,
    cfg_control,
):
    """B 方案 + Bezier vertical-first 轨迹 + release extension。

    phase_local u:
        0 ~ 1:
            执行 B 方案 swing:
            dq = q_IK(s_start + (1 - s_start) * u) - q_IK(0)

        1 ~ 1 + extend_ratio:
            不再继续 IK 轨迹；
            把 swing 终点 offset 平滑释放到 0。

    返回:
        ff_offsets: [num_envs, num_actions]
        单位 rad，应该加到 joint_pos_target 上。
    """

    num_envs = phase_local.shape[0]
    ff_offsets = torch.zeros(num_envs, num_actions, device=device)

    if not torch.any(active):
        return ff_offsets

    # ========== 几何参数 ==========
    l1 = float(_cfg_get(cfg_control, "stair_ff_l1", 0.25))
    l2 = float(_cfg_get(cfg_control, "stair_ff_l2", 0.25))
    rw = float(_cfg_get(cfg_control, "stair_ff_wheel_radius", 0.085))
    hs = float(_cfg_get(cfg_control, "stair_ff_step_height", 0.15))

    x0 = float(_cfg_get(cfg_control, "stair_ff_x0", 0.0))
    x1 = float(_cfg_get(cfg_control, "stair_ff_x1", 0.10))

    x_hip = float(_cfg_get(cfg_control, "stair_ff_x_hip", x0))
    z_hip = float(_cfg_get(cfg_control, "stair_ff_z_hip", 0.45))

    # 旧参数，保留兼容
    h_margin = float(_cfg_get(cfg_control, "stair_ff_h_margin", 0.055))

    # 新 Bezier 轨迹参数
    clear_margin = float(_cfg_get(cfg_control, "stair_ff_clear_margin", h_margin))
    p1_z_ratio = float(_cfg_get(cfg_control, "stair_ff_bezier_p1_z_ratio", 0.70))
    p2_x_ratio = float(_cfg_get(cfg_control, "stair_ff_bezier_p2_x_ratio", 0.05))
    p3_x_ratio = float(_cfg_get(cfg_control, "stair_ff_bezier_p3_x_ratio", 0.65))
    p4_z_ratio = float(_cfg_get(cfg_control, "stair_ff_bezier_p4_z_ratio", 0.25))

    # B 方案：触发后从模板轨迹的这个相位开始
    s_start = float(_cfg_get(cfg_control, "stair_ff_phase_start", 0.10))

    # 进入前馈的平滑时间比例
    ramp_ratio = float(_cfg_get(cfg_control, "stair_ff_ramp_ratio", 0.06))

    # 函数内部限幅；外面还有 stair_ff_k 和 stair_ff_final_max_offset
    max_offset = float(_cfg_get(cfg_control, "stair_ff_max_offset", 1.20))

    # swing 结束后的 release 阶段
    extend_enabled = bool(_cfg_get(cfg_control, "stair_ff_extend_enabled", True))
    extend_ratio = float(_cfg_get(cfg_control, "stair_ff_extend_ratio", 0.30))
    extend_ratio = max(extend_ratio, 0.0)

    phase_end = 1.0 + extend_ratio if extend_enabled else 1.0

    # ========== q_IK(0)，作为相对初始姿态的基准 ==========
    s_zero = torch.zeros(num_envs, device=device)

    q1_zero, q2_zero = _single_leg_ik_template(
        s=s_zero,
        l1=l1,
        l2=l2,
        rw=rw,
        hs=hs,
        x0=x0,
        x1=x1,
        x_hip=x_hip,
        z_hip=z_hip,
        h_margin=h_margin,
        clear_margin=clear_margin,
        p1_z_ratio=p1_z_ratio,
        p2_x_ratio=p2_x_ratio,
        p3_x_ratio=p3_x_ratio,
        p4_z_ratio=p4_z_ratio,
    )

    side_joint_names = [
        ("FL_thigh_joint", "FL_calf_joint"),
        ("FR_thigh_joint", "FR_calf_joint"),
    ]

    for side in range(2):
        side_active = active[:, side]

        if not torch.any(side_active):
            continue

        # 注意：这里允许 u 增长到 1 + extend_ratio。
        u_total = torch.clamp(phase_local[:, side], 0.0, phase_end)

        # IK swing 轨迹本身仍然只走 0~1。
        # u_total > 1 后，不再继续推进 IK，而是保持 swing 终点，然后释放 offset。
        u_swing = torch.clamp(u_total, 0.0, 1.0)

        # B 方案：局部相位 u 映射到完整模板轨迹的 s_start~1
        s = s_start + (1.0 - s_start) * u_swing

        q1, q2 = _single_leg_ik_template(
            s=s,
            l1=l1,
            l2=l2,
            rw=rw,
            hs=hs,
            x0=x0,
            x1=x1,
            x_hip=x_hip,
            z_hip=z_hip,
            h_margin=h_margin,
            clear_margin=clear_margin,
            p1_z_ratio=p1_z_ratio,
            p2_x_ratio=p2_x_ratio,
            p3_x_ratio=p3_x_ratio,
            p4_z_ratio=p4_z_ratio,
        )

        # B 方案核心：相对完整轨迹初始姿态的前馈偏置
        dq1 = q1 - q1_zero
        dq2 = q2 - q2_zero

        # swing 开始时平滑打开，避免触发瞬间太冲
        if ramp_ratio > 1e-6:
            ramp = _smoothstep(u_swing / ramp_ratio)
            dq1 = dq1 * ramp
            dq2 = dq2 * ramp

        # swing 结束后平滑释放 offset，让腿重新伸展回支撑姿态
        if extend_enabled and extend_ratio > 1e-6:
            ext_u = torch.clamp((u_total - 1.0) / extend_ratio, 0.0, 1.0)
            release_gate = 1.0 - _smoothstep(ext_u)
            dq1 = dq1 * release_gate
            dq2 = dq2 * release_gate

        # 限幅
        dq1 = torch.clamp(dq1, -max_offset, max_offset)
        dq2 = torch.clamp(dq2, -max_offset, max_offset)

        # 没激活的环境置零
        dq1 = dq1 * side_active.float()
        dq2 = dq2 * side_active.float()

        thigh_name, calf_name = side_joint_names[side]

        if thigh_name in dof_names:
            thigh_idx = dof_names.index(thigh_name)
            ff_offsets[:, thigh_idx] += dq1

        if calf_name in dof_names:
            calf_idx = dof_names.index(calf_name)
            ff_offsets[:, calf_idx] += dq2

    return ff_offsets