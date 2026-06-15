# utils/stair_ik_feedforward.py

import math
import torch


def _cfg_get(cfg, name, default):
    return getattr(cfg, name, default)


def _smoothstep(x: torch.Tensor) -> torch.Tensor:
    x = torch.clamp(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


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
):
    """
    由完整模板相位 s 计算单腿 IK 绝对角。

    坐标定义：
        世界坐标：x 向前，z 向上；
        髋关节局部坐标：x 向前，z 向下。

    关节角定义：
        q1：大腿角，竖直向下为 0，后摆为正；
        q2：小腿角，相对大腿伸直为 0，前摆为负。

    输入：
        s: [num_envs]，范围 0~1。

    输出：
        q1_abs, q2_abs: [num_envs]，单位 rad。
    """

    s = torch.clamp(s, 0.0, 1.0)
    pi = math.pi

    # 摆线进度函数
    sigma = s - torch.sin(2.0 * pi * s) / (2.0 * pi)

    # 轮心高度
    z0 = rw
    z1 = hs + rw

    # 15 cm 台阶时，原来的抬腿高度一般不够，所以这里按台阶高度自适应
    h_lift = 0.5 * hs + h_margin

    # 世界坐标下轮心轨迹
    xc = x0 + (x1 - x0) * sigma
    zc = z0 + (z1 - z0) * sigma + h_lift * torch.sin(pi * s) ** 2

    # 转到髋关节局部坐标
    x_rel = xc - x_hip
    z_rel = z_hip - zc

    # 二连杆 IK
    D = (x_rel ** 2 + z_rel ** 2 - l1 ** 2 - l2 ** 2) / (2.0 * l1 * l2)
    D = torch.clamp(D, -1.0, 1.0)

    q2 = torch.atan2(
        -torch.sqrt(torch.clamp(1.0 - D ** 2, min=0.0)),
        D,
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
    """
    B 方案：触发后直接从完整轨迹的 s_start 相位开始，
    但是前馈偏置按 q_IK(s) - q_IK(0) 计算。

    这意味着：
        1. 触发后会立刻给一部分抬腿偏置；
        2. 前馈输出是 rad 单位的关节目标偏置；
        3. 这个偏置应该加到 joint_pos_target 上，而不是直接作为 action。

    参数：
        phase_local: [num_envs, 2]
            触发后的局部相位 u，范围 0~1。
            第 0 列对应 FL， 第 1 列对应 FR。

        active: [num_envs, 2]
            左右腿前馈是否激活。

        dof_names:
            当前环境的 dof_names。

        num_actions:
            action 维度，一般是 8。

        cfg_control:
            self.cfg.control。

    返回：
        ff_offsets: [num_envs, num_actions]
            单位 rad。
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

    x0 = float(_cfg_get(cfg_control, "stair_ff_x0", 0.05))
    x1 = float(_cfg_get(cfg_control, "stair_ff_x1", 0.35))

    # 初始时让髋关节在轮心正上方
    x_hip = float(_cfg_get(cfg_control, "stair_ff_x_hip", x0))
    z_hip = float(_cfg_get(cfg_control, "stair_ff_z_hip", 0.40))

    h_margin = float(_cfg_get(cfg_control, "stair_ff_h_margin", 0.05))

    # B 方案：触发后从模板轨迹的这个相位开始
    s_start = float(_cfg_get(cfg_control, "stair_ff_phase_start", 0.30))

    # 进入前馈的平滑时间比例。
    # B 方案允许初始跳变，所以这个值不要太大；0.05~0.15 比较合适。
    ramp_ratio = float(_cfg_get(cfg_control, "stair_ff_ramp_ratio", 0.10))

    # 前馈偏置限幅，防止 IK 偏置过大直接抽腿
    max_offset = float(_cfg_get(cfg_control, "stair_ff_max_offset", 0.65))

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
    )

    side_joint_names = [
        ("FL_thigh_joint", "FL_calf_joint"),
        ("FR_thigh_joint", "FR_calf_joint"),
    ]

    for side in range(2):
        side_active = active[:, side]

        if not torch.any(side_active):
            continue

        # 触发后的局部相位 u: 0~1
        u = torch.clamp(phase_local[:, side], 0.0, 1.0)

        # B 方案：局部相位 u 映射到完整模板轨迹的 s_start~1
        s = s_start + (1.0 - s_start) * u

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
        )

        # B 方案核心：相对完整轨迹初始姿态的前馈偏置
        dq1 = q1 - q1_zero
        dq2 = q2 - q2_zero

        # 平滑打开，避免触发瞬间太冲
        if ramp_ratio > 1e-6:
            ramp = _smoothstep(u / ramp_ratio)
            dq1 = dq1 * ramp
            dq2 = dq2 * ramp

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