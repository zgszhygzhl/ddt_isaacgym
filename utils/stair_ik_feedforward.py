# utils/stair_ik_feedforward.py

import math

import torch


def _cfg_get(cfg, name, default):
    return getattr(cfg, name, default)


def _smoothstep(x: torch.Tensor) -> torch.Tensor:
    x = torch.clamp(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def _clamp_float(x: float, lo: float, hi: float) -> float:
    return min(max(x, lo), hi)


def _solve_leg_ik_from_point(
    xc: torch.Tensor,
    zc: torch.Tensor,
    l1: float,
    l2: float,
    x_hip: float,
    z_hip: float,
):
    """
    Solve two-link IK from wheel-center point.

    World frame:
        x forward positive
        z upward positive

    Hip local frame:
        x forward positive
        z_down downward positive

    Joint convention:
        q1: thigh angle, vertical-down is 0, rear swing positive
        q2: calf angle, straight is 0, forward bend negative
    """
    x_rel = xc - x_hip
    z_rel = z_hip - zc

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
    """
    Rounded vertical-first 5th-order Bezier wheel-center trajectory.

    It is not a hard vertical-horizontal-vertical broken line.
    It is a smooth trajectory with:
        1. early upward lift;
        2. high forward transfer;
        3. smooth landing to upper step.
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
    clear_margin: float,
    p1_z_ratio: float,
    p2_x_ratio: float,
    p3_x_ratio: float,
    p4_z_ratio: float,
):
    """
    Compute absolute IK joint angles from Bezier stair trajectory phase.
    """
    s = torch.clamp(s, 0.0, 1.0)

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

    q1, q2 = _solve_leg_ik_from_point(
        xc=xc,
        zc=zc,
        l1=l1,
        l2=l2,
        x_hip=x_hip,
        z_hip=z_hip,
    )

    return q1, q2


def _compute_support_offset_float(
    l1: float,
    l2: float,
    x_hip: float,
    z_hip: float,
    q1_default: float,
    q2_default: float,
    support_hip_lift: float,
    support_k: float,
    support_max_offset: float,
):
    """
    Compute stance-leg extension offset by IK.

    Physical meaning:
        keep the support wheel center fixed;
        raise desired hip height by support_hip_lift;
        solve IK again;
        q_support - q_default is the support compensation.

    This avoids guessing the signs of thigh/calf compensation manually.
    """

    # FK from default hip pose to default support wheel center.
    x_rel = -l1 * math.sin(q1_default) - l2 * math.sin(q1_default + q2_default)
    z_down = l1 * math.cos(q1_default) + l2 * math.cos(q1_default + q2_default)

    wheel_x = x_hip + x_rel
    wheel_z = z_hip - z_down

    z_hip_support = z_hip + support_hip_lift

    x_rel_support = wheel_x - x_hip
    z_rel_support = z_hip_support - wheel_z

    d = (
        x_rel_support * x_rel_support
        + z_rel_support * z_rel_support
        - l1 * l1
        - l2 * l2
    ) / (2.0 * l1 * l2)
    d = _clamp_float(d, -1.0, 1.0)

    q2_support = math.atan2(-math.sqrt(max(1.0 - d * d, 0.0)), d)

    q1_support = math.atan2(
        -l2 * math.sin(q2_support),
        l1 + l2 * math.cos(q2_support),
    ) - math.atan2(x_rel_support, z_rel_support)

    dq1 = q1_support - q1_default
    dq2 = q2_support - q2_default

    dq1 = support_k * _clamp_float(dq1, -support_max_offset, support_max_offset)
    dq2 = support_k * _clamp_float(dq2, -support_max_offset, support_max_offset)

    return dq1, dq2


def compute_stair_ik_ff_offsets_b(
    phase_local: torch.Tensor,
    active: torch.Tensor,
    dof_names,
    num_actions: int,
    device,
    cfg_control,
):
    """
    Stair feedforward = swing-leg Bezier IK + opposite support-leg compensation.

    For each active swing leg:
        1. apply Bezier swing offset to that leg;
        2. apply stance support compensation to the opposite leg.

    phase_local u:
        0 ~ 1:
            swing phase

        1 ~ 1 + extend_ratio:
            swing offset releases back to zero

    The returned value is in radians and is added to joint_pos_target.
    """

    num_envs = phase_local.shape[0]
    ff_offsets = torch.zeros(num_envs, num_actions, device=device)

    if not torch.any(active):
        return ff_offsets

    # ========== geometry ==========
    l1 = float(_cfg_get(cfg_control, "stair_ff_l1", 0.25))
    l2 = float(_cfg_get(cfg_control, "stair_ff_l2", 0.25))
    rw = float(_cfg_get(cfg_control, "stair_ff_wheel_radius", 0.085))
    hs = float(_cfg_get(cfg_control, "stair_ff_step_height", 0.15))

    x0 = float(_cfg_get(cfg_control, "stair_ff_x0", 0.0))
    x1 = float(_cfg_get(cfg_control, "stair_ff_x1", 0.10))

    x_hip = float(_cfg_get(cfg_control, "stair_ff_x_hip", x0))
    z_hip = float(_cfg_get(cfg_control, "stair_ff_z_hip", 0.45))

    # Backward compatibility: if clear_margin is not set, use old h_margin.
    old_h_margin = float(_cfg_get(cfg_control, "stair_ff_h_margin", 0.055))
    clear_margin = float(_cfg_get(cfg_control, "stair_ff_clear_margin", old_h_margin))

    p1_z_ratio = float(_cfg_get(cfg_control, "stair_ff_bezier_p1_z_ratio", 0.70))
    p2_x_ratio = float(_cfg_get(cfg_control, "stair_ff_bezier_p2_x_ratio", 0.05))
    p3_x_ratio = float(_cfg_get(cfg_control, "stair_ff_bezier_p3_x_ratio", 0.65))
    p4_z_ratio = float(_cfg_get(cfg_control, "stair_ff_bezier_p4_z_ratio", 0.25))

    # ========== swing shaping ==========
    s_start = float(_cfg_get(cfg_control, "stair_ff_phase_start", 0.10))
    ramp_ratio = float(_cfg_get(cfg_control, "stair_ff_ramp_ratio", 0.06))
    max_offset = float(_cfg_get(cfg_control, "stair_ff_max_offset", 1.20))

    extend_enabled = bool(_cfg_get(cfg_control, "stair_ff_extend_enabled", True))
    extend_ratio = float(_cfg_get(cfg_control, "stair_ff_extend_ratio", 0.30))
    extend_ratio = max(extend_ratio, 0.0)

    phase_end = 1.0 + extend_ratio if extend_enabled else 1.0

    # ========== support compensation parameters ==========
    support_enabled = bool(_cfg_get(cfg_control, "stair_ff_support_enabled", True))
    support_hip_lift = float(_cfg_get(cfg_control, "stair_ff_support_hip_lift", 0.040))
    support_k = float(_cfg_get(cfg_control, "stair_ff_support_k", 0.80))
    support_ramp_ratio = float(_cfg_get(cfg_control, "stair_ff_support_ramp_ratio", 0.25))
    support_max_offset = float(_cfg_get(cfg_control, "stair_ff_support_max_offset", 0.35))

    q1_default_support = float(_cfg_get(cfg_control, "stair_ff_support_default_thigh", 0.8))
    q2_default_support = float(_cfg_get(cfg_control, "stair_ff_support_default_calf", -1.5))

    support_dq1, support_dq2 = _compute_support_offset_float(
        l1=l1,
        l2=l2,
        x_hip=x_hip,
        z_hip=z_hip,
        q1_default=q1_default_support,
        q2_default=q2_default_support,
        support_hip_lift=support_hip_lift,
        support_k=support_k,
        support_max_offset=support_max_offset,
    )

    support_dq1 = torch.tensor(support_dq1, device=device)
    support_dq2 = torch.tensor(support_dq2, device=device)

    # ========== q_IK(0), swing reference ==========
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

        u_total = torch.clamp(phase_local[:, side], 0.0, phase_end)
        u_swing = torch.clamp(u_total, 0.0, 1.0)

        # Local phase -> template phase.
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
            clear_margin=clear_margin,
            p1_z_ratio=p1_z_ratio,
            p2_x_ratio=p2_x_ratio,
            p3_x_ratio=p3_x_ratio,
            p4_z_ratio=p4_z_ratio,
        )

        # ========== swing offset for active leg ==========
        dq1 = q1 - q1_zero
        dq2 = q2 - q2_zero

        if ramp_ratio > 1e-6:
            swing_ramp = _smoothstep(u_swing / ramp_ratio)
            dq1 = dq1 * swing_ramp
            dq2 = dq2 * swing_ramp

        if extend_enabled and extend_ratio > 1e-6:
            ext_u = torch.clamp((u_total - 1.0) / extend_ratio, 0.0, 1.0)
            release_gate = 1.0 - _smoothstep(ext_u)
            dq1 = dq1 * release_gate
            dq2 = dq2 * release_gate
        else:
            release_gate = torch.ones_like(u_total)

        dq1 = torch.clamp(dq1, -max_offset, max_offset)
        dq2 = torch.clamp(dq2, -max_offset, max_offset)

        dq1 = dq1 * side_active.float()
        dq2 = dq2 * side_active.float()

        thigh_name, calf_name = side_joint_names[side]

        if thigh_name in dof_names:
            thigh_idx = dof_names.index(thigh_name)
            ff_offsets[:, thigh_idx] += dq1

        if calf_name in dof_names:
            calf_idx = dof_names.index(calf_name)
            ff_offsets[:, calf_idx] += dq2

        # ========== support offset for opposite leg ==========
        if support_enabled:
            opposite_side = 1 - side
            support_thigh_name, support_calf_name = side_joint_names[opposite_side]

            if support_ramp_ratio > 1e-6:
                support_gate = _smoothstep(u_swing / support_ramp_ratio)
            else:
                support_gate = torch.ones_like(u_swing)

            # When the swing leg starts release, support compensation also releases.
            support_gate = support_gate * release_gate
            support_gate = support_gate * side_active.float()

            support_q1 = support_dq1 * support_gate
            support_q2 = support_dq2 * support_gate

            if support_thigh_name in dof_names:
                support_thigh_idx = dof_names.index(support_thigh_name)
                ff_offsets[:, support_thigh_idx] += support_q1

            if support_calf_name in dof_names:
                support_calf_idx = dof_names.index(support_calf_name)
                ff_offsets[:, support_calf_idx] += support_q2

    return ff_offsets