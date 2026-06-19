"""Generate a single CSV that demonstrates a two-leg stair feedforward
with swing-leg Bezier trajectory, landing hold, and high-step support scaling.

This file is for visualization in motion-editor.cyoahs.dev.

It is still a visualization script, not a dynamics simulation:
    - base is fixed;
    - swing leg follows rounded vertical-first Bezier IK;
    - opposite ground-side support leg extends strongly;
    - after a leg lands on the high step, it does not immediately recover to
      default, and its support compensation is weakened to avoid lifting the
      high-step side too much.
"""

import csv
import math
import os
import xml.etree.ElementTree as ET


# ---------------- current config / visualization parameters ----------------
FPS = 50
PRE_HOLD = 0.20
POST_HOLD = 0.50

DURATION = 0.55
FOLLOWUP_PHASE = 0.850
FOLLOWUP_PHASE_INIT = 0.00

K_FF = 1.0
PHASE_START = 0.0
RAMP_RATIO = 0.06
FINAL_MAX_OFFSET = 0.65

L1 = 0.25
L2 = 0.25
RW = 0.085
HS = 0.15

# Wheel-center swing trajectory start/end.
X0 = 0.0
X1 = 0.15
X_HIP = X0
Z_HIP = 0.45

# Rounded vertical-first Bezier clearance.
CLEAR_MARGIN = 0.055

# Swing-leg release/recovery phase.
STANCE_EXTEND_DURATION = 0.55

# ---------------- landing / high-step support logic ----------------
# After the first leg lands on the upper step, do not release its swing offset
# completely while the second leg is still swinging.  This keeps the high-step
# leg slightly tucked, avoiding excessive height on that side.
LANDING_HOLD_RATIO = 0.35

# When the opposite leg local phase reaches this value, the landed leg can
# start releasing fully toward default.
LANDING_RELEASE_OPPOSITE_PHASE = 0.90

# If the support leg itself has already landed on the high step, weaken its
# support compensation.  Ground-side support remains full strength.
HIGH_STEP_SUPPORT_SCALE = 0.25

# ---------------- support-leg compensation ----------------
# Physical meaning:
#   When one leg swings, the other leg is asked to become slightly "longer".
#   We compute this by keeping the stance wheel center fixed and solving IK for
#   a slightly higher hip height.  In real contact, this generates upward support.
SUPPORT_COMP_ENABLED = True

# Desired extra virtual hip height carried by the stance leg.
SUPPORT_HIP_LIFT = 0.040

# Scale the IK-derived support offset.
SUPPORT_K = 0.80

# How fast the support leg reaches the compensation after the other leg starts swing.
SUPPORT_RAMP_RATIO = 0.25

# Limit support compensation itself.
SUPPORT_MAX_OFFSET = 0.35

# ---------------- static base visualization ----------------
BASE_X0 = 0.0
BASE_Z0 = 0.50

DEFAULT_Q = {
    "FL_hip_joint": 0.0,
    "FL_thigh_joint": 0.8,
    "FL_calf_joint": -1.5,
    "FL_foot_joint": 0.0,
    "FR_hip_joint": 0.0,
    "FR_thigh_joint": 0.8,
    "FR_calf_joint": -1.5,
    "FR_foot_joint": 0.0,
}

OUT_NAME = "d1h_stair_bezier_landing_hold_highstep_support_demo_static_base_urdf_order.csv"


def smoothstep(x: float) -> float:
    x = min(max(x, 0.0), 1.0)
    return x * x * (3.0 - 2.0 * x)


def clamp(x: float, lo: float, hi: float) -> float:
    return min(max(x, lo), hi)


def fk_wheel_from_hip(q1: float, q2: float):
    """Forward kinematics from hip to wheel center.

    Local hip frame:
        x forward positive;
        z_down downward positive.
    """
    x_rel = -L1 * math.sin(q1) - L2 * math.sin(q1 + q2)
    z_down = L1 * math.cos(q1) + L2 * math.cos(q1 + q2)
    return x_rel, z_down


def solve_ik_point(xc: float, zc: float, x_hip: float, z_hip: float):
    """Solve two-link IK for a wheel center point in world frame."""
    x_rel = xc - x_hip
    z_rel = z_hip - zc

    d = (x_rel * x_rel + z_rel * z_rel - L1 * L1 - L2 * L2) / (2.0 * L1 * L2)
    d = clamp(d, -1.0, 1.0)

    q2 = math.atan2(-math.sqrt(max(1.0 - d * d, 0.0)), d)

    q1 = math.atan2(
        -L2 * math.sin(q2),
        L1 + L2 * math.cos(q2),
    ) - math.atan2(x_rel, z_rel)

    return q1, q2


def bezier5_point(s: float):
    """Rounded vertical-first 5th-order Bezier wheel-center trajectory."""
    s = clamp(s, 0.0, 1.0)

    z0 = RW
    z1 = HS + RW
    dx = X1 - X0
    z_peak = HS + RW + CLEAR_MARGIN

    p0 = (X0, z0)
    p1 = (X0, z0 + 0.70 * (z_peak - z0))
    p2 = (X0 + 0.05 * dx, z_peak)
    p3 = (X0 + 0.65 * dx, z_peak)
    p4 = (X1, z1 + 0.25 * (z_peak - z1))
    p5 = (X1, z1)

    b0 = (1.0 - s) ** 5
    b1 = 5.0 * s * (1.0 - s) ** 4
    b2 = 10.0 * s ** 2 * (1.0 - s) ** 3
    b3 = 10.0 * s ** 3 * (1.0 - s) ** 2
    b4 = 5.0 * s ** 4 * (1.0 - s)
    b5 = s ** 5

    xc = (
        b0 * p0[0]
        + b1 * p1[0]
        + b2 * p2[0]
        + b3 * p3[0]
        + b4 * p4[0]
        + b5 * p5[0]
    )
    zc = (
        b0 * p0[1]
        + b1 * p1[1]
        + b2 * p2[1]
        + b3 * p3[1]
        + b4 * p4[1]
        + b5 * p5[1]
    )

    return xc, zc


def ik_template(s: float):
    """IK of the Bezier wheel-center swing trajectory."""
    s = clamp(s, 0.0, 1.0)
    xc, zc = bezier5_point(s)
    return solve_ik_point(xc, zc, X_HIP, Z_HIP)


Q_ZERO = ik_template(0.0)


def current_b_offset(u: float):
    """Swing-leg B-scheme offset before adding to default joint pose."""
    u = clamp(u, 0.0, 1.0)

    s = PHASE_START + (1.0 - PHASE_START) * u
    q1, q2 = ik_template(s)

    dq1 = q1 - Q_ZERO[0]
    dq2 = q2 - Q_ZERO[1]

    if RAMP_RATIO > 1e-9:
        r = smoothstep(u / RAMP_RATIO)
        dq1 *= r
        dq2 *= r

    dq1 = clamp(dq1, -FINAL_MAX_OFFSET, FINAL_MAX_OFFSET)
    dq2 = clamp(dq2, -FINAL_MAX_OFFSET, FINAL_MAX_OFFSET)

    return K_FF * dq1, K_FF * dq2


FINAL_SWING_OFFSET = current_b_offset(1.0)


def leg_phase_at(t: float, start: float, init_u: float = 0.0):
    """Return the local phase used for visualization/gating.

    phase:
        <0: not started
        0~1: swing
        1~2: recovery/release, because STANCE_EXTEND_DURATION == DURATION here
    """
    if t < start:
        return -1.0

    tau = t - start

    if tau <= DURATION:
        return init_u + (1.0 - init_u) * (tau / DURATION)

    if tau <= DURATION + STANCE_EXTEND_DURATION:
        return 1.0 + (tau - DURATION) / STANCE_EXTEND_DURATION

    return 1.0 + STANCE_EXTEND_DURATION / DURATION


def leg_is_active(t: float, start: float):
    return start <= t <= start + DURATION + STANCE_EXTEND_DURATION


def leg_swing_offset_at(
    t: float,
    start: float,
    init_u: float,
    opposite_active: bool,
    opposite_phase: float,
):
    """Swing current B offset, then release with landing hold.

    During recovery, if the opposite leg is still in its swing phase, this leg
    keeps LANDING_HOLD_RATIO of the final swing offset instead of fully returning
    to default.  This prevents the landed high-step leg from becoming too long.
    """
    if t < start:
        return 0.0, 0.0

    tau = t - start

    if tau <= DURATION:
        u = init_u + (1.0 - init_u) * (tau / DURATION)
        return current_b_offset(u)

    if tau <= DURATION + STANCE_EXTEND_DURATION:
        r = smoothstep((tau - DURATION) / STANCE_EXTEND_DURATION)

        hold_needed = opposite_active and (opposite_phase < LANDING_RELEASE_OPPOSITE_PHASE)
        hold_floor = LANDING_HOLD_RATIO if hold_needed else 0.0

        release_gate = hold_floor + (1.0 - hold_floor) * (1.0 - r)

        return FINAL_SWING_OFFSET[0] * release_gate, FINAL_SWING_OFFSET[1] * release_gate

    return 0.0, 0.0


def support_gate_at(t: float, other_leg_start: float):
    """Support demand caused by the other leg's swing/recovery."""
    if t < other_leg_start:
        return 0.0

    tau = t - other_leg_start

    if tau <= DURATION:
        ramp_time = max(DURATION * SUPPORT_RAMP_RATIO, 1e-6)
        return smoothstep(tau / ramp_time)

    if tau <= DURATION + STANCE_EXTEND_DURATION:
        return 1.0 - smoothstep((tau - DURATION) / STANCE_EXTEND_DURATION)

    return 0.0


def compute_support_offset():
    """Compute stance-leg extension offset from IK.

    Keep the default stance wheel center fixed, then solve IK for a higher hip.
    The resulting joint difference is a physically meaningful support extension.
    """
    q1_default = DEFAULT_Q["FL_thigh_joint"]
    q2_default = DEFAULT_Q["FL_calf_joint"]

    x_rel, z_down = fk_wheel_from_hip(q1_default, q2_default)

    wheel_x = X_HIP + x_rel
    wheel_z = Z_HIP - z_down

    q1_support, q2_support = solve_ik_point(
        xc=wheel_x,
        zc=wheel_z,
        x_hip=X_HIP,
        z_hip=Z_HIP + SUPPORT_HIP_LIFT,
    )

    dq1 = q1_support - q1_default
    dq2 = q2_support - q2_default

    dq1 = SUPPORT_K * clamp(dq1, -SUPPORT_MAX_OFFSET, SUPPORT_MAX_OFFSET)
    dq2 = SUPPORT_K * clamp(dq2, -SUPPORT_MAX_OFFSET, SUPPORT_MAX_OFFSET)

    return dq1, dq2


SUPPORT_OFFSET = compute_support_offset()


def support_offset_at(
    t: float,
    other_leg_start: float,
    support_leg_active: bool,
    support_leg_phase: float,
):
    """Return support compensation applied to the stance leg.

    Ground-side support remains full strength.
    If the support leg has already landed on the high step, scale it down to
    HIGH_STEP_SUPPORT_SCALE.
    """
    if not SUPPORT_COMP_ENABLED:
        return 0.0, 0.0

    g = support_gate_at(t, other_leg_start)

    support_leg_landed = support_leg_active and (support_leg_phase >= 1.0)
    if support_leg_landed:
        g *= HIGH_STEP_SUPPORT_SCALE

    return SUPPORT_OFFSET[0] * g, SUPPORT_OFFSET[1] * g


def parse_joint_order():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(script_dir, "robot.urdf"),
        os.path.join(os.getcwd(), "robot.urdf"),
        "/mnt/data/robot.urdf",
    ]

    for path in candidates:
        if os.path.exists(path):
            root = ET.parse(path).getroot()
            order = []
            for joint in root.findall("joint"):
                if joint.get("type") != "fixed":
                    order.append(joint.get("name"))
            if order:
                return order

    return [
        "FL_hip_joint",
        "FL_thigh_joint",
        "FL_calf_joint",
        "FL_foot_joint",
        "FR_hip_joint",
        "FR_thigh_joint",
        "FR_calf_joint",
        "FR_foot_joint",
    ]


def main():
    joint_order = parse_joint_order()

    fl_start = PRE_HOLD
    fr_start = PRE_HOLD + FOLLOWUP_PHASE * DURATION

    total_t = (
        PRE_HOLD
        + FOLLOWUP_PHASE * DURATION
        + DURATION
        + STANCE_EXTEND_DURATION
        + POST_HOLD
    )
    n = int(round(total_t * FPS)) + 1

    rows = []

    for i in range(n):
        t = i / FPS

        bx = BASE_X0
        bz = BASE_Z0

        fl_active = leg_is_active(t, fl_start)
        fr_active = leg_is_active(t, fr_start)

        fl_phase = leg_phase_at(t, fl_start, 0.0)
        fr_phase = leg_phase_at(t, fr_start, FOLLOWUP_PHASE_INIT)

        # Swing offsets for each leg with landing hold.
        fl_swing_dth, fl_swing_dcf = leg_swing_offset_at(
            t=t,
            start=fl_start,
            init_u=0.0,
            opposite_active=fr_active,
            opposite_phase=fr_phase,
        )
        fr_swing_dth, fr_swing_dcf = leg_swing_offset_at(
            t=t,
            start=fr_start,
            init_u=FOLLOWUP_PHASE_INIT,
            opposite_active=fl_active,
            opposite_phase=fl_phase,
        )

        # Opposite support compensation:
        #   FR supports while FL swings;
        #   FL supports while FR swings.
        fl_support_dth, fl_support_dcf = support_offset_at(
            t=t,
            other_leg_start=fr_start,
            support_leg_active=fl_active,
            support_leg_phase=fl_phase,
        )
        fr_support_dth, fr_support_dcf = support_offset_at(
            t=t,
            other_leg_start=fl_start,
            support_leg_active=fr_active,
            support_leg_phase=fr_phase,
        )

        q = dict(DEFAULT_Q)

        q["FL_thigh_joint"] += fl_swing_dth + fl_support_dth
        q["FL_calf_joint"] += fl_swing_dcf + fl_support_dcf

        q["FR_thigh_joint"] += fr_swing_dth + fr_support_dth
        q["FR_calf_joint"] += fr_swing_dcf + fr_support_dcf

        row = [bx, 0.0, bz, 0.0, 0.0, 0.0, 1.0]
        row.extend(q.get(name, 0.0) for name in joint_order)
        rows.append(row)

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), OUT_NAME)

    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(rows)

    print(out_path)
    print(f"SUPPORT_OFFSET thigh={SUPPORT_OFFSET[0]:+.4f}, calf={SUPPORT_OFFSET[1]:+.4f}")
    print(f"FOLLOWUP_DELAY={FOLLOWUP_PHASE * DURATION:.4f}s")
    print(f"LANDING_HOLD_RATIO={LANDING_HOLD_RATIO:.2f}")
    print(f"HIGH_STEP_SUPPORT_SCALE={HIGH_STEP_SUPPORT_SCALE:.2f}")


if __name__ == "__main__":
    main()
