"""Generate a single CSV that demonstrates a complete two-leg stair feedforward.

This file is for visualization in motion-editor.cyoahs.dev.

This version uses a rounded vertical-first Bezier wheel-center trajectory:
    1) lift mostly upward first,
    2) transfer forward at a high clearance,
    3) settle down to the upper step height.

It is still a visualization script, not a dynamics simulation.
The base is fixed; only the leg joints move.
"""

import csv
import math
import os
import xml.etree.ElementTree as ET


# ---------------- current config / visualization parameters ----------------
FPS = 50
PRE_HOLD = 0.20
POST_HOLD = 0.50

# from current uploaded config / visualization setting
DURATION = 0.55
FOLLOWUP_PHASE = 0.70
FOLLOWUP_PHASE_INIT = 0.00
K_FF = 1.0
PHASE_START = 0.1
RAMP_RATIO = 0.06
FINAL_MAX_OFFSET = 0.65

L1 = 0.25
L2 = 0.25
RW = 0.085
HS = 0.2

# Wheel-center trajectory start/end.
# Current demo uses 10 cm horizontal send.
X0 = 0.0
X1 = 0.2
X_HIP = X0
Z_HIP = 0.45

# Recommended Bezier trajectory clearance.
# z_peak = upper-platform wheel-center height + CLEAR_MARGIN.
CLEAR_MARGIN = 0.045

# Extra demo-only extension/recovery phase.
# The swing part is current B scheme; the extension part blends the leg back to
# default so both legs do not remain tucked.
STANCE_EXTEND_DURATION = 0.55

# Static base visualization: keep body fixed and only move the legs.
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

OUT_NAME = "d1h_stair_current_b_bezier_extend_demo_static_base_urdf_order.csv"


def smoothstep(x: float) -> float:
    x = min(max(x, 0.0), 1.0)
    return x * x * (3.0 - 2.0 * x)


def bezier5_point(s: float):
    """Rounded vertical-first 5th-order Bezier wheel-center trajectory.

    Control-point design:
        P0: wheel-center start
        P1: almost same x as P0, so the initial tangent is mostly upward
        P2: near the start in x, already at high clearance
        P3: forward at high clearance
        P4: above the landing point, so the final tangent is mostly downward
        P5: wheel-center end on upper step

    This is smoother than a hard vertical-horizontal-vertical segmented path,
    but still lifts earlier than the old cycloid-like trajectory.
    """
    s = min(max(s, 0.0), 1.0)

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
    """IK of the recommended Bezier wheel-center trajectory."""
    s = min(max(s, 0.0), 1.0)

    xc, zc = bezier5_point(s)

    x_rel = xc - X_HIP
    z_rel = Z_HIP - zc

    d = (x_rel * x_rel + z_rel * z_rel - L1 * L1 - L2 * L2) / (2.0 * L1 * L2)
    d = min(max(d, -1.0), 1.0)

    q2 = math.atan2(-math.sqrt(max(1.0 - d * d, 0.0)), d)

    q1 = math.atan2(
        -L2 * math.sin(q2),
        L1 + L2 * math.cos(q2),
    ) - math.atan2(x_rel, z_rel)

    return q1, q2


Q_ZERO = ik_template(0.0)


def current_b_offset(u: float):
    """Current B-scheme offset shape, before adding to default joint pose."""
    u = min(max(u, 0.0), 1.0)

    s = PHASE_START + (1.0 - PHASE_START) * u
    q1, q2 = ik_template(s)

    dq1 = q1 - Q_ZERO[0]
    dq2 = q2 - Q_ZERO[1]

    if RAMP_RATIO > 1e-9:
        r = smoothstep(u / RAMP_RATIO)
        dq1 *= r
        dq2 *= r

    dq1 = min(max(dq1, -FINAL_MAX_OFFSET), FINAL_MAX_OFFSET)
    dq2 = min(max(dq2, -FINAL_MAX_OFFSET), FINAL_MAX_OFFSET)

    return K_FF * dq1, K_FF * dq2


FINAL_OFFSET = current_b_offset(1.0)


def leg_offset_at(t: float, start: float, init_u: float = 0.0):
    """Swing current B offset, then extend/release back to zero.

    Swing:
        u = init_u -> 1

    Extension:
        blend from final swing offset back to zero, which means returning to
        default/support leg posture instead of holding the tucked swing endpoint.
    """
    if t < start:
        return 0.0, 0.0

    tau = t - start

    if tau <= DURATION:
        u = init_u + (1.0 - init_u) * (tau / DURATION)
        return current_b_offset(u)

    if tau <= DURATION + STANCE_EXTEND_DURATION:
        r = smoothstep((tau - DURATION) / STANCE_EXTEND_DURATION)
        return FINAL_OFFSET[0] * (1.0 - r), FINAL_OFFSET[1] * (1.0 - r)

    return 0.0, 0.0


def parse_joint_order():
    # Prefer robot.urdf beside this script, then current working directory, then /mnt/data.
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

        # Keep the body fixed. Only the four leg joints change.
        bx = BASE_X0
        bz = BASE_Z0

        fl_dth, fl_dcf = leg_offset_at(t, fl_start, 0.0)
        fr_dth, fr_dcf = leg_offset_at(t, fr_start, FOLLOWUP_PHASE_INIT)

        q = dict(DEFAULT_Q)
        q["FL_thigh_joint"] += fl_dth
        q["FL_calf_joint"] += fl_dcf
        q["FR_thigh_joint"] += fr_dth
        q["FR_calf_joint"] += fr_dcf

        row = [bx, 0.0, bz, 0.0, 0.0, 0.0, 1.0]
        row.extend(q.get(name, 0.0) for name in joint_order)
        rows.append(row)

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), OUT_NAME)

    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(rows)

    print(out_path)


if __name__ == "__main__":
    main()
