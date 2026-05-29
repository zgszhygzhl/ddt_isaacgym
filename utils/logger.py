
import os

import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from multiprocessing import Process


class Logger:
    def __init__(self, dt):
        self.state_log = defaultdict(list)
        self.rew_log = defaultdict(list)
        self.dt = dt
        self.num_episodes = 0
        self.plot_process = None

    def log_state(self, key, value):
        self.state_log[key].append(value)

    def log_states(self, dict):
        for key, value in dict.items():
            self.log_state(key, value)

    def log_rewards(self, dict, num_episodes):
        for key, value in dict.items():
            if 'rew' in key:
                self.rew_log[key].append(value.item() * num_episodes)
        self.num_episodes += num_episodes

    def reset(self):
        self.state_log.clear()
        self.rew_log.clear()

    def plot_states(self, save_path=None, show=True, plot_metadata=None):
        if save_path is not None or not show:
            self._plot(save_path=save_path, show=show, plot_metadata=plot_metadata)
            return

        self.plot_process = Process(target=self._plot, kwargs={"plot_metadata": plot_metadata})
        self.plot_process.start()

    def _plot(self, save_path=None, show=True, plot_metadata=None):
        if not self.state_log:
            return

        plot_metadata = plot_metadata or {}
        joint_names = plot_metadata.get("joint_names", [])
        logged_joint_name = plot_metadata.get("logged_joint_name", "tracked_joint")
        contact_names = plot_metadata.get("contact_names", [])

        log = self.state_log
        time = None
        for _, value in log.items():
            if value:
                time = np.arange(len(value)) * self.dt
                break
        if time is None:
            return

        def append_handle(handle_list, line):
            if line is not None:
                handle_list.append(line)

        def dedup_handles(handles):
            unique_handles = []
            seen_labels = set()
            for handle in handles:
                label = handle.get_label()
                if label not in seen_labels:
                    unique_handles.append(handle)
                    seen_labels.add(label)
            return unique_handles

        fig, axs = plt.subplots(3, 3, figsize=(14, 9))
        main_handles = []

        a = axs[0, 0]
        if log["base_vel_x"]:
            append_handle(main_handles, a.plot(time, log["base_vel_x"], label="actual base vel x")[0])
        if log["command_x"]:
            append_handle(main_handles, a.plot(time, log["command_x"], label="commanded vel x")[0])
        a.set(xlabel="time [s]", ylabel="base lin vel [m/s]", title="Base Velocity X")

        a = axs[0, 1]
        if log["base_vel_y"]:
            append_handle(main_handles, a.plot(time, log["base_vel_y"], label="actual base vel y")[0])
        if log["command_y"]:
            append_handle(main_handles, a.plot(time, log["command_y"], label="commanded vel y")[0])
        a.set(xlabel="time [s]", ylabel="base lin vel [m/s]", title="Base Velocity Y")

        a = axs[0, 2]
        if log["base_vel_yaw"]:
            append_handle(main_handles, a.plot(time, log["base_vel_yaw"], label="actual yaw rate")[0])
        if log["command_yaw"]:
            append_handle(main_handles, a.plot(time, log["command_yaw"], label="commanded yaw rate")[0])
        a.set(xlabel="time [s]", ylabel="base ang vel [rad/s]", title="Base Velocity Yaw")

        a = axs[1, 0]
        if log["base_heading"]:
            append_handle(main_handles, a.plot(time, log["base_heading"], label="actual heading")[0])
        if log["command_heading"]:
            append_handle(main_handles, a.plot(time, log["command_heading"], label="commanded heading")[0])
        a.set(xlabel="time [s]", ylabel="heading [rad]", title="Base Heading")

        axs[1, 1].axis("off")

        a = axs[1, 2]
        if log["base_vel_z"]:
            append_handle(main_handles, a.plot(time, log["base_vel_z"], label="actual base vel z")[0])
        a.set(xlabel="time [s]", ylabel="base lin vel [m/s]", title="Base Velocity Z")

        a = axs[2, 0]
        if log["contact_forces_z"]:
            forces = np.array(log["contact_forces_z"])
            for index in range(forces.shape[1]):
                contact_label = contact_names[index] if index < len(contact_names) else f"contact {index}"
                append_handle(main_handles, a.plot(time, forces[:, index], label=contact_label)[0])
        a.set(xlabel="time [s]", ylabel="forces z [N]", title="Vertical Contact Forces")

        a = axs[2, 1]
        if log["dof_vel"] != [] and log["dof_torque"] != []:
            append_handle(
                main_handles,
                a.plot(log["dof_vel"], log["dof_torque"], "x", label=f"{logged_joint_name} torque-velocity")[0],
            )
        a.set(xlabel="joint vel [rad/s]", ylabel="joint torque [Nm]", title=f"Torque vs Velocity: {logged_joint_name}")

        a = axs[2, 2]
        if log["base_height"]:
            append_handle(main_handles, a.plot(time, log["base_height"], label="actual base height")[0])
        if log["command_height"]:
            append_handle(main_handles, a.plot(time, log["command_height"], label="target base height")[0])
        a.set(xlabel="time [s]", ylabel="base height [m]", title="Base Height")

        main_handles = dedup_handles(main_handles)
        if main_handles:
            fig.legend(
                handles=main_handles,
                loc="upper center",
                bbox_to_anchor=(0.5, 0.99),
                ncol=4,
                fontsize=8,
                framealpha=0.8,
                columnspacing=1.2,
                handlelength=1.8,
            )
        fig.suptitle("Inference Tracking Overview", fontsize=14)
        fig.tight_layout(rect=(0.02, 0.04, 0.98, 0.92))

        fig2 = None
        if log["torques"]:
            num_joints = len(log["torques"][0])
            fig2, axs2 = plt.subplots(4, num_joints // 4, figsize=(12, 9))
            joint_handles = []
            for joint_idx in range(num_joints):
                a = axs2[joint_idx % 4, joint_idx // 4]
                joint_name = joint_names[joint_idx] if joint_idx < len(joint_names) else f"joint_{joint_idx}"
                joint_torques = [torque[joint_idx] for torque in log["torques"]]
                append_handle(joint_handles, a.plot(time, joint_torques, label="torque [Nm]")[0])
                a.set(xlabel="time [s]", ylabel="joint torque [Nm]", title=joint_name)
                if log["velocities"]:
                    a2 = a.twinx()
                    joint_velocities = [vel[joint_idx] for vel in log["velocities"]]
                    append_handle(joint_handles, a2.plot(time, joint_velocities, "r--", label="velocity [rad/s]")[0])
                    a2.set_ylabel("joint velocity [rad/s]", color="r")

            joint_handles = dedup_handles(joint_handles)
            if joint_handles:
                fig2.legend(
                    handles=joint_handles,
                    loc="upper center",
                    bbox_to_anchor=(0.5, 0.99),
                    ncol=2,
                    fontsize=8,
                    framealpha=0.8,
                    handlelength=2.0,
                )
            fig2.suptitle("Per-Joint Torque and Velocity", fontsize=14)
            fig2.tight_layout(rect=(0.02, 0.04, 0.98, 0.94))

        if save_path is not None:
            fig.savefig(save_path, dpi=200, bbox_inches="tight")
            if fig2 is not None:
                root, ext = os.path.splitext(save_path)
                fig2.savefig(f"{root}_joints{ext}", dpi=200, bbox_inches="tight")

        if show:
            plt.show()
        else:
            plt.close(fig)
            if fig2 is not None:
                plt.close(fig2)

    def print_rewards(self):
        print("Average rewards per second:")
        for key, values in self.rew_log.items():
            mean = np.sum(np.array(values)) / self.num_episodes
            print(f" - {key}: {mean}")
        print(f"Total number of episodes: {self.num_episodes}")

    def __del__(self):
        if self.plot_process is not None:
            self.plot_process.kill()
