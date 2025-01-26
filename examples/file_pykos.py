"""Test PyKOS communication."""

import argparse
import asyncio
from dataclasses import dataclass, field
from pathlib import Path

from pykos import KOS

from digital_twin.actor.pykos_bot import PyKOSActor
from digital_twin.puppet.mujoco_puppet import MujocoPuppet


@dataclass
class RobotConfigs:
    joint_mapping: dict[str, int] = field(default_factory=dict)
    signs: dict[str, float] = field(default_factory=dict)

@dataclass
class ZbotConfigs(RobotConfigs):
    joint_mapping: dict[str, int] = field(default_factory=lambda: {
        # Left arm
        "left_shoulder_yaw": 11,
        "left_shoulder_pitch": 12,
        "left_elbow_yaw": 13,
        "left_gripper": 14,
        # Right arm
        "right_shoulder_yaw": 21,
        "right_shoulder_pitch": 22,
        "right_elbow_yaw": 23,
        "right_gripper": 24,
        # Left leg
        "left_hip_yaw": 31,
        "left_hip_roll": 32,
        "left_hip_pitch": 33,
        "left_knee_pitch": 34,
        "left_ankle_pitch": 35,
        # Right leg
        "right_hip_yaw": 41,
        "right_hip_roll": 42,
        "right_hip_pitch": 43,
        "right_knee_pitch": 44,
        "right_ankle_pitch": 45
    })

    signs: dict[str, float] = field(default_factory=lambda: {
        # Left arm
        "left_shoulder_yaw": 1,
        "left_shoulder_pitch": -1,
        "left_elbow_yaw": -1,
        "left_gripper": 1,
        # Right arm
        "right_shoulder_yaw": 1,
        "right_shoulder_pitch": 1,
        "right_elbow_yaw": 1,
        "right_gripper": 1,
        # Left leg
        "left_hip_yaw": 1,
        "left_hip_roll": -1,
        "left_hip_pitch": 1,
        "left_knee_pitch": 1,
        "left_ankle_pitch": 1,
        # Right leg
        "right_hip_yaw": 1,
        "right_hip_roll": 1,
        "right_hip_pitch": 1,
        "right_knee_pitch": 1,
        "right_ankle_pitch": 1
    })

@dataclass
class KbotConfigs(RobotConfigs):
    joint_mapping: dict[str, int] = field(default_factory=lambda: {
        # Left arm
        "L_shoulder_y_03": 11,
        "L_shoulder_x_03": 12,
        "L_shoulder_z_02": 13,
        "L_elbow_02": 14,
        "L_wrist_02": 15,
        # Right arm
        "R_shoulder_y_03": 21,
        "R_shoulder_x_03": 22,
        "R_shoulder_z_02": 23,
        "R_elbow_02": 24,
        "R_wrist_02": 25,
        # Left leg
        "L_hip_y_04": 31,
        "L_hip_x_03": 32,
        "L_hip_z_03": 33,
        "L_knee_04": 34,
        "L_ankle_02": 35,
        # Right leg
        "R_hip_y_04": 41,
        "R_hip_x_03": 42,
        "R_hip_z_03": 43,
        "R_knee_04": 44,
        "R_ankle_02": 45
    })

    signs: dict[str, float] = field(default_factory=lambda: {
        # Left arm
        "L_shoulder_y_03": 1,
        "L_shoulder_x_03": 1,
        "L_shoulder_z_02": 1,
        "L_elbow_02": 1,
        "L_wrist_02": 1,
        # Right arm
        "R_shoulder_y_03": 1,
        "R_shoulder_x_03": 1,
        "R_shoulder_z_02": 1,
        "R_elbow_02": 1,
        "R_wrist_02": 1,
        # Left leg
        "L_hip_y_04": 1,
        "L_hip_x_03": 1,
        "L_hip_z_03": 1,
        "L_knee_04": -1,
        "L_ankle_02": 1,
        # Right leg
        "R_hip_y_04": 1,
        "R_hip_x_03": 1,
        "R_hip_z_03": 1,
        "R_knee_04": 1,
        "R_ankle_02": 1
    })

@dataclass
class KbotNakedConfigs(RobotConfigs):
    joint_mapping: dict[str, int] = field(default_factory=lambda: {
        # Left arm
        "left_shoulder_pitch_03": 11,
        "left_shoulder_roll_03": 12,
        "left_shoulder_yaw_02": 13,
        "left_elbow_02": 14,
        "left_wrist_02": 15,
        # Right arm
        "right_shoulder_pitch_03": 21,
        "right_shoulder_roll_03": 22,
        "right_shoulder_yaw_02": 23,
        "right_elbow_02": 24,
        "right_wrist_02": 25,
        # Left leg
        "left_hip_pitch_04": 31,
        "left_hip_roll_03": 32,
        "left_hip_yaw_03": 33,
        "left_knee_04": 34,
        "left_ankle_02": 35,
        # Right leg
        "right_hip_pitch_04": 41,
        "right_hip_roll_03": 42,
        "right_hip_yaw_03": 43,
        "right_knee_04": 44,
        "right_ankle_02": 45
    })

    signs: dict[str, float] = field(default_factory=lambda: {
        # Left arm
        "left_shoulder_pitch_03": 1,
        "left_shoulder_roll_03": -1,
        "left_shoulder_yaw_02": 1,
        "left_elbow_02": 1,
        "left_wrist_02": 1,
        # Right arm
        "right_shoulder_pitch_03": 1,
        "right_shoulder_roll_03": 1,
        "right_shoulder_yaw_02": -1,
        "right_elbow_02": 1,
        "right_wrist_02": 1,
        # Left leg
        "left_hip_pitch_04": 1,
        "left_hip_roll_03": 1,
        "left_hip_yaw_03": 1,
        "left_knee_04": -1,  # Note: has negative limit range
        "left_ankle_02": 1,
        # Right leg
        "right_hip_pitch_04": 1,
        "right_hip_roll_03": 1,
        "right_hip_yaw_03": 1,
        "right_knee_04": -1,
        "right_ankle_02": -1
    })

async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mjcf_name", type=str, help="Name of the Mujoco model in the K-Scale API")
    parser.add_argument("--ip", type=str, default="localhost", help="IP address of the robot")
    args = parser.parse_args()

    configs: RobotConfigs
    match args.mjcf_name:
        case "zbot-v2":
            configs = ZbotConfigs()
        case "kbot-v1":
            configs = KbotConfigs()
        case "kbot-v1-naked":
            configs = KbotNakedConfigs()
        case _:
            raise ValueError(f"No configs for {args.mjcf_name}")

    async with KOS(ip=args.ip) as kos:
        actor = PyKOSActor(kos, configs.joint_mapping, kos_signs=configs.signs)
        await actor.offset_in_place()
        print(f"Offsets: {actor.get_offsets()}")
        puppet = MujocoPuppet(args.mjcf_name)

        puppet.mjcf_path = Path("xmls/robot_fixed.xml")

        while True:
            # Update joint angles as before
            orn = await actor.get_orientation()
            joint_angles = await actor.get_joint_angles()
            await puppet.set_joint_angles(joint_angles)
            await puppet.set_orientation((orn[0], orn[1], orn[2], orn[3]))
            await asyncio.sleep(0.01)


if __name__ == "__main__":
    asyncio.run(main())
