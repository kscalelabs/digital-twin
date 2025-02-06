"""Test PyKOS communication."""

import argparse
import asyncio
from dataclasses import dataclass, field

from pykos import KOS

from ks_digital_twin.actor.pykos_bot import PyKOSActor
from ks_digital_twin.puppet.mujoco_puppet import MujocoPuppet


@dataclass
class RobotConfigs:
    joint_mapping: dict[str, int] = field(default_factory=dict)
    signs: dict[str, int] = field(default_factory=dict)


@dataclass
class KbotConfigs(RobotConfigs):
    joint_mapping: dict[str, int] = field(
        default_factory=lambda: {
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
            "right_ankle_02": 45,
        }
    )

    signs: dict[str, int] = field(
        default_factory=lambda: {
            # Left arm
            "left_shoulder_pitch_03": 1,
            "left_shoulder_roll_03": 1,
            "left_shoulder_yaw_02": 1,
            "left_elbow_02": -1,
            "left_wrist_02": 1,
            # Right arm
            "right_shoulder_pitch_03": 1,
            "right_shoulder_roll_03": 1,
            "right_shoulder_yaw_02": 1,
            "right_elbow_02": -1,
            "right_wrist_02": 1,
            # Left leg
            "left_hip_pitch_04": 1,
            "left_hip_roll_03": 1,
            "left_hip_yaw_03": 1,
            "left_knee_04": 1,
            "left_ankle_02": 1,
            # Right leg
            "right_hip_pitch_04": 1,
            "right_hip_roll_03": 1,
            "right_hip_yaw_03": 1,
            "right_knee_04": 1,
            "right_ankle_02": -1,
        }
    )


@dataclass
class KbotNakedConfigs(RobotConfigs):
    joint_mapping: dict[str, int] = field(
        default_factory=lambda: {
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
            "right_ankle_02": 45,
        }
    )

    signs: dict[str, int] = field(
        default_factory=lambda: {
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
            "right_ankle_02": -1,
        }
    )


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mjcf_name", type=str, help="Name of the Mujoco model in the K-Scale API")
    parser.add_argument("--ip", type=str, default="localhost", help="IP address of the robot")
    args = parser.parse_args()

    configs: RobotConfigs
    match args.mjcf_name:
        case "zbot-v2":
            raise NotImplementedError("jx skill issue")
        case "kbot-v1":
            configs = KbotConfigs()
        case _:
            raise ValueError(f"No configs for {args.mjcf_name}")

    async with KOS(ip=args.ip) as kos:
        actor = PyKOSActor(kos, configs.joint_mapping, kos_signs=configs.signs)
        await actor.offset_in_place()
        print(f"Offsets: {actor.get_offsets()}")
        puppet = MujocoPuppet(args.mjcf_name)

        while True:
            joint_angles = await actor.get_joint_angles()
            print(joint_angles)
            await puppet.set_joint_angles(joint_angles)
            await asyncio.sleep(0.01)


if __name__ == "__main__":
    asyncio.run(main())
