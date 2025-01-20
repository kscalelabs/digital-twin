"""Test PyKOS communication."""

import argparse
import asyncio

from pykos import KOS

from digital_twin.actor.pykos_bot import PyKOSActor
from digital_twin.puppet.mujoco_puppet import MujocoPuppet

joint_mapping = {
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
}

signs: dict[str, float] | None = {
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
}


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mjcf_name", type=str, help="Name of the Mujoco model in the K-Scale API")
    parser.add_argument("--ip", type=str, default="localhost", help="IP address of the robot")
    args = parser.parse_args()

    kos = KOS(ip=args.ip)
    actor = PyKOSActor(kos, joint_mapping, kos_signs=signs)
    actor.offset_in_place()
    print(f"Offsets: {actor.get_offsets()}")
    puppet = MujocoPuppet(args.mjcf_name)

    while True:
        joint_angles = await actor.get_joint_angles()
        print(joint_angles)
        await puppet.set_joint_angles(joint_angles)
        await asyncio.sleep(0.01)


if __name__ == "__main__":
    asyncio.run(main())
