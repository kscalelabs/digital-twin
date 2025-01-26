"""Test PyKOS communication."""

import argparse
import asyncio
import logging

import colorlogging
from pykos import KOS

from digital_twin.actor.pykos_bot import PyKOSActor
from digital_twin.puppet.mujoco_puppet import MujocoPuppet

logger = logging.getLogger(__name__)


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
    parser.add_argument("name", type=str, help="Name of the model in the K-Scale API")
    parser.add_argument("--ip", type=str, default="localhost", help="IP address of the robot")
    args = parser.parse_args()

    colorlogging.configure()

    kos = KOS(ip=args.ip)

    # Gets the actor and sets the zero positions in-place.
    actor = PyKOSActor(args.name, kos)
    actor.offset_in_place()
    logger.info("Offsets: %s", actor.get_offsets())

    puppet = MujocoPuppet(args.name)

    while True:
        joint_angles = await actor.get_joint_angles()
        logger.info("Joint angles: %s", joint_angles)
        await puppet.set_joint_angles(joint_angles)
        await asyncio.sleep(0.01)


if __name__ == "__main__":
    asyncio.run(main())
