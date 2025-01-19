"""Displays a digital twin of a robot using Mujoco."""

import argparse
import asyncio
import logging

import colorlogging

from digital_twin.models.source import KeyboardSourceRobot
from digital_twin.models.target import MujocoTargetRobot

logger = logging.getLogger(__name__)


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mjcf_name", type=str, help="Name of the Mujoco model in the K-Scale API")
    parser.add_argument("--no-skip-root", action="store_true", help="Do not skip the root joint")
    args = parser.parse_args()

    colorlogging.configure()

    target_robot = MujocoTargetRobot(args.mjcf_name)
    joint_names = await target_robot.get_joint_names()
    if not args.no_skip_root:
        joint_names = joint_names[1:]
    source_robot = KeyboardSourceRobot(joint_names)

    while True:
        joint_angles = await source_robot.get_joint_angles()
        await target_robot.set_joint_angles(joint_angles)
        await asyncio.sleep(0.01)


if __name__ == "__main__":
    # python -m digital_twin.scripts.keyboard_to_mujoco
    asyncio.run(main())
