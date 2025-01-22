"""Test PyKOS communication."""

import argparse
import asyncio
import logging

import colorlogging
from pykos import KOS

from digital_twin.actor.pykos_bot import PyKOSActor
from digital_twin.puppet.mujoco_puppet import MujocoPuppet

logger = logging.getLogger(__name__)


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
