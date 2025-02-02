"""Test Mujoco puppet communication."""

import argparse
import asyncio

import colorlogging

from ks_digital_twin.actor.sinusoid import SinusoidActor
from ks_digital_twin.puppet.mujoco_puppet import MujocoPuppet


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mjcf_name", type=str, help="Name of the Mujoco model in the K-Scale API")
    parser.add_argument("--no-skip-root", action="store_true", help="Do not skip the root joint")
    args = parser.parse_args()

    colorlogging.configure()

    # Create the puppet and get its joint names
    puppet = MujocoPuppet(args.mjcf_name)
    joint_names = await puppet.get_joint_names()

    # Optionally skip the root joint
    if not args.no_skip_root:
        joint_names = joint_names[1:]

    # Create a sinusoidal actor that will control the joints
    actor = SinusoidActor(joint_names)

    try:
        while True:
            # Get joint angles from the actor
            joint_angles = await actor.get_joint_angles()
            # Apply them to the puppet
            await puppet.set_joint_angles(joint_angles)
            # Small sleep to control the update rate
            await asyncio.sleep(0.01)
    except KeyboardInterrupt:
        print("\nShutting down...")


if __name__ == "__main__":
    asyncio.run(main())
