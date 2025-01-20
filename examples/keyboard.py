"""Example script to test the keyboard source robot."""

import time

from digital_twin.actor.keyboard import KeyboardActor


def main() -> None:
    robot = KeyboardActor(joint_names=["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint"])
    while True:
        angles = robot.get_joint_angles()
        print(angles)
        time.sleep(0.1)


if __name__ == "__main__":
    main()
