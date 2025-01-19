"""Implements the target robot model using PyBullet."""

import argparse
import asyncio
import logging
from pathlib import Path

import colorlogging
import kscale

from digital_twin.source.sinusoid import SinusoidSourceRobot
from digital_twin.target.base import TargetRobot

try:
    import pybullet
    import pybullet_data
except ImportError:
    raise ImportError("PyBullet is not installed, please install it using `pip install pybullet`")

logger = logging.getLogger(__name__)


class PyBulletTargetRobot(TargetRobot):
    """Target robot model using PyBullet."""

    def __init__(self, name: str, fixed_base: bool = True) -> None:
        self.name = name
        self.fixed_base = fixed_base
        self.action_lock = asyncio.Lock()
        self.urdf_path = None
        self.physics_client = None
        self.robot_id = None
        self.joint_info = None
        self.last_time = None

        # FPS tracking
        self.last_render_time: float | None = None
        self.fps_window_size = 60  # Calculate average over last 60 frames
        self.frame_times: list[float] = []
        self.next_fps_log = 0  # Time when we should next log FPS
        self.fps_log_interval = 5.0  # Log FPS every 5 seconds

    async def get_urdf_path(self) -> Path:
        """Get the path to the URDF file for the robot."""
        if self.urdf_path is not None:
            return self.urdf_path
        async with self.action_lock:
            if self.urdf_path is None:
                api = kscale.K()
                urdf_dir = await api.download_and_extract_urdf(self.name)
                self.urdf_path = next(Path(urdf_dir).glob("*.urdf"))
                logger.info("Downloaded URDF model %s", self.urdf_path)
        return self.urdf_path

    async def get_physics_client(self) -> int:
        """Get or create the PyBullet physics client."""
        if self.physics_client is not None:
            return self.physics_client
        async with self.action_lock:
            if self.physics_client is None:
                # Connect to PyBullet in GUI mode with direct control disabled
                self.physics_client = pybullet.connect(pybullet.GUI, options="--direct")
                # Disable mouse picking/dragging of objects
                pybullet.configureDebugVisualizer(
                    pybullet.COV_ENABLE_MOUSE_PICKING, 0, physicsClientId=self.physics_client
                )
                # Disable GUI controls and rendering options
                pybullet.configureDebugVisualizer(pybullet.COV_ENABLE_GUI, 0, physicsClientId=self.physics_client)
                pybullet.setAdditionalSearchPath(pybullet_data.getDataPath())

                # Load ground plane if not fixed base
                if not self.fixed_base:
                    pybullet.loadURDF("plane.urdf")

                # Set gravity based on fixed_base setting
                pybullet.setGravity(0, 0, 0 if self.fixed_base else -9.81)

                logger.info("Connected to PyBullet physics server with GUI control disabled")
        return self.physics_client

    async def get_robot(self) -> tuple[int, dict]:
        """Get or load the robot model."""
        if self.robot_id is not None and self.joint_info is not None:
            return self.robot_id, self.joint_info

        urdf_path = await self.get_urdf_path()
        physics_client = await self.get_physics_client()

        async with self.action_lock:
            if self.robot_id is None or self.joint_info is None:
                # Load the robot
                self.robot_id = pybullet.loadURDF(
                    str(urdf_path), useFixedBase=self.fixed_base, physicsClientId=physics_client
                )

                # Get joint information and disable default motor control
                self.joint_info = {}
                for joint_id in range(pybullet.getNumJoints(self.robot_id, physicsClientId=physics_client)):
                    info = pybullet.getJointInfo(self.robot_id, joint_id, physicsClientId=physics_client)
                    if info[2] != pybullet.JOINT_FIXED:  # Skip fixed joints
                        # Disable default motor control and add damping
                        pybullet.setJointMotorControl2(
                            self.robot_id,
                            joint_id,
                            pybullet.VELOCITY_CONTROL,
                            targetVelocity=0,
                            force=0,
                            physicsClientId=physics_client,
                        )
                        # Add joint damping to prevent free motion
                        pybullet.changeDynamics(
                            self.robot_id, joint_id, jointDamping=1.0, physicsClientId=physics_client
                        )

                        self.joint_info[info[1].decode("utf-8")] = {
                            "id": joint_id,
                            "type": info[2],
                            "lower_limit": info[8],
                            "upper_limit": info[9],
                            "max_force": info[10],
                            "max_velocity": info[11],
                        }

                logger.info("Loaded robot with %d controllable joints", len(self.joint_info))

        return self.robot_id, self.joint_info

    async def get_joint_names(self) -> list[str]:
        """Get list of joint names."""
        _, joint_info = await self.get_robot()
        return list(joint_info.keys())

    async def set_joint_angles(self, joint_angles: dict[str, float]) -> None:
        """Set joint angles for the robot."""
        robot_id, joint_info = await self.get_robot()
        physics_client = await self.get_physics_client()

        # Ensure physics client is running
        if not pybullet.isConnected(physicsClientId=physics_client):
            raise RuntimeError("PyBullet physics server is not running")

        # Set joint positions using position control
        for joint_name, target_pos in joint_angles.items():
            if joint_name in joint_info:
                joint_id = joint_info[joint_name]["id"]
                pybullet.setJointMotorControl2(
                    robot_id,
                    joint_id,
                    pybullet.POSITION_CONTROL,
                    target_pos,
                    force=joint_info[joint_name]["max_force"],
                    physicsClientId=physics_client,
                )

        # Step simulation
        if self.last_time is None:
            self.last_time = asyncio.get_running_loop().time()
        current_time = asyncio.get_running_loop().time()
        sim_time = current_time - self.last_time
        self.last_time = current_time

        # PyBullet timestep is 1/240 seconds
        num_steps = int(sim_time * 240)
        for _ in range(num_steps):
            pybullet.stepSimulation(physicsClientId=physics_client)

        # Track and log FPS
        now = current_time
        if self.last_render_time is not None:
            frame_time = now - self.last_render_time
            self.frame_times.append(frame_time)
            if len(self.frame_times) > self.fps_window_size:
                self.frame_times.pop(0)

            # Log FPS periodically
            if now >= self.next_fps_log:
                avg_frame_time = sum(self.frame_times) / len(self.frame_times)
                fps = 1.0 / avg_frame_time if avg_frame_time > 0 else 0
                logger.info("Rendering at %.2f FPS", fps)
                self.next_fps_log = now + self.fps_log_interval

        self.last_render_time = now


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mjcf_name", type=str, help="Name of the Mujoco model in the K-Scale API")
    parser.add_argument("--no-skip-root", action="store_true", help="Do not skip the root joint")
    args = parser.parse_args()

    colorlogging.configure()

    target_robot = PyBulletTargetRobot(args.mjcf_name)
    joint_names = await target_robot.get_joint_names()
    if not args.no_skip_root:
        joint_names = joint_names[1:]
    source_robot = SinusoidSourceRobot(joint_names)

    while True:
        joint_angles = await source_robot.get_joint_angles()
        await target_robot.set_joint_angles(joint_angles)
        await asyncio.sleep(0.01)


if __name__ == "__main__":
    # python -m digital_twin.target.pybullet
    asyncio.run(main())
