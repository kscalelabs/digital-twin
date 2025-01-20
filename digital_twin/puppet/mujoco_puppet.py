"""Implements the target robot model using Mujoco."""

import argparse
import asyncio
import logging
from pathlib import Path

import colorlogging
import kscale

from digital_twin.actor.sinusoid import SinusoidActor
from digital_twin.puppet.base import Puppet

try:
    import mujoco
    import mujoco_viewer
    from mujoco import MjData, MjModel
except ImportError:
    raise ImportError("Mujoco is not installed, please install it using `pip install mujoco mujoco-python-viewer`")

logger = logging.getLogger(__name__)


class MujocoPuppet(Puppet):
    """Puppet robot model using Mujoco."""

    def __init__(self, name: str, fixed_base: bool = True) -> None:
        self.name = name
        self.fixed_base = fixed_base
        self.action_lock = asyncio.Lock()
        self.mjcf_path = None
        self.mj_model: MjModel | None = None
        self.mj_data: MjData | None = None
        self.mj_viewer: mujoco_viewer.MujocoViewer | None = None
        self.sensor_ids: dict[str, int] | None = None
        self.actuator_ids: dict[str, int] | None = None
        self.last_time: float | None = None

        # FPS tracking
        self.last_render_time: float | None = None
        self.fps_window_size = 60  # Calculate average over last 60 frames
        self.frame_times: list[float] = []
        self.next_fps_log = 0  # Time when we should next log FPS
        self.fps_log_interval = 5.0  # Log FPS every 5 seconds

    async def get_mjcf_path(self) -> None:
        if self.mjcf_path is not None:
            return self.mjcf_path
        async with self.action_lock:
            if self.mjcf_path is None:
                api = kscale.K()
                mjcf_dir = await api.download_and_extract_urdf(self.name)
                self.mjcf_path = next(Path(mjcf_dir).glob("*.mjcf"))
                logger.info("Downloaded Mujoco model %s", self.mjcf_path)
        return self.mjcf_path

    async def get_mj_model_and_data(self) -> tuple[MjModel, MjData]:
        if self.mj_model is not None and self.mj_data is not None:
            return self.mj_model, self.mj_data
        mjcf_path = await self.get_mjcf_path()
        async with self.action_lock:
            if self.mj_model is None or self.mj_data is None:
                self.mj_model = mujoco.MjModel.from_xml_path(str(mjcf_path))

                # If fixed_base is True, disable gravity to make the robot float in place
                if self.fixed_base:
                    self.mj_model.opt.gravity[2] = 0  # Set vertical gravity to 0
                    logger.info("Disabled gravity for fixed base mode")

                self.mj_data = mujoco.MjData(self.mj_model)

                # Reset velocities and accelerations.
                self.mj_data.qvel[:] = 0
                self.mj_data.qacc[:] = 0

                # Runs one step.
                mujoco.mj_forward(self.mj_model, self.mj_data)
                mujoco.mj_step(self.mj_model, self.mj_data)

                self.sensor_ids = {self.mj_model.sensor(i).name: i for i in range(self.mj_model.nsensor)}
                self.actuator_ids = {self.mj_model.actuator(i).name: i for i in range(self.mj_model.nu)}

        return self.mj_model, self.mj_data

    async def get_mj_viewer(self) -> mujoco_viewer.MujocoViewer:
        if self.mj_viewer is not None:
            return self.mj_viewer
        mj_model, mj_data = await self.get_mj_model_and_data()
        async with self.action_lock:
            if self.mj_viewer is None:
                self.mj_viewer = mujoco_viewer.MujocoViewer(mj_model, mj_data)
        return self.mj_viewer

    async def get_joint_names(self) -> list[str]:
        mj_model, _ = await self.get_mj_model_and_data()
        joint_names = [mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(mj_model.njnt)]
        return joint_names

    async def set_joint_angles(self, joint_angles: dict[str, float]) -> None:
        mj_model, mj_data = await self.get_mj_model_and_data()
        mj_viewer = await self.get_mj_viewer()
        if not mj_viewer.is_alive:
            mj_viewer.close()
            raise RuntimeError("MuJoCo viewer is not running")

        # TODO: This is not implemented correctly.
        kp = 1.0
        for name, target_pos in joint_angles.items():
            actuator_id = self.actuator_ids[name]
            if actuator_id <= 0:
                continue
            current_pos = mj_data.qpos[actuator_id]
            tau = kp * (target_pos - current_pos)
            mj_data.ctrl[actuator_id] = tau

        # Steps to the current time.
        if self.last_time is None:
            self.last_time = asyncio.get_running_loop().time()
        current_time = asyncio.get_running_loop().time()
        sim_time = current_time - self.last_time
        self.last_time = current_time
        while sim_time > 0:
            mujoco.mj_step(mj_model, mj_data)
            sim_time -= mj_model.opt.timestep

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
        mj_viewer.render()


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mjcf_name", type=str, help="Name of the Mujoco model in the K-Scale API")
    parser.add_argument("--no-skip-root", action="store_true", help="Do not skip the root joint")
    args = parser.parse_args()

    colorlogging.configure()

    puppet = MujocoPuppet(args.mjcf_name)
    joint_names = await puppet.get_joint_names()
    if not args.no_skip_root:
        joint_names = joint_names[1:]
    actor = SinusoidActor(joint_names)

    while True:
        joint_angles = await actor.get_joint_angles()
        await puppet.set_joint_angles(joint_angles)
        await asyncio.sleep(0.01)


if __name__ == "__main__":
    # python -m digital_twin.target.mujoco
    asyncio.run(main())
