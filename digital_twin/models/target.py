"""Defines the target robot model, which mirrors the source robot actions."""

import asyncio
import logging
from abc import ABC, abstractmethod
from pathlib import Path

import kscale
import mujoco
import mujoco_viewer
from mujoco import MjData, MjModel

logger = logging.getLogger(__name__)


class TargetRobot(ABC):
    """Target robot model."""

    @abstractmethod
    def get_joint_names(self) -> list[str]: ...

    @abstractmethod
    async def set_joint_angles(self, joint_angles: dict[str, float]) -> None: ...


class MujocoTargetRobot(TargetRobot):
    """Target robot model using Mujoco."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.action_lock = asyncio.Lock()
        self.mjcf_path = None
        self.mj_model: MjModel | None = None
        self.mj_data: MjData | None = None
        self.mj_viewer: mujoco_viewer.MujocoViewer | None = None

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
                self.mj_data = mujoco.MjData(self.mj_model)
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
        for name, angle in joint_angles.items():
            joint_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_JOINT, name)
            mj_data.qpos[joint_id] = angle
        mujoco.mj_step(mj_model, mj_data)
        mj_viewer.render()
