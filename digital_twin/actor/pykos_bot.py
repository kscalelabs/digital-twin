"""Defines an actor robot model that communicates with a robot using PyKOS."""

import asyncio
import logging
import math

import kscale
from kscale.web.gen.api import RobotURDFMetadataOutput
import numpy as np
import scipy
from pykos import KOS

from digital_twin.actor.base import ActorRobot

logger = logging.getLogger(__name__)


class PyKOSActor(ActorRobot):
    """Interface for communicating with a robot using PyKOS."""

    def __init__(self, name: str, kos: KOS) -> None:
        """Initialize the PyKOS actor.

        Args:
            name: The name of the robot.
            kos: The PyKOS instance to use.
        """
        self.name = name
        self.kos = kos

        self._action_lock = asyncio.Lock()
        self._metadata: RobotURDFMetadataOutput | None = None
        self._joint_ids: list[int] | None = None
        self._joint_ids_to_names: dict[int, str] | None = None

    async def get_metadata(self) -> RobotURDFMetadataOutput:
        if self._metadata is not None:
            return self._metadata
        async with self._action_lock:
            api = kscale.K()
            robot_class = await api.get_robot_class(self.name)
            metadata = robot_class.metadata
            if metadata is None:
                raise ValueError(f"No metadata found for robot {self.name}")
            self._metadata = metadata
        return self._metadata

    async def get_joint_ids_to_names(self) -> dict[int, str]:
        if self._joint_ids_to_names is not None:
            return self._joint_ids_to_names
        metadata = await self.get_metadata()
        async with self._action_lock:
            self._joint_ids_to_names = {v.id: k for k, v in metadata.joint_name_to_metadata.items()}
        return self._joint_ids_to_names

    async def get_joint_ids(self) -> list[int]:
        if self._joint_ids is not None:
            return self._joint_ids
        joint_ids_to_names = await self.get_joint_ids_to_names()
        self._joint_ids = list(sorted(joint_ids_to_names.keys()))
        return self._joint_ids
        self.joint_names_to_ids = joint_names
        self.joint_ids_to_names = {v: k for k, v in joint_names.items()}
        if len(self.joint_ids_to_names) != len(self.joint_names_to_ids):
            raise ValueError("Joint IDs must be unique")
        self.joint_ids = list(sorted(list(joint_names.values())))

        self.current_offsets = kos_offsets
        self.orn_offset = None
        self.current_signs = {k: 1.0 for k in self.joint_names_to_ids} if kos_signs is None else kos_signs

    def get_raw_angles(self) -> dict[int, float]:
        joint_ids = self.get_joint_ids()
        states = self.kos.actuator.get_actuators_state(joint_ids)
        return {state.actuator_id: state.position for state in states.states}

    async def get_named_angles(self, radians: bool = True) -> dict[str, float]:
        joint_ids_to_names = await self.get_joint_ids_to_names()
        return {
            joint_ids_to_names[id]: math.radians(angle) if radians else angle
            for id, angle in self.get_raw_angles().items()
        }

    async def offset_in_place(self) -> None:
        self.current_offsets = {k: v * -1 for k, v in (await self.get_named_angles(radians=False)).items()}

        initial_orientation = await self.get_orientation()
        self.orn_offset = scipy.spatial.transform.Rotation.from_quat(initial_orientation).inv().as_quat()

    async def get_joint_angles(self) -> dict[str, float]:
        if self.current_offsets is None:
            logger.info("No offsets set, returning values directly")
            return await self.get_named_angles(radians=True)
        return {
            name: math.radians((angle + self.current_offsets[name]) * (1.0 if self.current_signs[name] else -1.0))
            for name, angle in self.get_named_angles(radians=False).items()
        }

    async def get_orientation(self) -> tuple[float, float, float, float]:
        angles = await self.kos.imu.get_euler_angles()
        current_quat = scipy.spatial.transform.Rotation.from_euler("xyz", np.deg2rad([angles.roll, angles.pitch, angles.yaw])).as_quat()
        if self.orn_offset is not None:
            # Apply the offset by quaternion multiplication
            offset_rot = scipy.spatial.transform.Rotation.from_quat(self.orn_offset)
            current_rot = scipy.spatial.transform.Rotation.from_quat(current_quat)
            return (offset_rot * current_rot).as_quat()
        return current_quat

    def get_offsets(self) -> dict[str, float] | None:
        return self.current_offsets

