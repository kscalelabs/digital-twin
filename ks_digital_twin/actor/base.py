"""Defines the base actor robot model, which generates actions."""

import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class ActorRobot(ABC):
    """Abstract base class for robot models."""

    @abstractmethod
    async def get_joint_angles(self) -> dict[str, float]: ...
