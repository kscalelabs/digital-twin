"""Defines the base source robot model, which generates the actions."""

import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class SourceRobot(ABC):
    """Abstract base class for robot models."""

    @abstractmethod
    async def get_joint_angles(self) -> dict[str, float]: ...
