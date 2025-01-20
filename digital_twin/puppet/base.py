"""Defines the base target robot model, which mirrors the source robot actions."""

from abc import ABC, abstractmethod


class TargetRobot(ABC):
    """Target robot model."""

    @abstractmethod
    def get_joint_names(self) -> list[str]: ...

    @abstractmethod
    async def set_joint_angles(self, joint_angles: dict[str, float]) -> None: ...
