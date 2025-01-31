"""Defines the base puppet robot model, which mirrors the actor robot actions."""

from abc import ABC, abstractmethod


class Puppet(ABC):
    """Puppet robot model."""

    @abstractmethod
    async def get_joint_names(self) -> list[str]: ...

    @abstractmethod
    async def set_joint_angles(self, joint_angles: dict[str, float]) -> None: ...
