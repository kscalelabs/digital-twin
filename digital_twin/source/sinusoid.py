"""Defines a source robot model that generates sinusoidal joint angles."""

import math
import time

from digital_twin.source.base import SourceRobot


class SinusoidSourceRobot(SourceRobot):
    """Source robot model that generates sinusoidal joint angles."""

    def __init__(self, joint_names: list[str], amplitude: float = math.radians(30), frequency: float = 1.0) -> None:
        self.joint_names = joint_names
        self.amplitude = amplitude
        self.frequency = frequency
        self.start_time = time.time()

    async def get_joint_angles(self) -> dict[str, float]:
        current_time = time.time()
        delta_time = current_time - self.start_time
        position = self.amplitude * math.sin(delta_time * self.frequency * 2 * math.pi)
        return {name: position for name in self.joint_names}
