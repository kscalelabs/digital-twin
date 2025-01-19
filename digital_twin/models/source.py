"""Defines the source robot model, which generates the actions."""

import logging
import threading
import tkinter as tk
from abc import ABC, abstractmethod

from pykos import KOS

logger = logging.getLogger(__name__)


class SourceRobot(ABC):
    """Abstract base class for robot models."""

    @abstractmethod
    async def get_joint_angles(self) -> dict[str, float]: ...


class KeyboardSourceRobot(SourceRobot):
    """Source robot model that allows for keyboard control."""

    def __init__(self, joint_names: list[str]) -> None:
        self.current_index = 0
        self.joint_names = joint_names
        self.current_joint_angles = {name: 0.0 for name in joint_names}
        self.pressed_keys = set()

        # Create Tkinter window
        self.window = tk.Tk()
        self.window.title("Robot Joint Control")
        self.window.geometry("400x300")

        # Create label to show current joint
        self.joint_label = tk.Label(self.window, text=f"Controlling joint: {joint_names[self.current_index]}")
        self.joint_label.pack(pady=10)

        # Add angle display label
        self.angle_label = tk.Label(self.window, text="Current angle: 0.0")
        self.angle_label.pack(pady=5)

        # Create label to show controls
        controls_text = "Controls:\n" "Tab: Switch joint\n" "Up/Down: Adjust joint angle\n" "Esc: Quit"
        tk.Label(self.window, text=controls_text).pack(pady=10)

        # Bind keyboard events
        self.window.bind("<Tab>", lambda e: self._switch_joint())
        self.window.bind("<Up>", lambda e: self._update_angle(0.1))
        self.window.bind("<Down>", lambda e: self._update_angle(-0.1))
        self.window.bind("<Escape>", lambda e: self.window.quit())

        # No need to start mainloop here - it will be started in the main thread

    def _switch_joint(self) -> None:
        """Switch to the next joint."""
        self.current_index = (self.current_index + 1) % len(self.joint_names)
        self.joint_label.config(text=f"Controlling joint: {self.joint_names[self.current_index]}")
        # Update angle display for new joint
        current_joint = self.joint_names[self.current_index]
        self.angle_label.config(text=f"Current angle: {self.current_joint_angles[current_joint]:.2f}")

    def _update_angle(self, delta: float) -> None:
        """Update the angle of the current joint."""
        current_joint = self.joint_names[self.current_index]
        self.current_joint_angles[current_joint] += delta
        # Update angle display
        self.angle_label.config(text=f"Current angle: {self.current_joint_angles[current_joint]:.2f}")

    async def get_joint_angles(self) -> dict[str, float]:
        """Return the current joint angles."""
        # No need to call window.update() anymore since mainloop is running
        return self.current_joint_angles.copy()


class PyKOSRobot(SourceRobot):
    """Interface for communicating with a robot using PyKOS."""

    def __init__(
        self,
        kos: KOS,
        joint_names: dict[str, int],
    ) -> None:
        self.kos = kos
        self.joint_names_to_ids = joint_names
        self.joint_ids_to_names = {v: k for k, v in joint_names.items()}
        if len(self.joint_ids_to_names) != len(self.joint_names_to_ids):
            raise ValueError("Joint IDs must be unique")
        self.joint_ids = list(sorted(list(joint_names.values())))

    async def get_joint_angles(self) -> dict[str, float]:
        states = self.kos.actuator.get_actuators_state(self.joint_ids)
        id_to_angle = {state.actuator_id: state.position for state in states.states}
        return {self.joint_ids_to_names[id]: angle for id, angle in id_to_angle.items()}
