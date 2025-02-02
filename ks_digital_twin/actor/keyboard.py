"""Defines an actor robot model that allows for keyboard control."""

from pynput import keyboard
from pynput.keyboard import Key, KeyCode

from ks_digital_twin.actor.base import ActorRobot


class KeyboardActor(ActorRobot):
    """Actor robot model that allows for keyboard control."""

    def __init__(self, joint_names: list[str]) -> None:
        self.current_index = 0
        self.joint_names = joint_names
        self.current_joint_angles = {name: 0.0 for name in joint_names}

        # Print controls info
        print("Robot Joint Control")
        print("-----------------")
        print(f"Currently controlling: {joint_names[self.current_index]}")
        print("\nControls:")
        print("Tab: Switch joint")
        print("Up/Down: Adjust joint angle")
        print("Esc: Quit")

        # Set up keyboard listener
        self.listener = keyboard.Listener(on_press=self._on_press)
        self.listener.start()

    def _on_press(self, key: Key | KeyCode | None) -> None:
        """Handle keyboard press events."""
        if key is None:
            return

        try:
            if key == Key.tab:
                self._switch_joint()
            elif key == Key.up:
                self._update_angle(0.1)
            elif key == Key.down:
                self._update_angle(-0.1)
            elif key == Key.esc:
                self.listener.stop()
        except AttributeError:
            pass

    def _switch_joint(self) -> None:
        """Switch to the next joint."""
        self.current_index = (self.current_index + 1) % len(self.joint_names)
        print(f"\nNow controlling: {self.joint_names[self.current_index]}")
        print(f"Current angle: {self.current_joint_angles[self.joint_names[self.current_index]]:.2f}")

    def _update_angle(self, delta: float) -> None:
        """Update the angle of the current joint."""
        current_joint = self.joint_names[self.current_index]
        self.current_joint_angles[current_joint] += delta
        print(f"Joint {current_joint}: {self.current_joint_angles[current_joint]:.2f}")

    async def get_joint_angles(self) -> dict[str, float]:
        """Return the current joint angles."""
        if self.listener.is_alive():
            return self.current_joint_angles.copy()
        else:
            raise KeyboardInterrupt
