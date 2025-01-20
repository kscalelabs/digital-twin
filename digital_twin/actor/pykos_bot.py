"""Defines an actor robot model that communicates with a robot using PyKOS."""

import math

from pykos import KOS

from digital_twin.actor.base import ActorRobot


class PyKOSActor(ActorRobot):
    """Interface for communicating with a robot using PyKOS."""

    def __init__(
        self,
        kos: KOS,
        joint_names: dict[str, int],
        kos_offsets: dict[str, float] | None = None,
        kos_signs: dict[str, float] | None = None,
    ) -> None:
        """Initialize the PyKOS actor.

        Args:
            kos: The PyKOS instance to use.
            joint_names: A dictionary mapping joint names to their IDs.
            kos_offsets: A dictionary mapping joint names to their offsets in degrees.
            kos_signs: A dictionary mapping joint names to their signs.
        """
        self.kos = kos
        self.joint_names_to_ids = joint_names
        self.joint_ids_to_names = {v: k for k, v in joint_names.items()}
        if len(self.joint_ids_to_names) != len(self.joint_names_to_ids):
            raise ValueError("Joint IDs must be unique")
        self.joint_ids = list(sorted(list(joint_names.values())))

        self.current_offsets = kos_offsets
        self.current_signs = kos_signs
        if kos_signs is None:
            self.current_signs = {k: 1 for k in self.joint_names_to_ids}

    def get_raw_angles(self) -> dict[int, float]:
        states = self.kos.actuator.get_actuators_state(self.joint_ids)
        return {state.actuator_id: state.position for state in states.states}

    def get_named_angles(self, radians: bool = True) -> dict[str, float]:
        return {self.joint_ids_to_names[id]: math.radians(angle) if radians else angle for id, angle in self.get_raw_angles().items()}

    def offset_in_place(self) -> None:
        self.current_offsets = {k: v * -1 for k, v in self.get_named_angles(radians=False).items()}

    async def get_joint_angles(self) -> dict[str, float]:
        if self.current_offsets is None:
            print("No offsets set, returning values directly")
            return self.get_named_angles(radians=True)
        return {
            name: math.radians((angle + self.current_offsets[name]) * self.current_signs[name])
            for name, angle in self.get_named_angles(radians=False).items()
        }

    def get_offsets(self) -> dict[str, float] | None:
        return self.current_offsets
