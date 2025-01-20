"""Defines a source robot model that communicates with a robot using PyKOS."""

from pykos import KOS

from digital_twin.actor.base import SourceRobot


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
