"""Test PyKOS communication."""

from pykos import KOS

from digital_twin.actor.pykos_bot import PyKOSActor


def main() -> None:
    kos = KOS()
    _actor = PyKOSActor(kos, {"shoulder": 0, "elbow": 1, "wrist": 2})


if __name__ == "__main__":
    main()
