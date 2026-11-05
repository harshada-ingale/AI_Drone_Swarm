"""Flight telemetry provider and state containers."""

from dataclasses import dataclass


@dataclass
class Telemetry:
    """Drone flight and attitude state."""

    altitude_m: float = 30.0
    xy: tuple[float, float] = (0.0, 0.0)
    yaw_deg: float = 0.0
    hfov_deg: float = 84.0


class StaticTelemetry:
    """Static telemetry provider for testing and offline simulations.

    Note: In production hardware setups, a MAVLink/ROS 2 telemetry provider
    exposing the exact same .get() -> Telemetry interface will replace this class.
    """

    def __init__(
        self,
        altitude_m: float = 30.0,
        xy: tuple[float, float] = (0.0, 0.0),
        yaw_deg: float = 0.0,
        hfov_deg: float = 84.0,
    ) -> None:
        self._telemetry = Telemetry(
            altitude_m=altitude_m,
            xy=xy,
            yaw_deg=yaw_deg,
            hfov_deg=hfov_deg,
        )

    def get(self) -> Telemetry:
        """Returns current flight telemetry state."""
        return self._telemetry
