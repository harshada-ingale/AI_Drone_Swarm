"""Abstract interface and result models for swarm communication routing."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Mapping

from controller.models import DroneState


class RoutingError(Exception):
    """Base exception for routing subsystem failures."""


class NoRouteFoundError(RoutingError):
    """Raised when no communication path exists between source and destination."""


@dataclass
class RouteResult:
    """Outcome of a route discovery query."""

    success: bool
    path: list[str] = field(default_factory=list)  # e.g., ["D7", "D4", "D3", "GS"]
    total_cost: float = float("inf")
    hop_count: int = 0
    error_message: str | None = None

    def __bool__(self) -> bool:
        return self.success


class RoutingInterface(ABC):
    """Clean abstract interface for swarm routing engines."""

    @abstractmethod
    def find_route(
        self,
        source: str,
        destination: str,
        nodes: Mapping[str, DroneState],
        excluded_nodes: set[str] | None = None,
    ) -> RouteResult:
        """Finds optimal communication route from source to destination.

        Args:
            source: Node ID initiating transmission (e.g. 'D7').
            destination: Final target node ID (e.g. 'GS').
            nodes: Map of node IDs to their current DroneState.
            excluded_nodes: Optional set of node IDs (failed, isolated, compromised) to bypass.

        Returns:
            RouteResult containing success flag, ordered path list, and metrics.
        """
        raise NotImplementedError
