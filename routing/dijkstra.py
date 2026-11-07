"""Weighted Dijkstra routing engine for drone swarm mesh communications."""

import heapq
import math
from typing import Mapping

from controller.models import DroneHealthStatus, DroneState
from routing.router import RouteResult, RoutingInterface


class DijkstraRouter(RoutingInterface):
    """Computes optimal communication paths using multi-factor weighted Dijkstra search."""

    def __init__(
        self,
        max_comm_range_m: float = 500.0,
        ground_station_id: str = "GS",
        ground_station_pos: tuple[float, float] = (0.0, 0.0),
        weight_distance: float = 1.0,
        weight_battery: float = 2.0,
        weight_signal: float = 1.5,
        weight_packet_loss: float = 3.0,
        weight_latency: float = 1.0,
        weight_threat: float = 10.0,
    ) -> None:
        self.max_comm_range_m = max_comm_range_m
        self.ground_station_id = ground_station_id
        self.ground_station_pos = ground_station_pos

        self.w_dist = weight_distance
        self.w_bat = weight_battery
        self.w_sig = weight_signal
        self.w_loss = weight_packet_loss
        self.w_lat = weight_latency
        self.w_threat = weight_threat

        # Optional explicit topology overrides: {node_id: set_of_neighbor_ids}
        self._manual_neighbors: dict[str, set[str]] = {}

    def set_neighbor_override(self, u: str, neighbors: set[str]) -> None:
        """Sets explicit connectivity topology overriding geometric distance."""
        self._manual_neighbors[u] = set(neighbors)

    def clear_neighbor_overrides(self) -> None:
        """Clears explicit neighbor connectivity overrides."""
        self._manual_neighbors.clear()

    def _get_node_position(self, node_id: str, nodes: Mapping[str, DroneState]) -> tuple[float, float]:
        """Returns (x, y) coordinates for a node or Ground Station."""
        if node_id == self.ground_station_id:
            # Check if GS is registered in nodes
            if node_id in nodes:
                return nodes[node_id].position
            return self.ground_station_pos
        if node_id in nodes:
            return nodes[node_id].position
        return (0.0, 0.0)

    def _distance(self, pos1: tuple[float, float], pos2: tuple[float, float]) -> float:
        """Calculates 2D Euclidean distance in metres."""
        dx = pos1[0] - pos2[0]
        dy = pos1[1] - pos2[1]
        return math.hypot(dx, dy)

    def _calculate_edge_cost(
        self,
        u: str,
        v: str,
        dist: float,
        nodes: Mapping[str, DroneState],
    ) -> float:
        """Calculates multi-metric link cost for traversing hop u -> v."""
        # 1. Normalized distance cost (0.0 - 1.0+)
        dist_factor = dist / max(1.0, self.max_comm_range_m)

        # Metrics of target node v
        if v == self.ground_station_id:
            # Ground Station has unlimited power and zero threat
            bat_factor = 0.0
            sig_factor = 0.1
            loss_factor = 0.0
            lat_factor = 0.1
            threat_factor = 0.0
        elif v in nodes:
            st = nodes[v]
            # Lower battery -> higher penalty: 100% -> 0.0, 20% -> 0.8
            bat_factor = max(0.0, (100.0 - st.battery) / 100.0)
            # Weaker signal -> higher penalty (-30 dBm -> 0.3, -90 dBm -> 0.9)
            sig_factor = min(1.0, max(0.0, abs(st.signal_strength) / 100.0))
            # Packet loss (0.0 - 1.0)
            loss_factor = min(1.0, max(0.0, st.packet_loss))
            # Latency (normalized by 200ms)
            lat_factor = min(2.0, max(0.0, st.latency / 200.0))
            # Threat level (0.0 - 1.0)
            threat_factor = min(1.0, max(0.0, st.threat_level))
        else:
            bat_factor = 0.5
            sig_factor = 0.5
            loss_factor = 0.2
            lat_factor = 0.5
            threat_factor = 0.0

        edge_weight = (
            self.w_dist * dist_factor
            + self.w_bat * bat_factor
            + self.w_sig * sig_factor
            + self.w_loss * loss_factor
            + self.w_lat * lat_factor
            + self.w_threat * threat_factor
            + 0.1  # Hop penalty to avoid excessively long paths
        )
        return max(0.01, edge_weight)

    def find_route(
        self,
        source: str,
        destination: str,
        nodes: Mapping[str, DroneState],
        excluded_nodes: set[str] | None = None,
    ) -> RouteResult:
        """Finds optimal weighted path between source and destination, excluding specified nodes."""
        excluded = set(excluded_nodes or set())

        # If source equals destination
        if source == destination:
            return RouteResult(success=True, path=[source], total_cost=0.0, hop_count=0)

        # If source or destination are explicitly excluded
        if source in excluded:
            return RouteResult(
                success=False,
                error_message=f"Source node '{source}' is in the excluded set.",
            )
        if destination in excluded:
            return RouteResult(
                success=False,
                error_message=f"Destination node '{destination}' is in the excluded set.",
            )

        # Collect all eligible nodes
        eligible_node_ids = set(nodes.keys())
        eligible_node_ids.add(self.ground_station_id)
        eligible_node_ids.add(source)
        eligible_node_ids.add(destination)
        eligible_node_ids -= excluded

        if source not in eligible_node_ids:
            return RouteResult(
                success=False,
                error_message=f"Source node '{source}' is not available.",
            )
        if destination not in eligible_node_ids:
            return RouteResult(
                success=False,
                error_message=f"Destination node '{destination}' is not available.",
            )

        # Build adjacency graph
        adj: dict[str, list[tuple[str, float]]] = {n: [] for n in eligible_node_ids}

        for u in eligible_node_ids:
            pos_u = self._get_node_position(u, nodes)

            # Check if manual neighbors are specified
            if u in self._manual_neighbors:
                for v in self._manual_neighbors[u]:
                    if v in eligible_node_ids:
                        pos_v = self._get_node_position(v, nodes)
                        dist = self._distance(pos_u, pos_v)
                        cost = self._calculate_edge_cost(u, v, dist, nodes)
                        adj[u].append((v, cost))
            else:
                for v in eligible_node_ids:
                    if u == v:
                        continue
                    pos_v = self._get_node_position(v, nodes)
                    dist = self._distance(pos_u, pos_v)
                    if dist <= self.max_comm_range_m:
                        cost = self._calculate_edge_cost(u, v, dist, nodes)
                        adj[u].append((v, cost))

        # Dijkstra priority queue: (cumulative_cost, current_node, path)
        pq: list[tuple[float, str, list[str]]] = [(0.0, source, [source])]
        min_costs: dict[str, float] = {source: 0.0}

        while pq:
            cost, curr, path = heapq.heappop(pq)

            if curr == destination:
                return RouteResult(
                    success=True,
                    path=path,
                    total_cost=round(cost, 4),
                    hop_count=len(path) - 1,
                )

            if cost > min_costs.get(curr, float("inf")):
                continue

            for neighbor, edge_weight in adj.get(curr, []):
                new_cost = cost + edge_weight
                if new_cost < min_costs.get(neighbor, float("inf")):
                    min_costs[neighbor] = new_cost
                    heapq.heappush(pq, (new_cost, neighbor, path + [neighbor]))

        return RouteResult(
            success=False,
            path=[],
            total_cost=float("inf"),
            hop_count=0,
            error_message=f"No communication path found between '{source}' and '{destination}'.",
        )

