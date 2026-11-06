"""Central Self-Healing Swarm Controller orchestrating the MONITOR-DETECT-ISOLATE-REROUTE-RECOVER-VERIFY workflow."""

import logging
import time
from typing import Any, Mapping

from controller.adapters import (
    AIThreatAdapter,
    SecurityAdapter,
    SimulatorAdapter,
    VisionAlertAdapter,
)
from controller.config import SelfHealingConfig
from controller.detector import DetectionResult, FailureDetector
from controller.models import (
    DroneHealthStatus,
    DroneState,
    EventType,
    FailureReason,
    SelfHealingEvent,
    ThreatReport,
)
from controller.recovery import RecoveryManager
from routing.dijkstra import DijkstraRouter
from routing.router import NoRouteFoundError, RouteResult, RoutingInterface
from vision.emergency_alert import EmergencyAlert

logger = logging.getLogger("AI_Drone_Swarm.Controller")


class SelfHealingController:
    """Coordinates drone swarm self-healing, health monitoring, isolation, rerouting, and safe recovery."""

    def __init__(
        self,
        config: SelfHealingConfig | None = None,
        router: RoutingInterface | None = None,
        detector: FailureDetector | None = None,
        recovery_manager: RecoveryManager | None = None,
        security_adapter: SecurityAdapter | None = None,
    ) -> None:
        self.config = config or SelfHealingConfig()
        self.router = router or DijkstraRouter(
            max_comm_range_m=self.config.max_comm_range_m,
            ground_station_id=self.config.ground_station_id,
        )
        self.detector = detector or FailureDetector(self.config)
        self.recovery = recovery_manager or RecoveryManager(self.config, self.detector)
        self.security = security_adapter or SecurityAdapter()
        self.vision_adapter = VisionAlertAdapter(self.security)

        # Internal swarm state storage: {drone_id: DroneState}
        self._drones: dict[str, DroneState] = {}

        # Active communication routes: {route_id: {"source": str, "destination": str, "path": list[str], "original_path": list[str]}}
        self._active_routes: dict[str, dict[str, Any]] = {}

        # Event audit log: list of SelfHealingEvent
        self._events: list[SelfHealingEvent] = []

    # --------------------------------------------------------------------------
    # DRONE REGISTRATION & STATE ACCESS
    # --------------------------------------------------------------------------

    def register_drone(
        self,
        drone_id: str,
        initial_state: DroneState | dict[str, Any] | None = None,
    ) -> DroneState:
        """Registers a drone in the swarm controller inventory."""
        if not self.config.is_valid_drone_id(drone_id):
            raise ValueError(f"Invalid drone_id '{drone_id}'. Does not conform to swarm naming rules.")

        if initial_state is None:
            state = DroneState(drone_id=drone_id)
        elif isinstance(initial_state, DroneState):
            state = initial_state
            state.drone_id = drone_id
        elif isinstance(initial_state, dict):
            state_data = dict(initial_state)
            state_data["drone_id"] = drone_id
            state = DroneState.from_dict(state_data)
        else:
            raise TypeError("initial_state must be a DroneState, dict, or None.")

        state.validate()
        self._drones[drone_id] = state
        return state

    def get_drone_state(self, drone_id: str) -> DroneState | None:
        """Returns the DroneState for a given drone ID, or None if not registered."""
        return self._drones.get(drone_id)

    def get_all_drones(self) -> dict[str, DroneState]:
        """Returns copy of all registered drone states."""
        return dict(self._drones)

    def get_isolated_and_failed_drones(self) -> set[str]:
        """Returns set of node IDs currently ineligible for routing."""
        excluded = set()
        for d_id, st in self._drones.items():
            if st.operational_status in (
                DroneHealthStatus.ISOLATED,
                DroneHealthStatus.FAILED,
                DroneHealthStatus.RECOVERING,
            ):
                excluded.add(d_id)
            elif st.is_compromised:
                excluded.add(d_id)
        return excluded

    # --------------------------------------------------------------------------
    # MONITOR & DETECT: TELEMETRY & HEALTH INGESTION
    # --------------------------------------------------------------------------

    def update_drone_health(
        self,
        update: dict[str, Any] | DroneState,
        current_time: float | None = None,
    ) -> DetectionResult:
        """Main entry point to receive telemetry updates from simulator or drone node.

        Evaluates health thresholds, updates consecutive counters, triggers isolation
        or progresses recovery.
        """
        now = time.time() if current_time is None else current_time

        if isinstance(update, dict):
            parsed = SimulatorAdapter.parse_telemetry(update)
            drone_id = parsed["drone_id"]
        elif isinstance(update, DroneState):
            drone_id = update.drone_id
            parsed = update.to_dict()
        else:
            raise TypeError("Update must be a dictionary or DroneState instance.")

        if not self.config.is_valid_drone_id(drone_id):
            raise ValueError(f"Invalid drone_id '{drone_id}'.")

        # Get or register drone
        if drone_id not in self._drones:
            self.register_drone(drone_id)

        state = self._drones[drone_id]

        # Update telemetry fields (preserve existing state on partial dictionary updates)
        if isinstance(update, dict):
            if "position" in update:
                state.position = parsed["position"]
            if "altitude" in update:
                state.altitude = float(parsed["altitude"])
            if "heading" in update:
                state.heading = float(parsed["heading"])
            if "battery" in update:
                state.battery = float(parsed["battery"])
            if "signal_strength" in update:
                state.signal_strength = float(parsed["signal_strength"])
            if "packet_loss" in update:
                state.packet_loss = float(parsed["packet_loss"])
            if "latency" in update:
                state.latency = float(parsed["latency"])
            if "threat_level" in update:
                state.threat_level = float(parsed["threat_level"])
        else:
            state.position = parsed["position"]
            state.altitude = float(parsed["altitude"])
            state.heading = float(parsed["heading"])
            state.battery = float(parsed["battery"])
            state.signal_strength = float(parsed["signal_strength"])
            state.packet_loss = float(parsed["packet_loss"])
            state.latency = float(parsed["latency"])
            state.threat_level = float(parsed["threat_level"])

        state.last_seen = now
        state.validate()

        # Run failure detection evaluation
        result = self.detector.evaluate(state, current_time=now)

        # ----------------------------------------------------------------------
        # STATE MACHINE TRANSITIONS
        # ----------------------------------------------------------------------
        current_status = state.operational_status

        if current_status == DroneHealthStatus.ISOLATED:
            # Check if drone is ready to attempt recovery
            if result.is_healthy:
                success, msg, rec_evt = self.recovery.start_recovery(state, current_time=now)
                if success and rec_evt:
                    self._record_event(rec_evt)

        elif current_status == DroneHealthStatus.RECOVERING:
            # Check subsequent recovery verification
            success, msg, rec_evt = self.recovery.process_recovery_check(state, current_time=now)
            if rec_evt:
                self._record_event(rec_evt)
            if success:
                # Drone is fully restored to HEALTHY!
                if self.config.auto_restore_routes:
                    self.check_route_restorations()

        elif current_status in (DroneHealthStatus.HEALTHY, DroneHealthStatus.DEGRADED):
            if result.is_failed:
                # Failure condition detected
                if result.should_isolate_immediately:
                    # Immediate isolation (e.g. security breach)
                    reason = result.failure_reasons[0] if result.failure_reasons else FailureReason.SECURITY_COMPROMISED
                    self.isolate_drone(
                        drone_id,
                        reason=reason,
                        details={"metrics": result.metrics_exceeded, "immediate": True},
                        current_time=now,
                    )
                else:
                    # Fluctuation dampening: track consecutive failures
                    state.consecutive_failures += 1
                    state.consecutive_healthy = 0
                    if state.consecutive_failures >= self.config.consecutive_failures_to_isolate:
                        reason = result.failure_reasons[0] if result.failure_reasons else FailureReason.HARDWARE_FAULT
                        self.isolate_drone(
                            drone_id,
                            reason=reason,
                            details={
                                "metrics": result.metrics_exceeded,
                                "consecutive_failures": state.consecutive_failures,
                            },
                            current_time=now,
                        )
                    else:
                        # Mark as degraded while awaiting confirmation
                        if state.operational_status != DroneHealthStatus.DEGRADED:
                            state.previous_status = state.operational_status
                            state.operational_status = DroneHealthStatus.DEGRADED
                            self._record_event(
                                SelfHealingEvent(
                                    event_type=EventType.STATUS_CHANGED,
                                    drone_id=drone_id,
                                    timestamp=now,
                                    previous_state=state.previous_status.value,
                                    new_state=DroneHealthStatus.DEGRADED.value,
                                    details={"reason": "Transient degradation under confirmation", "metrics": result.metrics_exceeded},
                                )
                            )
            elif result.is_degraded:
                # Degraded condition (non-critical)
                state.consecutive_failures = 0
                if state.operational_status != DroneHealthStatus.DEGRADED:
                    state.previous_status = state.operational_status
                    state.operational_status = DroneHealthStatus.DEGRADED
                    self._record_event(
                        SelfHealingEvent(
                            event_type=EventType.STATUS_CHANGED,
                            drone_id=drone_id,
                            timestamp=now,
                            previous_state=state.previous_status.value,
                            new_state=DroneHealthStatus.DEGRADED.value,
                            details={"metrics": result.metrics_exceeded},
                        )
                    )
            else:
                # Healthy
                state.consecutive_failures = 0
                if state.operational_status != DroneHealthStatus.HEALTHY:
                    prev = state.operational_status.value
                    state.previous_status = state.operational_status
                    state.operational_status = DroneHealthStatus.HEALTHY
                    self._record_event(
                        SelfHealingEvent(
                            event_type=EventType.STATUS_CHANGED,
                            drone_id=drone_id,
                            timestamp=now,
                            previous_state=prev,
                            new_state=DroneHealthStatus.HEALTHY.value,
                            details={"message": "Telemetry normalized back to HEALTHY."},
                        )
                    )

        return result

    def heartbeat(self, drone_id: str, current_time: float | None = None) -> None:
        """Records keepalive heartbeat timestamp for a drone without requiring full telemetry."""
        now = time.time() if current_time is None else current_time
        if drone_id in self._drones:
            self._drones[drone_id].last_seen = now
            self._drones[drone_id].consecutive_failures = 0

    def heartbeat_active_drones(self, current_time: float | None = None) -> None:
        """Records keepalive heartbeat for all healthy non-isolated drones in the swarm."""
        now = time.time() if current_time is None else current_time
        for d_id, st in self._drones.items():
            if d_id != self.config.ground_station_id and st.operational_status == DroneHealthStatus.HEALTHY:
                st.last_seen = now
                st.consecutive_failures = 0

    def check_heartbeats(self, current_time: float | None = None) -> list[str]:
        """Scans all registered drones for heartbeat timeouts.

        Returns list of drone IDs that were isolated due to disconnection.
        """
        now = time.time() if current_time is None else current_time
        isolated_ids: list[str] = []

        for drone_id, state in list(self._drones.items()):
            # Ground Station is command infrastructure, not an aerial drone subject to flight heartbeats
            if drone_id == self.config.ground_station_id:
                continue

            if state.operational_status in (DroneHealthStatus.ISOLATED, DroneHealthStatus.FAILED):
                continue

            time_since_seen = now - state.last_seen
            if time_since_seen > self.config.heartbeat_timeout_s:
                state.consecutive_failures += 1
                if state.consecutive_failures >= self.config.consecutive_failures_to_isolate:
                    self.isolate_drone(
                        drone_id,
                        reason=FailureReason.DISCONNECTION,
                        details={
                            "time_since_seen_s": round(time_since_seen, 2),
                            "heartbeat_timeout_s": self.config.heartbeat_timeout_s,
                        },
                        current_time=now,
                    )
                    isolated_ids.append(drone_id)
            else:
                # Reset failure counter on healthy heartbeat interval
                state.consecutive_failures = 0

        return isolated_ids

    # --------------------------------------------------------------------------
    # ISOLATE: DRONE QUARANTINE & REROUTING TRIGGER
    # --------------------------------------------------------------------------

    def isolate_drone(
        self,
        drone_id: str,
        reason: FailureReason,
        details: dict[str, Any] | None = None,
        current_time: float | None = None,
    ) -> SelfHealingEvent:
        """Immediately isolates a drone, preventing its selection for new communication routes

        and automatically rerouting active communications traversing it.
        """
        now = time.time() if current_time is None else current_time
        if drone_id not in self._drones:
            self.register_drone(drone_id)

        state = self._drones[drone_id]
        prev_status = state.operational_status
        state.previous_status = prev_status
        state.operational_status = DroneHealthStatus.ISOLATED
        state.isolation_timestamp = now
        if reason not in state.failure_reasons:
            state.failure_reasons.append(reason)

        if reason == FailureReason.SECURITY_COMPROMISED:
            state.is_compromised = True
            state.security_cleared = False

        event = SelfHealingEvent(
            event_type=EventType.DRONE_ISOLATED,
            drone_id=drone_id,
            timestamp=now,
            previous_state=prev_status.value,
            new_state=DroneHealthStatus.ISOLATED.value,
            details={"reason": reason.value, **(details or {})},
        )
        self._record_event(event)

        # Reroute any active communication streams traversing this drone
        self.reroute_active_paths(drone_id, current_time=now)

        return event

    # --------------------------------------------------------------------------
    # REROUTE: ROUTING INTEGRATION & RECOVERY DETOURS
    # --------------------------------------------------------------------------

    def get_route(
        self,
        source: str,
        destination: str | None = None,
        raise_on_failure: bool = True,
    ) -> list[str] | None:
        """Finds an optimal communication route through healthy drones.

        Args:
            source: Source node ID (e.g. 'D7').
            destination: Target node ID (defaults to Ground Station 'GS').
            raise_on_failure: If True, raises NoRouteFoundError on route failure.

        Returns:
            List of node IDs (e.g. ["D7", "D4", "D3", "GS"]).
        """
        dest = destination or self.config.ground_station_id
        excluded = self.get_isolated_and_failed_drones()

        res = self.router.find_route(source, dest, self._drones, excluded_nodes=excluded)

        if not res.success:
            if raise_on_failure:
                raise NoRouteFoundError(
                    f"No route available from '{source}' to '{dest}'. Error: {res.error_message}"
                )
            return None

        return res.path

    def register_active_route(
        self,
        route_id: str,
        source: str,
        destination: str | None = None,
    ) -> list[str]:
        """Establishes and registers an active communication path for monitoring."""
        dest = destination or self.config.ground_station_id
        path = self.get_route(source, dest, raise_on_failure=True)
        if path is None:
            raise NoRouteFoundError(f"Cannot register active route '{route_id}': no path available.")

        self._active_routes[route_id] = {
            "source": source,
            "destination": dest,
            "path": list(path),
            "original_path": list(path),
        }
        return path

    def get_active_route(self, route_id: str) -> list[str] | None:
        """Returns the current path for an active communication route."""
        rec = self._active_routes.get(route_id)
        return list(rec["path"]) if rec else None

    def reroute_active_paths(
        self,
        failed_node_id: str,
        current_time: float | None = None,
    ) -> dict[str, list[str] | None]:
        """Inspects all active routes traversing failed_node_id and computes alternative paths.

        Returns dict mapping route_id to the new path (or None if rerouting failed).
        """
        now = time.time() if current_time is None else current_time
        rerouted_results: dict[str, list[str] | None] = {}

        for route_id, route_info in list(self._active_routes.items()):
            curr_path = route_info["path"]
            # Check if failed_node is an intermediate hop or endpoint
            if failed_node_id in curr_path:
                source = route_info["source"]
                destination = route_info["destination"]

                try:
                    new_path = self.get_route(source, destination, raise_on_failure=True)
                    route_info["path"] = list(new_path)
                    rerouted_results[route_id] = new_path

                    self._record_event(
                        SelfHealingEvent(
                            event_type=EventType.ROUTE_REROUTED,
                            drone_id=failed_node_id,
                            timestamp=now,
                            details={
                                "route_id": route_id,
                                "previous_path": curr_path,
                                "new_path": new_path,
                                "reason": f"Bypassing failed/isolated node '{failed_node_id}'",
                            },
                        )
                    )
                except NoRouteFoundError as err:
                    route_info["path"] = []
                    rerouted_results[route_id] = None

                    self._record_event(
                        SelfHealingEvent(
                            event_type=EventType.ROUTING_FAILED,
                            drone_id=failed_node_id,
                            timestamp=now,
                            details={
                                "route_id": route_id,
                                "previous_path": curr_path,
                                "error": str(err),
                            },
                        )
                    )

        return rerouted_results

    def check_route_restorations(self, current_time: float | None = None) -> list[str]:
        """Examines active detour routes to see if original routes can be safely restored."""
        now = time.time() if current_time is None else current_time
        restored_routes: list[str] = []
        excluded = self.get_isolated_and_failed_drones()

        for route_id, info in self._active_routes.items():
            curr_path = info["path"]
            orig_path = info["original_path"]

            if curr_path != orig_path:
                # Check if all nodes in original path are now healthy
                is_safe = all(node not in excluded for node in orig_path)
                if is_safe:
                    # Verify route is valid and computes cost
                    source = info["source"]
                    dest = info["destination"]
                    res = self.router.find_route(source, dest, self._drones, excluded_nodes=excluded)
                    if res.success:
                        # Restore original path
                        info["path"] = list(res.path)
                        restored_routes.append(route_id)
                        self._record_event(
                            SelfHealingEvent(
                                event_type=EventType.ROUTE_RESTORED,
                                drone_id=source,
                                timestamp=now,
                                details={
                                    "route_id": route_id,
                                    "detour_path": curr_path,
                                    "restored_path": res.path,
                                    "message": "Original path safely restored after node recovery.",
                                },
                            )
                        )

        return restored_routes

    # --------------------------------------------------------------------------
    # RECOVER & VERIFY
    # --------------------------------------------------------------------------

    def recover_drone(
        self,
        drone_id: str,
        force: bool = False,
        current_time: float | None = None,
    ) -> tuple[bool, str]:
        """Attempts or forces drone recovery."""
        now = time.time() if current_time is None else current_time
        if drone_id not in self._drones:
            return False, f"Drone '{drone_id}' not found."

        state = self._drones[drone_id]

        if force:
            state.operational_status = DroneHealthStatus.HEALTHY
            state.failure_reasons.clear()
            state.consecutive_failures = 0
            state.consecutive_healthy = 0
            state.is_compromised = False
            state.security_cleared = True
            state.isolation_timestamp = None

            evt = SelfHealingEvent(
                event_type=EventType.DRONE_RECOVERED,
                drone_id=drone_id,
                timestamp=now,
                new_state=DroneHealthStatus.HEALTHY.value,
                details={"message": "Drone recovery was forced by operator override."},
            )
            self._record_event(evt)
            if self.config.auto_restore_routes:
                self.check_route_restorations()
            return True, f"Drone '{drone_id}' forcibly recovered."

        # Progressive verification recovery
        if state.operational_status == DroneHealthStatus.ISOLATED:
            success, msg, evt = self.recovery.start_recovery(state, current_time=now)
            if evt:
                self._record_event(evt)
            return success, msg

        if state.operational_status == DroneHealthStatus.RECOVERING:
            success, msg, evt = self.recovery.process_recovery_check(state, current_time=now)
            if evt:
                self._record_event(evt)
            if success and self.config.auto_restore_routes:
                self.check_route_restorations()
            return success, msg

        return False, f"Drone '{drone_id}' is in state {state.operational_status.value}, not eligible for recovery."

    def grant_security_clearance(
        self,
        drone_id: str,
        clearance_token: str | None = None,
        current_time: float | None = None,
    ) -> None:
        """Grants security clearance allowing an isolated compromised drone to enter recovery."""
        if drone_id not in self._drones:
            raise KeyError(f"Drone '{drone_id}' not found.")
        state = self._drones[drone_id]
        evt = self.recovery.grant_security_clearance(state, clearance_token, current_time=current_time)
        self._record_event(evt)

    # --------------------------------------------------------------------------
    # AI & VISION MODULE INTEGRATION
    # --------------------------------------------------------------------------

    def report_threat(self, report: ThreatReport | dict[str, Any]) -> DetectionResult:
        """Processes AI threat report for anomaly or security breach."""
        parsed = AIThreatAdapter.parse_threat(report)
        drone_id = parsed.drone_id

        if drone_id not in self._drones:
            self.register_drone(drone_id)

        state = self._drones[drone_id]
        state.threat_level = parsed.threat_level
        if parsed.threat_level >= self.config.threat_critical:
            state.is_compromised = True

        self._record_event(
            SelfHealingEvent(
                event_type=EventType.SECURITY_ALERT,
                drone_id=drone_id,
                timestamp=parsed.timestamp,
                details={
                    "threat_type": parsed.threat_type,
                    "threat_level": parsed.threat_level,
                    "details": parsed.details,
                },
            )
        )

        # Trigger health update to evaluate threat immediately
        return self.update_drone_health(state, current_time=parsed.timestamp)

    def dispatch_emergency_alert(
        self,
        alert: EmergencyAlert,
        origin_drone_id: str | None = None,
    ) -> dict[str, Any]:
        """Dispatches an EmergencyAlert along a self-healed, encrypted route to Ground Station.

        If a relay on the route is isolated, re-routing occurs and the transmission completes.
        """
        source = origin_drone_id or alert.drone_id
        dest = self.config.ground_station_id

        # Compute or retrieve route
        route = self.get_route(source, dest, raise_on_failure=True)
        if route is None:
            raise NoRouteFoundError(f"Cannot dispatch alert {alert.alert_id}: no route to {dest}.")

        return self.vision_adapter.transmit_alert(alert, route)

    # --------------------------------------------------------------------------
    # EVENT AUDIT LOG
    # --------------------------------------------------------------------------

    def _record_event(self, event: SelfHealingEvent) -> None:
        """Records self-healing event, suppressing duplicate events within 100ms."""
        if self._events:
            last = self._events[-1]
            if (
                last.event_type == event.event_type
                and last.drone_id == event.drone_id
                and abs(last.timestamp - event.timestamp) < 0.1
                and last.details == event.details
            ):
                return
        self._events.append(event)
        logger.info(event.summary())

    def get_event_log(
        self,
        drone_id: str | None = None,
        event_type: EventType | None = None,
        limit: int | None = None,
    ) -> list[SelfHealingEvent]:
        """Retrieves filtered event audit log."""
        filtered = self._events
        if drone_id is not None:
            filtered = [e for e in filtered if e.drone_id == drone_id]
        if event_type is not None:
            filtered = [e for e in filtered if e.event_type == event_type]
        if limit is not None:
            filtered = filtered[-limit:]
        return list(filtered)

    def get_swarm_status(self) -> dict[str, Any]:
        """Provides consolidated dashboard status overview of the drone swarm."""
        total = len(self._drones)
        status_counts = {s.value: 0 for s in DroneHealthStatus}
        for st in self._drones.values():
            status_counts[st.operational_status.value] += 1

        return {
            "total_drones": total,
            "status_counts": status_counts,
            "active_routes_count": len(self._active_routes),
            "isolated_drones": [
                d_id
                for d_id, s in self._drones.items()
                if s.operational_status == DroneHealthStatus.ISOLATED
            ],
            "total_events": len(self._events),
        }

