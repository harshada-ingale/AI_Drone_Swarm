"""Streamlit Dashboard: AI Drone Swarm Command Center.

Real-time situational awareness, network topology, dynamic self-healing rerouting,
threat quarantine, post-quantum security verification, and disaster detection.
"""

from datetime import datetime, timezone
import json
import time
from typing import Any

import pandas as pd
import streamlit as st

from dashboard.backend import SwarmService
from dashboard.network_map import render_network_map_matplotlib, render_network_map_svg

# ==============================================================================
# PAGE CONFIGURATION & STYLES
# ==============================================================================

st.set_page_config(
    page_title="AI Drone Swarm Command Center",
    page_icon="🛸",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for dark-themed operational aesthetic
st.markdown(
    """
    <style>
    .reportview-container {
        background: #0b1120;
    }
    .status-badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 4px;
        font-weight: 700;
        font-size: 11px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .badge-sim {
        background: #ef4444;
        color: #ffffff;
        box-shadow: 0 0 10px rgba(239, 68, 68, 0.4);
    }
    .badge-optimal {
        background: #10b981;
        color: #ffffff;
    }
    .badge-detour {
        background: #f59e0b;
        color: #ffffff;
    }
    .badge-noroute {
        background: #64748b;
        color: #ffffff;
    }
    .metric-card {
        background: #1e293b;
        border-radius: 8px;
        padding: 12px 16px;
        border: 1px solid #334155;
    }
    .alert-panel {
        background: #111827;
        border: 1px solid #374151;
        border-radius: 8px;
        padding: 16px;
        margin-top: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ==============================================================================
# SESSION STATE INITIALIZATION
# ==============================================================================

if "service" not in st.session_state:
    st.session_state.service = SwarmService(initial_sim_time=1000.0)

if "running" not in st.session_state:
    st.session_state.running = False

service: SwarmService = st.session_state.service

# ==============================================================================
# SIDEBAR CONTROLS
# ==============================================================================

with st.sidebar:
    st.title("🛸 Swarm Ops")
    st.caption("AI Drone Swarm Ground Command")

    # 1. Simulation Control
    st.subheader("⏱️ Simulation Clock")
    col_c1, col_c2 = st.columns(2)
    with col_c1:
        if st.session_state.running:
            if st.button("⏸️ Pause", width="stretch"):
                st.session_state.running = False
                service.pause()
                st.rerun()
        else:
            if st.button("▶️ Start", width="stretch"):
                st.session_state.running = True
                service.start()
                st.rerun()

    with col_c2:
        if st.button("⏭️ Step +1s", width="stretch"):
            service.step(1.0)
            st.rerun()

    col_c3, col_c4 = st.columns(2)
    with col_c3:
        if st.button("⏩ Step +5s", width="stretch"):
            service.step(5.0)
            st.rerun()
    with col_c4:
        if st.button("🔄 Reset", width="stretch"):
            st.session_state.service = SwarmService(initial_sim_time=1000.0)
            st.session_state.running = False
            st.rerun()

    st.markdown("---")

    # 2. Node Operations
    st.subheader("🛠️ Swarm Node Actions")
    drone_options = ["D1", "D2", "D3", "D4", "D5", "D6", "D7"]
    selected_drone = st.selectbox("Select Target Drone", drone_options, index=3)  # default D4

    col_a1, col_a2 = st.columns(2)
    with col_a1:
        if st.button("❌ Fail Node", width="stretch"):
            success, msg = service.fail_drone(selected_drone)
            if success:
                st.toast(msg, icon="⚠️")
            else:
                st.error(msg)
            st.rerun()

    with col_a2:
        if st.button("🩹 Recover Node", width="stretch"):
            success, msg = service.recover_drone(selected_drone)
            if success:
                st.toast(msg, icon="✅")
            else:
                st.warning(msg)
            st.rerun()

    # Attack Injection
    st.markdown("**Inject Adverse Condition / Attack:**")
    attack_kind = st.selectbox(
        "Attack Type",
        ["Jamming (RF Degradation)", "DoS (High Latency/Loss)", "Spoofing (GPS Anomaly)"],
    )
    if st.button("⚡ Inject Attack", width="stretch"):
        kind_map = {
            "Jamming (RF Degradation)": "jamming",
            "DoS (High Latency/Loss)": "dos",
            "Spoofing (GPS Anomaly)": "spoofing",
        }
        success, msg = service.inject_attack(selected_drone, kind_map[attack_kind])
        if success:
            st.toast(msg, icon="🔥")
        st.rerun()

    # Signed Clearance Action
    if st.button("🔏 Grant Signed Clearance", width="stretch"):
        try:
            res = service.grant_clearance(selected_drone)
            st.toast(f"Cryptographic clearance verified for {selected_drone}!", icon="🛡️")
        except Exception as exc:
            st.error(f"Clearance error: {exc}")
        st.rerun()

    st.markdown("---")

    # 3. Mission & Detection Commands
    st.subheader("🎯 Mission & Vision")

    # AI Threat Detection (Honest: not implemented yet)
    if st.button("🤖 Run AI Threat Detector", width="stretch"):
        try:
            service.run_detection()
        except NotImplementedError as exc:
            st.info(f"AI Threat Detector: {exc}")

    # Vision Alert Trigger
    vision_src = st.selectbox("Vision Source", ["Scripted Test Data", "test.mp4 (Clip)"])
    if st.button("👁️ Run Vision & Alert", width="stretch"):
        src_val = "scripted" if vision_src.startswith("Scripted") else "test.mp4"
        res = service.run_vision(source=src_val)
        if res.get("success"):
            st.toast(f"Emergency Alert dispatched from D7 to GS!", icon="🚨")
        else:
            st.error(f"Vision failure: {res.get('error')}")
        st.rerun()

    if st.button("🚨 Dispatch Alert Direct", width="stretch"):
        service.send_alert()
        st.toast("Direct emergency alert dispatched to GS!", icon="🚨")
        st.rerun()

    # Mobility Control (Disabled with explanation)
    st.button("🗺️ Move Drones (Simulation)", disabled=True, width="stretch")
    st.caption("ℹ️ *Mobility simulator not implemented yet*")

# ==============================================================================
# MAIN DISPLAY & SNAPSHOT
# ==============================================================================

snapshot = service.snapshot()
drones = snapshot["drones"]
stats = snapshot["stats"]["latest"]
active_route = snapshot["active_route"]
net_status = snapshot["network_status"]
sec_info = snapshot["security"]

# ------------------------------------------------------------------------------
# 1. TOP STATUS STRIP
# ------------------------------------------------------------------------------

sim_state_str = "RUNNING" if snapshot["running"] else "PAUSED"
sim_state_color = "🟢" if snapshot["running"] else "⏸️"

top_col1, top_col2, top_col3, top_col4 = st.columns([1.5, 1.2, 1.2, 3.1])

with top_col1:
    st.markdown(
        '<span class="status-badge badge-sim">⚠️ SIMULATED DATA</span> '
        f'<span style="font-size:12px; font-weight:600; color:#94a3b8; margin-left:8px;">{sim_state_color} {sim_state_str}</span>',
        unsafe_allow_html=True,
    )
    st.caption(f"Simulation Clock: **{snapshot['sim_time']:.1f} s**")

with top_col2:
    healthy_c = stats.get("healthy_count", 0)
    isolated_c = stats.get("isolated_count", 0)
    st.metric("Healthy / Isolated", f"{healthy_c} / {isolated_c}")

with top_col3:
    route_badge = "badge-optimal" if net_status == "OPTIMAL" else ("badge-detour" if "DETOUR" in net_status else "badge-noroute")
    st.markdown(f'<span class="status-badge {route_badge}">{net_status}</span>', unsafe_allow_html=True)
    st.caption(f"Active Hops: **{len(active_route)}** nodes")

with top_col4:
    # Truthful PQC status banner
    if sec_info.get("is_post_quantum", False):
        st.success(f"🔒 **{sec_info.get('verdict', 'ML-KEM ACTIVE')}** — {sec_info.get('kem_algorithm')} + {sec_info.get('sig_algorithm')}")
    else:
        st.warning(f"⚠️ **{sec_info.get('verdict', 'CLASSICAL FALLBACK (not post-quantum)')}** — {sec_info.get('kem_algorithm')} + {sec_info.get('sig_algorithm')}")

st.markdown("---")

# ------------------------------------------------------------------------------
# 2. MAIN PANELS (TABS)
# ------------------------------------------------------------------------------

tab_map, tab_drones, tab_threats, tab_healing, tab_vision, tab_security, tab_stats, tab_events = st.tabs([
    "🗺️ Network Map",
    "📋 Drone Status",
    "🛡️ Threat & Quarantine",
    "🩹 Self-Healing",
    "👁️ Disaster Detection",
    "🔐 Post-Quantum Security",
    "📈 Telemetry Charts",
    "📜 Audit Event Log",
])

# ------------------------------------------------------------------------------
# TAB 1: NETWORK MAP
# ------------------------------------------------------------------------------
with tab_map:
    st.subheader("Swarm Mesh Network Topology")
    col_map_top, col_map_opt = st.columns([4, 1])
    with col_map_top:
        route_str = " ──► ".join(active_route) if active_route else "NO ACTIVE ROUTE"
        st.markdown(f"**Current Route:** `{route_str}` &nbsp;|&nbsp; Status: **{net_status}**")
    with col_map_opt:
        map_mode = st.radio("Render Engine", ["Interactive SVG (Fast)", "Matplotlib"], horizontal=True)

    if map_mode.startswith("Interactive"):
        svg_content = render_network_map_svg(snapshot, width=860, height=430)
        st.markdown(svg_content, unsafe_allow_html=True)
    else:
        fig = render_network_map_matplotlib(snapshot)
        st.pyplot(fig)

    st.caption("Hover over any node in the SVG to view real-time battery, link signal, and latency telemetry.")

# ------------------------------------------------------------------------------
# TAB 2: DRONE STATUS TABLE
# ------------------------------------------------------------------------------
with tab_drones:
    st.subheader("Swarm Node Telemetry & Operational State")
    table_rows = []
    for d_id, d in drones.items():
        table_rows.append({
            "Drone ID": d_id,
            "Role": d.get("role", "Node"),
            "Status": d["operational_status"],
            "Position (X, Y)": f"({d['position'][0]:.0f}, {d['position'][1]:.0f})",
            "Battery (%)": d["battery"],
            "Signal (dBm)": d["signal_strength"],
            "Packet Loss (%)": round(d["packet_loss"] * 100, 1),
            "Latency (ms)": d["latency"],
            "Threat Level": d["threat_level"],
            "Compromised": "YES ⚠️" if d["is_compromised"] else "No",
            "Cleared": "Yes" if d["security_cleared"] else "No",
            "Age (s)": d["last_seen_age"],
        })
    df_drones = pd.DataFrame(table_rows)
    st.dataframe(df_drones, width="stretch", hide_index=True)

# ------------------------------------------------------------------------------
# TAB 3: THREAT & QUARANTINE PANEL
# ------------------------------------------------------------------------------
with tab_threats:
    st.subheader("Security Anomaly Quarantine & Adversarial Threats")
    compromised_drones = [d for d in drones.values() if d["is_compromised"] or d["operational_status"] == "ISOLATED"]

    if not compromised_drones:
        st.success("✅ No nodes currently quarantined. All active drones verified within normal operational boundaries.")
    else:
        for cd in compromised_drones:
            cid = cd["drone_id"]
            with st.container():
                st.markdown(f"#### ⚠️ Node **{cid}** — {cd['operational_status']}")
                reasons_str = ", ".join(cd["failure_reasons"]) if cd["failure_reasons"] else "None recorded"
                st.write(f"- **Failure Reasons**: `{reasons_str}`")
                st.write(f"- **Threat Level**: `{cd['threat_level']}` | **Compromised Flag**: `{cd['is_compromised']}`")
                st.write(f"- **Security Clearance**: `{'GRANTED' if cd['security_cleared'] else 'PENDING OPERATOR SIGNATURE'}`")

                col_q1, col_q2 = st.columns([1, 4])
                with col_q1:
                    if not cd["security_cleared"] and cd["is_compromised"]:
                        if st.button(f"🔏 Grant Clearance for {cid}", key=f"clr_{cid}"):
                            service.grant_clearance(cid)
                            st.toast(f"Clearance issued for {cid}", icon="🛡️")
                            st.rerun()
                    else:
                        if st.button(f"🩹 Start Recovery for {cid}", key=f"rec_{cid}"):
                            service.recover_drone(cid)
                            st.toast(f"Recovery initiated for {cid}", icon="✅")
                            st.rerun()
                st.markdown("---")

# ------------------------------------------------------------------------------
# TAB 4: SELF-HEALING & REROUTING PANEL
# ------------------------------------------------------------------------------
with tab_healing:
    st.subheader("Autonomous Self-Healing Routing Controller")
    initial_route = snapshot["initial_route"]

    c_heal1, c_heal2 = st.columns(2)
    with c_heal1:
        st.markdown("**Primary Optimal Route:**")
        st.code(" ──► ".join(initial_route))
        st.markdown("**Current Active Mesh Route:**")
        st.code(" ──► ".join(active_route) if active_route else "NO ROUTE AVAILABLE")

    with c_heal2:
        st.markdown("**Route State & Stabilization:**")
        if active_route == initial_route:
            st.success("✅ Mesh operating on optimal minimum-latency path.")
        elif not active_route:
            st.error("❌ Network partitioned! No viable path connects Scout to Ground Station.")
        else:
            st.warning("⚠️ Active route is bypassing isolated nodes via dynamic detour.")

    # Recovering nodes progress
    rec_progress = snapshot.get("recovering_progress", {})
    if rec_progress:
        st.markdown("#### Progressive Recovery Health Verification")
        for rid, prog in rec_progress.items():
            checks = prog["checks"]
            req = prog["required"]
            pct = min(1.0, checks / req if req > 0 else 1.0)
            st.write(f"Node **{rid}**: {checks} / {req} consecutive healthy telemetry passes")
            st.progress(pct)

# ------------------------------------------------------------------------------
# TAB 5: DISASTER DETECTION & GEOLOCATION PANEL
# ------------------------------------------------------------------------------
with tab_vision:
    st.subheader("Disaster Detection, Geolocation & Evidence")
    latest_alert = snapshot.get("latest_alert")

    if not latest_alert:
        st.info("ℹ️ No emergency alert received yet. Use the sidebar to dispatch an alert or run the vision pipeline.")
    else:
        st.markdown(f"### Alert: `{latest_alert.get('alert_id')}`")
        col_v1, col_v2, col_v3, col_v4 = st.columns(4)
        col_v1.metric("Priority", latest_alert.get("priority", "UNKNOWN"))
        col_v2.metric("Persons Detected", latest_alert.get("persons", 0))
        col_v3.metric("Origin Node", latest_alert.get("origin_drone_id", "D7"))
        col_v4.metric("Delivered to GS", "YES ✅" if latest_alert.get("delivered") else "NO ❌")

        # Honest hazard reporting
        hazards = latest_alert.get("hazards")
        if hazards is None:
            st.info("ℹ️ **Hazards Classification**: *Not supported by current detector model (null, not verified clear)*")
        else:
            st.write(f"**Hazards Identified**: {hazards}")

        # Geolocation Table
        loc_gps = latest_alert.get("location_gps", [])
        loc_local = latest_alert.get("location_local", [])
        if loc_gps:
            st.markdown("#### Per-Person Real-World Geolocation")
            geo_rows = []
            for i, p_gps in enumerate(loc_gps):
                p_loc = loc_local[i] if i < len(loc_local) else {}
                geo_rows.append({
                    "Track ID": p_gps.get("track_id", i + 1),
                    "Latitude (WGS84)": f"{p_gps.get('lat', 0.0):.6f}° N" if p_gps.get("lat") else "N/A",
                    "Longitude (WGS84)": f"{p_gps.get('lon', 0.0):.6f}° W" if p_gps.get("lon") else "N/A",
                    "Uncertainty (m)": f"±{p_gps.get('uncertainty_m', 0.0)} m",
                    "Local East (m)": f"{p_loc.get('east_m', 0.0):.1f} m",
                    "Local North (m)": f"{p_loc.get('north_m', 0.0):.1f} m",
                    "Status": p_gps.get("location_status", "VALID"),
                })
            st.dataframe(pd.DataFrame(geo_rows), width="stretch", hide_index=True)

        # Drone Pose & Reference Frame Metadata
        drone_pose = latest_alert.get("drone_pose")
        if drone_pose:
            with st.expander("Telemetry & Reference Frame Coordinates"):
                c_p1, c_p2 = st.columns(2)
                with c_p1:
                    st.write("**Drone Pose:**")
                    st.json(drone_pose)
                with c_p2:
                    st.write("**Reference Frame Definition:**")
                    st.json(latest_alert.get("reference_frame", {}))

        # Evidence Thumbnails
        evidence_list = latest_alert.get("evidence", [])
        if evidence_list:
            st.markdown("#### Cropped Evidence Thumbnails")
            cols = st.columns(min(len(evidence_list), 4))
            for idx, item in enumerate(evidence_list[:4]):
                thumb_b64 = item.get("thumbnail_b64")
                if thumb_b64:
                    with cols[idx]:
                        st.image(
                            f"data:image/jpeg;base64,{thumb_b64}",
                            caption=f"Track #{item.get('track_id')} (Conf: {item.get('confidence', 0.0):.2f})",
                            width="stretch",
                        )

# ------------------------------------------------------------------------------
# TAB 6: POST-QUANTUM SECURITY PANEL
# ------------------------------------------------------------------------------
with tab_security:
    st.subheader("Cryptographic Security & Identity Subsystem")
    col_s1, col_s2 = st.columns(2)

    with col_s1:
        st.markdown("#### Cryptographic Primitives")
        st.write(f"- **Operation Mode**: `{sec_info.get('mode', 'classical').upper()}`")
        st.write(f"- **Key Encapsulation (KEM)**: `{sec_info.get('kem_algorithm')}`")
        st.write(f"- **Signature Mechanism**: `{sec_info.get('sig_algorithm')}`")
        st.write(f"- **Post-Quantum Status**: `{'ACTIVE (NIST FIPS 203/204)' if sec_info.get('is_post_quantum') else 'CLASSICAL FALLBACK'}`")
        st.write(f"- **Wire Cipher**: `AES-256-GCM (Authenticated Encryption)`")
        st.write(f"- **Replay Protection**: `64-Packet Sliding Anti-Replay Window`")

    with col_s2:
        st.markdown("#### Ground Station Decrypt Verification")
        if latest_alert and latest_alert.get("delivered"):
            st.success(
                f"✅ Last alert payload safely unsealed and authenticated at Ground Station.\n\n"
                f"Intermediate relays (`{' -> '.join(latest_alert.get('hops_traversed', []))}`) "
                f"forwarded ciphertext verbatim without relay decryption."
            )
        else:
            st.info("Awaiting emergency transmission to verify end-to-end unsealing.")

    st.markdown("#### Security Clearance Authority")
    st.write(
        f"Registered Operator Identity: `{service.operator_identity.peer_id}` "
        f"({service.operator_identity.algorithm})"
    )

# ------------------------------------------------------------------------------
# TAB 7: STATISTICS LINE CHARTS
# ------------------------------------------------------------------------------
with tab_stats:
    st.subheader("Historical Telemetry Trends (Last 100 Sim Steps)")
    hist = snapshot["stats"]["history"]
    if hist:
        df_hist = pd.DataFrame(hist)
        if "sim_time" in df_hist.columns:
            df_hist = df_hist.set_index("sim_time")

        c_st1, c_st2 = st.columns(2)
        with c_st1:
            st.markdown("**Packet Loss Trend (%)**")
            st.line_chart(df_hist[["avg_packet_loss"]])
        with c_st2:
            st.markdown("**Average Latency (ms)**")
            st.line_chart(df_hist[["avg_latency"]])

        c_st3, c_st4 = st.columns(2)
        with c_st3:
            st.markdown("**Battery Degradation (%)**")
            st.line_chart(df_hist[["avg_battery"]])
        with c_st4:
            st.markdown("**Swarm Node Counts**")
            st.line_chart(df_hist[["healthy_count", "isolated_count", "recovering_count"]])
    else:
        st.info("Stepping simulation will populate telemetry history charts.")

# ------------------------------------------------------------------------------
# TAB 8: AUDIT EVENT LOG
# ------------------------------------------------------------------------------
with tab_events:
    st.subheader("Swarm Controller Audit Trail")
    raw_events = snapshot.get("events", [])
    if raw_events:
        evt_rows = []
        for e in raw_events:
            evt_rows.append({
                "Sim Time (s)": e["timestamp"],
                "Event Type": e["event_type"],
                "Drone ID": e["drone_id"],
                "Prev State": e.get("previous_state") or "-",
                "New State": e.get("new_state") or "-",
                "Details": json.dumps(e.get("details", {})),
            })
        st.dataframe(pd.DataFrame(evt_rows), width="stretch", hide_index=True)
    else:
        st.info("No audit events recorded yet.")

# ==============================================================================
# AUTO-STEPPING LOOP WHEN RUNNING
# ==============================================================================

if st.session_state.running:
    time.sleep(0.5)
    service.step(dt=1.0)
    st.rerun()
