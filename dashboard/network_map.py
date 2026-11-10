"""Network topology visualizer for AI Drone Swarm Command Center.

Provides both responsive, high-performance SVG generation and Matplotlib
figure rendering for mesh connectivity, active routes, and drone node states.
"""

import html
import math
from typing import Any


def _map_coords(
    x: float,
    y: float,
    width: int = 800,
    height: int = 420,
    min_x: float = -60.0,
    max_x: float = 660.0,
    min_y: float = -140.0,
    max_y: float = 160.0,
    pad_x: float = 60.0,
    pad_y: float = 50.0,
) -> tuple[float, float]:
    """Projects cartesian simulation (X, Y) into SVG pixel coordinates."""
    usable_w = width - 2 * pad_x
    usable_h = height - 2 * pad_y

    norm_x = (x - min_x) / (max_x - min_x)
    norm_y = (y - min_y) / (max_y - min_y)

    svg_x = pad_x + norm_x * usable_w
    svg_y = height - (pad_y + norm_y * usable_h)
    return round(svg_x, 1), round(svg_y, 1)


def render_network_map_svg(snapshot: dict[str, Any], width: int = 800, height: int = 420) -> str:
    """Generates standalone, styled SVG markup visualizing swarm mesh topology."""
    drones = snapshot.get("drones", {})
    links = snapshot.get("links", [])
    active_route = snapshot.get("active_route", [])

    # Pre-compute positions
    coords: dict[str, tuple[float, float]] = {}
    for d_id, d in drones.items():
        pos = d.get("position", [0.0, 0.0])
        coords[d_id] = _map_coords(pos[0], pos[1], width=width, height=height)

    # Active route segment pairs
    active_segments = set()
    for idx in range(len(active_route) - 1):
        n1 = active_route[idx]
        n2 = active_route[idx + 1]
        active_segments.add((n1, n2))
        active_segments.add((n2, n1))

    svg_parts = [
        f'<svg viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg" '
        f'style="width: 100%; height: auto; max-height: 480px; background: #0f172a; border-radius: 8px; border: 1px solid #1e293b;">',
        '<defs>',
        '  <filter id="glow" x="-20%" y="-20%" width="140%" height="140%">',
        '    <feGaussianBlur stdDeviation="3" result="blur" />',
        '    <feMerge>',
        '      <feMergeNode in="blur" />',
        '      <feMergeNode in="SourceGraphic" />',
        '    </feMerge>',
        '  </filter>',
        '  <marker id="route-arrow" viewBox="0 0 10 10" refX="22" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">',
        '    <path d="M 0 1 L 10 5 L 0 9 z" fill="#00e5ff" />',
        '  </marker>',
        '</defs>',
    ]

    # Draw grid lines for context
    svg_parts.append('<g stroke="#1e293b" stroke-width="1" stroke-dasharray="4,4">')
    for gx in range(100, width, 120):
        svg_parts.append(f'<line x1="{gx}" y1="0" x2="{gx}" y2="{height}" />')
    for gy in range(60, height, 80):
        svg_parts.append(f'<line x1="0" y1="{gy}" x2="{width}" y2="{gy}" />')
    svg_parts.append('</g>')

    # 1. Background links (in-range communication channels)
    svg_parts.append('<g class="links">')
    for link in links:
        src = link["source"]
        tgt = link["target"]
        if src in coords and tgt in coords:
            x1, y1 = coords[src]
            x2, y2 = coords[tgt]
            is_active_route = (src, tgt) in active_segments
            if not is_active_route and link.get("in_range", False):
                stroke_color = "#334155" if link.get("active", False) else "#1e293b"
                stroke_dash = "3,3" if not link.get("active", False) else "none"
                svg_parts.append(
                    f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{stroke_color}" '
                    f'stroke-width="1.5" stroke-dasharray="{stroke_dash}">'
                    f'<title>Link {src} - {tgt}: {link["distance"]}m</title>'
                    f'</line>'
                )
    svg_parts.append('</g>')

    # 2. Highlighted Active Route (prominent glowing cyan lines)
    if len(active_route) > 1:
        svg_parts.append('<g class="active-route" filter="url(#glow)">')
        for idx in range(len(active_route) - 1):
            n1 = active_route[idx]
            n2 = active_route[idx + 1]
            if n1 in coords and n2 in coords:
                x1, y1 = coords[n1]
                x2, y2 = coords[n2]
                svg_parts.append(
                    f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="#00e5ff" '
                    f'stroke-width="3.5" marker-end="url(#route-arrow)">'
                    f'<title>Active Route Hop: {n1} -> {n2}</title>'
                    f'</line>'
                )
        svg_parts.append('</g>')

    # 3. Nodes
    svg_parts.append('<g class="nodes">')
    for d_id, d in drones.items():
        if d_id not in coords:
            continue
        cx, cy = coords[d_id]
        status = d.get("operational_status", "UNKNOWN")
        is_compromised = d.get("is_compromised", False)
        is_gs = (d_id == "GS")
        battery = d.get("battery", 100.0)
        signal = d.get("signal_strength", -50.0)
        latency = d.get("latency", 20.0)
        loss = d.get("packet_loss", 0.0)

        # Determine node color and icon
        if is_gs:
            fill_color = "#3b82f6"
            stroke_color = "#60a5fa"
            status_text = "Ground Command Station"
        elif is_compromised:
            fill_color = "#ef4444"
            stroke_color = "#f87171"
            status_text = "SECURITY COMPROMISED"
        elif status == "HEALTHY":
            fill_color = "#10b981"
            stroke_color = "#34d399"
            status_text = "HEALTHY"
        elif status == "DEGRADED":
            fill_color = "#f59e0b"
            stroke_color = "#fbbf24"
            status_text = "DEGRADED"
        elif status == "ISOLATED":
            fill_color = "#475569"
            stroke_color = "#64748b"
            status_text = "ISOLATED / QUARANTINED"
        elif status == "RECOVERING":
            fill_color = "#06b6d4"
            stroke_color = "#22d3ee"
            status_text = "RECOVERING"
        else:
            fill_color = "#94a3b8"
            stroke_color = "#cbd5e1"
            status_text = status

        tooltip = (
            f"Drone: {d_id}\n"
            f"Role: {d.get('role', 'Node')}\n"
            f"Status: {status_text}\n"
            f"Battery: {battery}%\n"
            f"Signal: {signal} dBm\n"
            f"Latency: {latency} ms\n"
            f"Packet Loss: {round(loss * 100, 1)}%"
        )

        svg_parts.append(f'<g transform="translate({cx},{cy})" style="cursor: pointer;">')
        svg_parts.append(f'<title>{html.escape(tooltip)}</title>')

        # Node body shape
        if is_gs:
            # Hexagonal or diamond shape for ground station
            svg_parts.append(
                f'<polygon points="0,-22 19,-11 19,11 0,22 -19,11 -19,-11" '
                f'fill="{fill_color}" stroke="{stroke_color}" stroke-width="2.5" />'
            )
        else:
            # Circle
            svg_parts.append(
                f'<circle r="18" fill="{fill_color}" stroke="{stroke_color}" stroke-width="2.5" />'
            )

        # Overlay symbol for isolated / compromised
        if status == "ISOLATED" and not is_gs:
            svg_parts.append(
                '<line x1="-8" y1="-8" x2="8" y2="8" stroke="#ffffff" stroke-width="2.5" />'
                '<line x1="8" y1="-8" x2="-8" y2="8" stroke="#ffffff" stroke-width="2.5" />'
            )
        elif is_compromised and not is_gs:
            svg_parts.append(
                '<text x="0" y="5" font-family="sans-serif" font-weight="bold" font-size="14" '
                'fill="#ffffff" text-anchor="middle">!</text>'
            )
        else:
            # Node ID text
            svg_parts.append(
                f'<text x="0" y="4" font-family="sans-serif" font-weight="bold" font-size="11" '
                f'fill="#ffffff" text-anchor="middle">{d_id}</text>'
            )

        # Label below node
        label_y = 32 if not is_gs else 34
        svg_parts.append(
            f'<text x="0" y="{label_y}" font-family="sans-serif" font-size="10" '
            f'fill="#94a3b8" text-anchor="middle">{d_id} ({int(battery)}%)</text>'
        )

        svg_parts.append('</g>')

    svg_parts.append('</g>')

    # 4. Legend
    svg_parts.append(
        '<g transform="translate(15, 15)" font-family="sans-serif" font-size="10" fill="#94a3b8">'
        '<rect x="0" y="0" width="135" height="115" rx="6" fill="#0b1120" opacity="0.85" stroke="#1e293b" />'
        '<circle cx="15" cy="18" r="6" fill="#10b981" />'
        '<text x="28" y="21">Healthy</text>'
        '<circle cx="15" cy="36" r="6" fill="#f59e0b" />'
        '<text x="28" y="39">Degraded</text>'
        '<circle cx="15" cy="54" r="6" fill="#475569" />'
        '<text x="28" y="57">Isolated</text>'
        '<circle cx="15" cy="72" r="6" fill="#ef4444" />'
        '<text x="28" y="75">Compromised</text>'
        '<line x1="8" y1="92" x2="22" y2="92" stroke="#00e5ff" stroke-width="3" />'
        '<text x="28" y="95">Active Route</text>'
        '</g>'
    )

    svg_parts.append('</svg>')
    return "\n".join(svg_parts)


def render_network_map_matplotlib(snapshot: dict[str, Any]) -> Any:
    """Renders Network Map as a Matplotlib Figure fallback."""
    import os
    import tempfile
    if "MPLCONFIGDIR" not in os.environ:
        os.environ["MPLCONFIGDIR"] = tempfile.gettempdir()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    drones = snapshot.get("drones", {})
    links = snapshot.get("links", [])
    active_route = snapshot.get("active_route", [])

    fig, ax = plt.subplots(figsize=(9, 5), facecolor="#0f172a")
    ax.set_facecolor("#0f172a")

    # Inactive / normal links
    for link in links:
        s, t = link["source"], link["target"]
        if s in drones and t in drones and link.get("in_range", False):
            p1 = drones[s]["position"]
            p2 = drones[t]["position"]
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color="#334155", linestyle="--", linewidth=1.2, zorder=1)

    # Active route
    if len(active_route) > 1:
        rx = [drones[node]["position"][0] for node in active_route if node in drones]
        ry = [drones[node]["position"][1] for node in active_route if node in drones]
        ax.plot(rx, ry, color="#00e5ff", linewidth=2.8, zorder=2, label="Active Route")

    # Nodes
    for d_id, d in drones.items():
        pos = d["position"]
        st = d.get("operational_status", "UNKNOWN")
        comp = d.get("is_compromised", False)

        if d_id == "GS":
            c = "#3b82f6"
            marker = "s"
        elif comp:
            c = "#ef4444"
            marker = "o"
        elif st == "HEALTHY":
            c = "#10b981"
            marker = "o"
        elif st == "DEGRADED":
            c = "#f59e0b"
            marker = "o"
        elif st == "ISOLATED":
            c = "#475569"
            marker = "X"
        else:
            c = "#06b6d4"
            marker = "o"

        ax.scatter(pos[0], pos[1], s=350, color=c, edgecolors="#ffffff", linewidth=1.5, zorder=3, marker=marker)
        ax.text(pos[0], pos[1] - 22, f"{d_id} ({int(d['battery'])}%)", color="#cbd5e1", fontsize=9, ha="center", va="top")

    ax.set_title("Swarm Mesh Topology & Routing", color="#f8fafc", fontsize=13, pad=10)
    ax.set_xlabel("Local East (m)", color="#94a3b8")
    ax.set_ylabel("Local North (m)", color="#94a3b8")
    ax.tick_params(colors="#64748b")
    for spine in ax.spines.values():
        spine.set_color("#334155")
    ax.grid(True, linestyle=":", color="#1e293b", alpha=0.7)

    fig.tight_layout()
    return fig
