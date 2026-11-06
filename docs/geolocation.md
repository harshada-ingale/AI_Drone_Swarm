# Real-World Per-Person Geolocation System

## 1. Overview & Coordinate Conventions

The edge drone vision pipeline performs 3D ray-casting geolocation from pixel coordinates directly to real-world coordinates on the WGS-84 ellipsoid and local flat-ground tangent planes.

### Coordinate Systems & Frames

1. **Camera Frame ($C$)**:
   - Origin: Camera optical center.
   - $+X_c$: Right along the image sensor.
   - $+Y_c$: Down along the image sensor ($v$ pixel coordinate increases downward).
   - $+Z_c$: Forward along the lens optical axis.
   - Pixel coordinates $(u, v)$ unproject through the pinhole model:
     $$x_c = \frac{u - c_x}{f_x}, \quad y_c = \frac{v - c_y}{f_y}, \quad z_c = 1.0$$

2. **Gimbal & Body Frame ($B$)**:
   - $+X_b$: Forward along drone longitudinal axis.
   - $+Y_b$: Starboard (right) along drone lateral axis.
   - $+Z_b$: Downward along drone vertical axis.
   - **Gimbal Pitch ($\gamma$)**: Angle relative to horizontal:
     - $\gamma = -90^\circ$: Nadir (straight down). The optical axis points directly along $+Z_b$, top of image points forward along $+X_b$.
     - $\gamma = 0^\circ$: Forward-facing horizontal camera.
     - $\gamma > -75^\circ$: Oblique camera view; feet/ground contact points are picked at the **bottom-centre** of detection bounding boxes $(cx, y_2)$ rather than box centres.

3. **Earth Frame (NED & WGS-84)**:
   - True North-East-Down: Ray is rotated by full 3D attitude sequence:
     $$R = R_z(\text{yaw}) \cdot R_y(\text{pitch}) \cdot R_x(\text{roll})$$
   - Intersection with the ground surface at altitude $H$ ($H = \text{alt\_agl\_m}$) yields true offsets $(\text{north\_m}, \text{east\_m})$ from the sub-drone nadir point.
   - **Important**: Heading rotation is applied **strictly once** inside `geolocate_pixel`. Offset positions are directly composed with the drone's position without any secondary yaw rotation.

---

## 2. The Two Position Systems and Origin

Every raised emergency alert provides dual, synchronized coordinate sets for all visible confirmed detections:

1. **`location_gps`**:
   - Format: `[{"track_id": int, "lat": float, "lon": float, "uncertainty_m": float}, ...]`
   - Reference: WGS-84 geographic coordinates (decimal degrees) and 1-sigma error radius in metres.

2. **`location_local`**:
   - Format: `[{"track_id": int, "east_m": float, "north_m": float}, ...]`
   - Reference: Local ENU (East-North-Up) Cartesian grid in metres ($x = \text{east}$, $y = \text{north}$) measured from the configured `origin`.

3. **`reference_frame`**:
   - Identifies the origin and coordinate definitions:
     ```json
     {
       "gps": "WGS84 degrees",
       "local": "ENU metres from origin (x=east, y=north)",
       "origin": {
         "name": "ground_station" | "home" | "simulation" | "first_drone_fix",
         "lat": 16.70512,
         "lon": 74.24341
       }
     }
     ```
   - In GPS mode without explicit `--origin-lat` / `--origin-lon`, the origin auto-initializes to the first valid drone 3D GPS fix.

> **Deprecated Field Notice**: The legacy `"location": [x, y]` key is retained solely for backward compatibility with older dashboard receivers. Do not use for tactical localization; rely exclusively on `location_gps` and `location_local`.

---

## 3. Required Flight Telemetry

The telemetry receiver (`MavlinkTelemetry`) runs on a background thread and digests the following MAVLink messages:

| MAVLink Message | Fields Extracted | Units & Notes |
| :--- | :--- | :--- |
| `GLOBAL_POSITION_INT` | `lat`, `lon`, `alt`, `relative_alt`, `hdg` | $10^7$ deg $\to$ deg; $\text{mm} \to \text{m}$; $\text{cdeg} \to \text{deg}$. If `hdg == 65535`, fallback to attitude yaw. |
| `ATTITUDE` | `roll`, `pitch`, `yaw` | Radians $\to$ degrees. |
| `GPS_RAW_INT` | `fix_type` | Requires $\ge 3$ (3D fix). Values $< 3$ invalidate poses. |
| `MOUNT_ORIENTATION` | `pitch` | Gimbal pitch angle in degrees (if mounted). |

### Altitude Sources
- **`alt_source="relative"`**: Uses `relative_alt` directly from laser/barometer AGL estimation.
- **`alt_source="msl_minus_ground"`**: Computes $\text{alt\_agl} = \text{alt\_msl} - \text{ground\_elevation\_msl\_m}$ for terrain elevation references. Requires `--ground-elevation-msl`.

---

## 4. Measuring Camera HFOV

Accurate horizontal field of view (HFOV) is critical to prevent radial scaling errors:

1. Mount the camera horizontally facing a flat wall at distance $D$ (e.g., $3.0\text{ m}$).
2. Align two visual markers at the extreme left ($u = 0$) and right ($u = W$) edges of the video feed.
3. Measure the physical distance $W_{\text{scene}}$ between markers on the wall.
4. Calculate HFOV:
   $$\text{HFOV} = 2 \cdot \arctan\left(\frac{W_{\text{scene}}}{2 \cdot D}\right)$$
5. Pass the measured angle via `--camera-hfov <deg>`.

---

## 5. Factors Limiting Accuracy

- **Camera Latency**: Time gap between optical sensor exposure and MAVLink packet reception. Compensated via `PoseBuffer.pose_at(frame_time - latency)`.
- **GPS Fix Precision**: Horizontal GNSS dilution of precision (typically $\pm 1.5\text{ m}$ to $\pm 3.0\text{ m}$ without RTK).
- **Attitude Errors**: A $1^\circ$ error in drone pitch/roll at $50\text{ m}$ altitude introduces $\approx 0.87\text{ m}$ displacement at nadir, expanding rapidly at oblique angles.
- **Grazing Rays**: For depression angles $< 5^\circ$ (camera looking near or above the horizon), ray-plane intersections diverge towards infinity. The system suppresses coordinates and marks them as `"grazing_ray"`.

---

## 6. Field Validation & Calibration Procedure

1. **Ground Control Points (GCPs)**:
   - Place 3 high-contrast visual markers on an open field with surveyed GPS coordinates.
2. **Multi-Altitude Hover Tests**:
   - Hover the drone at $15\text{ m}$, $30\text{ m}$, and $50\text{ m}$ AGL directly above GCP 1.
   - Compare geolocated coordinates of GCP 2 and GCP 3 against ground truth.
3. **Oblique Gimbal Tilt Tests**:
   - Tilt the gimbal to $-60^\circ$ and $-45^\circ$.
   - Verify that bottom-centre pixel selection accurately identifies the marker base.
4. **Latency Tuning**:
   - Fly in a lateral figure-8 or yaw rotation over a stationary target.
   - Adjust `--camera-latency-ms` until detected target coordinates remain stationary regardless of rotation direction.
5. **Tune ErrorModel**:
   - Measure residual error circle radius over 50 frames; adjust `gps_sigma_m` and `attitude_sigma_deg` so reported 1-sigma uncertainty captures $68\%$ of experimental variance.
