#!/usr/bin/env python3
"""Drive the built-in TurtleBot3 Burger around obstacles to a marked goal."""
import argparse
import asyncio
import io
import json
import math
import os
import sys
import time
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
REFERENCE_PROJECT = Path(os.environ.get(
    "NEXT_FORGE_REFERENCE_PROJECT",
    "/data/eurobin/next-forge-sim2real-isaac-sim",
))
if not REFERENCE_PROJECT.is_dir():
    raise RuntimeError(
        "Reference project not found. Set NEXT_FORGE_REFERENCE_PROJECT to "
        "the next-forge-sim2real-isaac-sim directory."
    )
sys.path.insert(0, str(REFERENCE_PROJECT))

from taskboard_runtime import (  # noqa: E402
    parse_common_args,
    resolve_output,
    simulation_app_config,
)
from turtlebot3_web import PreviewServer  # noqa: E402


def add_demo_arguments(parser):
    parser.add_argument("--goal-hold-steps", type=int, default=90,
                        help="steps to hold the robot at the goal for the video")
    parser.add_argument("--inject-runtime-obstacle", action="store_true",
                        help="insert a test obstacle ahead of the moving robot")
    parser.add_argument("--inject-low-obstacle", action="store_true",
                        help="insert a 12 cm tall obstacle that the lidar cannot see; "
                             "only the depth camera detects it")
    parser.add_argument("--no-camera", action="store_true",
                        help="navigate with the lidar only (no depth camera)")
    parser.add_argument("--http-stream", action="store_true",
                        help="serve a live MJPEG camera preview for VS Code SSH port forwarding")
    parser.add_argument("--http-stream-port", type=int, default=8765,
                        help="localhost port for the live browser preview")
    parser.add_argument("--http-stream-fps", type=float, default=10.0,
                        help="live browser preview frame rate")


args = parse_common_args(
    "TurtleBot3 Burger obstacle-avoidance goal demo",
    add_extra=add_demo_arguments,
)
if args.goal_hold_steps < 0:
    raise SystemExit("--goal-hold-steps must be >= 0")
if args.http_stream_port <= 0 or args.http_stream_port > 65535:
    raise SystemExit("--http-stream-port must be between 1 and 65535")
if args.http_stream_fps <= 0.0:
    raise SystemExit("--http-stream-fps must be > 0")
output = resolve_output(args, "turtlebot3_goal")

# Isaac Sim normally keeps its shader and data caches inside its own install
# folder.  On this shared install that folder is read-only for students, so
# nothing is cached and every launch recompiles all shaders (about 80 s).
# Point Kit at a cache folder we own instead; only the first launch is slow.
KIT_PORTABLE_ROOT = Path(os.environ.get(
    "TURTLEBOT3_KIT_CACHE", Path.home() / ".cache" / "isaac-sim-turtlebot3"))
if "--portable-root" not in sys.argv:
    KIT_PORTABLE_ROOT.mkdir(parents=True, exist_ok=True)
    sys.argv += ["--portable-root", str(KIT_PORTABLE_ROOT)]

preview = PreviewServer(args.http_stream_port) if args.http_stream else None
if preview is not None:
    preview.start()
    print(
        f"[preview] listening at http://127.0.0.1:{args.http_stream_port}/ "
        "(forward this port in VS Code)",
        flush=True,
    )
    if not preview.state.wait_for_start():
        print("[preview] stopped before simulation startup", flush=True)
        preview.close()
        raise SystemExit(0)

try:
    from isaacsim import SimulationApp  # noqa: E402

    simulation_app = SimulationApp(simulation_app_config(args))
except Exception as error:
    if preview is not None:
        message = f"{type(error).__name__}: {error}"
        print(f"[preview] Isaac Sim failed to start: {message}", file=sys.stderr, flush=True)
        preview.state.set_error(message)
        try:
            while "stop" not in preview.state.drain_commands():
                time.sleep(0.1)
        finally:
            preview.close()
    raise

try:
    import numpy as np
    import omni.usd
    from pxr import Gf, UsdGeom, UsdPhysics
    from isaacsim.core.api import World
    from isaacsim.robot.experimental.wheeled_robots.controllers.differential_controller import (
        DifferentialController,
    )
    from isaacsim.robot.experimental.wheeled_robots.robots.wheeled_robot import (
        WheeledRobot,
    )
    from isaacsim.sensors.experimental.physics import Raycast, RaycastSensor
    from isaacsim.storage.native import get_assets_root_path_async
    from taskboard_capture import CaptureManager, render_now
    from reactive_navigation import TangentBugNavigator

    DT = 1.0 / 60.0
    START = (-2.35, -1.8)
    GOAL = (0.8, 1.3)
    GOAL_TOLERANCE = 0.03                     # stop when the body centre is this close (m)
    BODY_CENTER_OFFSET = (-0.033, 0.0)        # body centre relative to base_link (wheel axle)
    SCAN_FOV = math.radians(360.0)
    SCAN_RAY_COUNT = 181
    SCAN_MAX_RANGE = 3.5
    LIDAR_OFFSET = np.array([-0.0328, 0.0])   # base_scan relative to base_link (m)
    # Forward depth camera, mounted low on the chassis so it also sees obstacles
    # below the lidar plane (the lidar scans 0.22 m above the floor).
    CAMERA_PATH = "/World/TurtleBot3/base_link/FrontDepthCamera"
    CAMERA_WIDTH, CAMERA_HEIGHT = 160, 120
    CAMERA_FOCAL_LENGTH = 12.0                # with the aperture below: 82 x 66 degree view
    CAMERA_APERTURE = 20.955
    CAMERA_MOUNT = (0.045, 0.0, 0.09)         # relative to base_link (m)
    CAMERA_PITCH = math.radians(20.0)         # tilted down
    CAMERA_RANGE = 3.0
    CAMERA_INTERVAL = 6                       # physics steps per depth frame (10 Hz)
    OBSTACLE_MIN_HEIGHT = 0.04                # depth points above this are obstacles
    LINEAR_ACCELERATION = 0.45
    LINEAR_DECELERATION = 0.9
    ANGULAR_ACCELERATION = 3.0
    OBSTACLES = [
        {"name": "Long Wall 1", "center": (-1.105, -0.9), "size": (3.61, 0.12, 0.7),
         "color": (0.25, 0.58, 0.52)},
        {"name": "Long Wall 2", "center": (0.805, 0.45), "size": (4.21, 0.12, 0.7),
         "color": (0.25, 0.58, 0.52)},
    ]

    def make_box(stage, path, center, size, color, collision=False):
        cube = UsdGeom.Cube.Define(stage, path)
        cube.CreateSizeAttr(1.0)
        cube.CreateDisplayColorAttr([Gf.Vec3f(*color)])
        xform = UsdGeom.Xformable(cube.GetPrim())
        xform.AddTranslateOp().Set(Gf.Vec3d(*center))
        xform.AddScaleOp().Set(Gf.Vec3d(*size))
        if collision:
            UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
        return cube.GetPrim()

    def make_marker(stage, path, point, radius, color):
        marker = UsdGeom.Cylinder.Define(stage, path)
        marker.CreateRadiusAttr(radius)
        marker.CreateHeightAttr(0.025)
        marker.CreateDisplayColorAttr([Gf.Vec3f(*color)])
        UsdGeom.Xformable(marker.GetPrim()).AddTranslateOp().Set(
            Gf.Vec3d(point[0], point[1], 0.018)
        )

    def yaw_from_quaternion(quaternion):
        w, x, y, z = [float(value) for value in quaternion]
        return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

    def follow_robot_camera(capture, position, quaternion):
        yaw = yaw_from_quaternion(quaternion)
        forward_x, forward_y = math.cos(yaw), math.sin(yaw)
        eye = [position[0] - 1.5 * forward_x - 0.45 * forward_y, position[1] - 1.5 * forward_y + 0.45 * forward_x, 1.45]
        target = [position[0] + 0.1 * forward_x, position[1] + 0.1 * forward_y, 0.12]
        capture.set_view(eye, target)

    def wrap_angle(angle):
        return (angle + math.pi) % (2.0 * math.pi) - math.pi

    def clamp(value, lower, upper):
        return max(lower, min(upper, value))

    startup_clock = time.monotonic()

    def startup_stage(name):
        """Print how long each startup stage took (helps find slow starts)."""
        global startup_clock
        now = time.monotonic()
        print(f"[startup] {name}: {now - startup_clock:.1f}s", flush=True)
        startup_clock = now

    world = World(stage_units_in_meters=1.0, physics_dt=DT, rendering_dt=DT)
    world.scene.add_default_ground_plane()
    stage = omni.usd.get_context().get_stage()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)

    make_box(stage, "/World/Course/LeftBoundary", (0.0, -2.45, 0.18),
             (5.9, 0.08, 0.36), (0.17, 0.22, 0.25), True)
    make_box(stage, "/World/Course/RightBoundary", (0.0, 2.45, 0.18),
             (5.9, 0.08, 0.36), (0.17, 0.22, 0.25), True)
    make_box(stage, "/World/Course/LowerXBoundary", (-2.95, 0.0, 0.18),
             (0.08, 4.9, 0.36), (0.17, 0.22, 0.25), True)
    make_box(stage, "/World/Course/UpperXBoundary", (2.95, 0.0, 0.18),
             (0.08, 4.9, 0.36), (0.17, 0.22, 0.25), True)
    for index, obstacle in enumerate(OBSTACLES):
        x, y = obstacle["center"]
        width, depth, height = obstacle["size"]
        make_box(stage, f"/World/Obstacles/Obstacle_{index + 1}",
                 (x, y, height / 2.0), (width, depth, height),
                 obstacle["color"], True)
        make_box(stage, f"/World/Obstacles/Obstacle_{index + 1}_cap",
                 (x, y, height + 0.025), (width + 0.05, depth + 0.05, 0.05),
                 (0.12, 0.16, 0.19))

    make_marker(stage, "/World/Markers/Start", START, 0.20, (0.24, 0.67, 0.43))
    make_marker(stage, "/World/Markers/Goal", GOAL, 0.27, (0.15, 0.72, 0.73))

    startup_stage("world and course built")
    assets_root = asyncio.get_event_loop().run_until_complete(
        get_assets_root_path_async()
    )
    if not assets_root:
        raise RuntimeError("Isaac Sim could not resolve its built-in asset root.")
    robot_usd = assets_root + "/Isaac/Robots/Turtlebot/Turtlebot3/turtlebot3_burger.usd"
    startup_stage("asset server located")
    robot = WheeledRobot(
        "/World/TurtleBot3",
        wheel_dof_names=["wheel_left_joint", "wheel_right_joint"],
        usd_path=robot_usd,
        positions=[START[0], START[1], 0.12],
        orientations=[1.0, 0.0, 0.0, 0.0],
    )
    startup_stage("robot model loaded")
    world.reset()
    startup_stage("physics reset")

    scan_angles = np.linspace(-SCAN_FOV / 2.0, SCAN_FOV / 2.0, SCAN_RAY_COUNT)
    ray_directions = [
        [float(math.cos(angle)), float(math.sin(angle)), 0.0]
        for angle in scan_angles
    ]
    lidar_authoring = Raycast.create(
        "/World/TurtleBot3/base_scan/ReactiveLidar",
        min_range=0.05,
        max_range=SCAN_MAX_RANGE,
        ray_origins=[[0.0, 0.0, 0.0]] * SCAN_RAY_COUNT,
        ray_directions=ray_directions,
        output_frame="SENSOR",
        report_hit_prim_paths=True,
        translations=[[0.0, 0.0, 0.04]],
    )
    lidar = RaycastSensor(lidar_authoring)
    injected_obstacle = False
    if args.inject_runtime_obstacle:
        make_box(stage, "/World/RuntimeObstacle",
                 (START[0] + 0.9, START[1], 0.3), (0.42, 0.52, 0.6),
                 (0.92, 0.18, 0.46), True)
        injected_obstacle = True
    injected_low_obstacle = False
    if args.inject_low_obstacle:
        # 12 cm tall: entirely below the lidar plane, so only the camera sees it.
        make_box(stage, "/World/LowObstacle", (-0.45, -1.5, 0.06), (0.5, 0.4, 0.12),
                 (0.95, 0.55, 0.12), True)
        injected_low_obstacle = True

    camera_annotator = None
    onboard_annotator = None
    if not args.no_camera:
        import omni.replicator.core as rep

        camera = UsdGeom.Camera.Define(stage, CAMERA_PATH)
        camera.CreateFocalLengthAttr(CAMERA_FOCAL_LENGTH)
        camera.CreateHorizontalApertureAttr(CAMERA_APERTURE)
        camera.CreateVerticalApertureAttr(CAMERA_APERTURE * CAMERA_HEIGHT / CAMERA_WIDTH)
        camera.CreateClippingRangeAttr(Gf.Vec2f(0.02, 20.0))
        # USD cameras look along -Z with +Y up.  Rows: camera X (robot right),
        # camera Y (up, tilted), camera Z (backward, tilted), translation.
        cos_pitch, sin_pitch = math.cos(CAMERA_PITCH), math.sin(CAMERA_PITCH)
        UsdGeom.Xformable(camera.GetPrim()).AddTransformOp().Set(Gf.Matrix4d(
            0.0, -1.0, 0.0, 0.0,
            sin_pitch, 0.0, cos_pitch, 0.0,
            -cos_pitch, 0.0, sin_pitch, 0.0,
            CAMERA_MOUNT[0], CAMERA_MOUNT[1], CAMERA_MOUNT[2], 1.0,
        ))
        rep.orchestrator.set_capture_on_play(False)
        camera_product = rep.create.render_product(
            CAMERA_PATH, (CAMERA_WIDTH, CAMERA_HEIGHT), name="FrontDepthCamera")
        camera_annotator = rep.AnnotatorRegistry.get_annotator("distance_to_image_plane")
        camera_annotator.attach(camera_product)
        if args.http_stream:
            # Colour image from the same camera for the dashboard's inset view.
            onboard_annotator = rep.AnnotatorRegistry.get_annotator("rgb")
            onboard_annotator.attach(camera_product)
        focal_pixels = CAMERA_WIDTH * CAMERA_FOCAL_LENGTH / CAMERA_APERTURE
        pixel_u, pixel_v = np.meshgrid(np.arange(CAMERA_WIDTH) + 0.5,
                                       np.arange(CAMERA_HEIGHT) + 0.5)
        pixel_right = (pixel_u - CAMERA_WIDTH / 2.0) / focal_pixels
        pixel_up = -(pixel_v - CAMERA_HEIGHT / 2.0) / focal_pixels

    startup_stage("lidar and depth camera set up")
    controller = DifferentialController(
        wheel_radius=0.033,
        wheel_base=0.160,
        max_linear_speed=0.24,
        max_angular_speed=1.1,
        max_wheel_speed=10.0,
    )
    capture = None
    preview_only = args.http_stream and not (args.capture_images or args.record_video)
    if output is not None and (args.capture_images or args.record_video or args.http_stream):
        capture_args = args
        if preview_only:
            import copy
            capture_args = copy.copy(args)
            capture_args.capture_images = True
            capture_args.capture_interval = 0
            capture_args.image_width = 960
            capture_args.image_height = 540
        capture = CaptureManager(capture_args, output, stage, DT)
        position, orientation = robot.get_world_poses()
        follow_robot_camera(capture, position.numpy()[0], orientation.numpy()[0])
        capture.start()
        if preview_only:
            capture.images_enabled = False
        else:
            capture.milestone("initial")

    preview_size = (
        capture.video_size
        if capture is not None and capture.video_enabled
        else (960, 540)
    )
    preview_interval = max(1, int(round(1.0 / (args.http_stream_fps * DT))))
    if args.http_stream:
        if capture is None:
            raise RuntimeError("Live preview requires an output directory; use --headless or --output-dir.")
        capture._target(preview_size)

    startup_stage("video capture set up")
    for _ in range(30):
        world.step(render=not args.headless)
    startup_stage("first 30 simulation steps")
    if preview is not None:
        preview.state.set_status("running")

    def sensor_scan_to_distances(reading):
        """Return lidar distances, hit mask and robot-frame hit points (x, y)."""
        if reading is None or not reading.is_valid or reading.ray_count != SCAN_RAY_COUNT:
            return None
        try:
            hit_paths = np.asarray(reading.hit_prim_paths).reshape(-1)
            hit_positions = np.asarray(reading.hit_positions, dtype=np.float32)
            if hit_paths.size != SCAN_RAY_COUNT or hit_positions.shape[0] != SCAN_RAY_COUNT:
                return None
            hit_mask = np.asarray([
                bool(path)
                and str(path) not in ("", "None")
                and not str(path).startswith("/World/TurtleBot3/")
                for path in hit_paths
            ])
            distances = np.full(SCAN_RAY_COUNT, SCAN_MAX_RANGE, dtype=np.float32)
            measured = np.linalg.norm(hit_positions[:, :3], axis=1)
            usable_hits = hit_mask & np.isfinite(measured) & (measured >= 0.05)
            distances[usable_hits] = np.minimum(measured[usable_hits], SCAN_MAX_RANGE)
            points = hit_positions[usable_hits, :2].astype(float) + LIDAR_OFFSET
            return distances, usable_hits, points
        except (AttributeError, TypeError, ValueError, IndexError):
            return None

    latest_depth = None   # last depth image (m), kept for the dashboard

    def camera_obstacle_points(robot_height):
        """Robot-frame (x, y) of obstacles in the depth image: nearest per 2 degrees.

        Pixels are projected into the robot frame; anything higher than
        OBSTACLE_MIN_HEIGHT above the floor counts as an obstacle.
        """
        data = camera_annotator.get_data()
        if data is None:
            return None
        depth = np.asarray(data, dtype=np.float32)
        if depth.size != CAMERA_WIDTH * CAMERA_HEIGHT:
            return None
        depth = depth.reshape(CAMERA_HEIGHT, CAMERA_WIDTH)
        global latest_depth
        latest_depth = depth
        valid = np.isfinite(depth) & (depth > 0.02) & (depth < CAMERA_RANGE + 1.0)
        if not np.any(valid):
            return np.zeros((0, 2))
        forward = depth[valid].astype(float)
        right = pixel_right[valid] * forward
        up = pixel_up[valid] * forward
        cos_pitch, sin_pitch = math.cos(CAMERA_PITCH), math.sin(CAMERA_PITCH)
        x = CAMERA_MOUNT[0] + forward * cos_pitch + up * sin_pitch
        y = CAMERA_MOUNT[1] - right
        height = robot_height + CAMERA_MOUNT[2] - forward * sin_pitch + up * cos_pitch
        distance = np.hypot(x, y)
        obstacle = (height > OBSTACLE_MIN_HEIGHT) & (height < 1.0) & (distance < CAMERA_RANGE)
        if not np.any(obstacle):
            return np.zeros((0, 2))
        x, y, distance = x[obstacle], y[obstacle], distance[obstacle]
        bins = np.round(np.degrees(np.arctan2(y, x)) / 2.0).astype(int)
        order = np.lexsort((distance, bins))
        first = np.ones(len(order), dtype=bool)
        first[1:] = bins[order][1:] != bins[order][:-1]
        keep = order[first]
        return np.stack([x[keep], y[keep]], axis=1)

    def publish_onboard_frame():
        """Send the robot camera's latest colour frame to the dashboard inset."""
        try:
            frame = np.asarray(onboard_annotator.get_data())
        except Exception:  # annotator not ready yet
            return
        if frame.ndim != 3 or frame.shape[0] == 0 or frame.shape[2] < 3:
            return
        from PIL import Image
        image = Image.fromarray(np.ascontiguousarray(frame[..., :3]).astype(np.uint8), mode="RGB")
        image = image.resize((CAMERA_WIDTH * 2, CAMERA_HEIGHT * 2), Image.BILINEAR)
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=80)
        preview.state.publish_onboard(buffer.getvalue())

    def depth_to_jpeg(depth, robot_height):
        """Colour a depth image: red/yellow = near, blue = far, black = nothing.

        Pixels the navigator treats as obstacles are drawn at full brightness;
        floor pixels are dimmed, so the image shows what the camera detects.
        """
        from PIL import Image
        valid = np.isfinite(depth) & (depth > 0.02) & (depth < CAMERA_RANGE + 1.0)
        near = np.clip(1.0 - np.where(valid, depth, CAMERA_RANGE) / CAMERA_RANGE, 0.0, 1.0)
        stops = np.array([0.0, 0.35, 0.65, 0.85, 1.0])
        colours = np.array([[10, 20, 60], [30, 90, 200], [40, 200, 200], [250, 220, 60], [235, 60, 40]], float)
        rgb = np.stack([np.interp(near, stops, colours[:, c]) for c in range(3)], axis=-1)
        forward = np.where(valid, depth, 0.0)
        height = (robot_height + CAMERA_MOUNT[2] - forward * math.sin(CAMERA_PITCH)
                  + pixel_up * forward * math.cos(CAMERA_PITCH))
        floor = valid & (height <= OBSTACLE_MIN_HEIGHT)
        rgb[floor] *= 0.3
        rgb[~valid] = 0.0
        image = Image.fromarray(rgb.astype(np.uint8), mode="RGB")
        image = image.resize((CAMERA_WIDTH * 2, CAMERA_HEIGHT * 2), Image.NEAREST)
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=85)
        return buffer.getvalue()

    def publish_sensor_data(goal_distance, goal_bearing, command, robot_height):
        """Send live sensor readings to the dashboard's Sensors panel."""
        def nearest(points):
            return round(float(np.min(np.hypot(points[:, 0], points[:, 1]))), 3) if len(points) else None

        lidar = last_lidar_points if last_lidar_points is not None else np.zeros((0, 2))
        camera = camera_points if camera_points is not None else np.zeros((0, 2))
        preview.state.update_sensors({
            "range": SCAN_MAX_RANGE,
            "lidar": np.round(lidar, 3).tolist(),
            "camera": np.round(camera, 3).tolist(),
            "lidar_rays": SCAN_RAY_COUNT,
            "lidar_nearest": nearest(lidar),
            "camera_nearest": nearest(camera),
            "camera_enabled": camera_annotator is not None,
            "camera_frames": camera_frames,
            "goal_distance": round(goal_distance, 3),
            "goal_bearing": round(goal_bearing, 4),
            "heading": round(nav.heading, 4) if nav is not None else None,
            "clearance": round(nav.clearance, 3) if nav is not None else None,
            "linear": round(float(command[0]), 3),
            "angular": round(float(command[1]), 3),
            "mode": nav.mode if nav is not None else None,
        })
        if latest_depth is not None:
            preview.state.publish_depth(depth_to_jpeg(latest_depth, robot_height))

    def front_distance(points, half_angle=math.radians(10.0)):
        if points is None or not len(points):
            return SCAN_MAX_RANGE
        bearing = np.abs(np.arctan2(points[:, 1], points[:, 0]))
        ahead = (bearing <= half_angle) & (points[:, 0] > 0.0)
        return float(np.min(np.hypot(points[ahead, 0], points[ahead, 1]))) if np.any(ahead) else SCAN_MAX_RANGE

    def sector_distance(points, lower, upper):
        """Nearest obstacle distance among points with bearing in [lower, upper]."""
        if points is None or not len(points):
            return SCAN_MAX_RANGE
        bearing = np.arctan2(points[:, 1], points[:, 0])
        inside = (bearing >= lower) & (bearing <= upper)
        if not np.any(inside):
            return SCAN_MAX_RANGE
        return float(np.min(np.hypot(points[inside, 0], points[inside, 1])))

    navigator = TangentBugNavigator(
        GOAL,
        sensor_range=SCAN_MAX_RANGE,
        max_speed=0.22,
        max_turn=1.1,
    )
    camera_points = None
    camera_frames = 0
    camera_only_steps = 0
    nav = None
    last_lidar_points = np.zeros((0, 2))
    completed = False
    step_limit = args.max_steps or 7200
    final_position = [START[0], START[1], 0.0]
    selected_heading = 0.0
    minimum_obstacle_distance = SCAN_MAX_RANGE
    sensor_failure_steps = 0
    applied_linear_speed = 0.0
    applied_angular_speed = 0.0
    camera_failures = 0
    paused = False
    stop_requested = False
    for step in range(1, step_limit + 1):
        if preview is not None:
            while True:
                for action in preview.state.drain_commands():
                    if action == "pause":
                        paused = True
                        preview.state.set_status("paused")
                    elif action == "resume":
                        paused = False
                        preview.state.set_status("running")
                    elif action == "stop":
                        stop_requested = True
                        preview.state.set_status("stopping")
                if stop_requested or not paused:
                    break
                time.sleep(0.05)
        if stop_requested:
            break

        position, orientation = robot.get_world_poses()
        position = position.numpy()[0]
        quaternion = orientation.numpy()[0]
        robot_yaw = yaw_from_quaternion(quaternion)
        # Navigate and judge arrival by the middle of the robot body, not the axle.
        body_center = np.array([
            position[0] + BODY_CENTER_OFFSET[0] * math.cos(robot_yaw)
            - BODY_CENTER_OFFSET[1] * math.sin(robot_yaw),
            position[1] + BODY_CENTER_OFFSET[0] * math.sin(robot_yaw)
            + BODY_CENTER_OFFSET[1] * math.cos(robot_yaw),
        ])
        final_position = [float(body_center[0]), float(body_center[1]), float(position[2])]
        goal_dx = GOAL[0] - body_center[0]
        goal_dy = GOAL[1] - body_center[1]
        goal_distance = math.hypot(goal_dx, goal_dy)
        goal_bearing = wrap_angle(math.atan2(goal_dy, goal_dx) - robot_yaw)
        front_clearance = SCAN_MAX_RANGE
        left_clearance = SCAN_MAX_RANGE
        right_clearance = SCAN_MAX_RANGE
        selected_clearance = SCAN_MAX_RANGE
        selected_heading = 0.0
        command = [0.0, 0.0]

        if completed or goal_distance <= GOAL_TOLERANCE:
            completed = True
            command = [0.0, 0.0]
        else:
            if camera_annotator is not None and (camera_points is None or step % CAMERA_INTERVAL == 0):
                render_now()
                fresh = camera_obstacle_points(float(position[2]))
                if fresh is None:
                    camera_failures += 1
                else:
                    camera_points = fresh
                    camera_frames += 1
            sensor_data = sensor_scan_to_distances(lidar.get_sensor_reading())
            if sensor_data is None:
                # No lidar this step: stand still rather than drive blind.
                sensor_failure_steps += 1
                command = [0.0, 0.0]
            else:
                scan, _hit_mask, lidar_points = sensor_data
                last_lidar_points = lidar_points
                sensor_failure_steps = 0
                if camera_points is not None and len(camera_points):
                    points = np.vstack([lidar_points, camera_points])
                    lidar_front = sector_distance(lidar_points, -math.radians(25.0), math.radians(25.0))
                    camera_front = sector_distance(camera_points, -math.radians(25.0), math.radians(25.0))
                    if camera_front < lidar_front - 0.1:
                        camera_only_steps += 1
                else:
                    points = lidar_points
                if len(points):
                    minimum_obstacle_distance = min(
                        minimum_obstacle_distance,
                        float(np.min(np.hypot(points[:, 0], points[:, 1]))),
                    )
                front_clearance = sector_distance(points, -math.radians(8.0), math.radians(8.0))
                left_clearance = sector_distance(points, math.radians(18.0), math.radians(95.0))
                right_clearance = sector_distance(points, -math.radians(95.0), -math.radians(18.0))
                nav = navigator.update(body_center, robot_yaw, points, DT)
                selected_heading = nav.heading
                selected_clearance = nav.clearance
                command = [nav.linear, nav.angular]

        target_linear_speed, target_angular_speed = command
        linear_rate = (
            LINEAR_DECELERATION
            if abs(target_linear_speed) < abs(applied_linear_speed)
            or target_linear_speed * applied_linear_speed < 0.0
            else LINEAR_ACCELERATION
        )
        applied_linear_speed += clamp(
            target_linear_speed - applied_linear_speed,
            -linear_rate * DT,
            linear_rate * DT,
        )
        applied_angular_speed += clamp(
            target_angular_speed - applied_angular_speed,
            -ANGULAR_ACCELERATION * DT,
            ANGULAR_ACCELERATION * DT,
        )
        command = [applied_linear_speed, applied_angular_speed]
        robot.apply_wheel_actions(controller.forward(np.asarray(command)))
        world.step(render=not args.headless)
        sim_time = step * DT
        if preview is not None:
            preview.state.update_telemetry({
                "step": step,
                "simulated_seconds": sim_time,
                "position_x": float(position[0]),
                "position_y": float(position[1]),
                "goal_x": GOAL[0],
                "goal_y": GOAL[1],
                "goal_distance": goal_distance,
                "goal_bearing": goal_bearing,
                "front": front_clearance,
                "left": left_clearance,
                "right": right_clearance,
                "nearest": minimum_obstacle_distance,
                "navigation": (
                    "Goal reached" if completed
                    else f"{nav.mode}: {nav.detail}" if nav is not None
                    else "Starting"
                ),
                "goal_reached": completed,
            })
        if step % 120 == 0:
            print(
                f"[nav] step={step} pos=({position[0]:.2f},{position[1]:.2f}) "
                f"goal={goal_distance:.2f}m bearing={goal_bearing:.2f}rad "
                f"front={front_clearance:.2f}m left={left_clearance:.2f}m right={right_clearance:.2f}m "
                f"selected={selected_heading:.2f}rad clearance={selected_clearance:.2f}m "
                f"sensor_failures={sensor_failure_steps} camera_frames={camera_frames} "
                f"mode={nav.mode if nav else '-'} ({nav.detail if nav else ''}) "
                f"cmd=[{command[0]:.2f},{command[1]:.2f}]",
                flush=True,
            )
        if capture is not None:
            position, orientation = robot.get_world_poses()
            follow_robot_camera(capture, position.numpy()[0], orientation.numpy()[0])
            video_frames_before = capture.video_frames
            capture.on_step(step, sim_time)
            if preview is not None and step % preview_interval == 0:
                if capture.video_enabled and capture.video_frames > video_frames_before:
                    rgb = capture.targets[preview_size].read()
                else:
                    rgb = capture._render([preview_size])[preview_size]
                from PIL import Image
                jpeg_buffer = io.BytesIO()
                Image.fromarray(rgb, mode="RGB").save(jpeg_buffer, format="JPEG", quality=78)
                preview.state.publish(jpeg_buffer.getvalue())
                if onboard_annotator is not None:
                    publish_onboard_frame()
                publish_sensor_data(goal_distance, goal_bearing, command, final_position[2])
        if completed and step >= args.goal_hold_steps:
            break

    if capture is not None:
        capture.milestone("goal_reached" if completed else "goal_not_reached")
        capture.finish()
    if preview is not None:
        preview.state.set_status(
            "completed" if completed and not stop_requested else "stopped"
        )

    result = {
        "robot": "TurtleBot3 Burger",
        "asset": robot_usd,
        "goal": [GOAL[0], GOAL[1]],
        "final_position": final_position[:2],
        "final_goal_error_m": round(math.hypot(GOAL[0] - final_position[0], GOAL[1] - final_position[1]), 4),
        "goal_reached": completed,
        "steps": step,
        "simulated_seconds": round(step * DT, 3),
        "navigation": "reactive Tangent Bug (sensors + goal only; no map, path or waypoints)",
        "final_mode": nav.mode if nav is not None else None,
        "mode_switches": navigator.mode_switches,
        "depth_camera": camera_annotator is not None,
        "camera_frames": camera_frames,
        "camera_failures": camera_failures,
        "camera_only_detection_steps": camera_only_steps,
        "sensor_prim": "/World/TurtleBot3/base_scan/ReactiveLidar",
        "sensor_rays": SCAN_RAY_COUNT,
        "sensor_field_of_view_degrees": math.degrees(SCAN_FOV),
        "minimum_obstacle_distance": round(minimum_obstacle_distance, 3),
        "runtime_obstacle_injected": injected_obstacle,
        "low_obstacle_injected": injected_low_obstacle,
        "obstacles": OBSTACLES,
    }
    result_path = (output.root if output is not None else PROJECT_DIR / "output") / "results.json"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)
    if preview is not None and completed and not stop_requested:
        print("[preview] goal reached; dashboard remains available until Stop", flush=True)
        while True:
            if "stop" in preview.state.drain_commands():
                break
            time.sleep(0.1)
        preview.state.set_status("stopped")
    if not completed and not stop_requested:
        raise SystemExit(2)
finally:
    if preview is not None:
        preview.close()
    simulation_app.close()
