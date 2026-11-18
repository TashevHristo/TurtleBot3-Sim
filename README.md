# TurtleBot3 Obstacle Course

An Isaac Sim 6.0.1 demo in which the built-in TurtleBot3 Burger drives from a start marker to a goal marker through a course of two long walls.

The robot navigates **only from its sensors and the goal position**. There is no map, no planned path and no waypoints. Each control step it uses:

- its own pose (x, y, heading) and the goal position, and
- the obstacle points its sensors measure at that moment:
  - a 360° lidar (181 rays, 3.5 m range, 0.22 m above the floor);
  - a front depth camera (160×120, 82°×66° view, 3 m range), mounted low and tilted 20° down, so it also sees obstacles below the lidar plane.

## How the controller works

`reactive_navigation.py` contains `TangentBugNavigator`, a variant of the Tangent Bug algorithm with two modes:

- **Motion to goal.** If the straight line to the goal is free, the robot drives along it. If an obstacle blocks it, the robot finds the obstacle's two visible ends. It then heads past the end with the shorter estimated route: the distance to the end plus the straight-line distance from the end to the goal.
- **Boundary following.** If that estimate gets worse or stops improving, the robot is in a dead end. It follows the obstacle's edge until it sees open space that is closer to the goal than any obstacle point it saw while following. Then it goes back to motion to goal. If it detects that it has gone round in a loop, it switches to following the other side.

Speed is limited by the free distance in the chosen direction. Acceleration and turn-rate limits smooth the commands. The robot stops if the lidar reading fails.

## Run headless

```bash
cd /data/students/student3/project/turtlebot3
./run.sh
```

`run.sh` is the same as:

```bash
/data/isaac-sim/isaac-sim-6.0.1/python.sh turtlebot3_goal_demo.py \
  --headless --capture-images --capture-interval 120 --record-video \
  --output-dir ./output
```

A run writes these files:

- `output/images/`
- `output/videos/simulation.mp4`
- `output/logs/`
- `output/results.json`

`results.json` records whether the goal was reached, the time taken, the final mode and the number of mode switches. It also records depth-camera statistics:

- frames captured;
- failures;
- steps in which the camera saw an obstacle closer than the lidar did.

The first run downloads the TurtleBot3 asset from Isaac Sim's asset server.

**Startup time.** The shared Isaac Sim install folder is read-only for students, so Isaac Sim can't save its compiled shaders there and would recompile them on every launch, which takes about 80 s. The demo therefore keeps Isaac Sim's cache in `~/.cache/isaac-sim-turtlebot3`, about 0.9 GB; set `TURTLEBOT3_KIT_CACHE` to use another folder. The first launch takes about 95 s while the cache fills, and later launches start in about 8–10 s. If startup becomes slow again, for example after a GPU driver update, the cache is being rebuilt, and only that one launch is slow. Each launch prints `[startup]` lines showing how long each stage took. You need an RTX-capable GPU and FFmpeg for the MP4.

### Options

| Flag | Effect |
|---|---|
| `--inject-runtime-obstacle` | Adds a box 0.9 m ahead of the start that the controller has not been told about |
| `--inject-low-obstacle` | Adds a 12 cm tall box below the lidar plane, which only the depth camera can see |
| `--no-camera` | Navigates with the lidar only |
| `--max-steps N` | Step limit (default 7200, i.e. 120 s) |
| `--goal-hold-steps N` | Steps to hold at the goal for the video |
| `--output-dir DIR` | Where to write images, video and results |

## Browser dashboard over SSH

Start the dashboard server with Isaac Sim's Python launcher:

```bash
cd /data/students/student3/project/turtlebot3
/data/isaac-sim/isaac-sim-6.0.1/python.sh turtlebot3_web_server.py --port 8764
```

1. In VS Code Remote SSH, open the **Ports** view and forward port `8764`.
2. Open the forwarded address (normally `http://localhost:8764`) in your browser.
3. Click **Start simulation**. The dashboard shows a starting status while Isaac Sim loads, which takes about a minute.

While the simulation runs, the dashboard shows:

- the live camera;
- the robot's position;
- the goal distance and heading error;
- the lidar and camera clearances;
- the current navigation mode, e.g. `boundary following: wall on right`.
- a small **Robot camera** inset in the corner of the video, showing the robot's own front camera;
- a **Sensors** row under the video:
  - a top-down view of the lidar hits (blue) and depth-camera detections (orange), with the direction to the goal and the chosen heading;
  - the colour-coded depth image (red is near, blue is far; obstacles are bright and the floor is dimmed);
  - live readings: lidar hits, the closest obstacle seen by each sensor, the free distance ahead, and the speed and turn commands.

**Pause**, **Resume** and **Stop** control the run. After the robot reaches the goal, the last frame stays on screen until you press **Stop**.

After a run ends, the server starts a fresh simulation process and the **Start simulation** button comes back after a few seconds, so you can run again without going back to the terminal. Each new run reloads Isaac Sim, which takes about a minute. To shut the server down and free the port, press **Shut down server** in the dashboard and confirm, or press **Ctrl+C** in the terminal. **Shut down server** works at any time and stops a running simulation first. If you don't shut the server down, it keeps the port, and starting another server on the same port fails.

The server only listens on localhost. It streams over VS Code's SSH tunnel using HTTP/MJPEG.

Server options:

- `--port` picks another port, for example if 8764 is taken by another user on the machine.
- `--stream-fps` changes the preview frame rate (default 10).
- `--single-run` exits after one run instead of offering a new one.
- Other flags, such as `--inject-low-obstacle` or `--output-dir`, are passed through to the demo.

## Results

| Run | Result |
|---|---|
| Main course, 2026-11-18 (new controller) | Goal reached in 71.9 s, 4 mode switches, closest obstacle 0.15 m, 719 depth frames with 0 errors |
| Main course, 2026-11-05 (old map-and-waypoint version) | Stopped after 86 s, 0.43 m short of the goal |

The `--inject-runtime-obstacle` and `--inject-low-obstacle` runs have not yet been tested in Isaac Sim.

## Files

- `turtlebot3_goal_demo.py`: scene setup, sensors and the control loop
- `reactive_navigation.py`: the map-free Tangent Bug controller
- `turtlebot3_web.py`, `turtlebot3_web_server.py`: the browser dashboard
- `turtlebot3_goal_demo.py.bak`: the previous version, which used a map and waypoints

Set `ISAAC_SIM_ROOT` if Isaac Sim is installed somewhere else. Set `NEXT_FORGE_REFERENCE_PROJECT` if the reference repository is not at `/data/eurobin/next-forge-sim2real-isaac-sim`.
