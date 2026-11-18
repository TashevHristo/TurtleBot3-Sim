#!/usr/bin/env python3
"""Start the TurtleBot3 browser dashboard and launch Isaac Sim from its Start button.

Each simulation run is a separate child process.  When a run ends (Stop in the
browser, goal reached and stopped, or step limit), a fresh child is started so
the dashboard comes back with its Start button.  Press Ctrl+C in the terminal to
quit for good, or press Shut down server in the dashboard.
"""
import argparse
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

QUICK_FAILURE_SECONDS = 5.0


def main():
    parser = argparse.ArgumentParser(
        description="TurtleBot3 browser control server; the page starts Isaac Sim."
    )
    parser.add_argument("--port", type=int, default=8765,
                        help="localhost port for the dashboard (default: 8765)")
    parser.add_argument("--stream-fps", type=float, default=10.0,
                        help="camera preview frame rate (default: 10)")
    parser.add_argument("--single-run", action="store_true",
                        help="exit after one simulation instead of offering a new run")
    options, simulation_args = parser.parse_known_args()
    if not 1 <= options.port <= 65535:
        parser.error("--port must be between 1 and 65535")
    if options.stream_fps <= 0:
        parser.error("--stream-fps must be > 0")

    demo = Path(__file__).with_name("turtlebot3_goal_demo.py")
    command = [
        sys.executable,
        str(demo),
        "--headless",
        "--http-stream",
        "--http-stream-port",
        str(options.port),
        "--http-stream-fps",
        str(options.stream_fps),
        *simulation_args,
    ]

    # The dashboard writes this file when Shut down server is pressed, so the
    # launcher knows not to start another run however the child process exits.
    flag_dir = tempfile.mkdtemp(prefix="turtlebot3_dashboard_")
    shutdown_flag = os.path.join(flag_dir, "shutdown")
    environment = dict(os.environ, TURTLEBOT3_SHUTDOWN_FLAG=shutdown_flag)
    try:
        return supervise(command, environment, shutdown_flag, options)
    finally:
        try:
            os.remove(shutdown_flag)
        except OSError:
            pass
        os.rmdir(flag_dir)


def supervise(command, environment, shutdown_flag, options):
    run = 0
    while True:
        run += 1
        if run > 1:
            print(f"[dashboard] ready for run {run} at http://127.0.0.1:{options.port}/",
                  flush=True)
        started = time.monotonic()
        child = subprocess.Popen(command, env=environment)
        try:
            code = child.wait()
        except KeyboardInterrupt:
            # Ctrl+C also reaches the child (same process group); let it shut down.
            try:
                child.wait(timeout=30)
            except (subprocess.TimeoutExpired, KeyboardInterrupt):
                child.kill()
            print("[dashboard] stopped", flush=True)
            return 0
        if os.path.exists(shutdown_flag):
            print(f"[dashboard] shut down from the browser; port {options.port} is free",
                  flush=True)
            return 0
        if options.single_run:
            return code
        if code != 0 and time.monotonic() - started < QUICK_FAILURE_SECONDS:
            # Failing straight away (for example the port is taken): don't loop.
            print(f"[dashboard] simulation exited with code {code} right after "
                  "starting; not restarting", file=sys.stderr, flush=True)
            return code


if __name__ == "__main__":
    sys.exit(main())
