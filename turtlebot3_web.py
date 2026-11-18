"""Loopback web dashboard for the TurtleBot3 simulation."""
import json
import os
import queue
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit


INDEX_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>TurtleBot3 Control</title>
<style>
:root{color-scheme:dark;--bg:#101419;--panel:#1a2028;--panel2:#222b35;--line:#303b48;
--text:#e8edf3;--muted:#97a4b3;--accent:#56c99a;--blue:#5295ef;--red:#e46d72}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);
font:14px/1.5 system-ui,-apple-system,Segoe UI,sans-serif}
header{display:flex;align-items:center;gap:12px;padding:13px 20px;background:var(--panel);
border-bottom:1px solid var(--line)}h1{font-size:16px;margin:0;font-weight:650}
.status{margin-left:auto;color:var(--muted);display:flex;align-items:center;gap:8px}
.dot{width:9px;height:9px;border-radius:50%;background:#d39a45}.dot.live{background:var(--accent)}
.dot.stopped{background:var(--red)}main{display:grid;grid-template-columns:minmax(0,1fr) 330px;
gap:16px;padding:16px;max-width:1500px;margin:auto}
.screen{position:relative;background:#050709;border:1px solid var(--line);border-radius:9px;
overflow:hidden;min-height:240px;aspect-ratio:16/9;display:grid;place-items:center}
.screen img{width:100%;height:100%;object-fit:contain;position:absolute;inset:0}
.placeholder{color:var(--muted);padding:24px;text-align:center}
.caption{position:absolute;bottom:10px;left:10px;padding:4px 9px;border-radius:5px;
background:#080b0dcc;color:#e5e9ef;font:12px ui-monospace,monospace}
aside{display:flex;flex-direction:column;gap:12px}section{background:var(--panel);
border:1px solid var(--line);border-radius:9px;padding:13px 15px}
h2{font-size:11px;letter-spacing:.09em;text-transform:uppercase;color:var(--muted);
margin:0 0 11px;font-weight:650}.controls{display:flex;gap:8px}
button{flex:1;padding:9px 10px;border:1px solid var(--line);border-radius:6px;
background:var(--panel2);color:var(--text);font:inherit;font-weight:600;cursor:pointer}
button:hover:enabled{border-color:#657487}button:disabled{opacity:.42;cursor:default}
button.primary{background:#23694f;border-color:#328463}button.danger{color:#ffb4b4}
button.ghost{display:block;width:100%;margin-top:10px;background:transparent;color:var(--muted);font-weight:500}
button.ghost:hover:enabled{color:#ffb4b4;border-color:#8a4a4e}
dl{display:grid;grid-template-columns:1fr auto;gap:7px 12px;margin:0}
dt{color:var(--muted)}dd{margin:0;text-align:right;font-variant-numeric:tabular-nums}
.hint{color:var(--muted);font-size:12px;margin:10px 0 0}
@media(max-width:850px){main{grid-template-columns:1fr;padding:10px}header{padding:11px 13px}}
</style>
</head>
<body>
<header><h1>TurtleBot3 · Obstacle Course</h1>
<div class="status"><span class="dot" id="dot"></span><span id="status">Connecting…</span></div></header>
<main><div><div class="screen">
<img id="stream" alt="Live simulation camera">
<div class="placeholder" id="placeholder">Waiting for the first camera frame…</div>
<div class="caption" id="caption">Simulation starting</div>
</div><p class="hint">Live camera and navigation telemetry update as the simulation runs.</p></div>
<aside><section><h2>Simulation</h2><div class="controls">
<button id="start" class="primary">Start simulation</button>
<button id="pause" class="primary" hidden>Pause</button><button id="resume" hidden>Resume</button>
<button id="stop" class="danger" hidden>Stop</button></div><button id="shutdown" class="ghost">Shut down server</button><p class="hint" id="message">Dashboard ready. Start the simulation when you are ready.</p>
</section><section><h2>Navigation</h2><dl>
<dt>Mode</dt><dd id="mode">—</dd><dt>Step / time</dt><dd id="step">—</dd>
<dt>Position</dt><dd id="position">—</dd><dt>Goal</dt><dd id="goal">—</dd>
<dt>Distance to goal</dt><dd id="distance">—</dd><dt>Heading error</dt><dd id="heading">—</dd>
</dl></section><section><h2>Range sensor</h2><dl>
<dt>Front clearance</dt><dd id="front">—</dd><dt>Left clearance</dt><dd id="left">—</dd>
<dt>Right clearance</dt><dd id="right">—</dd><dt>Nearest detected</dt><dd id="nearest">—</dd>
</dl></section></aside></main>
<script>
const $=id=>document.getElementById(id);
let stopped=false,streamConnected=false,streamWanted=false,lastStatus="",runEnded=false,shutDown=false;
const FIELDS=["mode","step","position","goal","distance","heading","front","left","right","nearest"];
function resetRun(){
  streamConnected=false;$("stream").removeAttribute("src");
  $("placeholder").hidden=false;setText("placeholder","Click Start simulation to begin a run.");
  for(const id of FIELDS)setText(id,"\u2014");
}
function fmt(value,digits=2){return Number.isFinite(Number(value))?Number(value).toFixed(digits):"—"}
function setText(id,value){$(id).textContent=value}
function connectStream(){
  if(streamConnected||!streamWanted)return;
  streamConnected=true;$("stream").src="/stream.mjpg?ts="+Date.now();
}
async function send(action){
  try{
    const response=await fetch("/api/control",{method:"POST",headers:{"Content-Type":"application/json"},
      body:JSON.stringify({action})});
    const result=await response.json();
    if(!response.ok)throw new Error(result.error||"Request failed");
    setText("message",action==="shutdown"?"Shutting down the server…":action==="stop"?"Stopping simulation…":action==="pause"?"Pause requested":"Resume requested");
  }catch(error){setText("message",error.message)}
}
async function refresh(){
  try{
    const response=await fetch("/api/state",{cache:"no-store"});
    if(!response.ok)throw new Error("Simulation status unavailable");
    const data=await response.json(),t=data.telemetry||{},status=data.status||"starting";
    if(data.shutdown){shutDown=true;$("shutdown").disabled=true}
    if(status==="waiting"&&lastStatus!=="waiting"){resetRun();runEnded=false}
    if(["stopping","stopped"].includes(status))runEnded=true;
    lastStatus=status;
    setText("status",status==="waiting"?"Ready":status==="running"?"Live":status==="paused"?"Paused":
      status==="completed"?"Goal reached":
      status==="failed"?"Startup failed":
      status==="stopping"?"Stopping…":status==="stopped"?"Stopped":"Starting…");
    $("dot").className="dot"+(["running","completed"].includes(status)?" live":
      ["stopped","failed"].includes(status)?" stopped":"");
    $("start").hidden=status!=="waiting";
    for(const id of ["pause","resume"])$(id).hidden=!["running","paused"].includes(status);
    $("stop").hidden=["waiting","stopped"].includes(status);
    $("start").disabled=status!=="waiting";
    $("pause").disabled=status!=="running";$("resume").disabled=status!=="paused";
    $("stop").disabled=status==="stopping"||status==="stopped";stopped=status==="stopped";
    if(status==="waiting")setText("message","Dashboard ready. Click Start simulation to launch Isaac Sim.");
    else if(status==="starting")setText("message","Starting Isaac Sim… this can take a minute on first launch.");
    else if(status==="failed")setText("message",data.error||"Isaac Sim failed to start. Check the terminal output.");
    else if(status==="completed")setText("message","Goal reached. Press Stop to end this run; a new run can then be started.");
    else if(status==="stopped")setText("message",(t.goal_reached?"Goal reached. ":"Simulation stopped. ")+"Preparing a new run\u2026");
    else if(status==="paused")setText("message","Simulation paused");
    else if(status==="running")setText("message","Autonomous navigation active");
    setText("caption",status==="completed"?"Goal reached":status==="paused"?"Simulation paused":"Live camera");
    const x=Number(t.position_x),y=Number(t.position_y);
    if(Number.isFinite(x)&&Number.isFinite(y))setText("position",`${fmt(x)}, ${fmt(y)} m`);
    const gx=Number(t.goal_x),gy=Number(t.goal_y);
    if(Number.isFinite(gx)&&Number.isFinite(gy))setText("goal",`${fmt(gx)}, ${fmt(gy)} m`);
    if(t.step!==undefined)setText("step",`${t.step} / ${fmt(t.simulated_seconds,1)} s`);
    if(t.goal_distance!==undefined)setText("distance",`${fmt(t.goal_distance)} m`);
    if(t.goal_bearing!==undefined)setText("heading",`${fmt(t.goal_bearing)} rad`);
    for(const key of ["front","left","right","nearest"])
      if(t[key]!==undefined)setText(key,`${fmt(t[key])} m`);
    if(t.navigation)setText("mode",t.navigation);
    streamWanted=["running","completed"].includes(status);connectStream();
    if(stopped&&!streamConnected)setText("placeholder","Simulation stopped");
  }catch(error){
    if(shutDown){$("dot").className="dot stopped";setText("status","Server shut down");
      for(const id of ["start","pause","resume","stop","shutdown"])$(id).disabled=true;
      setText("message","The dashboard server has shut down and the port is free. Start it again from the terminal.")}
    else if(runEnded){$("dot").className="dot";setText("status","Restarting\u2026");
      setText("message","Preparing a new run. The Start button appears in a few seconds.")}
    else{setText("status","Connection lost");setText("message",error.message)}
  }
}
$("stream").addEventListener("load",()=>{$("placeholder").hidden=true});
$("stream").addEventListener("error",()=>{
  streamConnected=false;
  if(!stopped){setText("placeholder","Reconnecting to camera…");setTimeout(connectStream,1200)}
});
$("start").onclick=()=>send("start");
$("pause").onclick=()=>send("pause");$("resume").onclick=()=>send("resume");
$("stop").onclick=()=>send("stop");
$("shutdown").onclick=()=>{
  if(!confirm("Shut down the dashboard server? Any running simulation is stopped and the port is freed. You will need to start the server again from the terminal."))return;
  shutDown=true;runEnded=true;$("shutdown").disabled=true;send("shutdown");
};
window.addEventListener("load",()=>{refresh();setInterval(refresh,500)});
</script></body></html>"""


class PreviewState:
    def __init__(self):
        self._condition = threading.Condition()
        self._status = "waiting"
        self._telemetry = {}
        self._jpeg = None
        self._sequence = 0
        self._commands = queue.Queue()
        self._start_requested = False
        self._error = None
        self._shutdown = False

    def request_shutdown(self):
        """Stop any run and tell the launcher not to start another one."""
        flag = os.environ.get("TURTLEBOT3_SHUTDOWN_FLAG")
        if flag:
            try:
                with open(flag, "w") as handle:
                    handle.write("shutdown\n")
            except OSError:
                pass
        with self._condition:
            self._shutdown = True
            self._status = "stopped" if self._status == "waiting" else "stopping"
            self._condition.notify_all()
        self._commands.put("stop")

    def wait_for_start(self):
        with self._condition:
            self._condition.wait_for(
                lambda: self._start_requested or self._status in {"stopping", "stopped"}
            )
            return self._start_requested

    def request_start(self):
        with self._condition:
            if self._status != "waiting":
                return False
            self._start_requested = True
            self._status = "starting"
            self._condition.notify_all()
            return True

    def set_status(self, status):
        with self._condition:
            self._status = status
            self._condition.notify_all()

    def update_telemetry(self, telemetry):
        with self._condition:
            self._telemetry = dict(telemetry)
            self._condition.notify_all()

    def publish(self, jpeg):
        with self._condition:
            self._jpeg = jpeg
            self._sequence += 1
            self._condition.notify_all()

    def snapshot(self):
        with self._condition:
            return {
                "status": self._status,
                "telemetry": dict(self._telemetry),
                "error": self._error,
                "shutdown": self._shutdown,
            }

    def set_error(self, error):
        with self._condition:
            self._error = str(error)
            self._status = "failed"
            self._condition.notify_all()

    def wait_for_next_frame(self, sequence):
        with self._condition:
            self._condition.wait_for(
                lambda: self._sequence > sequence or self._status == "stopped",
                timeout=15.0,
            )
            return self._sequence, self._jpeg, self._status

    def submit(self, action):
        self._commands.put(action)

    def drain_commands(self):
        commands = []
        while True:
            try:
                commands.append(self._commands.get_nowait())
            except queue.Empty:
                return commands


class PreviewServer:
    def __init__(self, port):
        self.state = PreviewState()
        state = self.state

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                path = urlsplit(self.path).path
                if path in ("/", "/index.html"):
                    self._send(HTTPStatus.OK, INDEX_HTML.encode(), "text/html; charset=utf-8")
                elif path == "/api/state":
                    body = json.dumps(state.snapshot()).encode()
                    self._send(HTTPStatus.OK, body, "application/json")
                elif path == "/stream.mjpg":
                    self._stream()
                else:
                    self._send(HTTPStatus.NOT_FOUND, b"Not found", "text/plain; charset=utf-8")

            def do_POST(self):
                if self.path != "/api/control":
                    self._send(HTTPStatus.NOT_FOUND, b'{"error":"not found"}', "application/json")
                    return
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                    if length <= 0 or length > 4096:
                        raise ValueError("request body must be between 1 and 4096 bytes")
                    body = json.loads(self.rfile.read(length))
                    action = body.get("action") if isinstance(body, dict) else None
                    if not isinstance(action, str) or action not in {
                        "start", "pause", "resume", "stop", "shutdown"
                    }:
                        raise ValueError("action must be start, pause, resume, stop, or shutdown")
                except (ValueError, json.JSONDecodeError) as error:
                    response = json.dumps({"error": str(error)}).encode()
                    self._send(HTTPStatus.BAD_REQUEST, response, "application/json")
                    return
                current_status = state.snapshot()["status"]
                if action == "shutdown":
                    state.request_shutdown()
                    self._send(HTTPStatus.ACCEPTED, b'{"accepted":true}', "application/json")
                    return
                if action == "start":
                    if not state.request_start():
                        self._send(
                            HTTPStatus.CONFLICT,
                            b'{"error":"simulation has already been started or stopped"}',
                            "application/json",
                        )
                        return
                    state.submit(action)
                    self._send(HTTPStatus.ACCEPTED, b'{"accepted":true}', "application/json")
                    return
                if action == "stop" and current_status == "waiting":
                    state.set_status("stopped")
                    state.submit(action)
                    self._send(HTTPStatus.ACCEPTED, b'{"accepted":true}', "application/json")
                    return
                if action == "pause" and current_status == "running":
                    next_status = "paused"
                elif action == "resume" and current_status == "paused":
                    next_status = "running"
                elif action == "stop" and current_status in {
                    "starting", "running", "paused", "completed", "failed"
                }:
                    next_status = "stopping"
                else:
                    self._send(
                        HTTPStatus.CONFLICT,
                        b'{"error":"action is not available in the current simulation state"}',
                        "application/json",
                    )
                    return
                state.set_status(next_status)
                state.submit(action)
                self._send(HTTPStatus.ACCEPTED, b'{"accepted":true}', "application/json")

            def _stream(self):
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Connection", "close")
                self.end_headers()
                sequence = 0
                try:
                    while True:
                        sequence, jpeg, status = state.wait_for_next_frame(sequence)
                        if jpeg is not None:
                            self.wfile.write(
                                b"--frame\r\nContent-Type: image/jpeg\r\n"
                                + f"Content-Length: {len(jpeg)}\r\n\r\n".encode("ascii")
                                + jpeg + b"\r\n"
                            )
                            self.wfile.flush()
                            if status == "stopped":
                                return
                        else:
                            return
                except (BrokenPipeError, ConnectionResetError, OSError):
                    return

            def _send(self, status, body, content_type):
                self.send_response(status)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, _format, *_args):
                return

        self._server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
        self._server.daemon_threads = True
        self.port = self._server.server_address[1]
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    def start(self):
        self._thread.start()

    def close(self):
        self.state.set_status("stopped")
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=2.0)
