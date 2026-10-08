#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = ["laya-mlx"]
# ///
"""laya-mlx plays the Tetris web app over a WebSocket. No Playwright, no extra packages.

  uv run laya_tetris_ws.py                    # serves the page at http://localhost:8765/?ws=1, opens it, laya plays
  uv run laya_tetris_ws.py --mode primitive
  uv run laya_tetris_ws.py --policy greedy    # no model, checks the wiring
  uv run laya_tetris_ws.py --no-open          # just print the URL

This script is the server (HTTP for the page + WebSocket for the agent, one port); the web page is the client.
The page owns the game. Each turn the server asks it for the state line, hands that to laya with a
question and candidate actions, then sends the chosen action back. Messages are JSON:
  server -> page: {"id": 3, "cmd": "step", "action": "left"}      (cmds: step, reset, state, load)
  page -> server: {"id": 3, "obs": "TETRIS1 ...", "reward": 0, "done": false, "valid": true, "cleared": 0}
"""
import argparse, base64, hashlib, json, pathlib, queue, random, socket, struct, sys, threading, time, urllib.parse, webbrowser
from laya_tetris import Tetris, play_turn, add_agent_args, turn_kwargs, load_agent

GUID = b"258EAFA5-E914-47DA-95CA-C5AB0DC85B11"


# --- minimal WebSocket server side (text frames only) ---
def _exact(c, n):
    b = b""
    while len(b) < n:
        x = c.recv(n - len(b))
        if not x: raise ConnectionError("page disconnected")
        b += x
    return b


def serve(srv, html_path, ws_q, taken, lock):
    """One port, two jobs: plain HTTP GET / returns the game page; a WebSocket upgrade becomes the agent connection.
    Each connection gets its own thread, so idle browser pre-connections can't block the real one."""
    def handle(c):
        try:
            c.settimeout(10)
            data = b""
            while b"\r\n\r\n" not in data:
                x = c.recv(4096)
                if not x: return c.close()
                data += x
            lines = data.split(b"\r\n\r\n")[0].decode("latin1").split("\r\n")
            req = lines[0].split()
            hdr = {l.split(":", 1)[0].strip().lower(): l.split(":", 1)[1].strip() for l in lines[1:] if ":" in l}
            if hdr.get("upgrade", "").lower() == "websocket" and "sec-websocket-key" in hdr:
                with lock:
                    if taken.is_set():   # a stale tab reconnecting after the game started: turn it away
                        c.sendall(b"HTTP/1.1 409 Conflict\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"); return c.close()
                    acc = base64.b64encode(hashlib.sha1(hdr["sec-websocket-key"].encode() + GUID).digest())
                    c.sendall(b"HTTP/1.1 101 Switching Protocols\r\nUpgrade: websocket\r\nConnection: Upgrade\r\n"
                              b"Sec-WebSocket-Accept: " + acc + b"\r\n\r\n")
                    c.settimeout(None); taken.set(); ws_q.put(c)
                return
            path = urllib.parse.urlparse(req[1]).path if len(req) > 1 else "/"
            if req[0] == "GET" and path in ("/", "/index.html", "/tetris-env.html"):
                body = open(html_path, "rb").read()
                c.sendall(b"HTTP/1.1 200 OK\r\nContent-Type: text/html; charset=utf-8\r\nContent-Length: "
                          + str(len(body)).encode() + b"\r\nConnection: close\r\n\r\n" + body)
            else:
                c.sendall(b"HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\nConnection: close\r\n\r\n")
            c.close()
        except Exception:
            try: c.close()
            except Exception: pass
    while True:
        try: c, _ = srv.accept()
        except OSError: return
        threading.Thread(target=handle, args=(c,), daemon=True).start()


def ws_send(c, text):
    p = text.encode(); n = len(p)
    h = b"\x81" + (bytes([n]) if n < 126 else b"\x7e" + struct.pack(">H", n) if n < 65536 else b"\x7f" + struct.pack(">Q", n))
    c.sendall(h + p)


def ws_recv(c):
    msg = b""
    while True:
        b1, b2 = _exact(c, 2); op, n = b1 & 15, b2 & 127
        if n == 126: n = struct.unpack(">H", _exact(c, 2))[0]
        elif n == 127: n = struct.unpack(">Q", _exact(c, 8))[0]
        mask = _exact(c, 4) if b2 & 128 else None
        d = _exact(c, n)
        if mask: d = bytes(x ^ mask[i % 4] for i, x in enumerate(d))
        if op == 8: raise ConnectionError("page closed the socket")
        if op == 9: c.sendall(b"\x8a" + bytes([len(d)]) + d); continue  # ping -> pong
        msg += d
        if b1 & 128: return msg.decode()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--html", default=str(pathlib.Path(__file__).with_name("tetris-env.html")))
    ap.add_argument("--delay", type=float, default=0.25, help="seconds between actions")
    ap.add_argument("--max-moves", type=int, default=500)
    ap.add_argument("--wait", type=int, default=120, help="seconds to wait for the page to connect")
    ap.add_argument("--no-open", action="store_true")
    add_agent_args(ap)
    a = ap.parse_args()

    agent = None
    if a.policy == "laya":
        agent = load_agent(a.model, a.backend)
    rng, env = random.Random(a.seed), Tetris(a.seed)

    srv = socket.socket(); srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try: srv.bind(("127.0.0.1", a.port))
    except OSError: sys.exit(f"Port {a.port} is busy (an earlier run still alive?). Stop it or pass --port.")
    srv.listen(16)
    ws_q, taken = queue.Queue(), threading.Event()
    threading.Thread(target=serve, args=(srv, a.html, ws_q, taken, threading.Lock()), daemon=True).start()
    url = f"http://localhost:{a.port}/?ws=1"
    print("Game page (opens automatically unless --no-open):", url, flush=True)
    if not a.no_open: webbrowser.open(url)
    try: conn = ws_q.get(timeout=a.wait)
    except queue.Empty: sys.exit(f"No page connected within {a.wait}s. Open {url} in a browser.")
    print("Page connected.")

    mid = 0

    def call(**msg):
        nonlocal mid
        mid += 1
        ws_send(conn, json.dumps({"id": mid, **msg}))
        r = json.loads(ws_recv(conn))
        assert r.get("id") == mid, f"out-of-order reply: {r}"
        return r

    def sync(r):  # the page's state line is the source of truth
        env.load(r["obs"]); return r

    def do(action):
        r = sync(call(cmd="step", action=action)); time.sleep(a.delay); return r

    try:
        sync(call(cmd="reset", seed=a.seed))
        n = 0
        while not env.over and n < a.max_moves:
            label = play_turn(env, do, agent, rng, **turn_kwargs(a))
            n += 1
            print(f"#{n:3d} {label:70s} score={env.score} lines={env.lines}")

    except (ConnectionError, OSError) as e:
        sys.exit(f"Stopped: {e}")

    print(f"\nfinished: moves={n} score={env.score} lines={env.lines} over={env.over}")
    conn.close(); srv.close()


if __name__ == "__main__":
    main()
