#!/usr/bin/env python3
"""Tiny TCP relay: login-node 127.0.0.1:<lport> -> <node>:8080 (the viser server).

login1 can reach the compute node's 8080 directly, but VS Code only forwards ports that
LISTEN on the login node's localhost. This relay makes the viewer appear on login1:<lport>
so VS Code auto-forwards it to your laptop. Handles HTTP + websocket (raw TCP passthrough).
"""
import socket, sys, threading

LPORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8090
DEST = (sys.argv[2] if len(sys.argv) > 2 else "gpu048", int(sys.argv[3]) if len(sys.argv) > 3 else 8080)


def pipe(a, b):
    try:
        while True:
            d = a.recv(65536)
            if not d:
                break
            b.sendall(d)
    except OSError:
        pass
    finally:
        for s in (a, b):
            try:
                s.close()
            except OSError:
                pass


srv = socket.socket()
srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
srv.bind(("127.0.0.1", LPORT))
srv.listen(128)
print(f"viser proxy: 127.0.0.1:{LPORT} -> {DEST[0]}:{DEST[1]}", flush=True)
while True:
    c, _ = srv.accept()
    try:
        u = socket.create_connection(DEST, timeout=10)
    except OSError:
        c.close()
        continue
    threading.Thread(target=pipe, args=(c, u), daemon=True).start()
    threading.Thread(target=pipe, args=(u, c), daemon=True).start()
