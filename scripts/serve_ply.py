"""Open splats from the cluster directly in the live SuperSplat editor, without downloading them to your laptop's disk.

Serves ONLY the given files (via a private temp dir of symlinks) on 127.0.0.1 of the node you are logged into, with the
CORS / Private-Network headers the https SuperSplat page needs to fetch from http://localhost. Your SSH tunnel carries
the bytes; the browser keeps the splat in memory only.

On your laptop (one terminal; the tunnel and the server must be on the SAME login node, so run both in this session):
    ssh -L 8765:127.0.0.1:8765 knaskar_umass_edu@unity.rc.umass.edu
    python3 /work/pi_donghyunkim_umass_edu/kushaan/scan2sim/scripts/serve_ply.py <a.ply> [<b.ply> ...] [--port 8765]
Then open the printed https://superspl.at/editor?load=... link(s) in Chrome/Edge (Ctrl-C stops the server).
Login-node safe: a static file server, no compute.
"""
from __future__ import annotations
import argparse, functools, http.server, os, sys, tempfile, urllib.parse

EDITOR = "https://superspl.at/editor"


class Handler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Private-Network", "true")   # Chrome: https page -> localhost
        self.send_header("Access-Control-Allow-Headers", "*")
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(204); self.end_headers()

    def log_message(self, fmt, *a):
        sys.stderr.write("[serve_ply] %s\n" % (fmt % a))


class Server(http.server.ThreadingHTTPServer):
    def handle_error(self, *a):          # browser-cancelled transfers
        pass


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("plys", nargs="+")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--editor", default=EDITOR)
    args = ap.parse_args()
    root = tempfile.mkdtemp(prefix="serve_ply_")
    names = []
    for p in args.plys:
        p = os.path.realpath(p)
        if not os.path.isfile(p):
            sys.exit(f"missing: {p}")
        n = os.path.basename(p)
        while n in names:
            n = "_" + n
        os.symlink(p, os.path.join(root, n)); names.append(n)
    srv = Server(("127.0.0.1", args.port), functools.partial(Handler, directory=root))
    print(f"[serve_ply] serving {len(names)} file(s) on 127.0.0.1:{args.port} (host {os.uname().nodename}); "
          f"needs: ssh -L {args.port}:127.0.0.1:{args.port} to THIS host", flush=True)
    for n in names:
        size = os.path.getsize(os.path.join(root, n)) / 1e6
        print(f"  {size:8.0f} MB  {args.editor}?load=" + urllib.parse.quote(f"http://localhost:{args.port}/{n}", safe=":/"), flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
