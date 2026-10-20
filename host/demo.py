"""Read-only localhost dashboard for explicitly selected FPGA telemetry."""

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
from urllib.parse import urlsplit


ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--live-dir', type=Path, required=True)
    parser.add_argument('--port', type=int, default=8765)
    args = parser.parse_args()
    if not 0 <= args.port <= 65535:
        parser.error('port must be between 0 and 65535')
    live_file = args.live_dir.resolve() / 'latest.json'

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            path = urlsplit(self.path).path
            if path == '/':
                selected, content_type = ROOT / 'web/index.html', 'text/html; charset=utf-8'
            elif path == '/latest.json':
                selected, content_type = live_file, 'application/json; charset=utf-8'
            elif path == '/recorded.json':
                selected = ROOT / 'demo/recorded/latest.json'
                content_type = 'application/json; charset=utf-8'
            else:
                self.send_error(404)
                return
            try:
                if path == '/latest.json' and not selected.exists():
                    body = json.dumps({'status': 'waiting', 'samples': [], 'count_verified': 0,
                                       'throughput_hz': 0}).encode()
                else:
                    body = selected.read_bytes()
            except FileNotFoundError:
                self.send_error(404)
                return
            except OSError:
                self.send_error(500, 'Cannot read selected telemetry')
                return
            self.send_response(200)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format, *arguments):
            pass

    server = ThreadingHTTPServer(('127.0.0.1', args.port), Handler)
    print(f'FPGA dashboard: http://127.0.0.1:{server.server_port}', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == '__main__':
    main()
