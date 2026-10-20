"""Local orientation viewer driven by verified replies from an explicit FPGA USB device."""

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
from pathlib import Path
import signal
import threading
import time
from urllib.parse import urlsplit

from host.broad_protocol import build_request, parse_response
from host.ftdi import Ftdi
from host.orientation_sim import MODEL_DT_S, advance_pose, normalize_quaternion, sensors_for_step
from host.verify_probe import usb_serial_number
from host.verify_broad import positive_integer
from model.broad_integer import GoldenBroadPredictor, quantize_inputs


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_CYCLES = 124046


def exchange(connection, sequence, inputs, command=1):
    request = build_request(sequence, inputs, command)
    if connection.write(request) != len(request):
        raise TimeoutError('Incomplete FPGA request write')
    response = connection.read(210)
    if len(response) != 210:
        raise TimeoutError(f'FPGA response timeout: received {len(response)}/210 bytes')
    actual_sequence, hidden, raw, cycles = parse_response(response, hidden_size=96)
    if actual_sequence != sequence:
        raise ValueError('FPGA response sequence mismatch')
    return hidden, raw, cycles


class OrientationSession:
    def __init__(self, args):
        self.args = args
        self.lock = threading.Lock()
        self.connection = None
        self.predictor = None
        self.truth = [1., 0., 0., 0.]
        self.target = [1., 0., 0., 0.]
        self.state = self.snapshot('disconnected', 0, False)

    def snapshot(self, status, count, reset_verified, sample=None):
        return {'status': status, 'source': 'fpga_usb', 'count_verified': count,
                'usb_requested_baudrate': self.args.usb_requested_baudrate,
                'reset_verified': reset_verified, 'fixed_model_dt_s': MODEL_DT_S,
                'sensor_units': {'acc': 'm/s^2', 'gyr': 'rad/s', 'mag': 'uT'},
                'target_wxyz': self.target.copy(), 'sample': sample}

    def close(self):
        connection, self.connection = self.connection, None
        if connection is not None:
            connection.__exit__(None, None, None)

    def connect(self):
        if self.connection is not None:
            return self.state
        self.predictor = GoldenBroadPredictor.from_export(self.args.export_dir)
        if self.predictor.hidden_size != 96:
            raise ValueError('Orientation demo requires the frozen 96-state BROAD export')
        self.connection = Ftdi(self.args.usb_serial, self.args.libftdi,
                               requested_baudrate=self.args.usb_requested_baudrate).__enter__()
        try:
            return self.reset()
        except Exception:
            self.close()
            raise

    def reset(self):
        hidden, raw, cycles = exchange(self.connection, 65535, [0] * 9, command=2)
        if (hidden, raw, cycles) != ((0,) * 96, (0,) * 4, 0):
            raise ValueError('FPGA RESET did not clear state, output and cycles')
        self.predictor.reset()
        self.truth = [1., 0., 0., 0.]
        self.target = [1., 0., 0., 0.]
        self.state = self.snapshot('connected', 0, True)
        return self.state

    def step(self, target_wxyz=None):
        if self.connection is None:
            raise ValueError('Connect the FPGA before stepping')
        index = self.state['count_verified']
        target = normalize_quaternion(self.target if target_wxyz is None else target_wxyz)
        if sum(a * b for a, b in zip(target, self.target)) < 0:
            target = -target
        truth = advance_pose(self.truth, target)
        sensors = sensors_for_step(self.truth, truth)
        normalization = self.predictor.manifest['normalization']
        normalized = [(value - mean) / std for value, mean, std in
                      zip(sensors, normalization['mean'], normalization['std'])]
        quantized, clipping_count = quantize_inputs(normalized)
        inputs = quantized.tolist()
        expected_hidden, expected_raw = self.predictor.step(inputs)
        started = time.perf_counter()
        hidden, raw, cycles = exchange(self.connection, index & 65535, inputs)
        roundtrip_ms = (time.perf_counter() - started) * 1000
        if hidden != tuple(expected_hidden) or raw != tuple(expected_raw):
            raise ValueError('FPGA state/output differs from the golden integer model')
        if cycles != EXPECTED_CYCLES:
            raise ValueError(f'Unexpected FPGA cycle count: {cycles}')
        norm = math.sqrt(sum(value * value for value in raw))
        if norm == 0:
            raise ValueError('FPGA returned a zero quaternion')
        quaternion = [value / norm for value in raw]
        dot = abs(float(sum(normalize_quaternion(quaternion) * normalize_quaternion(truth))))
        self.truth = truth
        self.target = target.tolist()
        sample = {'index': index, 'sensors': sensors, 'quantized_inputs': inputs,
                  'hidden': list(hidden), 'raw_outputs': list(raw), 'cycles': cycles,
                  'quaternion_wxyz': quaternion,
                  'truth_wxyz': truth, 'target_wxyz': self.target.copy(),
                  'model_time_s': (index + 1) * MODEL_DT_S,
                  'error_deg': math.degrees(2 * math.acos(min(1., max(0., dot)))),
                  'physical_quaternion_wxyz': truth,
                  'verified': True, 'input_clipping_count': clipping_count,
                  'roundtrip_ms': roundtrip_ms, 'compute_us': cycles / 50}
        self.state = self.snapshot('running', index + 1, True, sample)
        return self.state

    def disconnect(self):
        self.close()
        self.state = self.snapshot('disconnected', self.state['count_verified'], False)
        return self.state

    def fail(self, error):
        try:
            self.close()
        except OSError as cleanup_error:
            error = f'{error}; cleanup failed: {cleanup_error}'
        self.state = {**self.state, 'status': 'error', 'reset_verified': False, 'error': str(error)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--usb-serial', required=True, type=usb_serial_number)
    parser.add_argument('--port', type=int, default=8766)
    parser.add_argument('--export-dir', type=Path, default=ROOT / 'artifacts/broad-integer-v1/export')
    parser.add_argument('--libftdi')
    parser.add_argument('--usb-requested-baudrate', type=positive_integer, default=460800)
    args = parser.parse_args()
    if not 0 <= args.port <= 65535:
        parser.error('port must be between 0 and 65535')
    session = OrientationSession(args)
    assets = {'/': ('index.html', 'text/html; charset=utf-8'),
              '/app.js': ('app.js', 'text/javascript; charset=utf-8'),
              '/scene.js': ('scene.js', 'text/javascript; charset=utf-8'),
              '/style.css': ('style.css', 'text/css; charset=utf-8'),
              '/vendor/three.module.js': ('vendor/three.module.js', 'text/javascript; charset=utf-8'),
              '/vendor/three.core.js': ('vendor/three.core.js', 'text/javascript; charset=utf-8')}

    class Handler(BaseHTTPRequestHandler):
        def respond(self, body, content_type, status=200):
            self.send_response(status)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.end_headers()
            self.wfile.write(body)

        def json_response(self, value, status=200):
            self.respond(json.dumps(value, allow_nan=False).encode(), 'application/json; charset=utf-8', status)

        def local_request(self):
            port = self.server.server_port
            host = self.headers.get('Host')
            origin = self.headers.get('Origin')
            if (host not in (f'127.0.0.1:{port}', f'localhost:{port}')
                    or (origin is not None and origin != f'http://{host}')):
                self.json_response({'error': 'Only the exact local application origin is allowed'}, 403)
                return False
            return True

        def do_GET(self):
            if not self.local_request():
                return
            path = urlsplit(self.path).path
            if path == '/api/state':
                self.json_response(session.state)
                return
            if path not in assets:
                self.send_error(404)
                return
            filename, content_type = assets[path]
            self.respond((ROOT / 'web/orientation' / filename).read_bytes(), content_type)

        def do_POST(self):
            if not self.local_request():
                return
            operations = {'/api/connect': session.connect, '/api/step': session.step,
                          '/api/disconnect': session.disconnect, '/api/reset': session.reset}
            path = urlsplit(self.path).path
            if path not in operations:
                self.send_error(404)
                return
            if self.headers.get_content_type() != 'application/json':
                self.json_response({'error': 'Content-Type must be application/json'}, 415)
                return
            try:
                length = int(self.headers.get('Content-Length', '0'))
                if length < 0 or self.headers.get('Transfer-Encoding') is not None:
                    raise ValueError('Invalid request length or transfer encoding')
                if length > 4096:
                    self.json_response({'error': 'Request exceeds 4096 bytes'}, 413)
                    return
                self.connection.settimeout(2)
                body = json.loads(self.rfile.read(length) or b'{}')
                allowed = {'target_wxyz'} if path == '/api/step' else set()
                if not isinstance(body, dict) or set(body) - allowed:
                    raise ValueError('Invalid request object or fields')
                if 'target_wxyz' in body:
                    target = body['target_wxyz']
                    if (not isinstance(target, list) or len(target) != 4
                            or any(type(value) not in (int, float) for value in target)):
                        raise ValueError('Expected four finite numeric quaternion components')
                    normalize_quaternion(target)
            except (OSError, ValueError, TypeError, OverflowError) as error:
                self.json_response({'error': str(error)}, 400)
                return
            if not session.lock.acquire(blocking=False):
                self.json_response({'error': 'An FPGA operation is already in flight'}, 409)
                return
            try:
                if path in ('/api/step', '/api/reset') and session.connection is None:
                    self.json_response({'error': 'Connect the FPGA first'}, 409)
                    return
                try:
                    state = (session.step(body.get('target_wxyz')) if path == '/api/step'
                             else operations[path]())
                    self.json_response(state)
                except TimeoutError as error:
                    session.fail(error)
                    self.json_response(session.state, 504)
                except (OSError, ValueError, KeyError, TypeError) as error:
                    session.fail(error)
                    self.json_response(session.state, 502)
            finally:
                session.lock.release()

        def log_message(self, format, *arguments):
            pass

    def stop(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, stop)
    server = ThreadingHTTPServer(('127.0.0.1', args.port), Handler)
    print(f'FPGA orientation: http://127.0.0.1:{server.server_port}', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        with session.lock:
            session.close()


if __name__ == '__main__':
    main()
