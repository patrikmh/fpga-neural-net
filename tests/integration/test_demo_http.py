"""Real localhost HTTP tracer for the FPGA live visualization application."""

from html.parser import HTMLParser
import json
from pathlib import Path
import socket
import subprocess
import tempfile
import time
import unittest
import urllib.error
import urllib.request


ROOT = Path(__file__).resolve().parents[2]


class Controls(HTMLParser):
    def __init__(self):
        super().__init__()
        self.canvases = 0
        self.buttons = []
        self.downloads = []
        self.in_button = False

    def handle_starttag(self, tag, attributes):
        attributes = dict(attributes)
        if tag == "canvas":
            self.canvases += 1
        if tag == "button":
            self.in_button = True
            self.buttons.append("")
        if tag == "a" and "download" in attributes:
            self.downloads.append(attributes.get("href", ""))

    def handle_endtag(self, tag):
        if tag == "button":
            self.in_button = False

    def handle_data(self, text):
        if self.in_button:
            self.buttons[-1] += text


class DemoHttpTracerTests(unittest.TestCase):
    def test_local_app_serves_visualization_and_latest_telemetry_without_exposing_other_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            live = workspace / "live"
            live.mkdir()
            (workspace / "secret.json").write_text('{"private": true}')
            with socket.socket() as reserved:
                reserved.bind(("127.0.0.1", 0))
                port = reserved.getsockname()[1]
            process = subprocess.Popen(
                [str(ROOT / ".venv/bin/python"), "-m", "host.demo",
                 "--live-dir", str(live), "--port", str(port)],
                cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            )
            base = f"http://127.0.0.1:{port}"
            try:
                deadline = time.monotonic() + 5
                while True:
                    if process.poll() is not None:
                        stdout, stderr = process.communicate()
                        self.fail("Actual HTTP app must start:\n" + stdout + stderr)
                    try:
                        with urllib.request.urlopen(base + "/", timeout=0.2) as response:
                            html = response.read().decode("utf-8")
                            self.assertEqual(response.status, 200)
                        break
                    except (urllib.error.URLError, TimeoutError):
                        if time.monotonic() >= deadline:
                            self.fail("Actual localhost HTTP app did not start within five seconds")
                        time.sleep(0.05)
                controls = Controls()
                controls.feed(html)
                self.assertGreater(controls.canvases, 0)
                self.assertTrue(any("paus" in label.lower() for label in controls.buttons),
                                "Visualization must provide a pause control")
                self.assertTrue(any("latest.json" in target for target in controls.downloads)
                                or any("download" in label.lower() or "ladda" in label.lower()
                                       for label in controls.buttons),
                                "Visualization must provide a latest-data download control")
                with urllib.request.urlopen(base + "/latest.json", timeout=1) as response:
                    self.assertEqual(json.load(response)["status"], "waiting")
                fixture = {"status": "running", "count_verified": 1, "throughput_hz": 123.5,
                           "samples": [{"index": 0, "input": 4096, "y": -2, "golden_y": -2,
                                        "hidden": list(range(8)), "cycles": 1052, "roundtrip_ms": 3.2}]}
                (live / "latest.json").write_text(json.dumps(fixture))
                with urllib.request.urlopen(base + "/latest.json", timeout=1) as response:
                    self.assertEqual(json.load(response), fixture)
                for path in ("/missing", "/../secret.json", "/..%2fsecret.json"):
                    with self.subTest(path=path), self.assertRaises(urllib.error.HTTPError) as rejected:
                        urllib.request.urlopen(base + path, timeout=1)
                    self.assertEqual(rejected.exception.code, 404)
            finally:
                if process.poll() is None:
                    process.terminate()
                try:
                    process.communicate(timeout=2)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.communicate(timeout=2)


if __name__ == "__main__":
    unittest.main()
