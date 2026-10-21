"""Actual HTTP orientation application through an external libftdi device."""

from html.parser import HTMLParser
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor,TimeoutError as FutureTimeout
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import socket
import struct
import subprocess
import tempfile
import time
import unittest
import urllib.error
import urllib.request

from model.broad_integer import GoldenBroadPredictor
from test_orientation_sim import reference_pose_step,reference_sensors,unit


ROOT = Path(__file__).resolve().parents[2]
EXPORT = ROOT / "artifacts/broad-integer-v1/export"


def crc_frame(payload):
    crc = 0xffff
    for byte in payload:
        crc ^= byte << 8
        for _ in range(8):
            crc = ((crc << 1) ^ 0x1021) & 0xffff if crc & 0x8000 else (crc << 1) & 0xffff
    return payload + struct.pack("<H", crc)


def identities():
    return {str(path.relative_to(EXPORT)):hashlib.sha256(path.read_bytes()).hexdigest()
            for path in EXPORT.rglob("*") if path.is_file()}


class OrientationPage(HTMLParser):
    def __init__(self):
        super().__init__()
        self.canvas_count = 0
        self.buttons = []
        self.modules = []
        self.in_button = False

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "canvas":
            self.canvas_count += 1
        if tag == "button":
            self.in_button = True
            self.buttons.append("")
        if tag == "script" and attrs.get("type") == "module" and attrs.get("src"):
            self.modules.append(attrs["src"])

    def handle_endtag(self, tag):
        if tag == "button":
            self.in_button = False

    def handle_data(self, text):
        if self.in_button:
            self.buttons[-1] += text


class OrientationDemoTracerTests(unittest.TestCase):
    def test_http_connect_step_disconnect_uses_physical_sensor_units_and_verified_fpga_quaternion(self):
        before = identities()
        artifact = json.loads((EXPORT / "manifest.json").read_text())
        self.assertEqual(artifact["config"]["hidden_size"], 96)
        sensors = [0., 0., 9.81, 0., 0., 0.,
                   *[45 * value for value in (-.02069254372432757, .35690462341735035, -.9339116170267592)]]
        normalized = [(value-mean)/std for value,mean,std in
                      zip(sensors,artifact["normalization"]["mean"],artifact["normalization"]["std"])]
        inputs = [min(32767,max(-32768,math.floor(value*2048+.5) if value>=0 else -math.floor(-value*2048+.5)))
                  for value in normalized]
        hidden, raw = GoldenBroadPredictor.from_export(EXPORT).step(inputs)
        norm = math.sqrt(sum(value*value for value in raw))
        self.assertGreater(norm,0)
        quaternion = [value/norm for value in raw]
        self.assertGreater(math.sqrt(sum((value-identity)**2 for value,identity in zip(quaternion,[1,0,0,0]))),1e-3,
                           "Fixture must distinguish the FPGA prediction from physical identity")
        requests = crc_frame(struct.pack("<BBH9h",0xa6,2,65535,*([0]*9)))
        requests += crc_frame(struct.pack("<BBH9h",0xa6,1,0,*inputs))
        replies = crc_frame(struct.pack("<BBH96h4hI",0xa6,0,65535,*([0]*96),*([0]*4),0))
        replies += crc_frame(struct.pack("<BBH96h4hI",0xa6,0,0,*hidden,*raw,124046))
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            source = (ROOT / "tests/integration/fixtures/fake_cfc_ftdi.c").read_text()
            source = re.sub(r"\b28\b","210",source)
            for old,new in (("expected[8]","expected[24]"),("size != 8","size != 24"),
                            ("fread(expected, 1, 8, requests) != 8","fread(expected, 1, 24, requests) != 24"),
                            ("memcmp(expected, data, 8)","memcmp(expected, data, 24)"),("data[4]","data[1]")):
                source = source.replace(old,new)
            c_path, library = workspace / "fake_orientation.c", workspace / "fake_orientation.dylib"
            c_path.write_text(source)
            compiled = subprocess.run([shutil.which("clang") or "clang","-shared","-fPIC",str(c_path),"-o",str(library)],
                                      cwd=ROOT,capture_output=True,text=True,timeout=30)
            self.assertEqual(compiled.returncode,0,compiled.stdout+compiled.stderr)
            request_file,reply_file,trace = (workspace/name for name in ("requests.bin","replies.bin","trace.txt"))
            request_file.write_bytes(requests)
            reply_file.write_bytes(replies)
            with socket.socket() as reserved:
                reserved.bind(("127.0.0.1",0))
                port = reserved.getsockname()[1]
            self.assertNotEqual(port,8765)
            process = subprocess.Popen(
                [str(ROOT/".venv/bin/python"),"-m","host.orientation_demo","--usb-serial","2023112818",
                 "--port",str(port),"--export-dir",str(EXPORT),"--libftdi",str(library)],
                cwd=ROOT,env={**os.environ,"FAKE_CFC_REQUESTS":str(request_file),"FAKE_CFC_REPLIES":str(reply_file),
                              "FAKE_CFC_TRACE":str(trace),"FAKE_CFC_FAULT":"","FAKE_CFC_DELAY_MS":"0"},
                stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,
            )
            base = f"http://127.0.0.1:{port}"
            def api(path, post=False):
                request = urllib.request.Request(base+path,data=b"{}" if post else None,
                                                 headers={"Content-Type":"application/json"} if post else {})
                with urllib.request.urlopen(request,timeout=3) as response:
                    self.assertEqual(response.status,200)
                    return json.load(response)
            try:
                deadline = time.monotonic()+5
                while True:
                    if process.poll() is not None:
                        stdout,stderr = process.communicate()
                        self.fail("Actual orientation app must start:\n"+stdout+stderr)
                    try:
                        with urllib.request.urlopen(base+"/",timeout=.2) as response:
                            html = response.read().decode()
                        break
                    except (urllib.error.URLError,TimeoutError):
                        if time.monotonic()>=deadline:
                            self.fail("Orientation app did not start within five seconds")
                        time.sleep(.05)
                page = OrientationPage()
                page.feed(html)
                self.assertGreater(page.canvas_count,0)
                labels = {label.strip().lower() for label in page.buttons}
                for alternatives in (("connect","anslut"),("step","steg","stega"),("disconnect","koppla från")):
                    self.assertTrue(any(label in labels for label in alternatives),f"Missing orientation control: {alternatives}")
                self.assertTrue(page.modules,"Page must load a real module asset")
                with urllib.request.urlopen(base+page.modules[0],timeout=1) as asset:
                    self.assertIn("javascript",asset.headers.get_content_type())
                    self.assertGreater(len(asset.read()),0)
                self.assertEqual(api("/api/state")["status"],"disconnected")
                self.assertFalse(trace.exists(),"USB context must not be allocated before connect")
                connected = api("/api/connect",post=True)
                self.assertEqual(connected["status"],"connected")
                self.assertTrue(connected["reset_verified"])
                self.assertEqual(connected["count_verified"],0)
                stepped = api("/api/step",post=True)
                self.assertEqual(stepped["status"],"running")
                self.assertEqual(stepped["source"],"fpga_usb")
                self.assertEqual(stepped["count_verified"],1)
                self.assertEqual(stepped["fixed_model_dt_s"],.0035)
                self.assertEqual(stepped["sensor_units"],{"acc":"m/s^2","gyr":"rad/s","mag":"uT"})
                sample = stepped["sample"]
                self.assertEqual(sample["index"],0)
                self.assertEqual(len(sample["sensors"]),9)
                for actual,expected in zip(sample["sensors"],sensors):
                    self.assertAlmostEqual(actual,expected,places=10)
                self.assertEqual(sample["quantized_inputs"],inputs)
                self.assertEqual(sample["hidden"],hidden)
                self.assertEqual(sample["raw_outputs"],raw)
                self.assertEqual(sample["cycles"],124046)
                self.assertTrue(sample["verified"])
                self.assertEqual(len(sample["quaternion_wxyz"]),4)
                for actual,expected in zip(sample["quaternion_wxyz"],quaternion):
                    self.assertAlmostEqual(actual,expected,places=10)
                self.assertAlmostEqual(sum(value*value for value in sample["quaternion_wxyz"]),1,places=10)
                self.assertEqual(api("/api/state"),stepped)
                disconnected = api("/api/disconnect",post=True)
                self.assertEqual(disconnected["status"],"disconnected")
                events = trace.read_text().splitlines()
                self.assertEqual(events.count("request RESET"),1)
                self.assertEqual(events.count("request STEP"),1)
                self.assertEqual(events[-2:],["close","free"])
                self.assertFalse(any(event.startswith("ERROR") for event in events),events)
                self.assertEqual(identities(),before)
            finally:
                if process.poll() is None:
                    process.terminate()
                try:
                    process.communicate(timeout=2)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.communicate(timeout=2)


@contextmanager
def rotated_native_app(workspace,requests,replies,native_fault="",delay_ms=10,fault_after_steps=0,
                       *,module="host.orientation_demo",export_dir=EXPORT):
    source=(ROOT/"tests/integration/fixtures/fake_cfc_ftdi.c").read_text()
    source=re.sub(r"\b28\b","210",source)
    for old,new in (("expected[8]","expected[24]"),("size != 8","size != 24"),
                    ("fread(expected, 1, 8, requests) != 8","fread(expected, 1, 24, requests) != 24"),
                    ("memcmp(expected, data, 8)","memcmp(expected, data, 24)"),("data[4]","data[1]")):
        source=source.replace(old,new)
    source=source.replace("static FILE *requests, *replies;","static FILE *requests, *replies;\nstatic int fixture_steps;")
    source=source.replace('requests = fopen(getenv("FAKE_CFC_REQUESTS"), "rb");',
                          'configured=0; stale=0; drained=0; offset=210; command=0; delayed=0; fixture_steps=0;\n'
                          '    requests = fopen(getenv("FAKE_CFC_REQUESTS"), "rb");')
    source=source.replace('const char *fault = getenv("FAKE_CFC_FAULT");',
                          'const char *fault = getenv("FAKE_CFC_FAULT");\n'
                          '    const char *after = getenv("FAKE_CFC_FAULT_AFTER_STEPS");\n'
                          '    if (after && fixture_steps <= strtol(after, NULL, 10)) fault=NULL;')
    source=source.replace("offset = 0; command = data[1]; delayed = 0;",
                          "offset = 0; command = data[1]; delayed = 0; if(command==1) fixture_steps++;")
    c_path,library=workspace/"rotated.c",workspace/"rotated.dylib"
    c_path.write_text(source)
    compiled=subprocess.run([shutil.which("clang") or "clang","-shared","-fPIC",str(c_path),"-o",str(library)],
                            cwd=ROOT,capture_output=True,text=True,timeout=30)
    if compiled.returncode:
        raise AssertionError(compiled.stdout+compiled.stderr)
    request_path,reply_path,trace=(workspace/name for name in ("requests.bin","replies.bin","trace.txt"))
    request_path.write_bytes(requests)
    reply_path.write_bytes(replies)
    with socket.socket() as reserved:
        reserved.bind(("127.0.0.1",0))
        port=reserved.getsockname()[1]
    process=subprocess.Popen(
        [str(ROOT/".venv/bin/python"),"-m",module,"--usb-serial","2023112818",
         "--port",str(port),"--export-dir",str(export_dir),"--libftdi",str(library)],
        cwd=ROOT,env={**os.environ,"FAKE_CFC_REQUESTS":str(request_path),"FAKE_CFC_REPLIES":str(reply_path),
                      "FAKE_CFC_TRACE":str(trace),"FAKE_CFC_FAULT":native_fault,"FAKE_CFC_DELAY_MS":str(delay_ms),
                      "FAKE_CFC_FAULT_AFTER_STEPS":str(fault_after_steps)},
        stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,
    )
    base=f"http://127.0.0.1:{port}"
    try:
        deadline=time.monotonic()+5
        while True:
            if process.poll() is not None:
                stdout,stderr=process.communicate()
                raise AssertionError("Actual orientation app must start:\n"+stdout+stderr)
            try:
                with urllib.request.urlopen(base+"/",timeout=.2) as response:
                    response.read()
                break
            except (urllib.error.URLError,TimeoutError):
                if time.monotonic()>=deadline:
                    raise AssertionError("Orientation app startup deadline")
                time.sleep(.05)
        yield base,trace
    finally:
        if process.poll() is None:
            process.terminate()
        try:
            process.communicate(timeout=2)
        except subprocess.TimeoutExpired:
            process.kill()
            process.communicate(timeout=2)


class OrientationRotatedHttpTests(unittest.TestCase):
    def test_rotated_steps_retain_target_and_advance_fixed_model_time_despite_usb_delay(self):
        artifact=json.loads((EXPORT/"manifest.json").read_text())
        predictor=GoldenBroadPredictor.from_export(EXPORT)
        targets=[unit([1,0,0,1]),None,unit([math.cos(math.pi/8),math.sin(math.pi/8),0,0]),
                 -unit([math.cos(math.pi/8),math.sin(math.pi/8),0,0])]
        desired=truth=unit([1,0,0,0])
        requests=crc_frame(struct.pack("<BBH9h",0xa6,2,65535,*([0]*9)))
        replies=crc_frame(struct.pack("<BBH96h4hI",0xa6,0,65535,*([0]*96),*([0]*4),0))
        expected=[]
        for index,target in enumerate(targets):
            if target is not None:
                desired=target
            previous=truth
            truth=reference_pose_step(previous,desired)
            sensors=reference_sensors(previous,truth)
            normalized=[(float(value)-mean)/std for value,mean,std in
                        zip(sensors,artifact["normalization"]["mean"],artifact["normalization"]["std"])]
            inputs=[min(32767,max(-32768,math.floor(value*2048+.5) if value>=0 else -math.floor(-value*2048+.5)))
                    for value in normalized]
            hidden,raw=predictor.step(inputs)
            norm=math.sqrt(sum(value*value for value in raw))
            self.assertGreater(norm,0)
            quaternion=[value/norm for value in raw]
            requests+=crc_frame(struct.pack("<BBH9h",0xa6,1,index,*inputs))
            replies+=crc_frame(struct.pack("<BBH96h4hI",0xa6,0,index,*hidden,*raw,124046))
            expected.append(dict(truth=truth.copy(),target=desired.copy(),sensors=sensors.copy(),inputs=inputs,
                                 hidden=hidden,raw=raw,quaternion=quaternion))
        before=identities()
        with tempfile.TemporaryDirectory() as directory:
            with rotated_native_app(Path(directory),requests,replies) as (base,trace):
                def post(path,payload):
                    request=urllib.request.Request(base+path,data=json.dumps(payload).encode(),
                                                   headers={"Content-Type":"application/json"})
                    try:
                        with urllib.request.urlopen(request,timeout=3) as response:
                            return json.load(response)
                    except urllib.error.HTTPError as error:
                        self.fail("Target-aware physical STEP must match independent native requests: "+error.read().decode())
                self.assertTrue(post("/api/connect",{})["reset_verified"])
                started=time.monotonic()
                for index,(target,reference) in enumerate(zip(targets,expected)):
                    payload={} if target is None else {"target_wxyz":target.tolist()}
                    state=post("/api/step",payload)
                    self.assertEqual(state["count_verified"],index+1)
                    self.assertEqual(state["source"],"fpga_usb")
                    sample=state["sample"]
                    self.assertEqual(sample["quantized_inputs"],reference["inputs"])
                    self.assertEqual(sample["hidden"],reference["hidden"])
                    self.assertEqual(sample["raw_outputs"],reference["raw"])
                    self.assertEqual(sample["cycles"],124046)
                    self.assertTrue(sample["verified"])
                    self.assertAlmostEqual(sample["model_time_s"],(index+1)*.0035,places=12)
                    for actual,wanted in zip(sample["sensors"],reference["sensors"]):
                        self.assertAlmostEqual(actual,float(wanted),places=8)
                    for name,key in (("truth_wxyz","truth"),("target_wxyz","target")):
                        actual=unit(sample[name])
                        self.assertAlmostEqual(abs(float(sum(actual*reference[key]))),1,places=10)
                    desired_state=unit(state["target_wxyz"])
                    self.assertAlmostEqual(abs(float(sum(desired_state*reference["target"]))),1,places=10)
                    for actual,wanted in zip(sample["quaternion_wxyz"],reference["quaternion"]):
                        self.assertAlmostEqual(actual,wanted,places=10)
                    dot=abs(float(sum(unit(sample["quaternion_wxyz"])*reference["truth"])))
                    error_deg=math.degrees(2*math.acos(min(1,max(0,dot))))
                    self.assertAlmostEqual(sample["error_deg"],error_deg,places=8)
                self.assertGreaterEqual(time.monotonic()-started,.04)
                self.assertEqual(post("/api/disconnect",{})["status"],"disconnected")
                events=trace.read_text().splitlines()
                self.assertEqual(events.count("request RESET"),1)
                self.assertEqual(events.count("request STEP"),4)
                self.assertEqual(events[-2:],["close","free"])
        self.assertEqual(identities(),before)


def orientation_stream(actions):
    artifact=json.loads((EXPORT/"manifest.json").read_text())
    predictor=GoldenBroadPredictor.from_export(EXPORT)
    requests=[]
    replies=[]
    samples=[]
    truth=target=unit([1,0,0,0])
    sequence=0
    for command,requested in actions:
        if command==2:
            predictor.reset()
            truth=target=unit([1,0,0,0])
            sequence=0
            requests.append(crc_frame(struct.pack("<BBH9h",0xa6,2,65535,*([0]*9))))
            replies.append(crc_frame(struct.pack("<BBH96h4hI",0xa6,0,65535,*([0]*96),*([0]*4),0)))
            continue
        if requested is not None:
            target=unit(requested)
        previous=truth
        truth=reference_pose_step(previous,target)
        sensors=reference_sensors(previous,truth)
        normalized=[(float(value)-mean)/std for value,mean,std in
                    zip(sensors,artifact["normalization"]["mean"],artifact["normalization"]["std"])]
        inputs=[min(32767,max(-32768,math.floor(value*2048+.5) if value>=0 else -math.floor(-value*2048+.5)))
                for value in normalized]
        hidden,raw=predictor.step(inputs)
        requests.append(crc_frame(struct.pack("<BBH9h",0xa6,1,sequence,*inputs)))
        replies.append(crc_frame(struct.pack("<BBH96h4hI",0xa6,0,sequence,*hidden,*raw,124046)))
        samples.append({"hidden":hidden,"raw_outputs":raw,"truth":truth.copy(),"target":target.copy()})
        sequence+=1
    return b"".join(requests),replies,samples


class OrientationSessionGuardTests(unittest.TestCase):
    def request(self,base,path,body=None,headers=None):
        actual_headers={"Content-Type":"application/json"} if body is not None else {}
        actual_headers.update(headers or {})
        request=urllib.request.Request(base+path,data=body,headers=actual_headers)
        try:
            with urllib.request.urlopen(request,timeout=3) as response:
                status=response.status
                data=response.read()
        except urllib.error.HTTPError as error:
            status=error.code
            data=error.read()
        try:
            decoded=json.loads(data)
        except (ValueError,UnicodeDecodeError):
            decoded={"body":data.decode(errors="replace")}
        return status,decoded

    def post(self,base,path,payload=None):
        return self.request(base,path,json.dumps({} if payload is None else payload).encode())

    def test_verified_reset_clears_state_pose_target_and_count_before_cold_step(self):
        target=unit([1,0,0,1]).tolist()
        requests,replies,samples=orientation_stream([(2,None),(1,target),(2,None),(1,None)])
        with tempfile.TemporaryDirectory() as directory:
            with rotated_native_app(Path(directory),requests,b"".join(replies)) as (base,trace):
                self.assertEqual(self.post(base,"/api/connect")[0],200)
                code,first=self.post(base,"/api/step",{"target_wxyz":target})
                self.assertEqual(code,200,first)
                self.assertEqual(first["count_verified"],1)
                code,reset=self.post(base,"/api/reset")
                self.assertEqual(code,200,reset)
                self.assertEqual(reset["status"],"connected")
                self.assertTrue(reset["reset_verified"])
                self.assertEqual(reset["count_verified"],0)
                self.assertIsNone(reset["sample"])
                self.assertEqual(reset["target_wxyz"],[1,0,0,0])
                code,cold=self.post(base,"/api/step")
                self.assertEqual(code,200,cold)
                self.assertEqual(cold["sample"]["hidden"],samples[1]["hidden"])
                self.assertEqual(cold["sample"]["raw_outputs"],samples[1]["raw_outputs"])
                self.assertEqual(cold["sample"]["truth_wxyz"],[1,0,0,0])
                self.assertEqual(cold["sample"]["model_time_s"],.0035)
                self.assertEqual(trace.read_text().splitlines().count("request RESET"),2)
                self.assertEqual(self.post(base,"/api/disconnect")[0],200)
                with urllib.request.urlopen(base+"/",timeout=1) as response:
                    page=OrientationPage()
                    page.feed(response.read().decode())
                self.assertTrue(any(label.strip().lower() in ("reset","återställ") for label in page.buttons),
                                "Viewer must expose a reset control")

    def test_invalid_reset_acknowledgment_closes_device_without_clearing_last_verified_state(self):
        target=unit([1,0,0,1]).tolist()
        requests,replies,_=orientation_stream([(2,None),(1,target),(2,None)])
        bad=bytearray(replies[2][:-2])
        bad[4]=1
        replies[2]=crc_frame(bytes(bad))
        with tempfile.TemporaryDirectory() as directory:
            with rotated_native_app(Path(directory),requests,b"".join(replies)) as (base,trace):
                self.assertEqual(self.post(base,"/api/connect")[0],200)
                code,first=self.post(base,"/api/step",{"target_wxyz":target})
                self.assertEqual(code,200,first)
                code,failed=self.post(base,"/api/reset")
                self.assertEqual(code,502,failed)
                self.assertEqual(failed["status"],"error")
                self.assertFalse(failed["reset_verified"])
                self.assertEqual(failed["count_verified"],1)
                self.assertEqual(failed["sample"],first["sample"])
                self.assertEqual(failed["target_wxyz"],first["target_wxyz"])
                self.assertEqual(trace.read_text().splitlines()[-2:],["close","free"])

    def test_step_and_reset_return_conflict_promptly_while_a_native_step_is_in_flight(self):
        requests,replies,_=orientation_stream([(2,None),(1,None)])
        with tempfile.TemporaryDirectory() as directory:
            with rotated_native_app(Path(directory),requests,b"".join(replies),delay_ms=600) as (base,trace):
                self.assertEqual(self.post(base,"/api/connect")[0],200)
                with ThreadPoolExecutor(max_workers=3) as pool:
                    pending=pool.submit(self.post,base,"/api/step")
                    deadline=time.monotonic()+1
                    while "request STEP" not in trace.read_text():
                        if time.monotonic()>=deadline:
                            self.fail("Native STEP did not start")
                        time.sleep(.005)
                    conflicts={path:pool.submit(self.post,base,path) for path in ("/api/step","/api/reset")}
                    for path,future in conflicts.items():
                        try:
                            code,body=future.result(timeout=.25)
                        except FutureTimeout:
                            self.fail(f"{path} queued behind hardware instead of returning 409 promptly")
                        self.assertEqual(code,409,body)
                    self.assertEqual(pending.result(timeout=2)[0],200)
                self.assertEqual(trace.read_text().splitlines().count("request STEP"),1)
                self.assertEqual(self.post(base,"/api/disconnect")[0],200)

    def test_native_faults_stop_session_preserve_last_sample_and_allow_verified_reconnect(self):
        for fault in ("timeout","crc","sequence","cycles","hidden","raw","zero_raw"):
            with self.subTest(fault=fault),tempfile.TemporaryDirectory() as directory:
                requests,replies,_=orientation_stream([(2,None),(1,None),(1,None)])
                native="silent" if fault=="timeout" else ""
                if fault!="timeout":
                    if fault=="crc":
                        replies[2]=replies[2][:-1]+bytes([replies[2][-1]^1])
                    else:
                        changed=bytearray(replies[2][:-2])
                        if fault=="sequence":
                            changed[2]^=1
                        elif fault=="cycles":
                            struct.pack_into("<I",changed,204,124047)
                        elif fault=="hidden":
                            changed[4]^=1
                        elif fault=="raw":
                            changed[196]^=1
                        else:
                            changed[196:204]=b"\x00"*8
                        replies[2]=crc_frame(bytes(changed))
                with rotated_native_app(Path(directory),requests,b"".join(replies),native_fault=native,fault_after_steps=1) as (base,trace):
                    self.assertEqual(self.post(base,"/api/connect")[0],200)
                    code,first=self.post(base,"/api/step")
                    self.assertEqual(code,200,first)
                    code,failed=self.post(base,"/api/step")
                    self.assertEqual(code,504 if fault=="timeout" else 502,failed)
                    self.assertEqual(failed["status"],"error")
                    self.assertEqual(failed["count_verified"],1)
                    self.assertEqual(failed["sample"],first["sample"])
                    self.assertEqual(trace.read_text().splitlines()[-2:],["close","free"])
                    code,reconnected=self.post(base,"/api/connect")
                    self.assertEqual(code,200,reconnected)
                    self.assertTrue(reconnected["reset_verified"])
                    self.assertEqual(reconnected["count_verified"],0)
                    self.assertIsNone(reconnected["sample"])
                    code,cold=self.post(base,"/api/step")
                    self.assertEqual(code,200,cold)
                    self.assertEqual(cold["sample"]["hidden"],first["sample"]["hidden"])
                    self.assertEqual(cold["sample"]["raw_outputs"],first["sample"]["raw_outputs"])
                    self.assertEqual(self.post(base,"/api/disconnect")[0],200)
                    events=trace.read_text().splitlines()
                    self.assertEqual(events.count("request RESET"),2)
                    self.assertEqual(events.count("close"),2)
                    self.assertEqual(events.count("free"),2)

    def test_invalid_json_targets_size_and_content_type_preserve_valid_session_without_usb_writes(self):
        requests,replies,_=orientation_stream([(2,None),(1,None)])
        with tempfile.TemporaryDirectory() as directory:
            with rotated_native_app(Path(directory),requests,b"".join(replies)) as (base,trace):
                code,connected=self.post(base,"/api/connect")
                self.assertEqual(code,200,connected)
                before=trace.read_text()
                cases=[(b'{"target_wxyz":['+b"9"*400+b',0,0,1]}',{},400),
                       (b"{",{},400),(b"[]",{},400),(b'{"target_wxyz":[0,0,0,0]}',{},400),
                       (b'{"target_wxyz":[NaN,0,0,1]}',{},400),(b"x"*4097,{},413),
                       (b"{}",{"Content-Type":"text/plain"},415)]
                for body,headers,expected in cases:
                    try:
                        code,error=self.request(base,"/api/step",body,headers)
                    except (urllib.error.URLError,ConnectionError) as error:
                        self.fail(f"Invalid target must return an HTTP error without closing the connection: {error}")
                    self.assertEqual(code,expected,error)
                    self.assertEqual(self.request(base,"/api/state")[1],connected)
                    self.assertEqual(trace.read_text(),before)
                self.assertEqual(self.post(base,"/api/step")[0],200)
                self.assertEqual(self.post(base,"/api/disconnect")[0],200)

    def test_host_origin_and_traversal_guards_reject_before_usb_writes(self):
        requests,replies,_=orientation_stream([(2,None),(1,None)])
        with tempfile.TemporaryDirectory() as directory:
            with rotated_native_app(Path(directory),requests,b"".join(replies)) as (base,trace):
                code,connected=self.post(base,"/api/connect")
                self.assertEqual(code,200,connected)
                before=trace.read_text()
                for path in ("/../host/orientation_demo.py","/..%2fhost/orientation_demo.py"):
                    self.assertEqual(self.request(base,path)[0],404)
                port=int(base.rsplit(":",1)[1])
                wrong_port=port+1 if port<65535 else port-1
                hostile=[{"Origin":"https://attacker.example"},{"Origin":f"http://127.0.0.1:{wrong_port}"},
                         {"Host":"attacker.example"},{"Host":f"127.0.0.1:{wrong_port}"}]
                for headers in hostile:
                    code,error=self.request(base,"/api/step",b"{}",headers)
                    self.assertEqual(code,403,error)
                    self.assertEqual(self.request(base,"/api/state")[1],connected)
                    self.assertEqual(trace.read_text(),before)
                self.assertEqual(self.post(base,"/api/step")[0],200)
                self.assertEqual(self.post(base,"/api/disconnect")[0],200)


if __name__=="__main__":
    unittest.main()
