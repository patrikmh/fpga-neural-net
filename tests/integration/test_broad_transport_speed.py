"""THICKENING transport behaviors:
1. Default and faster UART pins preserve complete H96 RESET/STEP replies.
2. Explicit USB requests cross native FTDI, HTTP and verifier boundaries honestly.
3. A nondefault divisor reaches a real routed build and invalid values fail early.
"""

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

import torch

from integration.test_broad_uart import reference_crc
from integration.test_orientation_metrics import write_checkpoint, write_dataset, poses_z, sha256
from integration.test_probe_uart import executable
from model.broad_integer import GoldenBroadPredictor
from test_orientation_sim import reference_sensors


ROOT = Path(__file__).resolve().parents[2]
PYTHON = str(ROOT / '.venv/bin/python')
EXPORT = ROOT / 'artifacts/broad-integer-v1/export'
TOOLS = ROOT / '.tools/tang-toolchain/oss-cad-suite/bin'


def frame(payload):
    return payload + struct.pack('<H', reference_crc(payload))


def h96_exchange():
    manifest = json.loads((EXPORT / 'manifest.json').read_text())
    sensors = reference_sensors([1, 0, 0, 0], [1, 0, 0, 0])
    values = [(value-mean)/std for value, mean, std in zip(
        sensors, manifest['normalization']['mean'], manifest['normalization']['std'])]
    inputs = [min(32767, max(-32768, int(math.copysign(math.floor(abs(value)*2048+.5), value))))
              for value in values]
    hidden, raw = GoldenBroadPredictor.from_export(EXPORT).step(inputs)
    requests = frame(struct.pack('<BBH9h', 0xa6, 2, 65535, *([0]*9)))
    requests += frame(struct.pack('<BBH9h', 0xa6, 1, 0, *inputs))
    replies = frame(struct.pack('<BBH96h4hI', 0xa6, 0, 65535, *([0]*100), 0))
    replies += frame(struct.pack('<BBH96h4hI', 0xa6, 0, 0, *hidden, *raw, 124046))
    return inputs, hidden, raw, requests, replies


def pin_bench(banks, divisor):
    parameter = '' if divisor == 434 else f'.UART_DIVISOR({divisor}),'
    return f'''`timescale 1ns/1ps
module speed_tb;
 localparam integer BIT_NS={divisor*20};
 reg clk=0,rst=1,rx=1;
 wire tx;
 reg [7:0] request[0:47];
 reg [7:0] value;
 integer capture,byte_number,bit_number,count=0;
 string requests_path,replies_path;
 broad_top #({parameter}.HIDDEN_SIZE(96),.WEIGHTS_DIR({json.dumps(str(banks))}),
  .TANH_FILE({json.dumps(str(EXPORT/'tanh.hex'))}),
  .SIGMOID_FILE({json.dumps(str(EXPORT/'sigmoid.hex'))}))
 dut(.clk(clk),.rst(rst),.uart_rx(rx),.uart_tx(tx));
 always #10 clk=~clk;
 task automatic send_byte(input [7:0] data);
  integer bit_index;
  begin
   rx=0; #(BIT_NS);
   for(bit_index=0;bit_index<8;bit_index=bit_index+1) begin rx=data[bit_index]; #(BIT_NS); end
   rx=1; #(BIT_NS);
  end
 endtask
 initial begin
  wait(rst==0);
  for(byte_number=0;byte_number<420;byte_number=byte_number+1) begin
   @(negedge tx); #(BIT_NS/2);
   if(tx!==0) $fatal(1,"UART start bit at configured divisor");
   for(bit_number=0;bit_number<8;bit_number=bit_number+1) begin #(BIT_NS); value[bit_number]=tx; end
   #(BIT_NS); if(tx!==1) $fatal(1,"UART stop bit at configured divisor");
   $fdisplay(capture,"%02x",value); count=count+1;
  end
 end
 initial begin
  if(!$value$plusargs("REQ=%s",requests_path) || !$value$plusargs("RESP=%s",replies_path)) $fatal;
  capture=$fopen(replies_path,"w"); $readmemh(requests_path,request);
  repeat(40) @(posedge clk); @(negedge clk); rst=0;
  repeat(1024) @(posedge clk); @(negedge clk);
  for(integer transaction=0;transaction<2;transaction=transaction+1) begin
   for(integer index=0;index<24;index=index+1) send_byte(request[transaction*24+index]);
   wait(count>=(transaction+1)*210); #(BIT_NS*2);
  end
  $fclose(capture); $finish;
 end
 initial begin #{50000000 if divisor == 434 else 12000000}; $fatal(1,"No complete H96 reply at configured UART divisor"); end
endmodule
'''


def native_fixture(workspace):
    source = (ROOT / 'tests/integration/fixtures/fake_cfc_ftdi.c').read_text()
    source = re.sub(r'\b28\b', '210', source)
    for old, new in (('expected[8]', 'expected[24]'), ('size != 8', 'size != 24'),
                     ('fread(expected, 1, 8, requests) != 8', 'fread(expected, 1, 24, requests) != 24'),
                     ('memcmp(expected, data, 8)', 'memcmp(expected, data, 24)'), ('data[4]', 'data[1]')):
        source = source.replace(old, new)
    source = source.replace('if (baudrate != 460800)', 'if (baudrate != atoi(getenv("EXPECTED_USB_REQUEST")))')
    source = source.replace('configured |= 8; trace("requested baudrate 460800"); return 0;',
                            'char event[80]; snprintf(event,sizeof(event),"requested baudrate %d",baudrate); '
                            'configured |= 8; trace(event); return 0;')
    path, library = workspace / 'native.c', workspace / 'native.dylib'
    path.write_text(source)
    result = subprocess.run([shutil.which('clang') or 'clang', '-shared', '-fPIC', str(path), '-o', str(library)],
                            capture_output=True, text=True, timeout=30)
    if result.returncode:
        raise AssertionError(result.stdout + result.stderr)
    return library


class BroadTransportSpeedTests(unittest.TestCase):
    def run_command(self, command, env=None, timeout=60):
        return subprocess.run(command, cwd=ROOT, env=env, capture_output=True, text=True, timeout=timeout)

    def test_default_and_faster_uart_pins_return_all_h96_states_outputs_crc_sequence_and_same_cycles(self):
        _, _, _, requests, expected = h96_exchange()
        with tempfile.TemporaryDirectory() as temporary:
            workspace = Path(temporary)
            banks = workspace / 'banks'
            banks.mkdir()
            words = (EXPORT / 'weights.hex').read_text().split()
            for number, offset in enumerate(range(0, len(words), 1024)):
                bank = words[offset:offset+1024]
                (banks / f'weights_{number:03d}.hex').write_text('\n'.join(bank + ['0000']*(1024-len(bank)))+'\n')
            request_path = workspace / 'request.hex'
            request_path.write_text(''.join(f'{value:02x}\n' for value in requests))
            for divisor in (434, 50):
                with self.subTest(divisor=divisor):
                    bench, simulation, response = (workspace / f'{divisor}.{suffix}' for suffix in ('sv', 'vvp', 'hex'))
                    bench.write_text(pin_bench(banks, divisor))
                    compiled = self.run_command([executable('iverilog'), '-g2012', '-s', 'speed_tb', '-o', str(simulation),
                                                 'rtl/uart.sv', 'rtl/broad_core.sv', 'rtl/broad_top.sv', str(bench)])
                    self.assertEqual(compiled.returncode, 0, compiled.stdout+compiled.stderr)
                    result = self.run_command([executable('vvp'), str(simulation), f'+REQ={request_path}',
                                               f'+RESP={response}'], timeout=120)
                    self.assertEqual(result.returncode, 0, 'Actual UART pins must support configured speed:\n'
                                     + result.stdout+result.stderr)
                    self.assertEqual(bytes.fromhex(response.read_text()), expected,
                                     'Complete RESET and STEP must preserve all96 states,4 outputs,CRC/seq/cycles')

    def test_requested_usb_rate_reaches_native_http_and_verifier_without_changing_legacy_default(self):
        inputs, hidden, raw, requests, replies = h96_exchange()
        with tempfile.TemporaryDirectory() as temporary:
            workspace = Path(temporary)
            library = native_fixture(workspace)
            request_path, reply_path = workspace / 'requests.bin', workspace / 'replies.bin'
            request_path.write_bytes(requests)
            reply_path.write_bytes(replies)
            def environment(name, requested=4000000, fault=''):
                return {**os.environ, 'FAKE_CFC_REQUESTS': str(request_path), 'FAKE_CFC_REPLIES': str(reply_path),
                        'FAKE_CFC_TRACE': str(workspace / (name+'.trace')), 'FAKE_CFC_FAULT': fault,
                        'FAKE_CFC_DELAY_MS': '0', 'EXPECTED_USB_REQUEST': str(requested)}
            vectors = workspace / 'vectors'
            vectors.mkdir()
            vector_file = vectors / 'vectors.hex'
            vector_file.write_text(''.join(f'{value&65535:04x}' for value in inputs+hidden+raw)+'\n')
            artifact = json.loads((EXPORT / 'manifest.json').read_text())
            (vectors / 'manifest.json').write_text(json.dumps({
                'count': 1, 'hidden_size': 96, 'initial_hidden': [0]*96, 'numeric': artifact['numeric'],
                'export_manifest_sha256': sha256(EXPORT / 'manifest.json'),
                'checkpoint_sha256': artifact['checkpoint_sha256'], 'vectors_sha256': sha256(vector_file)}))
            base_arguments = [PYTHON, '-m', 'host.verify_broad', '--usb-serial', '2023112818', '--export-dir', str(EXPORT),
                              '--vectors-dir', str(vectors), '--count', '1', '--expected-cycles', '124046',
                              '--libftdi', str(library)]
            default = self.run_command(base_arguments, env=environment('default', 460800))
            self.assertEqual(default.returncode, 0, default.stderr)
            default_report = json.loads(default.stdout)
            self.assertEqual((default_report['baudrate'], default_report['usb_requested_baudrate']), (115200, 460800))
            arguments = base_arguments + ['--usb-requested-baudrate', '4000000', '--uart-divisor', '50']
            result = self.run_command(arguments, env=environment('verifier'))
            self.assertEqual(result.returncode, 0, 'Explicit USB request must cross actual native verifier:\n'+result.stderr)
            report = json.loads(result.stdout)
            self.assertEqual((report['count_verified'], report['hidden_size'], report['usb_requested_baudrate']),
                             (1, 96, 4000000))
            self.assertEqual((report['uart_divisor'], report['expected_fpga_baudrate']), (50, 1000000))
            self.assertEqual(report['baudrate'], 1000000)
            failed_output = workspace / 'failed.json'
            failed = self.run_command(arguments+['--output', str(failed_output)], env=environment('fault', fault='short'))
            self.assertNotEqual(failed.returncode, 0)
            self.assertFalse(failed_output.exists())
            self.assertEqual((workspace / 'fault.trace').read_text().splitlines()[-2:], ['close', 'free'])
            with socket.socket() as reserved:
                reserved.bind(('127.0.0.1', 0))
                port = reserved.getsockname()[1]
            self.assertNotIn(port, (8765, 8766))
            process = subprocess.Popen([PYTHON, '-m', 'host.orientation_demo', '--usb-serial', '2023112818',
                '--export-dir', str(EXPORT), '--libftdi', str(library), '--port', str(port),
                '--usb-requested-baudrate', '4000000'], cwd=ROOT, env=environment('app'),
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            base = f'http://127.0.0.1:{port}'
            def api(path, post=False):
                request = urllib.request.Request(base+path, data=b'{}' if post else None,
                    headers={'Content-Type': 'application/json'} if post else {})
                with urllib.request.urlopen(request, timeout=3) as response:
                    return json.load(response)
            try:
                deadline = time.monotonic()+5
                while True:
                    if process.poll() is not None:
                        stdout, stderr = process.communicate()
                        self.fail('Actual HTTP application must accept requested USB rate:\n'+stdout+stderr)
                    try:
                        state = api('/api/state')
                        break
                    except (urllib.error.URLError, TimeoutError):
                        if time.monotonic() >= deadline:
                            self.fail('HTTP app startup deadline')
                        time.sleep(.05)
                self.assertEqual(state['usb_requested_baudrate'], 4000000)
                self.assertTrue(api('/api/connect', True)['reset_verified'])
                step = api('/api/step', True)
                self.assertEqual(step['sample']['hidden'], hidden)
                self.assertEqual(step['sample']['raw_outputs'], raw)
                self.assertEqual(step['sample']['cycles'], 124046)
                api('/api/disconnect', True)
            finally:
                if process.poll() is None:
                    process.terminate()
                process.communicate(timeout=3)
            for name in ('default', 'verifier', 'app'):
                events = (workspace / (name+'.trace')).read_text().splitlines()
                self.assertIn('requested baudrate '+('460800' if name == 'default' else '4000000'), events)
                self.assertIn('latency 2', events)
                self.assertEqual(events.count('request RESET'), 1)
                self.assertEqual(events.count('request STEP'), 1)
                self.assertEqual(events[-2:], ['close', 'free'])
                self.assertFalse(any(event.startswith('ERROR') for event in events))
            invalid = self.run_command([PYTHON, '-c', 'from host.ftdi import Ftdi\n'
                'for value in (0,-1,1.5):\n'
                ' try: Ftdi("2023112818", "/nonexistent-native-library", requested_baudrate=value)\n'
                ' except ValueError: pass\n'
                ' else: raise AssertionError("Invalid rate accepted")'])
            self.assertEqual(invalid.returncode, 0, 'Invalid USB rate must fail before native library access:\n'+invalid.stderr)

    def test_nondefault_divisor_is_bound_to_routed_bitstream_and_invalid_values_fail_before_tools(self):
        with tempfile.TemporaryDirectory() as temporary:
            workspace = Path(temporary)
            dataset = workspace / 'dataset'
            write_dataset(dataset, [('hold', poses_z([0, 0]))])
            checkpoint = workspace / 'tiny.pt'
            write_checkpoint(checkpoint, dataset)
            from model.broad_network import BroadPredictor
            saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
            saved['config']['hidden_size'] = 3
            torch.manual_seed(17)
            saved['state_dict'] = BroadPredictor(hidden_size=3).state_dict()
            torch.save(saved, checkpoint)
            exported, output = workspace / 'export', workspace / 'build'
            exported_result = self.run_command([PYTHON, '-m', 'model.broad_export_integer', '--checkpoint',
                                                str(checkpoint), '--output-dir', str(exported)])
            self.assertEqual(exported_result.returncode, 0, exported_result.stderr)
            command = [PYTHON, 'scripts/build_broad.py', '--export-dir', str(exported)]
            for invalid in ('0', '-1', '1.5'):
                with self.subTest(invalid_divisor=invalid):
                    rejected = self.run_command(command+['--build-dir', str(workspace / ('invalid-'+invalid)),
                        '--uart-divisor', invalid], env={**os.environ, 'PATH': ''})
                    self.assertNotEqual(rejected.returncode, 0)
                    self.assertRegex(rejected.stderr.lower(), 'uart-divisor')
                    self.assertNotIn('required tool is missing', rejected.stderr.lower())
            result = self.run_command(command+['--build-dir', str(output), '--uart-divisor', '50', '--seed', '3'],
                env={**os.environ, 'PATH': str(TOOLS)+os.pathsep+os.environ.get('PATH', '')}, timeout=1900)
            self.assertEqual(result.returncode, 0, 'Actual nondefault tiny build must complete:\n'+result.stdout+result.stderr)
            report = json.loads((output / 'report.json').read_text())
            self.assertEqual(report['placement']['seed'], 3)
            self.assertEqual((report['hidden_size'], report['clock_mhz'], report['uart_divisor']), (3, 50, 50))
            self.assertEqual(report['expected_fpga_baudrate'], 1000000)
            synthesized = json.loads((output / 'synth.json').read_text())
            self.assertEqual(int(synthesized['modules']['broad_top']['parameter_default_values']['UART_DIVISOR'], 2), 50)
            self.assertGreaterEqual(report['timing']['achieved_mhz'], 50)
            for source in ('rtl/uart.sv', 'rtl/broad_top.sv', 'scripts/build_broad.py'):
                self.assertEqual(report['sources'][source], sha256(ROOT / source))
            bitstream = output / report['bitstream']['file']
            self.assertGreater(bitstream.stat().st_size, 0)
            self.assertEqual(report['bitstream']['sha256'], sha256(bitstream))
            self.assertEqual(report['export']['manifest_sha256'], sha256(exported / 'manifest.json'))


if __name__ == '__main__':
    unittest.main()
