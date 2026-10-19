"""CLI integration tests for the read-only FPGA environment report."""

import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import unittest

# Behaviors covered by ENV-2:
# 1. Missing executables prevent readiness and have no process exit code.
# 2. Nonzero probes prevent readiness and preserve exit codes and diagnostics.
# 3. Successful probes may report their version on stderr.
# 4. Timed-out probes return promptly without a process exit code.
# 5. Invalid timeouts are rejected before any executable is invoked.


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "scripts" / "check_env.py"
PROBES = {
    "uv": "--version",
    "yosys": "-V",
    "nextpnr-himbaechel": "--version",
    "gowin_pack": "--help",
    "openFPGALoader": "--version",
    "iverilog": "-V",
    "vvp": "-V",
}


class EnvironmentReportTests(unittest.TestCase):
    def test_cli_reports_ready_when_all_required_tool_probes_succeed(self):
        with tempfile.TemporaryDirectory() as directory:
            tool_directory = Path(directory)
            for name, argument in PROBES.items():
                executable = tool_directory / name
                executable.write_text(
                    f"#!{sys.executable}\n"
                    "import sys\n"
                    f"if sys.argv[1:] != [{argument!r}]:\n"
                    "    print('Unexpected probe arguments', file=sys.stderr)\n"
                    "    sys.exit(9)\n"
                    f"print({name + ' test-version 1.0'!r})\n"
                )
                executable.chmod(0o755)

            result = subprocess.run(
                [sys.executable, str(SCRIPT), "--json"],
                cwd=PROJECT_ROOT,
                env={**os.environ, "PATH": str(tool_directory)},
                capture_output=True,
                text=True,
                timeout=10,
            )

            self.assertEqual(
                result.returncode,
                0,
                f"Environment CLI must report ready with successful tools: {result.stderr}",
            )
            report = json.loads(result.stdout)
            self.assertTrue(report["ready"])
            self.assertFalse(report["hardware_verified"])
            self.assertTrue(report["python"]["supported"])
            self.assertEqual(report["platform"]["system"], platform.system())
            self.assertEqual(report["platform"]["machine"], platform.machine())
            self.assertEqual(set(report["tools"]), set(PROBES))
            for name in PROBES:
                with self.subTest(tool=name):
                    tool = report["tools"][name]
                    self.assertEqual(tool["status"], "ok")
                    self.assertEqual(tool["path"], str(tool_directory / name))
                    self.assertIn("detail", tool)
                    if name == "gowin_pack":
                        self.assertEqual(tool["probe_kind"], "help")
                        self.assertIsNone(tool["version"])
                    else:
                        self.assertEqual(tool["probe_kind"], "version")
                        self.assertEqual(tool["version"], f"{name} test-version 1.0")


class EnvironmentFailureReportTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.tool_directory = Path(temporary.name)
        for name in PROBES:
            self.write_tool(name, f"print({name + ' test-version 1.0'!r})\n")

    def write_tool(self, name, body):
        executable = self.tool_directory / name
        executable.write_text(f"#!{sys.executable}\n" + body)
        executable.chmod(0o755)

    def run_cli(self, *arguments):
        return subprocess.run(
            [sys.executable, str(SCRIPT), "--json", *arguments],
            cwd=PROJECT_ROOT,
            env={**os.environ, "PATH": str(self.tool_directory)},
            capture_output=True,
            text=True,
            timeout=8,
        )

    def test_missing_tool_prevents_readiness_without_an_exit_code(self):
        (self.tool_directory / "yosys").unlink()

        result = self.run_cli()

        self.assertEqual(result.returncode, 1, result.stderr)
        report = json.loads(result.stdout)
        self.assertFalse(report["ready"])
        tool = report["tools"]["yosys"]
        self.assertEqual(tool["status"], "missing")
        self.assertIsNone(tool["path"])
        self.assertIsNone(tool["version"])
        self.assertIn("exit_code", tool, "Missing probes must report exit_code=null")
        self.assertIsNone(tool["exit_code"])

    def test_nonzero_probe_preserves_failure_code_and_diagnostic(self):
        self.write_tool(
            "yosys",
            "import sys\n"
            "print('yosys test-version 1.0')\n"
            "print('cannot load synthesis library', file=sys.stderr)\n"
            "sys.exit(7)\n",
        )

        result = self.run_cli()

        self.assertEqual(result.returncode, 1, result.stderr)
        report = json.loads(result.stdout)
        self.assertFalse(report["ready"])
        tool = report["tools"]["yosys"]
        self.assertEqual(tool["status"], "failed")
        self.assertIsNone(tool["version"])
        self.assertIn("cannot load synthesis library", tool["detail"])
        self.assertIn("exit_code", tool, "Failed probes must preserve their exit code")
        self.assertEqual(tool["exit_code"], 7)

    def test_successful_stderr_version_is_reported_with_zero_exit_code(self):
        self.write_tool(
            "iverilog", "import sys\nprint('Icarus Verilog test-version', file=sys.stderr)\n"
        )

        result = self.run_cli()

        self.assertEqual(result.returncode, 0, result.stderr)
        report = json.loads(result.stdout)
        self.assertTrue(report["ready"])
        tool = report["tools"]["iverilog"]
        self.assertEqual(tool["status"], "ok")
        self.assertEqual(tool["version"], "Icarus Verilog test-version")
        self.assertIn("exit_code", tool, "Successful probes must report exit_code=0")
        self.assertEqual(tool["exit_code"], 0)

    def test_timeout_prevents_readiness_and_returns_without_waiting_for_tool(self):
        self.write_tool("yosys", "import time\ntime.sleep(2)\nprint('late version')\n")

        result = self.run_cli("--timeout-seconds", "0.2")

        self.assertEqual(result.returncode, 1, result.stderr)
        report = json.loads(result.stdout)
        self.assertFalse(report["ready"])
        tool = report["tools"]["yosys"]
        self.assertEqual(tool["status"], "timeout")
        self.assertIsNone(tool["version"])
        self.assertIn("exit_code", tool, "Timed-out probes must report exit_code=null")
        self.assertIsNone(tool["exit_code"])

    def test_invalid_timeout_is_rejected_before_any_tool_runs(self):
        marker = self.tool_directory / "probe-was-run"
        for name in PROBES:
            self.write_tool(
                name,
                f"from pathlib import Path\nPath({str(marker)!r}).touch()\nprint('version')\n",
            )

        for value in ("0", "-1", "nan", "inf"):
            with self.subTest(timeout=value):
                result = self.run_cli("--timeout-seconds", value)

                self.assertEqual(result.returncode, 2, result.stderr)
                self.assertIn("timeout", result.stderr.lower())
                self.assertFalse(marker.exists(), "Invalid timeout must not invoke tools")


if __name__ == "__main__":
    unittest.main()
