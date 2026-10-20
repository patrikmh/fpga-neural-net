PYTHON ?= .venv/bin/python
TOOLCHAIN_DIR ?= $(CURDIR)/.tools/tang-toolchain/oss-cad-suite
BUILD_DIR ?= build/probe
EXPORT_DIR ?= build/integer-v1
CFC_BUILD_DIR ?= build/cfc
CFC_VECTORS_DIR ?= $(EXPORT_DIR)/vectors
CFC_COUNT ?= 10000
BROAD_EXPORT_DIR ?= artifacts/broad-integer-v1/export
BROAD_BUILD_DIR ?= build/broad
BROAD_VECTORS_DIR ?= build/broad-usb-vectors-v1
BROAD_COUNT ?= 1000
BROAD_EXPECTED_CYCLES ?= 124046
DEMO_LIVE_DIR ?= build/live-session
DEMO_PORT ?= 8765
COUNT ?= 1000
export PATH := $(TOOLCHAIN_DIR)/bin:$(PATH)

.PHONY: doctor test check sim-probe probe-build cfc-build broad-build broad-verify verify-cfc verify-probe verify-usb demo

doctor:
	$(PYTHON) scripts/check_env.py

demo:
	$(PYTHON) -m host.demo --live-dir "$(DEMO_LIVE_DIR)" --port "$(DEMO_PORT)"

probe-build:
	$(PYTHON) scripts/build.py --build-dir "$(BUILD_DIR)"

cfc-build:
	$(PYTHON) scripts/build_cfc.py --export-dir "$(EXPORT_DIR)" --build-dir "$(CFC_BUILD_DIR)"

broad-build:
	$(PYTHON) scripts/build_broad.py --export-dir "$(BROAD_EXPORT_DIR)" --build-dir "$(BROAD_BUILD_DIR)"

broad-verify:
	DYLD_FALLBACK_LIBRARY_PATH="$(TOOLCHAIN_DIR)/lib$(if $(DYLD_FALLBACK_LIBRARY_PATH),:$(DYLD_FALLBACK_LIBRARY_PATH))" $(PYTHON) -m host.verify_broad --usb-serial "$(USB_SERIAL)" --export-dir "$(BROAD_EXPORT_DIR)" --vectors-dir "$(BROAD_VECTORS_DIR)" --count "$(BROAD_COUNT)" --expected-cycles "$(BROAD_EXPECTED_CYCLES)" $(if $(BROAD_OUTPUT),--output "$(BROAD_OUTPUT)")

verify-cfc:
	DYLD_FALLBACK_LIBRARY_PATH="$(TOOLCHAIN_DIR)/lib$(if $(DYLD_FALLBACK_LIBRARY_PATH),:$(DYLD_FALLBACK_LIBRARY_PATH))" $(PYTHON) -m host.verify_cfc --usb-serial "$(USB_SERIAL)" --export-dir "$(EXPORT_DIR)" --vectors-dir "$(CFC_VECTORS_DIR)" $(if $(CFC_DURATION),--duration "$(CFC_DURATION)",--count "$(CFC_COUNT)") $(if $(CFC_OUTPUT),--output "$(CFC_OUTPUT)") $(if $(CFC_LIVE_DIR),--live-dir "$(CFC_LIVE_DIR)")

verify-probe:
	$(PYTHON) -m host.verify_probe --port "$(PORT)" --count "$(COUNT)"

verify-usb:
	DYLD_FALLBACK_LIBRARY_PATH="$(TOOLCHAIN_DIR)/lib$(if $(DYLD_FALLBACK_LIBRARY_PATH),:$(DYLD_FALLBACK_LIBRARY_PATH))" $(PYTHON) -m host.verify_probe --usb-serial "$(USB_SERIAL)" --count "$(COUNT)"

test:
	$(PYTHON) -m unittest discover -s tests -v

sim-probe:
	$(PYTHON) -m unittest discover -s tests/integration -t tests -p 'test_probe_uart*.py' -v

check: test
	$(PYTHON) -m compileall -q scripts host tests
