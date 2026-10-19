PYTHON ?= .venv/bin/python
TOOLCHAIN_DIR ?= $(CURDIR)/.tools/tang-toolchain/oss-cad-suite
BUILD_DIR ?= build/probe
COUNT ?= 1000
export PATH := $(TOOLCHAIN_DIR)/bin:$(PATH)

.PHONY: doctor test check sim-probe probe-build verify-probe verify-usb

doctor:
	$(PYTHON) scripts/check_env.py

probe-build:
	$(PYTHON) scripts/build.py --build-dir "$(BUILD_DIR)"

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
