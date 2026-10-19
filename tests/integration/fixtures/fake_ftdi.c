/* Independent external libftdi device fixture; no production code is included. */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void trace(const char *event);
static int fault(const char *name) {
    const char *selected = getenv("FAKE_FTDI_FAULT");
    return selected && !strcmp(selected, name);
}

struct ftdi_context {
    void *usb_ctx;
    void *usb_dev;
    int usb_read_timeout;
    int usb_write_timeout;
};

struct ftdi_version_info {
    int major;
    int minor;
    int micro;
    const char *version_str;
    const char *snapshot_str;
};

struct ftdi_version_info ftdi_get_library_version(void) {
    struct ftdi_version_info version = {1, 5, 0, "1.5", "fixture"};
    if (fault("abi")) {
        trace("unsupported ABI");
        version.major = 2;
        version.minor = 0;
        version.version_str = "2.0";
    }
    return version;
}

static unsigned configured;
static int stale_sent, drained, sequence, response_offset;
static unsigned char response[9];
static int response_ready;

static void trace(const char *event) {
    FILE *output = fopen(getenv("FAKE_FTDI_TRACE"), "a");
    if (output) { fprintf(output, "%s\n", event); fclose(output); }
}

static int failure(const char *event) { trace(event); return -1; }

struct ftdi_context *ftdi_new(void) {
    struct ftdi_context *ctx = calloc(1, sizeof(*ctx));
    ctx->usb_read_timeout = 5000;
    ctx->usb_write_timeout = 5000;
    trace("new");
    return ctx;
}

int ftdi_set_interface(struct ftdi_context *ctx, int interface) {
    (void)ctx;
    if (interface != 2) return failure("ERROR interface must be B (2)");
    configured |= 1;
    trace("interface B");
    return 0;
}

int ftdi_usb_open_desc(struct ftdi_context *ctx, int vendor, int product,
                       const char *description, const char *serial) {
    (void)ctx; (void)description;
    if (fault("open")) return failure("ERROR injected open failure");
    if (!(configured & 1) || vendor != 0x0403 || product != 0x6010 ||
        !serial || strcmp(serial, "2023112818"))
        return failure("ERROR USB identity or interface ordering");
    configured |= 2;
    trace("open 0403:6010 2023112818");
    return 0;
}

int ftdi_usb_open_desc_index(struct ftdi_context *ctx, int vendor, int product,
                            const char *description, const char *serial, unsigned index) {
    if (index != 0) return failure("ERROR unexpected device index");
    return ftdi_usb_open_desc(ctx, vendor, product, description, serial);
}

int ftdi_set_bitmode(struct ftdi_context *ctx, unsigned char mask, unsigned char mode) {
    (void)ctx; (void)mask;
    if (fault("setup")) return failure("ERROR injected setup failure");
    if (mode != 0) return failure("ERROR bitmode must reset to UART");
    configured |= 4;
    trace("bitmode 0");
    return 0;
}

int ftdi_set_baudrate(struct ftdi_context *ctx, int baudrate) {
    (void)ctx;
    if (baudrate != 460800) return failure("ERROR requested baudrate must be 460800");
    configured |= 8;
    trace("requested baudrate 460800");
    return 0;
}

int ftdi_set_line_property(struct ftdi_context *ctx, int bits, int stop, int parity) {
    (void)ctx;
    if (bits != 8 || stop != 0 || parity != 0) return failure("ERROR UART must be 8N1");
    configured |= 16;
    trace("line 8N1");
    return 0;
}

int ftdi_setflowctrl(struct ftdi_context *ctx, int flow) {
    (void)ctx;
    if (flow != 0) return failure("ERROR flow control must be disabled");
    configured |= 32;
    trace("flow 0");
    return 0;
}

int ftdi_set_latency_timer(struct ftdi_context *ctx, unsigned char latency) {
    (void)ctx;
    if (latency != 2) return failure("ERROR latency must be 2 ms");
    configured |= 64;
    trace("latency 2");
    return 0;
}

int ftdi_usb_purge_buffers(struct ftdi_context *ctx) { (void)ctx; return 0; }
int ftdi_usb_purge_rx_buffer(struct ftdi_context *ctx) { (void)ctx; return 0; }
int ftdi_usb_purge_tx_buffer(struct ftdi_context *ctx) { (void)ctx; return 0; }
int ftdi_usb_reset(struct ftdi_context *ctx) { (void)ctx; return 0; }

static int check_ready(struct ftdi_context *ctx) {
    if (configured != 127) return failure("ERROR incomplete USB configuration");
    if (ctx->usb_read_timeout <= 0 || ctx->usb_read_timeout > 1000 ||
        ctx->usb_write_timeout <= 0 || ctx->usb_write_timeout > 1000)
        return failure("ERROR libftdi timeouts must be positive and bounded");
    return 0;
}

int ftdi_read_data(struct ftdi_context *ctx, unsigned char *data, int size) {
    if (check_ready(ctx) < 0 || size <= 0) return -1;
    if (fault("never_quiet")) {
        if (!stale_sent) trace("never quiet");
        stale_sent = 1;
        data[0] = 0xee;
        return 1;
    }
    if (drained && fault("read")) return failure("ERROR injected read failure");
    if (!drained) {
        if (!stale_sent) {
            data[0] = 0xee;
            stale_sent = 1;
            trace("drain stale");
            return 1;
        }
        drained = 1;
        trace("drain quiet");
        return 0;
    }
    if (!response_ready) return 0;
    if (fault("silent")) return 0;
    int count = 9 - response_offset;
    if (count > 2) count = 2;
    if (count > size) count = size;
    memcpy(data, response + response_offset, count);
    response_offset += count;
    trace("response fragment");
    if (response_offset == 9) response_ready = 0;
    return count;
}

int ftdi_write_data(struct ftdi_context *ctx, const unsigned char *data, int size) {
    static const int expected_a[3] = {-32768, 32767, 0};
    static const int expected_b[3] = {32767, -32768, 32767};
    static const int expected_address[3] = {0, 255, 1};
    if (check_ready(ctx) < 0) return -1;
    if (!drained) return failure("ERROR request sent before stale-data drain");
    if (fault("zero_write")) return 0;
    if (size != 10 || sequence >= 3 || response_ready)
        return failure("ERROR unexpected request size or order");
    unsigned char checksum = 0;
    for (int i = 0; i < size; i++) checksum ^= data[i];
    int seq = data[2] | (data[3] << 8);
    int a = (int16_t)(data[4] | (data[5] << 8));
    int b = (int16_t)(data[6] | (data[7] << 8));
    int address = data[8];
    if (data[0] != 0xa5 || data[1] != 0x5a || checksum || seq != sequence ||
        a != expected_a[sequence] || b != expected_b[sequence] ||
        address != expected_address[sequence])
        return failure("ERROR invalid request vector or framing");
    int rom = ((address * 4051 + 17389) & 65535) - 32768;
    uint32_t result = (uint32_t)(a * b + rom);
    response[0] = 0x5a; response[1] = 0xa5;
    response[2] = (unsigned char)seq; response[3] = (unsigned char)(seq >> 8);
    for (int i = 0; i < 4; i++) response[4 + i] = (unsigned char)(result >> (8 * i));
    response[8] = 0;
    for (int i = 0; i < 8; i++) response[8] ^= response[i];
    response_offset = 0;
    response_ready = 1;
    sequence++;
    trace("request verified");
    return size;
}

int ftdi_usb_close(struct ftdi_context *ctx) { (void)ctx; trace("close"); return 0; }
void ftdi_free(struct ftdi_context *ctx) { trace("free"); free(ctx); }
const char *ftdi_get_error_string(struct ftdi_context *ctx) {
    (void)ctx; return "External libftdi fixture rejected the operation; inspect trace";
}
