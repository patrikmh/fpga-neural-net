/* External libftdi ABI/device fixture. Requests and replies are independent data. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

struct ftdi_context { void *usb_ctx, *usb_dev; int usb_read_timeout, usb_write_timeout; };
struct ftdi_version_info { int major, minor, micro; const char *version_str, *snapshot_str; };
static FILE *requests, *replies;
static unsigned configured;
static int stale, drained, offset = 28, command, delayed;
static unsigned char response[28];

static void trace(const char *event) {
    FILE *output = fopen(getenv("FAKE_CFC_TRACE"), "a");
    if (output) { fprintf(output, "%s\n", event); fclose(output); }
}
static int fail(const char *event) { trace(event); return -1; }
struct ftdi_version_info ftdi_get_library_version(void) {
    struct ftdi_version_info version = {1, 5, 0, "1.5", "cfc-fixture"};
    return version;
}
struct ftdi_context *ftdi_new(void) {
    requests = fopen(getenv("FAKE_CFC_REQUESTS"), "rb");
    replies = fopen(getenv("FAKE_CFC_REPLIES"), "rb");
    if (!requests || !replies) return NULL;
    struct ftdi_context *ctx = calloc(1, sizeof(*ctx));
    ctx->usb_read_timeout = ctx->usb_write_timeout = 5000;
    trace("new");
    return ctx;
}
int ftdi_set_interface(struct ftdi_context *ctx, int interface) {
    (void)ctx;
    if (interface != 2) return fail("ERROR interface must be B");
    configured |= 1; trace("interface B"); return 0;
}
int ftdi_usb_open_desc(struct ftdi_context *ctx, int vendor, int product,
                       const char *description, const char *serial) {
    (void)ctx; (void)description;
    if (!(configured & 1) || vendor != 0x0403 || product != 0x6010 ||
        !serial || strcmp(serial, "2023112818")) return fail("ERROR USB selection");
    configured |= 2; trace("open 0403:6010 2023112818"); return 0;
}
int ftdi_set_bitmode(struct ftdi_context *ctx, unsigned char mask, unsigned char mode) {
    (void)ctx; (void)mask;
    if (mode) return fail("ERROR bitmode");
    configured |= 4; trace("bitmode 0"); return 0;
}
int ftdi_set_baudrate(struct ftdi_context *ctx, int baudrate) {
    (void)ctx;
    if (baudrate != 460800) return fail("ERROR USB baudrate");
    configured |= 8; trace("requested baudrate 460800"); return 0;
}
int ftdi_set_line_property(struct ftdi_context *ctx, int bits, int stop, int parity) {
    (void)ctx;
    if (bits != 8 || stop || parity) return fail("ERROR 8N1");
    configured |= 16; trace("line 8N1"); return 0;
}
int ftdi_setflowctrl(struct ftdi_context *ctx, int flow) {
    (void)ctx;
    if (flow) return fail("ERROR flow control");
    configured |= 32; trace("flow 0"); return 0;
}
int ftdi_set_latency_timer(struct ftdi_context *ctx, unsigned char latency) {
    (void)ctx;
    if (latency != 2) return fail("ERROR latency");
    configured |= 64; trace("latency 2"); return 0;
}
static int ready(struct ftdi_context *ctx) {
    if (configured != 127) return fail("ERROR incomplete USB setup");
    if (ctx->usb_read_timeout <= 0 || ctx->usb_read_timeout > 1000 ||
        ctx->usb_write_timeout <= 0 || ctx->usb_write_timeout > 1000)
        return fail("ERROR native timeout bounds");
    return 0;
}
int ftdi_read_data(struct ftdi_context *ctx, unsigned char *data, int size) {
    if (ready(ctx) < 0 || size <= 0) return -1;
    if (!drained) {
        if (!stale) { stale = 1; data[0] = 0xee; trace("drain stale"); return 1; }
        drained = 1; trace("drain quiet"); return 0;
    }
    if (offset == 28) return 0;
    if (!delayed) {
        const char *delay = getenv("FAKE_CFC_DELAY_MS");
        long milliseconds = delay ? strtol(delay, NULL, 10) : 0;
        if (milliseconds > 0 && milliseconds <= 1000) {
            struct timespec pause = {milliseconds / 1000, (milliseconds % 1000) * 1000000};
            nanosleep(&pause, NULL);
        }
        delayed = 1;
    }
    const char *fault = getenv("FAKE_CFC_FAULT");
    if (command == 1 && fault && !strcmp(fault, "silent")) return 0;
    if (command == 1 && fault && !strcmp(fault, "short") && offset >= 7) return 0;
    int count = 28 - offset;
    if (count > 3) count = 3;
    if (command == 1 && fault && !strcmp(fault, "short") && count > 7 - offset)
        count = 7 - offset;
    if (count > size) count = size;
    memcpy(data, response + offset, count); offset += count;
    trace("response fragment"); return count;
}
int ftdi_write_data(struct ftdi_context *ctx, const unsigned char *data, int size) {
    unsigned char expected[8];
    if (ready(ctx) < 0) return -1;
    if (!drained || size != 8 || offset != 28) return fail("ERROR request ordering or length");
    if (fread(expected, 1, 8, requests) != 8 || memcmp(expected, data, 8))
        return fail("ERROR request differs from independent expected frame");
    if (fread(response, 1, 28, replies) != 28) return fail("ERROR missing reply fixture");
    offset = 0; command = data[4]; delayed = 0;
    trace(command == 2 ? "request RESET" : "request STEP"); return size;
}
int ftdi_usb_close(struct ftdi_context *ctx) { (void)ctx; trace("close"); return 0; }
void ftdi_free(struct ftdi_context *ctx) {
    if (requests) fclose(requests);
    if (replies) fclose(replies);
    trace("free"); free(ctx);
}
const char *ftdi_get_error_string(struct ftdi_context *ctx) {
    (void)ctx; return "External CfC fixture rejected operation; inspect trace";
}
