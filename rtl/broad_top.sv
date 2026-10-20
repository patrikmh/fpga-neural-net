module broad_top #(
    parameter integer HIDDEN_SIZE = 96,
    parameter WEIGHTS_DIR = "build/broad-rtl-v1/rom",
    parameter TANH_FILE = "build/broad-rtl-v1/rom/tanh.hex",
    parameter SIGMOID_FILE = "build/broad-rtl-v1/rom/sigmoid.hex"
)(input wire clk, rst, uart_rx, output wire uart_tx);
    localparam integer PAYLOAD_BYTES = 16 + 2 * HIDDEN_SIZE;
    reg [7:0] startup = 0;
    always @(posedge clk) if (startup != 255) startup <= startup + 1;
    wire reset = rst || (startup != 255);
    wire [7:0] rx_data;
    wire rx_valid, tx_busy;
    reg tx_start = 0;
    reg [7:0] tx_data;
    uart_rx receiver(.clk(clk), .rst(reset), .rx(uart_rx), .data(rx_data), .valid(rx_valid));
    uart_tx transmitter(.clk(clk), .rst(reset), .start(tx_start), .data(tx_data), .tx(uart_tx), .busy(tx_busy));
    reg [7:0] request [0:21];
    wire [143:0] samples;
    genvar i;
    generate for (i = 0; i < 9; i = i + 1) begin: input_lanes
        assign samples[i*16 +: 16] = {request[5+i*2], request[4+i*2]};
    end endgenerate
    reg core_start = 0, core_reset = 0;
    wire core_busy, core_done;
    wire [HIDDEN_SIZE*16-1:0] core_hidden;
    wire [63:0] core_outputs;
    wire [31:0] core_cycles;
    broad_core #(.HIDDEN_SIZE(HIDDEN_SIZE), .WEIGHTS_DIR(WEIGHTS_DIR),
                 .TANH_FILE(TANH_FILE), .SIGMOID_FILE(SIGMOID_FILE))
        core(.clk(clk), .reset(reset || core_reset), .start(core_start), .input_values(samples),
             .hidden_values(core_hidden), .output_values(core_outputs), .busy(core_busy),
             .done(core_done), .cycles(core_cycles));
    wire [PAYLOAD_BYTES*8-1:0] payload = {core_cycles, core_outputs, core_hidden,
                                        request[3], request[2], 8'h00, 8'ha6};
    function automatic [15:0] crc_byte(input [15:0] previous, input [7:0] value);
        reg [15:0] crc;
        integer bit_number;
        begin
            crc = previous ^ {value, 8'h00};
            for (bit_number = 0; bit_number < 8; bit_number = bit_number + 1)
                crc = crc[15] ? (crc << 1) ^ 16'h1021 : crc << 1;
            crc_byte = crc;
        end
    endfunction
    integer rx_index = 0, tx_index = 0;
    reg [15:0] rx_crc = 16'hffff, tx_crc = 16'hffff;
    reg [7:0] crc_low;
    reg [2:0] state = 0;
    always @(posedge clk) begin
        tx_start <= 0;
        core_start <= 0;
        core_reset <= 0;
        if (reset) begin
            rx_index <= 0;
            tx_index <= 0;
            rx_crc <= 16'hffff;
            tx_crc <= 16'hffff;
            state <= 0;
        end else begin
            case (state)
                0: if (rx_valid) begin
                    if (rx_index == 0) begin
                        if (rx_data == 8'ha6) begin
                            request[0] <= rx_data;
                            rx_crc <= crc_byte(16'hffff, rx_data);
                            rx_index <= 1;
                        end
                    end else if (rx_index < 22) begin
                        request[rx_index] <= rx_data;
                        rx_crc <= crc_byte(rx_crc, rx_data);
                        rx_index <= rx_index + 1;
                    end else if (rx_index == 22) begin
                        crc_low <= rx_data;
                        rx_index <= 23;
                    end else begin
                        rx_index <= 0;
                        if ({rx_data, crc_low} == rx_crc) begin
                            if (request[1] == 1) begin
                                core_start <= 1;
                                state <= 1;
                            end else if (request[1] == 2) begin
                                core_reset <= 1;
                                state <= 3;
                            end
                        end
                    end
                end
                1: if (core_done) begin
                    tx_index <= 0;
                    tx_crc <= 16'hffff;
                    state <= 2;
                end
                2: if (!tx_busy && !tx_start) begin
                    tx_start <= 1;
                    if (tx_index < PAYLOAD_BYTES) begin
                        tx_data <= payload[tx_index*8 +: 8];
                        tx_crc <= crc_byte(tx_crc, payload[tx_index*8 +: 8]);
                        tx_index <= tx_index + 1;
                    end else if (tx_index == PAYLOAD_BYTES) begin
                        tx_data <= tx_crc[7:0];
                        tx_index <= tx_index + 1;
                    end else begin
                        tx_data <= tx_crc[15:8];
                        state <= 0;
                    end
                end
                3: begin
                    tx_index <= 0;
                    tx_crc <= 16'hffff;
                    state <= 2;
                end
                default: state <= 0;
            endcase
        end
    end
endmodule
