module cfc_top #(
    parameter WEIGHTS_FILE = "build/integer-v1/weights.hex",
    parameter TANH_FILE = "build/integer-v1/tanh.hex",
    parameter SIGMOID_FILE = "build/integer-v1/sigmoid.hex"
)(input wire clk, rst, uart_rx, output wire uart_tx);
    reg [7:0] startup = 0;
    always @(posedge clk) if (startup != 255) startup <= startup + 1;
    wire reset = rst || (startup != 255);
    wire [7:0] rx_data;
    wire rx_valid, tx_busy;
    reg tx_start = 0;
    reg [7:0] tx_data;
    uart_rx receiver(.clk(clk), .rst(reset), .rx(uart_rx), .data(rx_data), .valid(rx_valid));
    uart_tx transmitter(.clk(clk), .rst(reset), .start(tx_start), .data(tx_data), .tx(uart_tx), .busy(tx_busy));
    reg core_start = 0;
    wire core_busy, core_done;
    wire signed [15:0] core_y;
    wire [127:0] core_hidden;
    wire [31:0] core_cycles;
    reg [7:0] request [0:6];
    cfc_core #(.WEIGHTS_FILE(WEIGHTS_FILE), .TANH_FILE(TANH_FILE), .SIGMOID_FILE(SIGMOID_FILE))
        core(.clk(clk), .rst(reset), .start(core_start), .sample({request[6],request[5]}),
             .busy(core_busy), .done(core_done), .y(core_y), .hidden(core_hidden), .cycles(core_cycles));
    wire [215:0] payload = {core_cycles, core_hidden, core_y, 8'h00, request[3], request[2], 16'ha55a};
    reg [7:0] response_checksum;
    integer j;
    always @* begin
        response_checksum = 0;
        for (j = 0; j < 27; j = j + 1)
            response_checksum = response_checksum ^ payload[j*8 +: 8];
    end
    reg [2:0] rx_index = 0;
    reg [4:0] tx_index = 0;
    reg [7:0] checksum = 0;
    reg [1:0] state = 0;
    always @(posedge clk) begin
        tx_start <= 0;
        core_start <= 0;
        if (reset) begin
            rx_index <= 0;
            tx_index <= 0;
            checksum <= 0;
            state <= 0;
        end else begin
            case (state)
                0: if (rx_valid) begin
                    if (rx_index == 0) begin
                        if (rx_data == 8'ha5) begin
                            request[0] <= rx_data;
                            checksum <= rx_data;
                            rx_index <= 1;
                        end
                    end else if (rx_index == 1) begin
                        if (rx_data == 8'h5a) begin
                            request[1] <= rx_data;
                            checksum <= checksum ^ rx_data;
                            rx_index <= 2;
                        end else if (rx_data != 8'ha5) begin
                            rx_index <= 0;
                            checksum <= 0;
                        end
                    end else if (rx_index == 7) begin
                        rx_index <= 0;
                        checksum <= 0;
                        if (checksum == rx_data && request[4] == 1) begin
                            core_start <= 1;
                            state <= 1;
                        end
                    end else begin
                        request[rx_index] <= rx_data;
                        checksum <= checksum ^ rx_data;
                        rx_index <= rx_index + 1;
                    end
                end
                1: if (core_done) begin
                    tx_index <= 0;
                    state <= 2;
                end
                2: if (!tx_busy && !tx_start) begin
                    tx_data <= tx_index == 27 ? response_checksum : payload[tx_index*8 +: 8];
                    tx_start <= 1;
                    if (tx_index == 27) state <= 0;
                    else tx_index <= tx_index + 1;
                end
            endcase
        end
    end
endmodule
