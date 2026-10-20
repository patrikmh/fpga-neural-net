module probe_top #(parameter integer UART_DIVISOR = 434)(
    input wire clk, rst, uart_rx, output wire uart_tx
);
    reg [7:0] startup = 0;
    always @(posedge clk) begin
        if (startup != 255) startup <= startup + 1;
    end
    wire reset = rst || (startup != 255);
    wire [7:0] rx_data;
    wire rx_valid;
    reg tx_start = 0;
    reg [7:0] tx_data;
    wire tx_busy;
    uart_rx #(.CLKS_PER_BIT(UART_DIVISOR)) receiver(.clk(clk), .rst(reset), .rx(uart_rx), .data(rx_data), .valid(rx_valid));
    uart_tx #(.CLKS_PER_BIT(UART_DIVISOR)) transmitter(.clk(clk), .rst(reset), .start(tx_start), .data(tx_data), .tx(uart_tx), .busy(tx_busy));
    reg [7:0] request [0:8];
    reg [3:0] rx_index = 0;
    reg [7:0] checksum = 0;
    reg [2:0] state = 0;
    wire [7:0] address = request[8];
    wire signed [15:0] a = {request[5], request[4]};
    wire signed [15:0] b = {request[7], request[6]};
    wire signed [31:0] product = a * b;
    reg signed [15:0] rom [0:255];
    reg signed [15:0] rom_value;
    reg signed [31:0] ram [0:255];
    reg signed [31:0] ram_value;
    reg signed [31:0] result;
    initial $readmemh("rtl/probe_rom.hex", rom);
    always @(posedge clk) begin
        rom_value <= rom[address];
        if (state == 1) ram[address] <= product + {{16{rom_value[15]}}, rom_value};
        ram_value <= ram[address];
    end
    wire [7:0] response [0:8];
    assign response[0] = 8'h5a;
    assign response[1] = 8'ha5;
    assign response[2] = request[2];
    assign response[3] = request[3];
    assign response[4] = result[7:0];
    assign response[5] = result[15:8];
    assign response[6] = result[23:16];
    assign response[7] = result[31:24];
    assign response[8] = response[0] ^ response[1] ^ response[2] ^ response[3]
                       ^ response[4] ^ response[5] ^ response[6] ^ response[7];
    reg [3:0] tx_index = 0;
    always @(posedge clk) begin
        tx_start <= 0;
        if (reset) begin
            state <= 0;
            rx_index <= 0;
            checksum <= 0;
            tx_index <= 0;
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
                    end else if (rx_index == 9) begin
                        rx_index <= 0;
                        checksum <= 0;
                        if (checksum == rx_data && request[0] == 8'ha5 && request[1] == 8'h5a)
                            state <= 1;
                    end else begin
                        request[rx_index] <= rx_data;
                        checksum <= checksum ^ rx_data;
                        rx_index <= rx_index + 1;
                    end
                end
                1: state <= 2;
                2: state <= 3;
                3: begin
                    result <= ram_value;
                    tx_index <= 0;
                    state <= 4;
                end
                4: if (!tx_busy && !tx_start) begin
                    tx_data <= response[tx_index];
                    tx_start <= 1;
                    if (tx_index == 8) state <= 0;
                    else tx_index <= tx_index + 1;
                end
            endcase
        end
    end
endmodule
