module uart_rx #(parameter integer CLKS_PER_BIT = 434)(
    input wire clk, rst, rx,
    output reg [7:0] data, output reg valid
);
    reg sync1 = 1, sync2 = 1;
    reg [1:0] state = 0;
    integer count = 0;
    reg [2:0] bit_index = 0;
    always @(posedge clk) begin
        sync1 <= rx;
        sync2 <= sync1;
        valid <= 0;
        if (rst) begin
            state <= 0;
            count <= 0;
            data <= 0;
        end else begin
            case (state)
                0: if (!sync2) begin
                    count <= CLKS_PER_BIT / 2 - 1;
                    state <= 1;
                end
                1: if (count != 0) count <= count - 1;
                   else if (sync2) state <= 0;
                   else begin
                       count <= CLKS_PER_BIT - 1;
                       bit_index <= 0;
                       state <= 2;
                   end
                2: if (count != 0) count <= count - 1;
                   else begin
                       data[bit_index] <= sync2;
                       count <= CLKS_PER_BIT - 1;
                       if (bit_index == 7) state <= 3;
                       else bit_index <= bit_index + 1;
                   end
                3: if (count != 0) count <= count - 1;
                   else begin
                       valid <= sync2;
                       state <= 0;
                   end
            endcase
        end
    end
endmodule

module uart_tx #(parameter integer CLKS_PER_BIT = 434)(
    input wire clk, rst, start, input wire [7:0] data,
    output reg tx = 1, output reg busy = 0
);
    reg [9:0] frame;
    reg [3:0] bit_index;
    integer count;
    always @(posedge clk) begin
        if (rst) begin
            tx <= 1;
            busy <= 0;
            count <= 0;
            bit_index <= 0;
        end else if (!busy) begin
            if (start) begin
                frame <= {1'b1, data, 1'b0};
                tx <= 0;
                busy <= 1;
                bit_index <= 0;
                count <= CLKS_PER_BIT - 1;
            end
        end else if (count != 0) count <= count - 1;
        else if (bit_index == 9) begin
            busy <= 0;
            tx <= 1;
        end else begin
            bit_index <= bit_index + 1;
            tx <= frame[bit_index + 1];
            count <= CLKS_PER_BIT - 1;
        end
    end
endmodule
