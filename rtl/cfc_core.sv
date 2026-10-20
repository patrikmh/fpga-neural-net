module cfc_core #(
    parameter WEIGHTS_FILE = "build/integer-v1/weights.hex",
    parameter TANH_FILE = "build/integer-v1/tanh.hex",
    parameter SIGMOID_FILE = "build/integer-v1/sigmoid.hex"
)(
    input wire clk, rst, start,
    input wire signed [15:0] sample,
    output reg busy = 0,
    output reg done = 0,
    output reg signed [15:0] y,
    output wire [127:0] hidden,
    output reg [31:0] cycles
);
    reg signed [15:0] weights [0:328];
    reg signed [15:0] tanh_rom [0:4095];
    reg signed [15:0] sigmoid_rom [0:4095];
    reg [8:0] weight_address;
    reg [11:0] tanh_address, sigmoid_address;
    reg signed [15:0] weight_q, tanh_q, sigmoid_q;
    initial begin
        $readmemh(WEIGHTS_FILE, weights);
        $readmemh(TANH_FILE, tanh_rom);
        $readmemh(SIGMOID_FILE, sigmoid_rom);
    end
    always @(posedge clk) begin
        weight_q <= weights[weight_address];
        tanh_q <= tanh_rom[tanh_address];
        sigmoid_q <= sigmoid_rom[sigmoid_address];
    end

    reg signed [15:0] old_h [0:7];
    reg signed [15:0] new_h [0:7];
    reg signed [47:0] heads [0:31];
    reg signed [15:0] x, f1, f2;
    reg signed [15:0] mul_a, mul_b;
    wire signed [31:0] product = mul_a * mul_b;
    wire signed [47:0] extended_product = {{16{product[31]}}, product};
    wire signed [47:0] extended_weight = {{32{weight_q[15]}}, weight_q};
    wire signed [47:0] extended_f1 = {{32{f1[15]}}, f1};
    reg signed [47:0] acc;
    reg [1:0] head;
    reg [2:0] row, neuron;
    reg [3:0] column;
    reg [8:0] row_base;
    genvar h;
    generate for (h = 0; h < 8; h = h + 1) begin
        assign hidden[h*16 +: 16] = old_h[h];
    end endgenerate

    function automatic signed [47:0] round_shift(input signed [47:0] value, input integer bits);
        reg [47:0] magnitude;
        begin
            magnitude = value < 0 ? -value : value;
            magnitude = (magnitude + (48'd1 << (bits - 1))) >> bits;
            round_shift = value < 0 ? -$signed(magnitude) : $signed(magnitude);
        end
    endfunction

    function automatic [11:0] lookup_index(input signed [47:0] value);
        reg signed [47:0] index_value;
        begin
            index_value = round_shift(value, 18) + 48'sd2048;
            if (index_value < 0) lookup_index = 0;
            else if (index_value > 4095) lookup_index = 4095;
            else lookup_index = index_value[11:0];
        end
    endfunction

    function automatic signed [15:0] sat16(input signed [47:0] value);
        begin
            if (value > 32767) sat16 = 16'sh7fff;
            else if (value < -32768) sat16 = 16'sh8000;
            else sat16 = value[15:0];
        end
    endfunction

    localparam IDLE=0, BIAS_WAIT=1, BIAS_USE=2, WEIGHT_WAIT=3,
        WEIGHT_LOAD=4, MAC=5, AFFINE_DONE=6, NEURON=7,
        F1_WAIT=8, F1_READ=9, F2_WAIT=10, F2_READ=11,
        GATE_WAIT=12, GATE_READ=13, BLEND=14,
        OUT_BIAS_WAIT=15, OUT_BIAS_USE=16, OUT_WEIGHT_WAIT=17,
        OUT_WEIGHT_LOAD=18, OUT_MAC=19, COMPLETE=20;
    reg [4:0] state = IDLE;
    integer i;
    always @(posedge clk) begin
        done <= 0;
        if (rst) begin
            state <= IDLE;
            busy <= 0;
            y <= 0;
            cycles <= 0;
            for (i = 0; i < 8; i = i + 1) begin
                old_h[i] <= 0;
                new_h[i] <= 0;
            end
        end else begin
            if (busy) cycles <= cycles + 1;
            case (state)
                IDLE: if (start) begin
                    x <= sample;
                    busy <= 1;
                    cycles <= 1;
                    head <= 0;
                    row <= 0;
                    row_base <= 0;
                    weight_address <= 9;
                    state <= BIAS_WAIT;
                end
                BIAS_WAIT: state <= BIAS_USE;
                BIAS_USE: begin
                    acc <= extended_weight <<< 13;
                    column <= 0;
                    weight_address <= row_base;
                    state <= WEIGHT_WAIT;
                end
                WEIGHT_WAIT: state <= WEIGHT_LOAD;
                WEIGHT_LOAD: begin
                    mul_a <= column == 0 ? x : old_h[column - 1];
                    mul_b <= weight_q;
                    state <= MAC;
                end
                MAC: begin
                    acc <= acc + extended_product;
                    if (column == 8) state <= AFFINE_DONE;
                    else begin
                        column <= column + 1;
                        weight_address <= row_base + column + 1;
                        state <= WEIGHT_WAIT;
                    end
                end
                AFFINE_DONE: begin
                    heads[{head,row}] <= acc;
                    if (head == 3 && row == 7) begin
                        neuron <= 0;
                        state <= NEURON;
                    end else begin
                        if (row == 7) begin
                            head <= head + 1;
                            row <= 0;
                        end else row <= row + 1;
                        row_base <= row_base + 10;
                        weight_address <= row_base + 19;
                        state <= BIAS_WAIT;
                    end
                end
                NEURON: begin
                    tanh_address <= lookup_index(heads[{2'b00,neuron}]);
                    state <= F1_WAIT;
                end
                F1_WAIT: state <= F1_READ;
                F1_READ: begin
                    f1 <= tanh_q;
                    tanh_address <= lookup_index(heads[{2'b01,neuron}]);
                    state <= F2_WAIT;
                end
                F2_WAIT: state <= F2_READ;
                F2_READ: begin
                    f2 <= tanh_q;
                    sigmoid_address <= lookup_index(heads[{2'b10,neuron}] + heads[{2'b11,neuron}]);
                    state <= GATE_WAIT;
                end
                GATE_WAIT: state <= GATE_READ;
                GATE_READ: begin
                    mul_a <= f2 - f1;
                    mul_b <= sigmoid_q;
                    state <= BLEND;
                end
                BLEND: begin
                    new_h[neuron] <= sat16(extended_f1 + round_shift(extended_product, 13));
                    if (neuron == 7) begin
                        weight_address <= 328;
                        state <= OUT_BIAS_WAIT;
                    end else begin
                        neuron <= neuron + 1;
                        state <= NEURON;
                    end
                end
                OUT_BIAS_WAIT: state <= OUT_BIAS_USE;
                OUT_BIAS_USE: begin
                    acc <= extended_weight <<< 13;
                    column <= 0;
                    weight_address <= 320;
                    state <= OUT_WEIGHT_WAIT;
                end
                OUT_WEIGHT_WAIT: state <= OUT_WEIGHT_LOAD;
                OUT_WEIGHT_LOAD: begin
                    mul_a <= new_h[column];
                    mul_b <= weight_q;
                    state <= OUT_MAC;
                end
                OUT_MAC: begin
                    acc <= acc + extended_product;
                    if (column == 7) state <= COMPLETE;
                    else begin
                        column <= column + 1;
                        weight_address <= 320 + column + 1;
                        state <= OUT_WEIGHT_WAIT;
                    end
                end
                COMPLETE: begin
                    y <= sat16(round_shift(acc, 13));
                    for (i = 0; i < 8; i = i + 1) old_h[i] <= new_h[i];
                    busy <= 0;
                    done <= 1;
                    state <= IDLE;
                end
            endcase
        end
    end
endmodule
