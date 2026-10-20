module broad_core #(
    parameter integer HIDDEN_SIZE = 96,
    parameter WEIGHTS_DIR = "build/broad-rtl-v1/rom",
    parameter TANH_FILE = "build/broad-rtl-v1/rom/tanh.hex",
    parameter SIGMOID_FILE = "build/broad-rtl-v1/rom/sigmoid.hex"
)(
    input wire clk, reset, start,
    input wire [143:0] input_values,
    output wire [HIDDEN_SIZE*16-1:0] hidden_values,
    output reg [63:0] output_values,
    output reg busy = 0,
    output reg done = 0,
    output reg [31:0] cycles
);
    localparam integer ROW_WORDS = HIDDEN_SIZE + 10;
    localparam integer HEAD_WORDS = HIDDEN_SIZE * ROW_WORDS;
    localparam integer OUTPUT_BASE = 4 * HEAD_WORDS;
    localparam integer WEIGHT_COUNT = OUTPUT_BASE + 4 * (HIDDEN_SIZE + 1);
    localparam integer BANK_COUNT = (WEIGHT_COUNT + 1023) / 1024;
    localparam integer ADDRESS_WIDTH = $clog2(WEIGHT_COUNT) > 10 ? $clog2(WEIGHT_COUNT) : 11;
    localparam integer BANK_WIDTH = BANK_COUNT > 1 ? $clog2(BANK_COUNT) : 1;
    localparam integer INDEX_WIDTH = HIDDEN_SIZE > 1 ? $clog2(HIDDEN_SIZE) : 1;
    localparam integer COLUMN_WIDTH = $clog2(HIDDEN_SIZE + 9);

    reg [ADDRESS_WIDTH-1:0] weight_address;
    reg [BANK_WIDTH-1:0] selected_bank;
    wire signed [15:0] bank_output [0:BANK_COUNT-1];
    genvar bank;
    generate for (bank = 0; bank < BANK_COUNT; bank = bank + 1) begin : weight_banks
        reg signed [15:0] memory [0:1023];
        reg signed [15:0] value;
        initial $readmemh($sformatf("%s/weights_%03d.hex", WEIGHTS_DIR, bank), memory);
        always @(posedge clk) value <= memory[weight_address[9:0]];
        assign bank_output[bank] = value;
    end endgenerate
    always @(posedge clk) selected_bank <= BANK_WIDTH'(weight_address >> 10);
    wire signed [15:0] weight_q = bank_output[selected_bank];

    reg signed [15:0] tanh_rom [0:4095];
    reg signed [15:0] sigmoid_rom [0:4095];
    reg [11:0] tanh_address, sigmoid_address;
    reg signed [15:0] tanh_q, sigmoid_q;
    initial begin
        $readmemh(TANH_FILE, tanh_rom);
        $readmemh(SIGMOID_FILE, sigmoid_rom);
    end
    always @(posedge clk) begin
        tanh_q <= tanh_rom[tanh_address];
        sigmoid_q <= sigmoid_rom[sigmoid_address];
    end

    reg signed [15:0] samples [0:8];
    reg signed [15:0] old_hidden [0:HIDDEN_SIZE-1];
    reg signed [15:0] new_hidden [0:HIDDEN_SIZE-1];
    reg signed [15:0] raw_outputs [0:3];
    reg signed [47:0] ff1_acc, ff2_acc, time_a_acc, time_b_acc;
    reg signed [47:0] gate_sum;
    reg signed [47:0] accumulator;
    reg signed [15:0] f1, f2, mul_a, mul_b;
    wire signed [31:0] product = mul_a * mul_b;
    wire signed [47:0] extended_product = {{16{product[31]}}, product};
    wire signed [47:0] extended_weight = {{32{weight_q[15]}}, weight_q};
    wire signed [47:0] extended_f1 = {{32{f1[15]}}, f1};
    reg [INDEX_WIDTH-1:0] neuron;
    reg [COLUMN_WIDTH-1:0] column;
    reg [1:0] head, output_row;
    reg [ADDRESS_WIDTH-1:0] row_base, neuron_base;
    wire [INDEX_WIDTH-1:0] recurrent_column = INDEX_WIDTH'(column - COLUMN_WIDTH'(9));
    genvar lane;
    generate for (lane = 0; lane < HIDDEN_SIZE; lane = lane + 1) begin : hidden_lanes
        assign hidden_values[lane*16 +:16] = old_hidden[lane];
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
            else if (index_value > 48'sd4095) lookup_index = 4095;
            else lookup_index = index_value[11:0];
        end
    endfunction
    function automatic signed [15:0] sat16(input signed [47:0] value);
        begin
            if (value > 48'sd32767) sat16 = 16'sh7fff;
            else if (value < -48'sd32768) sat16 = 16'sh8000;
            else sat16 = value[15:0];
        end
    endfunction

    localparam IDLE=0, BIAS_WAIT=1, BIAS_USE=2, WEIGHT_WAIT=3,
        WEIGHT_LOAD=4, MAC=5, HEAD_DONE=6, ACTIVATE=7,
        F1_WAIT=8, F1_READ=9, F2_WAIT=10, F2_READ=11,
        GATE_WAIT=12, GATE_READ=13, BLEND=14,
        OUT_BIAS_WAIT=15, OUT_BIAS_USE=16, OUT_WEIGHT_WAIT=17,
        OUT_WEIGHT_LOAD=18, OUT_MAC=19, OUT_DONE=20, COMPLETE=21;
    reg [4:0] state = IDLE;
    wire signed [15:0] blend_value = sat16(extended_f1 + round_shift(extended_product, 13));
    genvar write_lane;
    generate for (write_lane = 0; write_lane < HIDDEN_SIZE; write_lane = write_lane + 1) begin : hidden_writes
        always @(posedge clk) begin
            if (reset) new_hidden[write_lane] <= 0;
            else if (state == BLEND && neuron == INDEX_WIDTH'(write_lane))
                new_hidden[write_lane] <= blend_value;
        end
    end endgenerate
    integer i;
    always @(posedge clk) begin
        done <= 0;
        if (reset) begin
            state <= IDLE;
            busy <= 0;
            cycles <= 0;
            output_values <= 0;
            for (i = 0; i < HIDDEN_SIZE; i = i + 1) begin
                old_hidden[i] <= 0;
            end
        end else begin
            if (busy) cycles <= cycles + 32'd1;
            case (state)
                IDLE: if (start) begin
                    for (i = 0; i < 9; i = i + 1) samples[i] <= input_values[i*16 +:16];
                    neuron <= 0;
                    head <= 0;
                    neuron_base <= 0;
                    row_base <= 0;
                    weight_address <= ADDRESS_WIDTH'(HIDDEN_SIZE + 9);
                    busy <= 1;
                    cycles <= 1;
                    state <= BIAS_WAIT;
                end
                BIAS_WAIT: state <= BIAS_USE;
                BIAS_USE: begin
                    accumulator <= extended_weight <<< 13;
                    column <= 0;
                    weight_address <= row_base;
                    state <= WEIGHT_WAIT;
                end
                WEIGHT_WAIT: state <= WEIGHT_LOAD;
                WEIGHT_LOAD: begin
                    mul_a <= column < COLUMN_WIDTH'(9) ? samples[4'(column)] : old_hidden[recurrent_column];
                    mul_b <= weight_q;
                    state <= MAC;
                end
                MAC: begin
                    accumulator <= accumulator + (column < COLUMN_WIDTH'(9) ? (extended_product <<< 2) : extended_product);
                    if (column == COLUMN_WIDTH'(HIDDEN_SIZE + 8)) state <= HEAD_DONE;
                    else begin
                        column <= column + COLUMN_WIDTH'(1);
                        weight_address <= row_base + ADDRESS_WIDTH'(column) + ADDRESS_WIDTH'(1);
                        state <= WEIGHT_WAIT;
                    end
                end
                HEAD_DONE: begin
                    case (head)
                        0: ff1_acc <= accumulator;
                        1: ff2_acc <= accumulator;
                        2: time_a_acc <= accumulator;
                        3: time_b_acc <= accumulator;
                    endcase
                    if (head == 3) state <= ACTIVATE;
                    else begin
                        head <= head + 2'd1;
                        row_base <= row_base + ADDRESS_WIDTH'(HEAD_WORDS);
                        weight_address <= row_base + ADDRESS_WIDTH'(HEAD_WORDS + HIDDEN_SIZE + 9);
                        state <= BIAS_WAIT;
                    end
                end
                ACTIVATE: begin
                    gate_sum <= time_a_acc + time_b_acc;
                    tanh_address <= lookup_index(ff1_acc);
                    state <= F1_WAIT;
                end
                F1_WAIT: state <= F1_READ;
                F1_READ: begin
                    f1 <= tanh_q;
                    tanh_address <= lookup_index(ff2_acc);
                    state <= F2_WAIT;
                end
                F2_WAIT: state <= F2_READ;
                F2_READ: begin
                    f2 <= tanh_q;
                    sigmoid_address <= lookup_index(gate_sum);
                    state <= GATE_WAIT;
                end
                GATE_WAIT: state <= GATE_READ;
                GATE_READ: begin
                    mul_a <= f2 - f1;
                    mul_b <= sigmoid_q;
                    state <= BLEND;
                end
                BLEND: begin
                    if (neuron == INDEX_WIDTH'(HIDDEN_SIZE - 1)) begin
                        output_row <= 0;
                        row_base <= ADDRESS_WIDTH'(OUTPUT_BASE);
                        weight_address <= ADDRESS_WIDTH'(OUTPUT_BASE + HIDDEN_SIZE);
                        state <= OUT_BIAS_WAIT;
                    end else begin
                        neuron <= neuron + INDEX_WIDTH'(1);
                        neuron_base <= neuron_base + ADDRESS_WIDTH'(ROW_WORDS);
                        row_base <= neuron_base + ADDRESS_WIDTH'(ROW_WORDS);
                        weight_address <= neuron_base + ADDRESS_WIDTH'(ROW_WORDS + HIDDEN_SIZE + 9);
                        head <= 0;
                        state <= BIAS_WAIT;
                    end
                end
                OUT_BIAS_WAIT: state <= OUT_BIAS_USE;
                OUT_BIAS_USE: begin
                    accumulator <= extended_weight <<< 13;
                    column <= 0;
                    weight_address <= row_base;
                    state <= OUT_WEIGHT_WAIT;
                end
                OUT_WEIGHT_WAIT: state <= OUT_WEIGHT_LOAD;
                OUT_WEIGHT_LOAD: begin
                    mul_a <= new_hidden[INDEX_WIDTH'(column)];
                    mul_b <= weight_q;
                    state <= OUT_MAC;
                end
                OUT_MAC: begin
                    accumulator <= accumulator + extended_product;
                    if (column == COLUMN_WIDTH'(HIDDEN_SIZE - 1)) state <= OUT_DONE;
                    else begin
                        column <= column + COLUMN_WIDTH'(1);
                        weight_address <= row_base + ADDRESS_WIDTH'(column) + ADDRESS_WIDTH'(1);
                        state <= OUT_WEIGHT_WAIT;
                    end
                end
                OUT_DONE: begin
                    raw_outputs[output_row] <= sat16(round_shift(accumulator, 13));
                    if (output_row == 3) state <= COMPLETE;
                    else begin
                        output_row <= output_row + 2'd1;
                        row_base <= row_base + ADDRESS_WIDTH'(HIDDEN_SIZE + 1);
                        weight_address <= row_base + ADDRESS_WIDTH'(2 * HIDDEN_SIZE + 1);
                        state <= OUT_BIAS_WAIT;
                    end
                end
                COMPLETE: begin
                    for (i = 0; i < HIDDEN_SIZE; i = i + 1) old_hidden[i] <= new_hidden[i];
                    for (i = 0; i < 4; i = i + 1) output_values[i*16 +:16] <= raw_outputs[i];
                    busy <= 0;
                    done <= 1;
                    state <= IDLE;
                end
                default: state <= IDLE;
            endcase
        end
    end
endmodule
