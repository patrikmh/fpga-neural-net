`timescale 1ns/1ps

module probe_uart_cases_tb;
    localparam realtime BIT_NS = 1.0e9 / 115200.0;
    reg clk = 0;
    reg rst = 1;
    reg uart_rx = 1;
    wire uart_tx;
    reg [7:0] request [0:255];
    reg [7:0] received;
    reg forbid_response = 0;
    string request_path, response_path;
    integer response_file;
    integer mode, n, frames;
    integer captured = 0;
    integer bit_index;

    probe_top dut(.clk(clk), .rst(rst), .uart_rx(uart_rx), .uart_tx(uart_tx));
    always #10 clk = ~clk;

    task automatic send_byte(input reg [7:0] value);
        integer bit_number;
        begin
            uart_rx = 0;
            #(BIT_NS);
            for (bit_number = 0; bit_number < 8; bit_number = bit_number + 1) begin
                uart_rx = value[bit_number];
                #(BIT_NS);
            end
            uart_rx = 1;
            #(BIT_NS);
        end
    endtask

    // Monitor starts before stimulus and runs throughout the final quiet window.
    initial begin
        wait (rst == 0);
        forever begin
            @(negedge uart_tx);
            if (forbid_response)
                $fatal(1, "Rejected request must produce no UART response");
            #(BIT_NS / 2.0);
            if (uart_tx !== 0) $fatal(1, "Invalid UART start bit");
            for (bit_index = 0; bit_index < 8; bit_index = bit_index + 1) begin
                #(BIT_NS);
                received[bit_index] = uart_tx;
            end
            #(BIT_NS);
            if (uart_tx !== 1) $fatal(1, "Invalid UART stop bit");
            $fdisplay(response_file, "%02x", received);
            captured = captured + 1;
        end
    end

    initial begin
        #10000000;
        $fatal(1, "Expected UART responses not received within 10 ms");
    end

    initial begin : stimulus
        integer index;
        integer start_index;
        integer expected_bytes;
        if (!$value$plusargs("REQ=%s", request_path) ||
            !$value$plusargs("RESP=%s", response_path) ||
            !$value$plusargs("MODE=%d", mode) ||
            !$value$plusargs("N=%d", n) ||
            !$value$plusargs("FRAMES=%d", frames))
            $fatal(1, "Missing scenario arguments");
        $readmemh(request_path, request, 0, n - 1);
        response_file = $fopen(response_path, "w");
        if (!response_file) $fatal(1, "Cannot open response file");
        repeat (40) @(posedge clk);
        @(negedge clk);
        rst = 0;
        repeat (1024) @(posedge clk);
        @(negedge clk);
        start_index = 0;
        if (mode == 1) begin
            forbid_response = 1;
            for (index = 0; index < 10; index = index + 1)
                send_byte(request[index]);
            #(BIT_NS * 20);
            forbid_response = 0;
            start_index = 10;
        end
        if (mode == 2) begin
            forbid_response = 1;
            for (index = 0; index < 5; index = index + 1)
                send_byte(request[index]);
            @(negedge clk);
            rst = 1;
            repeat (40) @(posedge clk);
            @(negedge clk);
            rst = 0;
            repeat (1024) @(posedge clk);
            @(negedge clk);
            forbid_response = 0;
            start_index = 5;
        end
        if (mode == 3) begin
            send_byte(request[0]);
            start_index = 1;
        end
        expected_bytes = 0;
        for (index = start_index; index < n; index = index + 1) begin
            send_byte(request[index]);
            if ((index - start_index + 1) % 10 == 0) begin
                expected_bytes = expected_bytes + 9;
                wait (captured >= expected_bytes);
                #(BIT_NS * 2);
            end
        end
        wait (captured >= frames * 9);
        #(BIT_NS * 20);
        if (captured != frames * 9)
            $fatal(1, "Unexpected extra UART response bytes: %0d", captured);
        $fclose(response_file);
        $finish;
    end
endmodule
