`timescale 1ns/1ps

module probe_uart_tb;
    localparam realtime BIT_NS = 1.0e9 / 115200.0;
    reg clk = 0;
    reg rst = 1;
    reg uart_rx = 1;
    wire uart_tx;
    reg [7:0] request [0:9];
    reg [7:0] received;
    string request_path;
    string response_path;
    integer response_file;
    integer byte_index;
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

    initial begin
        #10000000;
        $fatal(1, "UART exchange exceeded 10 ms watchdog");
    end

    initial begin
        if (!$value$plusargs("REQ=%s", request_path))
            $fatal(1, "Missing request fixture");
        if (!$value$plusargs("RESP=%s", response_path))
            $fatal(1, "Missing response destination");
        $readmemh(request_path, request);
        response_file = $fopen(response_path, "w");
        if (!response_file)
            $fatal(1, "Cannot open response destination");
        repeat (40) @(posedge clk);
        @(negedge clk);
        rst = 0;
        repeat (1024) @(posedge clk);
        @(negedge clk);
        fork
            begin : transmit_request
                integer index;
                for (index = 0; index < 10; index = index + 1)
                    send_byte(request[index]);
            end
            begin : capture_response
                for (byte_index = 0; byte_index < 9; byte_index = byte_index + 1) begin
                    @(negedge uart_tx);
                    #(BIT_NS / 2.0);
                    if (uart_tx !== 0)
                        $fatal(1, "Invalid UART start bit");
                    for (bit_index = 0; bit_index < 8; bit_index = bit_index + 1) begin
                        #(BIT_NS);
                        received[bit_index] = uart_tx;
                    end
                    #(BIT_NS);
                    if (uart_tx !== 1)
                        $fatal(1, "Invalid UART stop bit");
                    $fdisplay(response_file, "%02x", received);
                end
            end
        join
        $fclose(response_file);
        $finish;
    end
endmodule
