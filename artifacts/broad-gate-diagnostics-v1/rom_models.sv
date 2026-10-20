module SP (DO, DI, BLKSEL, AD, WRE, CLK, CE, OCE, RESET);

// 1 Enables output pipeline registers.
parameter READ_MODE = 1'b0;
// 0: no read on write, 1: transparent, 2: read-before-write
parameter WRITE_MODE = 2'b00;
parameter BIT_WIDTH = 32; // 1, 2, 4, 8, 16, 32
parameter BLK_SEL = 3'b000;
parameter RESET_MODE = "SYNC";
parameter INIT_RAM_00 = 256'h0;
parameter INIT_RAM_01 = 256'h0;
parameter INIT_RAM_02 = 256'h0;
parameter INIT_RAM_03 = 256'h0;
parameter INIT_RAM_04 = 256'h0;
parameter INIT_RAM_05 = 256'h0;
parameter INIT_RAM_06 = 256'h0;
parameter INIT_RAM_07 = 256'h0;
parameter INIT_RAM_08 = 256'h0;
parameter INIT_RAM_09 = 256'h0;
parameter INIT_RAM_0A = 256'h0;
parameter INIT_RAM_0B = 256'h0;
parameter INIT_RAM_0C = 256'h0;
parameter INIT_RAM_0D = 256'h0;
parameter INIT_RAM_0E = 256'h0;
parameter INIT_RAM_0F = 256'h0;
parameter INIT_RAM_10 = 256'h0;
parameter INIT_RAM_11 = 256'h0;
parameter INIT_RAM_12 = 256'h0;
parameter INIT_RAM_13 = 256'h0;
parameter INIT_RAM_14 = 256'h0;
parameter INIT_RAM_15 = 256'h0;
parameter INIT_RAM_16 = 256'h0;
parameter INIT_RAM_17 = 256'h0;
parameter INIT_RAM_18 = 256'h0;
parameter INIT_RAM_19 = 256'h0;
parameter INIT_RAM_1A = 256'h0;
parameter INIT_RAM_1B = 256'h0;
parameter INIT_RAM_1C = 256'h0;
parameter INIT_RAM_1D = 256'h0;
parameter INIT_RAM_1E = 256'h0;
parameter INIT_RAM_1F = 256'h0;
parameter INIT_RAM_20 = 256'h0;
parameter INIT_RAM_21 = 256'h0;
parameter INIT_RAM_22 = 256'h0;
parameter INIT_RAM_23 = 256'h0;
parameter INIT_RAM_24 = 256'h0;
parameter INIT_RAM_25 = 256'h0;
parameter INIT_RAM_26 = 256'h0;
parameter INIT_RAM_27 = 256'h0;
parameter INIT_RAM_28 = 256'h0;
parameter INIT_RAM_29 = 256'h0;
parameter INIT_RAM_2A = 256'h0;
parameter INIT_RAM_2B = 256'h0;
parameter INIT_RAM_2C = 256'h0;
parameter INIT_RAM_2D = 256'h0;
parameter INIT_RAM_2E = 256'h0;
parameter INIT_RAM_2F = 256'h0;
parameter INIT_RAM_30 = 256'h0;
parameter INIT_RAM_31 = 256'h0;
parameter INIT_RAM_32 = 256'h0;
parameter INIT_RAM_33 = 256'h0;
parameter INIT_RAM_34 = 256'h0;
parameter INIT_RAM_35 = 256'h0;
parameter INIT_RAM_36 = 256'h0;
parameter INIT_RAM_37 = 256'h0;
parameter INIT_RAM_38 = 256'h0;
parameter INIT_RAM_39 = 256'h0;
parameter INIT_RAM_3A = 256'h0;
parameter INIT_RAM_3B = 256'h0;
parameter INIT_RAM_3C = 256'h0;
parameter INIT_RAM_3D = 256'h0;
parameter INIT_RAM_3E = 256'h0;
parameter INIT_RAM_3F = 256'h0;

output reg [31:0] DO;
input [31:0] DI;
input [2:0] BLKSEL;
input [13:0] AD;
input WRE;
input CLK;
input CE;
input OCE;
input RESET;


localparam [16383:0] CONTENT = {INIT_RAM_3F, INIT_RAM_3E, INIT_RAM_3D, INIT_RAM_3C, INIT_RAM_3B, INIT_RAM_3A, INIT_RAM_39, INIT_RAM_38, INIT_RAM_37, INIT_RAM_36, INIT_RAM_35, INIT_RAM_34, INIT_RAM_33, INIT_RAM_32, INIT_RAM_31, INIT_RAM_30, INIT_RAM_2F, INIT_RAM_2E, INIT_RAM_2D, INIT_RAM_2C, INIT_RAM_2B, INIT_RAM_2A, INIT_RAM_29, INIT_RAM_28, INIT_RAM_27, INIT_RAM_26, INIT_RAM_25, INIT_RAM_24, INIT_RAM_23, INIT_RAM_22, INIT_RAM_21, INIT_RAM_20, INIT_RAM_1F, INIT_RAM_1E, INIT_RAM_1D, INIT_RAM_1C, INIT_RAM_1B, INIT_RAM_1A, INIT_RAM_19, INIT_RAM_18, INIT_RAM_17, INIT_RAM_16, INIT_RAM_15, INIT_RAM_14, INIT_RAM_13, INIT_RAM_12, INIT_RAM_11, INIT_RAM_10, INIT_RAM_0F, INIT_RAM_0E, INIT_RAM_0D, INIT_RAM_0C, INIT_RAM_0B, INIT_RAM_0A, INIT_RAM_09, INIT_RAM_08, INIT_RAM_07, INIT_RAM_06, INIT_RAM_05, INIT_RAM_04, INIT_RAM_03, INIT_RAM_02, INIT_RAM_01, INIT_RAM_00};
localparam DEPTH = 16384/BIT_WIDTH;
reg [BIT_WIDTH-1:0] rom [0:DEPTH-1];
integer k;
initial begin
 if ((BIT_WIDTH != 4) || READ_MODE != 0 || BLK_SEL != 0) $fatal(1,"Unsupported SP configuration");
 for(k=0;k<DEPTH;k=k+1) rom[k]=CONTENT[k*BIT_WIDTH +: BIT_WIDTH];
end
always @(posedge CLK) begin
 if(!CE || !OCE || WRE || RESET || BLKSEL != 0) $fatal(1,"Unsupported SP operation");
 DO <= {{32{1'b0}}} | rom[AD >> 2];
end
endmodule

module SPX9 (DO, DI, BLKSEL, AD, WRE, CLK, CE, OCE, RESET);

// 1 Enables output pipeline registers.
parameter READ_MODE = 1'b0;
// 0: no read on write, 1: transparent, 2: read-before-write
parameter WRITE_MODE = 2'b00;
parameter BIT_WIDTH = 36; // 9, 18, 36
parameter BLK_SEL = 3'b000;
parameter RESET_MODE = "SYNC";
parameter INIT_RAM_00 = 288'h0;
parameter INIT_RAM_01 = 288'h0;
parameter INIT_RAM_02 = 288'h0;
parameter INIT_RAM_03 = 288'h0;
parameter INIT_RAM_04 = 288'h0;
parameter INIT_RAM_05 = 288'h0;
parameter INIT_RAM_06 = 288'h0;
parameter INIT_RAM_07 = 288'h0;
parameter INIT_RAM_08 = 288'h0;
parameter INIT_RAM_09 = 288'h0;
parameter INIT_RAM_0A = 288'h0;
parameter INIT_RAM_0B = 288'h0;
parameter INIT_RAM_0C = 288'h0;
parameter INIT_RAM_0D = 288'h0;
parameter INIT_RAM_0E = 288'h0;
parameter INIT_RAM_0F = 288'h0;
parameter INIT_RAM_10 = 288'h0;
parameter INIT_RAM_11 = 288'h0;
parameter INIT_RAM_12 = 288'h0;
parameter INIT_RAM_13 = 288'h0;
parameter INIT_RAM_14 = 288'h0;
parameter INIT_RAM_15 = 288'h0;
parameter INIT_RAM_16 = 288'h0;
parameter INIT_RAM_17 = 288'h0;
parameter INIT_RAM_18 = 288'h0;
parameter INIT_RAM_19 = 288'h0;
parameter INIT_RAM_1A = 288'h0;
parameter INIT_RAM_1B = 288'h0;
parameter INIT_RAM_1C = 288'h0;
parameter INIT_RAM_1D = 288'h0;
parameter INIT_RAM_1E = 288'h0;
parameter INIT_RAM_1F = 288'h0;
parameter INIT_RAM_20 = 288'h0;
parameter INIT_RAM_21 = 288'h0;
parameter INIT_RAM_22 = 288'h0;
parameter INIT_RAM_23 = 288'h0;
parameter INIT_RAM_24 = 288'h0;
parameter INIT_RAM_25 = 288'h0;
parameter INIT_RAM_26 = 288'h0;
parameter INIT_RAM_27 = 288'h0;
parameter INIT_RAM_28 = 288'h0;
parameter INIT_RAM_29 = 288'h0;
parameter INIT_RAM_2A = 288'h0;
parameter INIT_RAM_2B = 288'h0;
parameter INIT_RAM_2C = 288'h0;
parameter INIT_RAM_2D = 288'h0;
parameter INIT_RAM_2E = 288'h0;
parameter INIT_RAM_2F = 288'h0;
parameter INIT_RAM_30 = 288'h0;
parameter INIT_RAM_31 = 288'h0;
parameter INIT_RAM_32 = 288'h0;
parameter INIT_RAM_33 = 288'h0;
parameter INIT_RAM_34 = 288'h0;
parameter INIT_RAM_35 = 288'h0;
parameter INIT_RAM_36 = 288'h0;
parameter INIT_RAM_37 = 288'h0;
parameter INIT_RAM_38 = 288'h0;
parameter INIT_RAM_39 = 288'h0;
parameter INIT_RAM_3A = 288'h0;
parameter INIT_RAM_3B = 288'h0;
parameter INIT_RAM_3C = 288'h0;
parameter INIT_RAM_3D = 288'h0;
parameter INIT_RAM_3E = 288'h0;
parameter INIT_RAM_3F = 288'h0;

output reg [35:0] DO;
input [35:0] DI;
input [2:0] BLKSEL;
input [13:0] AD;
input WRE;
input CLK;
input CE;
input OCE;
input RESET;


localparam [18431:0] CONTENT = {INIT_RAM_3F, INIT_RAM_3E, INIT_RAM_3D, INIT_RAM_3C, INIT_RAM_3B, INIT_RAM_3A, INIT_RAM_39, INIT_RAM_38, INIT_RAM_37, INIT_RAM_36, INIT_RAM_35, INIT_RAM_34, INIT_RAM_33, INIT_RAM_32, INIT_RAM_31, INIT_RAM_30, INIT_RAM_2F, INIT_RAM_2E, INIT_RAM_2D, INIT_RAM_2C, INIT_RAM_2B, INIT_RAM_2A, INIT_RAM_29, INIT_RAM_28, INIT_RAM_27, INIT_RAM_26, INIT_RAM_25, INIT_RAM_24, INIT_RAM_23, INIT_RAM_22, INIT_RAM_21, INIT_RAM_20, INIT_RAM_1F, INIT_RAM_1E, INIT_RAM_1D, INIT_RAM_1C, INIT_RAM_1B, INIT_RAM_1A, INIT_RAM_19, INIT_RAM_18, INIT_RAM_17, INIT_RAM_16, INIT_RAM_15, INIT_RAM_14, INIT_RAM_13, INIT_RAM_12, INIT_RAM_11, INIT_RAM_10, INIT_RAM_0F, INIT_RAM_0E, INIT_RAM_0D, INIT_RAM_0C, INIT_RAM_0B, INIT_RAM_0A, INIT_RAM_09, INIT_RAM_08, INIT_RAM_07, INIT_RAM_06, INIT_RAM_05, INIT_RAM_04, INIT_RAM_03, INIT_RAM_02, INIT_RAM_01, INIT_RAM_00};
localparam DEPTH = 18432/BIT_WIDTH;
reg [BIT_WIDTH-1:0] rom [0:DEPTH-1];
integer k;
initial begin
 if ((BIT_WIDTH != 9 && BIT_WIDTH != 18) || READ_MODE != 0 || BLK_SEL != 0) $fatal(1,"Unsupported SPX9 configuration");
 for(k=0;k<DEPTH;k=k+1) rom[k]=CONTENT[k*BIT_WIDTH +: BIT_WIDTH];
end
always @(posedge CLK) begin
 if(!CE || !OCE || WRE || RESET || BLKSEL != 0) $fatal(1,"Unsupported SPX9 operation");
 DO <= {{36{1'b0}}} | rom[AD >> (BIT_WIDTH == 9 ? 3 : 4)];
end
endmodule
