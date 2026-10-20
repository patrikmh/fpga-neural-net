// GW5A ALU LUT layout confirmed by nextpnr pack_luts.cc and apycula chipdb.py.
// S selects low/high XOR truth tables; C is carry generate truth table.
module DIAG_ALU(output SUM,COUT,input I0,I1,I2,I3,CIN);
parameter ALU_MODE=2;
parameter RAW_ALU_LUT=-1;
parameter CIN_NETTYPE="WIRE";
localparam [15:0] LUT = RAW_ALU_LUT >= 0 ? RAW_ALU_LUT[15:0] : ALU_MODE == "C2L" ? 16'h0000 : 16'h609a;
wire ci = CIN_NETTYPE == "GND" ? 1'b0 : CIN_NETTYPE == "VCC" ? 1'b1 : CIN;
wire s=LUT[4+8*I3+2*I1+I0];
wire c=LUT[2*I1+I0];
assign SUM=s^ci;
assign COUT=s?ci:c;
always @* if(I2 !== 1'b1) $fatal(1,"Unsupported ALU I2");
endmodule
