/*
	 example Xilinx Artex 7 FPGA
	 150MHz

	 Why FPGA?
	     flexibility (change the design)
	     speed (compared to software?)
	     low power
	 toolchain to generate chips:
	     - write Verilog code
	     - run FPGA synthesis
	     - run FPGA place and route
	     - generate bitstream
	     - download bitstream to FPGA

	 synopsys is already building AI tools to optimize the synthesis
	 Then AIs will improve the tools, then....
	 SKYNET
*/
`timescale 1ns/100ps
module neuron_tb;
	reg  [7:0]   a;            // holds 8 bit
	reg  [63:0]  b;            // holds 64 bits

	integer      i;
	real         r;            // 64-bit floating point number
	shortreal    sr;           // 32-bit floating point number
	wire         zzz;
	logic        clk;
	logic        rst;

	initial begin
		clk = 0;
		forever #1 clk = ~clk; // high for 1ns, low for 1ns
	end

	initial begin
		a = 1;
		for (int i = 0; i < 10; i++) begin
			#2
			a++;
			$display("a = %d", a);
		end
		$finish;
	end
endmodule
