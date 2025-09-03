/*
    f(sum(Wi*ai))  A single neuron in a neural network
*/
module neuron(input logic clk, input shortreal inputs[9:0], output shortreal output);
    shortreal sum;

    reg[15:0] wi[9:0]; // 2^10 = 1024 16-bit numbers

    initial begin
/*        for (int i = 0; i < 10; i++) begin
            wi[i] = $random;
        end
*/
        for (int i = 0; i < 1024; i++) begin
            wi[i] = .1;
        end
    end

    always @(posedge clk) begin
        sum = 0;
        for (int i = 0; i < 10; i++) begin
            sum += wi[i] * inputs[i];
        end
    end
endmodule
