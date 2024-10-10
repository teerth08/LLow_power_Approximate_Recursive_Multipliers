`include "exact_4x4.v"

module tb_mult_4x4;

    reg [3:0] a, b;    
    wire [7:0] Y;      

    mult_4x4 uut (
        .a(a), 
        .b(b), 
        .Y(Y)
    );

    integer i, j; // Loop variables

    initial begin
        $display("Testing all possible combinations of 4-bit inputs for a and b:");
        $display(" a     b     Y(a*b)");

        // Loop over all possible values of a and b (4-bit numbers: 0 to 15)
        for (i = 0; i < 16; i = i + 1) begin
            for (j = 0; j < 16; j = j + 1) begin
                a = i;    // Assign current value of i to a
                b = j;    // Assign current value of j to b

                          // THIS IS IMPORTANT STEP 
                #10;      // Wait for output to stabilize

                $display("%4d   %4d   %4d", a, b, Y);  // Print a, b, and the result Y
            end
        end

        $finish;  
    end

endmodule