`timescale 1ns / 1ps

module tb_multiplier_4x4;

    
    reg [3:0] A, X;  

    wire [7:0] product;

    integer i, j;      

    integer correct_results;  

    multiplier_4x4 uut (
        .A(A), 
        .X(X), 
        .product(product)
    );

    initial begin
        correct_results = 0;

        $display("Testing all possible combinations of 4-bit inputs for A and X:");
        $display("\n\n A   X  | Product (A * X)  | Expected  | Match\n");

        
        for (i = 0; i < 16; i = i + 1) begin
            for (j = 0; j < 16; j = j + 1) begin
                A = i;  
                X = j;  

                
                #10;    

                
                $display("%2d  %2d  |   %3d          |   %3d      | %d", A, X, product, i * j, (product == (i * j)) ? 1 : 0);

                
                correct_results = correct_results + ((product == i * j) ? 1 : 0);
            end
        end

        
        $display("\nTotal tests: %d", 256);
        $display("Correct results: %d", correct_results);
        $display("Accuracy: %f%%", (correct_results * 100.0) / 256);
        $display("Error   : %f%%", 100 - (correct_results * 100.0) / 256);

        $finish;  
    end
endmodule
