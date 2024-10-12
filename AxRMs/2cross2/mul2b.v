module mul2b (
    input wire a1, a0, b1, b0,  
    output wire c2, c1, c0      
);

    assign c2 = a1 & b1;
    assign c1 = a1 & b1;  
    assign c0 = a0 & b0;

endmodule
