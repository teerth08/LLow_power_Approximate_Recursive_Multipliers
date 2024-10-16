module multiplier_16x16 (
    input [15:0] A, 
    input [15:0] X, 
    output [31:0] product 
);

    wire [15:0] AL_XL, AH_XL, AL_XH, AH_XH;

    multiplier_8x8 m0 (.A(A[7:0]), .X(X[7:0]), .product(AL_XL));  
    multiplier_8x8 m1 (.A(A[15:8]), .X(X[7:0]), .product(AH_XL)); 
    multiplier_8x8 m2 (.A(A[7:0]), .X(X[15:8]), .product(AL_XH)); 
    multiplier_8x8 m3 (.A(A[15:8]), .X(X[15:8]), .product(AH_XH)); 

    wire [31:0] padded_AL_XL;
    wire [31:0] padded_AH_XL;
    wire [31:0] padded_AL_XH;
    wire [31:0] padded_AH_XH;

    
    assign padded_AL_XL = {16'b0, AL_XL};       
    assign padded_AH_XL = {8'b0, AH_XL, 8'b0};  
    assign padded_AL_XH = {8'b0, AL_XH, 8'b0};  
    assign padded_AH_XH = {AH_XH, 16'b0};      

    
    assign product = padded_AL_XL + padded_AH_XL + padded_AL_XH + padded_AH_XH;

endmodule
