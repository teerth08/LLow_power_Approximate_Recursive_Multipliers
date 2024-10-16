module multiplier_8x8 (
    input [7:0] A, 
    input [7:0] X, 
    output [15:0] product 
);

    wire [7:0] AL_XL, AH_XL, AL_XH, AH_XH;

    multiplier_4x4 m0 (.A(A[3:0]), .X(X[3:0]), .product(AL_XL)); 
    multiplier_4x4 m1 (.A(A[7:4]), .X(X[3:0]), .product(AH_XL)); 
    multiplier_4x4 m2 (.A(A[3:0]), .X(X[7:4]), .product(AL_XH)); 
    multiplier_4x4 m3 (.A(A[7:4]), .X(X[7:4]), .product(AH_XH)); 

    wire [15:0] padded_AL_XL;
    wire [15:0] padded_AH_XL;
    wire [15:0] padded_AL_XH;
    wire [15:0] padded_AH_XH;

   
    assign padded_AL_XL = {8'b0, AL_XL};       
    assign padded_AH_XL = {4'b0, AH_XL, 4'b0}; 
    assign padded_AL_XH = {4'b0, AL_XH, 4'b0}; 
    assign padded_AH_XH = {AH_XH, 8'b0};       

    // Sum the partial products
    assign product = padded_AL_XL + padded_AH_XL + padded_AL_XH + padded_AH_XH;

endmodule
