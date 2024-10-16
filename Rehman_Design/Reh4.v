module Reh2 (
    input [1:0]a,b,  
    output [3:0]Y
);
    assign Y[0] = (a[0] & b[1]) & (a[1] & b[0]);        
    assign Y[1] = (a[0] & b[1]) ^ (a[1] & b[0]);        
    assign Y[2] = ((a[0] & b[1]) & (a[1] & b[0])) ^ (a[1] & b[1]);        
    assign Y[3] = (a[0] & b[1]) & (a[1] & b[0]); 

endmodule



module multiplier_4x4 (
    input [3:0] A, X,
    output [7:0] Y
);

    wire [3:0] AL_XL, AH_XL, AL_XH, AH_XH;

    Reh2 m0 (.a(A[1:0]) , .b(X[1:0]), .Y(AL_XH));
    Reh2 m1 (.a(A[3:2]) , .b(X[1:0]), .Y(AH_XL));
    Reh2 m2 (.a(A[1:0]) , .b(X[3:2]), .Y(AL_XH));
    Reh2 m3 (.a(A[3:2]) , .b(X[3:2]), .Y(AH_XH));

    wire [7:0] padded_AL_XL;
    wire [7:0] padded_AH_XL;
    wire [7:0] padded_AL_XH;
    wire [7:0] padded_AH_XH;

    assign padded_AL_XL = {5'b0, AL_XL};       
    assign padded_AH_XL = {3'b0, AH_XL, 2'b0}; 
    assign padded_AL_XH = {3'b0, AL_XH, 2'b0}; 
    assign padded_AH_XH = {1'b0, AH_XH, 4'b0};       

    assign Y = padded_AL_XL + padded_AH_XL + padded_AL_XH + padded_AH_XH;

endmodule