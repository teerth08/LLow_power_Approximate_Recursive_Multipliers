module multiplier_2x2 (
    input wire a0,  
    input wire a1,  
    input wire b0,  
    input wire b1,  
    output wire out0, 
    output wire out1, 
    output wire out2, 
    output wire out3  
);
   
    wire and1, and2, and3, and4, xor1, xor2;
    wire or1, or2;

    assign and1 = a0 & b1;  
    assign and2 = a1 & b0;  
    assign and3 = a1 & b1; 

    assign and4 = and1 & and2; 
    assign xor1 = and1 ^ and2;
    assign xor2 = and4 ^ and3;

    assign out0 = and4;        
    assign out1 = xor1;         
    assign out2 = xor2;        
    assign out3 = and4;         
endmodule


module multiplier_4x4 (
    input [3:0] A, 
    input [3:0] X, 
    output [7:0] product 
);

    wire [3:0] AL_XL, AH_XL, AL_XH, AH_XH;

    multiplier_2x2 m0 (.a1(A[1]), .a0(A[0]), .b1(X[1]), .b0(X[0]), .out3(AL_XL[3]), .out2(AL_XL[2]), .out1(AL_XL[1]), .out0(AL_XL[0]));
    multiplier_2x2 m1 (.a1(A[3]), .a0(A[2]), .b1(X[1]), .b0(X[0]), .out3(AH_XL[3]), .out2(AH_XL[2]), .out1(AH_XL[1]), .out0(AH_XL[0]));
    multiplier_2x2 m2 (.a1(A[1]), .a0(A[0]), .b1(X[3]), .b0(X[2]), .out3(AL_XH[3]), .out2(AL_XH[2]), .out1(AL_XH[1]), .out0(AL_XH[0]));
    multiplier_2x2 m3 (.a1(A[3]), .a0(A[2]), .b1(X[3]), .b0(X[2]), .out3(AH_XH[3]), .out2(AH_XH[2]), .out1(AH_XH[1]), .out0(AH_XH[0]));


    wire [7:0] padded_AL_XL;
    wire [7:0] padded_AH_XL;
    wire [7:0] padded_AL_XH;
    wire [7:0] padded_AH_XH;

    assign padded_AL_XL = {5'b0, AL_XL};       
    assign padded_AH_XL = {3'b0, AH_XL, 2'b0}; 
    assign padded_AL_XH = {3'b0, AL_XH, 2'b0}; 
    assign padded_AH_XH = {1'b0, AH_XH, 4'b0};       

    assign product = padded_AL_XL + padded_AH_XL + padded_AL_XH + padded_AH_XH;

endmodule

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
