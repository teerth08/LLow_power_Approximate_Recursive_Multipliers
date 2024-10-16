module Kul2(
    input [1:0]a,b,
    output [3:0]Y
);
    assign Y[0] = a[0] & b[0];
    assign Y[1] = (a[1] & b[0]) | (a[0] & b[1]);
    assign Y[2] = a[1] & b[1];
    assign Y[3] = 0;
endmodule

module Kul4 (
    input [3:0] a, 
    input [3:0] b, 
    output [7:0] Y
);

    wire [3:0] AL_XL, AH_XL, AL_XH, AH_XH;

    Kul2 m0 (.a(a[1:0]), .b(b[1:0]), .Y(AL_XL));
    Kul2 m1 (.a(a[3:2]), .b(b[1:0]), .Y(AH_XL));
    Kul2 m2 (.a(a[1:0]), .b(b[3:2]), .Y(AL_XH));
    Kul2 m3 (.a(a[3:2]), .b(b[3:2]), .Y(AH_XH));
 
    wire [7:0] padded_AL_XL;
    wire [7:0] padded_AH_XL;
    wire [7:0] padded_AL_XH;
    wire [7:0] padded_AH_XH;

    assign padded_AL_XL = {4'b0, AL_XL};       
    assign padded_AH_XL = {2'b0, AH_XL, 2'b0}; 
    assign padded_AL_XH = {2'b0, AL_XH, 2'b0}; 
    assign padded_AH_XH = {AH_XH, 4'b0};       

    assign Y = padded_AL_XL + padded_AH_XL + padded_AL_XH + padded_AH_XH;

endmodule

module Kul8 (
    input [3:0] a, 
    input [3:0] b, 
    output [7:0] Y
);

    wire [3:0] AL_XL, AH_XL, AL_XH, AH_XH;

    Kul2 m0 (.a(a[1:0]), .b(b[1:0]), .Y(AL_XL));
    Kul2 m1 (.a(a[3:2]), .b(b[1:0]), .Y(AH_XL));
    Kul2 m2 (.a(a[1:0]), .b(b[3:2]), .Y(AL_XH));
    Kul2 m3 (.a(a[3:2]), .b(b[3:2]), .Y(AH_XH));

    wire [7:0] padded_AL_XL;
    wire [7:0] padded_AH_XL;
    wire [7:0] padded_AL_XH;
    wire [7:0] padded_AH_XH;

    assign padded_AL_XL = {4'b0, AL_XL};       
    assign padded_AH_XL = {2'b0, AH_XL, 2'b0}; 
    assign padded_AL_XH = {2'b0, AL_XH, 2'b0}; 
    assign padded_AH_XH = {AH_XH, 4'b0};       

    assign Y = padded_AL_XL + padded_AH_XL + padded_AL_XH + padded_AH_XH;

endmodule



module Kul16 (
    input [7:0] a, 
    input [7:0] b, 
    output [15:0] Y
);

    wire [7:0] AL_XL, AH_XL, AL_XH, AH_XH;

    Kul8 m0 (.a(a[3:0]), .b(b[3:0]), .Y(AL_XL));
    Kul8 m1 (.a(a[3:2]), .b(b[1:0]), .Y(AH_XL));
    Kul8 m2 (.a(a[1:0]), .b(b[3:2]), .Y(AL_XH));
    Kul8 m3 (.a(a[3:2]), .b(b[3:2]), .Y(AH_XH));

 
    wire [7:0] padded_AL_XL;
    wire [7:0] padded_AH_XL;
    wire [7:0] padded_AL_XH;
    wire [7:0] padded_AH_XH;

    assign padded_AL_XL = {4'b0, AL_XL};       
    assign padded_AH_XL = {2'b0, AH_XL, 2'b0}; 
    assign padded_AL_XH = {2'b0, AL_XH, 2'b0}; 
    assign padded_AH_XH = {AH_XH, 4'b0};       

    assign Y = padded_AL_XL + padded_AH_XL + padded_AL_XH + padded_AH_XH;

endmodule
