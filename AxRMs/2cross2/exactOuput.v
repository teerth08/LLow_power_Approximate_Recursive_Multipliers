module exactOutput_2cross2 (
    input wire [1:0] a,  
    input wire [1:0] b,  
    output wire [3:0] product  
);

    wire p0 = a[0] & b[0];
    wire p1 = a[0] & b[1];
    wire p2 = a[1] & b[0];
    wire p3 = a[1] & b[1];

    assign product[0] = p0;
    assign product[1] = p1 ^ p2;        
    assign product[2] = p3 ^ (p1 & p2);   
    assign product[3] = p3 & (p1 & p2);   

endmodule
