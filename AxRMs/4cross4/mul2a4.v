module mul2a4 (
    input wire [3:0] a,  
    input wire [3:0] b,  
    output wire [7:0] result  
);

    wire [3:0] product1, product2, product3, product4;

    mul2a mul1(.a(a[1:0]), .b(b[1:0]), .product(product1)); 
    mul2a mul2(.a(a[1:0]), .b(b[3:2]), .product(product2)); 
    mul2a mul3(.a(a[3:2]), .b(b[1:0]), .product(product3)); 
    mul2a mul4(.a(a[3:2]), .b(b[3:2]), .product(product4)); 

    assign result = product1 + (product2 << 2) + (product3 << 2) + (product4 << 4);

endmodule
