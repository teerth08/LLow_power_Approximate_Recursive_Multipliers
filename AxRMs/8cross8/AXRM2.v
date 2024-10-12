module AxRM1 (
    input wire [7:0] a,  
    input wire [7:0] b,  
    output wire [15:0] result  
);

    wire [3:0] product1, product2, product3, product4;
    wire [3:0] product5, product6, product7, product8;

    mul2b mul1 (.a(a[1:0]), .b(b[1:0]), .product(product1)); 
    mul2b mul2 (.a(a[1:0]), .b(b[3:2]), .product(product2)); 
    mul2b mul3 (.a(a[1:0]), .b(b[5:4]), .product(product3)); 
    mul2b mul4 (.a(a[1:0]), .b(b[7:6]), .product(product4));

    mul2b mul5 (.a(a[3:2]), .b(b[1:0]), .product(product5)); 
    mul2b mul6 (.a(a[3:2]), .b(b[3:2]), .product(product6)); 
    mul2b mul7 (.a(a[3:2]), .b(b[5:4]), .product(product7)); 
    mul2b mul8 (.a(a[3:2]), .b(b[7:6]), .product(product8)); 

    wire [3:0] product9, product10, product11, product12;
    wire [3:0] product13, product14, product15, product16;

    exactOutput_2cross2 mul9 (.a(a[5:4]), .b(b[1:0]), .product(product9)); 
    exactOutput_2cross2 mul10(.a(a[5:4]), .b(b[3:2]), .product(product10)); 
    exactOutput_2cross2 mul11(.a(a[5:4]), .b(b[5:4]), .product(product11)); 
    exactOutput_2cross2 mul12(.a(a[5:4]), .b(b[7:6]), .product(product12)); 

    exactOutput_2cross2 mul13(.a(a[7:6]), .b(b[1:0]), .product(product13));
    exactOutput_2cross2 mul14(.a(a[7:6]), .b(b[3:2]), .product(product14)); 
    exactOutput_2cross2 mul15(.a(a[7:6]), .b(b[5:4]), .product(product15));
    exactOutput_2cross2 mul16(.a(a[7:6]), .b(b[7:6]), .product(product16));


    wire [15:0] sum1, sum2, sum3, sum4, sum5, sum6, sum7;

    assign sum1 = {12'b0, product1} + ({10'b0, product2, 2'b0}) + ({8'b0, product3, 4'b0}) + ({6'b0, product4, 6'b0});
    assign sum2 = ({10'b0, product5, 2'b0}) + ({8'b0, product6, 4'b0}) + ({6'b0, product7, 6'b0}) + ({4'b0, product8, 8'b0});
    assign sum3 = ({8'b0, product9, 4'b0}) + ({6'b0, product10, 6'b0}) + ({4'b0, product11, 8'b0}) + ({2'b0, product12, 10'b0});
    assign sum4 = ({6'b0, product13, 6'b0}) + ({4'b0, product14, 8'b0}) + ({2'b0, product15, 10'b0}) + ({product16, 12'b0});

    // Final sum
    assign result = sum1 + sum2 + sum3 + sum4;

endmodule
