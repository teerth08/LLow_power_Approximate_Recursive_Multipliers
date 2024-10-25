# Low_power_Approximate_Recursive_Multipliers

## Ideas :-
0. Put all simulations of particular multiplier in their own simulation folder (`ORGANIZATION`)
1. Use matlab to generate uniform random distribution of inputs to feed into n16_5, n16_6, etc 
    You can convert the same multipliers to matlab, and do all the calculations there.
2. It matter WHAT IS THE DISTRIBUTION of inputs 
    => for special cases, there might be some pattern we could exploit ; For general cases => Uniform distribution
3. USE FPGA ( Make a Video demo, GAME CHANGER )


Explain to sir, WHY this paper implementation was hard ( refering to many other papers and implementing their designs )
Why we could not afford to chagne the paper
But in the spirit of what sir asked ( to incorporate what sir taught in the class :- delay, critical path, analysis from genus we spent time in THAT apart from the paper's requirements too)


## 8x8 Design ( 4 Novel Approaches )

Any `2N x 2N` multiplier can be built by placing 4 `NxN` multiplier.

In the n8_5 design inside 8x8 folder, in the line 123 and 122 => `you have the choice to toggle between which mulitplier to use` 

```bash
$ iverlog -o tb.vvp tb.v 
$ vvp tb.vvp
```

1. use exact multiplier and run simulation 
2. use n1 approx mutplier and run the simulation
3. use n1 approx mutplier and run the simulation ( from the n8_6_8x8.v file in the same folder, line number 108, 109 )


## 4x4 Design ( 2 Novel Approaches)
N1 and N2 design