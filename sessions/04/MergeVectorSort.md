# Merge VectorSort: What we are trying to do
SORT AS FAST AS POSSIBLE



1. Sort as many elements as you can on chip
   loaded 64 elements into ymm0..ymm7 registers
   loaded striped
   [a0, b0, c0, ...    h0]
   [a1, b1, c1, ...    h1]
   [a2, b2, c2, ...    h2]
   [a3, b3, c3, ...    h3]
   [a4, b4, c4, ...    h4]
   [a5, b5, c5, ...    h5]
   [a6, b6, c6, ...    h6]
   [a7, b7, c7, ...    h7]

   each list is a0, a1, ... a7
   Took exactly 19 compare/swaps (3 instructions each, 57)

2. transpose so each sorted list is in one vector registers
   [a0, a1, a2, a3, a4, a5, a6, a7]    counta = 8
   [b0, b1, b2, b3, b4, b5, b6, b7]    countb = 8
   [c0, c1, c2, c3, c4, c5, c6, c7]   
   [d0, d1, d2, d3, d4, d5, d6, d7]
   [e0, e1, e2, e3, e4, e5, e6, e7]
   [f0, f1, f2, f3, f4, f5, f6, f7]
   [g0, g1, g2, g3, g4, g5, g6, g7]   
   [h0, h1, h2, h3, h4, h5, h6, h7]


   [ a2, a3, a4, a5, a6, a7]    counta = 6
   [ b1, b2, b3, b4, b5, b6, b7]    countb = 7

   assume a0<b0 counta = 7
   [a0 a1 b0 
   assume a1<b0


   [a0, a1, a2, a3, a4, a5, a6, a7]    counta = 8
   [b0, b1, b2, b3, b4, b5, b6, b7]    countb = 8
   [c0, c1, c2, c3, c4, c5, c6, c7]    countc = 8
   [d0, d1, d2, d3, d4, d5, d6, d7]    countd = 8

   c0 < d0 and a0 < b0 and c0 < a0
    [a0, a1, a2, a3, a4, a5, a6, a7]    counta = 8
   [b0, b1, b2, b3, b4, b5, b6, b7]    countb = 8
   [c1, c2, c3, c4, c5, c6, c7]    countc = 7
   [d0, d1, d2, d3, d4, d5, d6, d7]    countd = 8

    merge into one sorted list of 32 numbers in ymm0, ymm1, ymm2, ymm3
   [c0, a0, a1, b0, c1, d1, d2, c2]

    (to save code, you can just REUSE ymm0, ymm1, ymm2, ymm3 after first writing them)
    merge into ymm4, ymm5, ymm6, ymm7


    write all numbers to memory (64 sorted at a time)

    with avx512, it would be 256 numbers at a time