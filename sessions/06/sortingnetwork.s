    .text

    .macro  vminmax reg1, reg2, tmp
        vpminsd \reg1, \reg2, \tmp     # tmp  = min(reg1, reg2)
        vpmaxsd \reg1, \reg2, \reg2    # reg2 = max(reg1, reg2)
        vmovdqa \tmp,  \reg1           # reg1 = tmp
    .endm

    .globl  sortnetwork8
sortnetwork8:                          # rcx -> 8 contiguous ymm vectors

    # load 8×8 int32 vectors: 32 bytes per ymm
    vmovdqa  (%rcx),       %ymm0
    vmovdqa  32(%rcx),     %ymm1
    vmovdqa  64(%rcx),     %ymm2
    vmovdqa  96(%rcx),     %ymm3
    vmovdqa  128(%rcx),    %ymm4
    vmovdqa  160(%rcx),    %ymm5
    vmovdqa  192(%rcx),    %ymm6
    vmovdqa  224(%rcx),    %ymm7

    # [(0,2),(1,3),(4,6),(5,7)]
    vminmax  %ymm0, %ymm2, %ymm8
    vminmax  %ymm1, %ymm3, %ymm8
    vminmax  %ymm4, %ymm6, %ymm8
    vminmax  %ymm5, %ymm7, %ymm8

    # [(0,4),(1,5),(2,6),(3,7)]
    vminmax  %ymm0, %ymm4, %ymm8
    vminmax  %ymm1, %ymm5, %ymm8
    vminmax  %ymm2, %ymm6, %ymm8
    vminmax  %ymm3, %ymm7, %ymm8

    # [(0,1),(2,3),(4,5),(6,7)]
    vminmax  %ymm0, %ymm1, %ymm8
    vminmax  %ymm2, %ymm3, %ymm8
    vminmax  %ymm4, %ymm5, %ymm8
    vminmax  %ymm6, %ymm7, %ymm8

    # [(2,4),(3,5)]
    vminmax  %ymm2, %ymm4, %ymm8
    vminmax  %ymm3, %ymm5, %ymm8

    # [(1,4),(3,6)]
    vminmax  %ymm1, %ymm4, %ymm8
    vminmax  %ymm3, %ymm6, %ymm8

    # [(1,2),(3,4),(5,6)]
    vminmax  %ymm1, %ymm2, %ymm8
    vminmax  %ymm3, %ymm4, %ymm8
    vminmax  %ymm5, %ymm6, %ymm8

    ret

    /*
       After this network, each ymmN holds a sorted 8-element vector.

       Example (conceptual):

       ymm0 = ( 1,  3,  9, 15, 22, 30, 38, 46)
       ymm1 = ( 3,  4,  8, 22, 23, 24, 25, 25)

       wi = {w1, w2, ...}
       ai = {a1, a2, ...}
    */
