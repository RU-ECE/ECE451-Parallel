
#
    .macro vminmax reg1, reg2, tmp
        vpminsd \reg1, \reg2, \tmp     # tmp = min(reg1, reg2)
        vpmaxsd \reg1, \reg2, \reg2    # reg2 = max(reg1, reg2)
        vmovdqa \tmp, \reg1            # reg1 = tmp
    .endm

    .global sortnetwork8
# ; this is a comment???
# this is a comment???
sortnetwork8: // ymm0, ymm1, ymm2, ymm3, ymm4, ymm5, ymm6, ymm7

    vmovdqa (%rcx),%ymm0
    vmovdqa (%rcx),%ymm1
    vmovdqa (%rcx),%ymm2
    vmovdqa (%rcx),%ymm3
    vmovdqa (%rcx),%ymm4
    vmovdqa (%rcx),%ymm5
    vmovdqa (%rcx),%ymm6
    vmovdqa (%rcx),%ymm7

#[(0,2),(1,3),(4,6),(5,7)]
    vminmax %ymm0, %ymm2, %ymm8
    vminmax %ymm1, %ymm3, %ymm8
    vminmax %ymm4, %ymm6, %ymm8
    vminmax %ymm5, %ymm7, %ymm8
#[(0,4),(1,5),(2,6),(3,7)]
    vminmax %ymm0, %ymm4, %ymm8
    vminmax %ymm1, %ymm5, %ymm8
    vminmax %ymm2, %ymm6, %ymm8
    vminmax %ymm3, %ymm7, %ymm8
#[(0,1),(2,3),(4,5),(6,7)]
    vminmax %ymm0, %ymm1, %ymm8 
    vminmax %ymm2, %ymm3, %ymm8
    vminmax %ymm4, %ymm5, %ymm8
    vminmax %ymm6, %ymm7, %ymm8
#[(2,4),(3,5)]
    vminmax %ymm2, %ymm4, %ymm8
    vminmax %ymm3, %ymm5, %ymm8
#[(1,4),(3,6)]
    vminmax %ymm1, %ymm4, %ymm8
    vminmax %ymm3, %ymm6, %ymm8
#[(1,2),(3,4),(5,6)]
    vminmax %ymm1, %ymm2, %ymm8
    vminmax %ymm3, %ymm4, %ymm8
    vminmax %ymm5, %ymm6, %ymm8
    ret
    // we have 8 groups of 8 sorted. Now..
    /*
       1    3   ...
       3    4
       9    8
       15   22
       22   23
       30   24
       38   25
       46   25


   ymm0 = (1,3,9,15,22,30,38,46)
   ymm1 = (3,4,8,22,23,24,25,25)


   wi = {w1, w2, ... }
   ai = {a1, a2, ... }
   
    */