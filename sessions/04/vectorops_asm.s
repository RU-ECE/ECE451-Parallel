.macro minmax a:req, b:req, tmp:req
    vpminsd   \a, \b, \tmp   # tmp = min(a,b), must be different!
    vpmaxsd   \a, \b, \b     # b = max(a, b)
    vmovaps   \tmp, \a       # a = tmp
.endm
#[(0,2),(1,3),(4,6),(5,7)]
#[(0,4),(1,5),(2,6),(3,7)]
#[(0,1),(2,3),(4,5),(6,7)]
#[(2,4),(3,5)]
#[(1,4),(3,6)]
#[(1,2),(3,4),(5,6)]
// by the end of this we will have 8 groups of 80

    .globl sort8
sort8:
    # write a lot of minmax preferably inline, why take the time to call?
    minmax %ymm0, %ymm2, %ymm8
    minmax %ymm1, %ymm3, %ymm8
    minmax %ymm4, %ymm6, %ymm8
    minmax %ymm5, %ymm7, %ymm8

    minmax %ymm0, %ymm4, %ymm8
    minmax %ymm1, %ymm5, %ymm8
    minmax %ymm2, %ymm6, %ymm8
    minmax %ymm3, %ymm7, %ymm8
 
    minmax %ymm0, %ymm1, %ymm8
    minmax %ymm2, %ymm3, %ymm8
    minmax %ymm4, %ymm5, %ymm8
    minmax %ymm6, %ymm7, %ymm8
 
    minmax %ymm2, %ymm4, %ymm8
    minmax %ymm3, %ymm5, %ymm8
 
    minmax %ymm1, %ymm4, %ymm8
    minmax %ymm3, %ymm6, %ymm8

    minmax %ymm1, %ymm2, %ymm8
    minmax %ymm3, %ymm4, %ymm8
    minmax %ymm5, %ymm6, %ymm8

    ret
    