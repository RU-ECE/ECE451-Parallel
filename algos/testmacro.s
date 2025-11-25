.macro minmax a:req, b:req, tmp:req
    vpminsd   \a, \b, \tmp   # tmp = min(a,b), must be different!
    vpmaxsd   \a, \b, \b     # b = max(a, b)
    vmovaps   \tmp, \a       # a = tmp
#    vpminsd   \tmp, \tmp, \a   # a = min(tmp, tmp) ugly hack because we don't know how to write a = tmp!
.endm

    .globl sort8
sort8:
    
    minmax  %ymm0, %ymm1, %ymm8
    minmax  %ymm2, %ymm3, %ymm8
    minmax  %ymm4, %ymm5, %ymm8
    minmax  %ymm6, %ymm7, %ymm8
    ret
