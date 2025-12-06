    .macro  minmax a:req, b:req, tmp:req
        vpminsd     \a, \b, \tmp         # tmp = min(a,b), must be different!
        vpmaxsd     \a, \b, \b           # b = max(a, b)
        vmovaps     \tmp, \a             # a = tmp
    .endm

    .globl  sort16
sort16:
    # [(0,13),(1,12),(2,15),(3,14),(4,8),(5,6),(7,11),(9,10)]
    # [(0,5),(1,7),(2,9),(3,4),(6,13),(8,14),(10,15),(11,12)]
    # [(0,1),(2,3),(4,5),(6,8),(7,9),(10,11),(12,13),(14,15)]
    # [(0,2),(1,3),(4,10),(5,11),(6,7),(8,9),(12,14),(13,15)]
    # [(1,2),(3,12),(4,6),(5,7),(8,10),(9,11),(13,14)]
    # [(1,4),(2,6),(5,8),(7,10),(9,13),(11,14)]
    # [(2,4),(3,6),(9,12),(11,13)]
    # [(3,5),(6,8),(7,9),(10,12)]
    # [(3,4),(5,6),(7,8),(9,10),(11,12)]
    # [(6,7),(8,9)]

    minmax      %zmm0,  %zmm13, %zmm17
    minmax      %zmm1,  %zmm12, %zmm17
    minmax      %zmm2,  %zmm5,  %zmm17
    minmax      %zmm3,  %zmm14, %zmm17
    minmax      %zmm4,  %zmm8,  %zmm17
    minmax      %zmm5,  %zmm6,  %zmm17
    minmax      %zmm7,  %zmm11, %zmm17
    minmax      %zmm9,  %zmm10, %zmm17

    minmax      %zmm0,  %zmm5,  %zmm17
    minmax      %zmm1,  %zmm7,  %zmm17
    minmax      %zmm2,  %zmm9,  %zmm17
    minmax      %zmm3,  %zmm4,  %zmm17
    minmax      %zmm6,  %zmm13, %zmm17
    minmax      %zmm8,  %zmm14, %zmm17
    minmax      %zmm10, %zmm15, %zmm17
    minmax      %zmm11, %zmm12, %zmm17

    minmax      %zmm0,  %zmm1,  %zmm17
    minmax      %zmm2,  %zmm3,  %zmm17
    minmax      %zmm4,  %zmm5,  %zmm17
    minmax      %zmm6,  %zmm8,  %zmm17
    minmax      %zmm7,  %zmm9,  %zmm17
    minmax      %zmm10, %zmm11, %zmm17
    minmax      %zmm12, %zmm13, %zmm17
    minmax      %zmm14, %zmm15, %zmm17

    ret

    .globl  main
main:
    call        sort16
    ret
