# minmax takes ymm0, ymm1 and returns with min values in ymm0 and max in ymm1
    .globl minmax
minmax:
    vpminsd   %ymm0, %ymm1, %ymm2   # ymm2 = min(ymm0,ymm1)
    vpmaxsd   %ymm0, %ymm1, %ymm1   # ymm1 = max(ymm0, ymm1)

    vpminsd   %ymm2, %ymm2, %ymm0   # ymm0 = min(ymm0,ymm1) 


#    vpxor   %ymm0, %ymm0, %ymm0     #ymm0 = 0
#    vpaddsd %ymm2, %ymm0            # horrible copy because we can't find mov instruction?
    ret

    .globl sort8colsasm
sort8colsasm:
    # write a lot of minmax preferably inline, why take the time to call?
    vpminsd   %ymm0, %ymm1, %ymm2   # ymm2 = min(ymm0,ymm1)
    vpmaxsd   %ymm0, %ymm1, %ymm1   # ymm1 = max(ymm0, ymm1)

    vpminsd   %ymm2, %ymm2, %ymm0   # ymm0 = min(ymm0,ymm1) 

    #... do the 19 swaps for the sorting network0
    ret
