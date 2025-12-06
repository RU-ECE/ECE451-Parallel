    .text

    .globl  sort8
sort8:
    # vector min(ymm0, ymm2), result in ymm0
    vpminsd %ymm2, %ymm0, %ymm0    # ymm0[i] = min(ymm0[i], ymm2[i])
    ret
