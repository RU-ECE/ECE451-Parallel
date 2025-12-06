    .text

    .globl  dovdot
dovdot:
    .cfi_startproc
    endbr64

    # SysV ABI: rdi = a, rsi = b, rdx = number_of_elements (must be multiple of 8)

    vxorps  %ymm0, %ymm0, %ymm0      # ymm0 = 0.0

    cmp     $0, %rdx                 # if n <= 0, just return 0
    jle     endit

.loop1:
    vmovups (%rdi), %ymm1            # load 8 floats from a
    vmulps  (%rsi), %ymm1, %ymm1     # ymm1 = a[i..i+7] * b[i..i+7]
    vaddps  %ymm1, %ymm0, %ymm0      # accumulate into ymm0

    add     $32, %rdi                # advance a by 8 floats (32 bytes)
    add     $32, %rsi                # advance b by 8 floats
    sub     $8,  %rdx                # processed 8 elements
    jg      .loop1

    # horizontally sum ymm0 to a single float in xmm0
    vextractf128    $1, %ymm0, %xmm1     # high 128 -> xmm1
    vaddps          %xmm1, %xmm0, %xmm0  # add high + low

    vpermilps       $0xB1, %xmm0, %xmm1  # swap pairs (1,0,3,2 pattern)
    vaddps          %xmm1, %xmm0, %xmm0

    vpermilps       $0x0E, %xmm0, %xmm1  # move high lane down
    vaddps          %xmm1, %xmm0, %xmm0  # xmm0[0] = sum of 8 floats

    # return value: scalar float in xmm0[0]

endit:
    ret
    .cfi_endproc
