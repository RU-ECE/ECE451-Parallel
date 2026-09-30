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

# AI is an IDIOT.
transpose8:
#    vmovdqa %ymm0, %ymm8    # aligned: be on multiple of 32 bytes OR DIE
#    vmovdqa %ymm1, %ymm9
#    vmovdqa %ymm2, %ymm10
#    vmovdqa %ymm3, %ymm11   
#    vmovdqa %ymm4, %ymm12
#    vmovdqa %ymm5, %ymm13
#    vmovdqa %ymm6, %ymm14
#    vmovdqa %ymm7, %ymm15
#    ret

# GAS / AT&T syntax
#
# Input:
#   %ymm0-%ymm7 = 8 rows of 8 x 32-bit values
#
# Output:
#   %ymm0-%ymm7 = transposed 8x8 matrix
#
# No memory accesses.
# %ymm8-%ymm15 are temporaries.

# ------------------------------------------------------------
# Stage 1: interleave 32-bit elements
# ------------------------------------------------------------

vmovdqa %ymm0, %ymm8
# [a0 a1 a2 a3 a4 a5 a6 a7]
vpunpckldq %ymm1, %ymm8, %ymm8      # [a00 a10 a01 a11 | a04 a14 a05 a15]
vmovdqa %ymm0, %ymm9
vpunpckhdq %ymm1, %ymm9, %ymm9      # [a02 a12 a03 a13 | a06 a16 a07 a17]

vmovdqa %ymm2, %ymm10
vpunpckldq %ymm3, %ymm10, %ymm10    # [a20 a30 a21 a31 | a24 a34 a25 a35]
vmovdqa %ymm2, %ymm11
vpunpckhdq %ymm3, %ymm11, %ymm11    # [a22 a32 a23 a33 | a26 a36 a27 a37]

vmovdqa %ymm4, %ymm12
vpunpckldq %ymm5, %ymm12, %ymm12    # [a40 a50 a41 a51 | a44 a54 a45 a55]
vmovdqa %ymm4, %ymm13
vpunpckhdq %ymm5, %ymm13, %ymm13    # [a42 a52 a43 a53 | a46 a56 a47 a57]

vmovdqa %ymm6, %ymm14
vpunpckldq %ymm7, %ymm14, %ymm14    # [a60 a70 a61 a71 | a64 a74 a65 a75]
vmovdqa %ymm6, %ymm15
vpunpckhdq %ymm7, %ymm15, %ymm15    # [a62 a72 a63 a73 | a66 a76 a67 a77]


# ------------------------------------------------------------
# Stage 2: interleave 64-bit elements
# ------------------------------------------------------------

vpunpcklqdq %ymm10, %ymm8, %ymm0    # [a00 a10 a20 a30 | a04 a14 a24 a34]
vpunpckhqdq %ymm10, %ymm8, %ymm1    # [a01 a11 a21 a31 | a05 a15 a25 a35]

vpunpcklqdq %ymm11, %ymm9, %ymm2    # [a02 a12 a22 a32 | a06 a16 a26 a36]
vpunpckhqdq %ymm11, %ymm9, %ymm3    # [a03 a13 a23 a33 | a07 a17 a27 a37]

vpunpcklqdq %ymm14, %ymm12, %ymm4   # [a40 a50 a60 a70 | a44 a54 a64 a74]
vpunpckhqdq %ymm14, %ymm12, %ymm5   # [a41 a51 a61 a71 | a45 a55 a65 a75]

vpunpcklqdq %ymm15, %ymm13, %ymm6   # [a42 a52 a62 a72 | a46 a56 a66 a76]
vpunpckhqdq %ymm15, %ymm13, %ymm7   # [a43 a53 a63 a73 | a47 a57 a67 a77]


# ------------------------------------------------------------
# Stage 3: exchange the 128-bit halves
#
# imm8 = 0x20:
#   low  128 = low  128 of first source
#   high 128 = low  128 of second source
#
# imm8 = 0x31:
#   low  128 = high 128 of first source
#   high 128 = high 128 of second source
# ------------------------------------------------------------

vperm2i128 $0x20, %ymm4, %ymm0, %ymm8    # [a00 a10 a20 a30 | a40 a50 a60 a70]
vperm2i128 $0x31, %ymm4, %ymm0, %ymm12   # [a04 a14 a24 a34 | a44 a54 a64 a74]

vperm2i128 $0x20, %ymm5, %ymm1, %ymm9     # [a01 a11 a21 a31 | a41 a51 a61 a71]
vperm2i128 $0x31, %ymm5, %ymm1, %ymm13   # [a05 a15 a25 a35 | a45 a55 a65 a75]

vperm2i128 $0x20, %ymm6, %ymm2, %ymm10    # [a02 a12 a22 a32 | a42 a52 a62 a72]
vperm2i128 $0x31, %ymm6, %ymm2, %ymm14   # [a06 a16 a26 a36 | a46 a56 a66 a76]

vperm2i128 $0x20, %ymm7, %ymm3, %ymm11    # [a03 a13 a23 a33 | a43 a53 a63 a73]
vperm2i128 $0x31, %ymm7, %ymm3, %ymm15   # [a07 a17 a27 a37 | a47 a57 a67 a77]


# ------------------------------------------------------------
# Final result: transpose is in ymm0-ymm7
# ------------------------------------------------------------

vmovdqa %ymm8,  %ymm0
vmovdqa %ymm9,  %ymm1
vmovdqa %ymm10, %ymm2
vmovdqa %ymm11, %ymm3
vmovdqa %ymm12, %ymm4
vmovdqa %ymm13, %ymm5
vmovdqa %ymm14, %ymm6
vmovdqa %ymm15, %ymm7