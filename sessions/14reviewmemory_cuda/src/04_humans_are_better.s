	.file	"03_dot_simd.cpp ALTERNATE"
	.globl	dovdot
dovdot:
	.cfi_startproc
	endbr64
# assumption: rdi=a, rsi=b, rcx = numberofelements
	vpxor   %ymm0, %ymm0  # ymm0 = 0
# note: we didn't check if the number is zero
# note: we didn't check if the number is not a multiple of 8 
#
    cmp $0, %rcx
	jeq  endit

.loop1:
	vmovups	(%rdi), %ymm1
	vmulps	(%rsi), %ymm1, %ymm1
	vaddps	%ymm1, %ymm0, %ymm0
	addq	$32, %rax
	sub     $8, %rcx
	jgt     .loop1

	# now, horizontally sum...




endit:
	ret
