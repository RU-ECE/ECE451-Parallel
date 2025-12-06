	# minmax takes ymm0, ymm1 and returns with min values in ymm0 and max in ymm1
	.globl	minmax
minmax:
	vpminsd	%ymm0, %ymm1, %ymm8	# ymm8 = min(ymm0,ymm1)
	vpmaxsd	%ymm0, %ymm1, %ymm1	# ymm1 = max(ymm0, ymm1)

	vpminsd	%ymm2, %ymm2, %ymm0	# ymm0 = min(ymm0,ymm1)

	# vpxor	%ymm0, %ymm0, %ymm0	# ymm0 = 0
	# vpaddsd %ymm2, %ymm0		# horrible copy because we can't find mov instruction?
	ret

	.macro	minmax a:req, b:req, tmp:req
		vpminsd	\a, \b, \tmp	# tmp = min(a,b)
		vpmaxsd	\a, \b, \b	# b = max(a,b)
		vpminsd	\tmp, \tmp, \a	# a = min(tmp,tmp)
	.endm

	#
	# ymm0, ymm1, ... ymm7 (8 regs containing 64 numbers)
	#
	# output: ymm0... ymm7 sorted vertically
	#
	.globl	sort8colsasm
sort8colsasm:
	# write a lot of minmax preferably inline, why take the time to call?
	minmax	%ymm0, %ymm1, %ymm8
	minmax	%ymm2, %ymm3, %ymm8
	minmax	%ymm4, %ymm5, %ymm8

	# ... do the 19 swaps for the sorting network
	ret
