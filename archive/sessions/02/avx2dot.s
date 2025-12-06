	.text
	.globl	dot_product_avx2

dot_product_avx2:
	# Input:
	#   %rdi = pointer to the first array (a)
	#   %rsi = pointer to the second array (b)
	#   %rcx = number of elements in the arrays (n)
	# Output:
	#   %rax = dot product result

	# Check if n is zero or negative, and return 0 in that case
	test	%rcx, %rcx
	jle		.done

	# Initialize accumulators
	vpxor	%ymm0, %ymm0, %ymm0	# Clear ymm0 to store the result
	vpxor	%ymm1, %ymm1, %ymm1	# Clear ymm1 for intermediate accumulations

.loop:
	# Load 32 bytes (8 floats) from each array into ymm2 and ymm3
	vmovaps	(%rdi), %ymm2
	vmovaps	(%rsi), %ymm3

	# Multiply the two vectors and accumulate the result in ymm1
	vfmadd231ps	%ymm2, %ymm3, %ymm1	# ymm1 = ymm1 + ymm2 * ymm3

	# Move to the next set of data
	add		$32, %rdi
	add		$32, %rsi
	sub		$8, %rcx

	# Check if we've processed all elements
	jnz		.loop

	# Horizontal sum of ymm1 into ymm0
	vphaddw	%ymm1, %ymm1, %ymm0
	vphaddd	%ymm0, %ymm0, %ymm0
	vphaddq	%ymm0, %ymm0, %ymm0

.done:
	# Extract the result from ymm0 into rax
	vextracti128	$1, %ymm0, %xmm1
	vpaddd			%xmm1, %xmm0, %xmm0
	movd			%xmm0, %eax

	ret
