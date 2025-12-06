#
# assembler used to use stack for passing parameters
# this involved memory, which is SLOW
#

# old way would use the stack

#   f(34, 23, 12)
#   pushl	$12			# 32-bit
#   pushl	$23
#   pushq	$34			# 64-bit
#   call	f


# intel "fastcall" convention uses up to 4 registers for parameters
# f(1, 2, 3, 4)
	mov		$1, %rsi
	mov		$2, %rdi
	mov		$3, %r8
	mov		$4, %r9
	call	f

# f(1, 2, 3, 4, 5)
	mov		$1, %rsi
	mov		$2, %rdi
	mov		$3, %r8
	mov		$4, %r9
	sub		$8, %rsp
	movq	$5, (%rsp)

# f(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0)
#   xmm0, xmm1, xmm2, xmm3, ... xmm7

	vaddsd	%xmm0, %xmm1, %xmm2	# xmm2 = xmm0 + xmm1
