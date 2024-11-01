	.file	"03_dot_simd.cpp"
	.text
	.p2align 4
	.type	_Z9dot_simd2PKfS0_i._omp_fn.0, @function
_Z9dot_simd2PKfS0_i._omp_fn.0:
.LFB7773:
	.cfi_startproc
	endbr64
	pushq	%r13
	.cfi_def_cfa_offset 16
	.cfi_offset 13, -16
	pushq	%r12
	.cfi_def_cfa_offset 24
	.cfi_offset 12, -24
	pushq	%rbp
	.cfi_def_cfa_offset 32
	.cfi_offset 6, -32
	pushq	%rbx
	.cfi_def_cfa_offset 40
	.cfi_offset 3, -40
	movq	%rdi, %rbx
	subq	$8, %rsp
	.cfi_def_cfa_offset 48
	movq	8(%rdi), %rbp
	movq	(%rdi), %r12
	call	omp_get_num_threads@PLT
	movl	%eax, %r13d
	call	omp_get_thread_num@PLT
	movl	%eax, %ecx
	movl	16(%rbx), %eax
	cltd
	idivl	%r13d
	cmpl	%edx, %ecx
	jl	.L2
.L6:
	imull	%eax, %ecx
	vxorps	%xmm1, %xmm1, %xmm1
	addl	%ecx, %edx
	addl	%edx, %eax
	cmpl	%eax, %edx
	jge	.L3
	movslq	%edx, %rdx
	.p2align 4,,10
	.p2align 3
.L4:
	vmovss	(%r12,%rdx,4), %xmm0
	vmulss	0(%rbp,%rdx,4), %xmm0, %xmm0
	addq	$1, %rdx
	vaddss	%xmm0, %xmm1, %xmm1
	cmpl	%edx, %eax
	jg	.L4
.L3:
	movl	20(%rbx), %edx
	leaq	20(%rbx), %rcx
.L5:
	vmovd	%edx, %xmm3
	movl	%edx, %eax
	vaddss	%xmm3, %xmm1, %xmm2
	vmovd	%xmm2, %esi
	lock cmpxchgl	%esi, (%rcx)
	jne	.L14
	addq	$8, %rsp
	.cfi_remember_state
	.cfi_def_cfa_offset 40
	popq	%rbx
	.cfi_def_cfa_offset 32
	popq	%rbp
	.cfi_def_cfa_offset 24
	popq	%r12
	.cfi_def_cfa_offset 16
	popq	%r13
	.cfi_def_cfa_offset 8
	ret
	.p2align 4,,10
	.p2align 3
.L2:
	.cfi_restore_state
	addl	$1, %eax
	xorl	%edx, %edx
	jmp	.L6
.L14:
	movl	%eax, %edx
	jmp	.L5
	.cfi_endproc
.LFE7773:
	.size	_Z9dot_simd2PKfS0_i._omp_fn.0, .-_Z9dot_simd2PKfS0_i._omp_fn.0
	.p2align 4
	.type	_Z9dot_simd3PKfS0_i._omp_fn.0, @function
_Z9dot_simd3PKfS0_i._omp_fn.0:
.LFB7774:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	pushq	%r14
	pushq	%r13
	pushq	%r12
	pushq	%rbx
	.cfi_offset 14, -24
	.cfi_offset 13, -32
	.cfi_offset 12, -40
	.cfi_offset 3, -48
	movq	%rdi, %rbx
	andq	$-32, %rsp
	subq	$64, %rsp
	movq	8(%rdi), %r12
	movq	(%rdi), %r13
	movq	%fs:40, %rax
	movq	%rax, 56(%rsp)
	xorl	%eax, %eax
	call	omp_get_num_threads@PLT
	movl	%eax, %r14d
	call	omp_get_thread_num@PLT
	movl	%eax, %ecx
	movl	16(%rbx), %eax
	cltd
	idivl	%r14d
	cmpl	%edx, %ecx
	jl	.L16
.L27:
	imull	%eax, %ecx
	vxorps	%xmm0, %xmm0, %xmm0
	movslq	%ecx, %rdi
	leal	(%rdx,%rdi), %ecx
	leal	(%rax,%rcx), %r8d
	cmpl	%r8d, %ecx
	jge	.L17
	vpxor	%xmm0, %xmm0, %xmm0
	leal	-1(%rax), %esi
	vmovdqa	%xmm0, (%rsp)
	vmovdqa	%xmm0, 16(%rsp)
	cmpl	$6, %esi
	jbe	.L24
	movslq	%edx, %rsi
	vxorps	%xmm1, %xmm1, %xmm1
	xorl	%edx, %edx
	addq	%rdi, %rsi
	movl	%eax, %edi
	salq	$2, %rsi
	shrl	$3, %edi
	leaq	0(%r13,%rsi), %r9
	salq	$5, %rdi
	addq	%r12, %rsi
	.p2align 4,,10
	.p2align 3
.L20:
	vmovups	(%r9,%rdx), %ymm2
	vmulps	(%rsi,%rdx), %ymm2, %ymm0
	addq	$32, %rdx
	vaddps	%ymm0, %ymm1, %ymm1
	cmpq	%rdx, %rdi
	jne	.L20
	movl	%eax, %edx
	vmovaps	%ymm1, (%rsp)
	andl	$-8, %edx
	addl	%edx, %ecx
	cmpl	%edx, %eax
	je	.L38
	vzeroupper
.L24:
	vmovss	(%rsp), %xmm1
	movslq	%ecx, %rax
	.p2align 4,,10
	.p2align 3
.L22:
	vmovss	0(%r13,%rax,4), %xmm0
	vmulss	(%r12,%rax,4), %xmm0, %xmm0
	addq	$1, %rax
	vaddss	%xmm0, %xmm1, %xmm1
	cmpl	%eax, %r8d
	jg	.L22
	vmovss	%xmm1, (%rsp)
.L21:
	movq	%rsp, %rax
	leaq	32(%rsp), %rdx
	vxorps	%xmm0, %xmm0, %xmm0
	.p2align 4,,10
	.p2align 3
.L19:
	vaddss	(%rax), %xmm0, %xmm0
	addq	$4, %rax
	cmpq	%rax, %rdx
	jne	.L19
.L17:
	movl	20(%rbx), %edx
	leaq	20(%rbx), %rcx
.L26:
	vmovd	%edx, %xmm4
	movl	%edx, %eax
	vaddss	%xmm4, %xmm0, %xmm3
	vmovd	%xmm3, %esi
	lock cmpxchgl	%esi, (%rcx)
	jne	.L41
	movq	56(%rsp), %rax
	subq	%fs:40, %rax
	jne	.L42
	leaq	-32(%rbp), %rsp
	popq	%rbx
	popq	%r12
	popq	%r13
	popq	%r14
	popq	%rbp
	.cfi_remember_state
	.cfi_def_cfa 7, 8
	ret
	.p2align 4,,10
	.p2align 3
.L16:
	.cfi_restore_state
	addl	$1, %eax
	xorl	%edx, %edx
	jmp	.L27
	.p2align 4,,10
	.p2align 3
.L38:
	vzeroupper
	jmp	.L21
.L42:
	call	__stack_chk_fail@PLT
.L41:
	movl	%eax, %edx
	jmp	.L26
	.cfi_endproc
.LFE7774:
	.size	_Z9dot_simd3PKfS0_i._omp_fn.0, .-_Z9dot_simd3PKfS0_i._omp_fn.0
	.p2align 4
	.type	_Z4prodPKfi._omp_fn.0, @function
_Z4prodPKfi._omp_fn.0:
.LFB7775:
	.cfi_startproc
	endbr64
	pushq	%r12
	.cfi_def_cfa_offset 16
	.cfi_offset 12, -16
	movq	(%rdi), %r12
	pushq	%rbp
	.cfi_def_cfa_offset 24
	.cfi_offset 6, -24
	pushq	%rbx
	.cfi_def_cfa_offset 32
	.cfi_offset 3, -32
	movq	%rdi, %rbx
	call	omp_get_num_threads@PLT
	movl	%eax, %ebp
	call	omp_get_thread_num@PLT
	movl	%eax, %ecx
	movl	8(%rbx), %eax
	cltd
	idivl	%ebp
	cmpl	%edx, %ecx
	jl	.L44
.L48:
	imull	%eax, %ecx
	vmovss	.LC1(%rip), %xmm0
	addl	%ecx, %edx
	leal	(%rax,%rdx), %ecx
	cmpl	%ecx, %edx
	jge	.L45
	movslq	%edx, %rdx
	subl	$1, %eax
	addq	%rdx, %rax
	leaq	(%r12,%rdx,4), %rcx
	leaq	4(%r12,%rax,4), %rax
	.p2align 4,,10
	.p2align 3
.L46:
	vmulss	(%rcx), %xmm0, %xmm0
	addq	$4, %rcx
	cmpq	%rax, %rcx
	jne	.L46
.L45:
	movl	12(%rbx), %edx
	leaq	12(%rbx), %rcx
.L47:
	vmovd	%edx, %xmm2
	movl	%edx, %eax
	vmulss	%xmm2, %xmm0, %xmm1
	vmovd	%xmm1, %esi
	lock cmpxchgl	%esi, (%rcx)
	jne	.L55
	popq	%rbx
	.cfi_remember_state
	.cfi_def_cfa_offset 24
	popq	%rbp
	.cfi_def_cfa_offset 16
	popq	%r12
	.cfi_def_cfa_offset 8
	ret
	.p2align 4,,10
	.p2align 3
.L44:
	.cfi_restore_state
	addl	$1, %eax
	xorl	%edx, %edx
	jmp	.L48
.L55:
	movl	%eax, %edx
	jmp	.L47
	.cfi_endproc
.LFE7775:
	.size	_Z4prodPKfi._omp_fn.0, .-_Z4prodPKfi._omp_fn.0
	.p2align 4
	.globl	_Z8dot_simdPKfS0_i
	.type	_Z8dot_simdPKfS0_i, @function
_Z8dot_simdPKfS0_i:
.LFB7285:
	.cfi_startproc
	endbr64
	movq	%rdi, %rcx
	testl	%edx, %edx
	jle	.L65
	leal	-1(%rdx), %r8d
	cmpl	$6, %r8d
	jbe	.L66
	movl	%edx, %edi
	xorl	%eax, %eax
	vxorps	%xmm0, %xmm0, %xmm0
	shrl	$3, %edi
	salq	$5, %rdi
	.p2align 4,,10
	.p2align 3
.L59:
	vmovups	(%rcx,%rax), %ymm4
	vmulps	(%rsi,%rax), %ymm4, %ymm1
	addq	$32, %rax
	vaddss	%xmm1, %xmm0, %xmm0
	vshufps	$85, %xmm1, %xmm1, %xmm3
	vshufps	$255, %xmm1, %xmm1, %xmm2
	vaddss	%xmm3, %xmm0, %xmm0
	vunpckhps	%xmm1, %xmm1, %xmm3
	vextractf128	$0x1, %ymm1, %xmm1
	vaddss	%xmm3, %xmm0, %xmm0
	vaddss	%xmm2, %xmm0, %xmm0
	vshufps	$85, %xmm1, %xmm1, %xmm2
	vaddss	%xmm1, %xmm0, %xmm0
	vaddss	%xmm2, %xmm0, %xmm0
	vunpckhps	%xmm1, %xmm1, %xmm2
	vshufps	$255, %xmm1, %xmm1, %xmm1
	vaddss	%xmm2, %xmm0, %xmm0
	vaddss	%xmm1, %xmm0, %xmm0
	cmpq	%rax, %rdi
	jne	.L59
	movl	%edx, %edi
	andl	$-8, %edi
	movl	%edi, %eax
	cmpl	%edi, %edx
	je	.L70
	vzeroupper
.L58:
	movl	%edx, %r9d
	subl	%edi, %r8d
	subl	%edi, %r9d
	cmpl	$2, %r8d
	jbe	.L62
	vmovups	(%rcx,%rdi,4), %xmm5
	vmulps	(%rsi,%rdi,4), %xmm5, %xmm1
	movl	%r9d, %edi
	andl	$-4, %edi
	addl	%edi, %eax
	vaddss	%xmm1, %xmm0, %xmm0
	vshufps	$85, %xmm1, %xmm1, %xmm2
	vaddss	%xmm2, %xmm0, %xmm0
	vunpckhps	%xmm1, %xmm1, %xmm2
	vshufps	$255, %xmm1, %xmm1, %xmm1
	vaddss	%xmm2, %xmm0, %xmm0
	vaddss	%xmm1, %xmm0, %xmm0
	cmpl	%edi, %r9d
	je	.L56
.L62:
	cltq
	.p2align 4,,10
	.p2align 3
.L64:
	vmovss	(%rcx,%rax,4), %xmm1
	vmulss	(%rsi,%rax,4), %xmm1, %xmm1
	addq	$1, %rax
	vaddss	%xmm1, %xmm0, %xmm0
	cmpl	%eax, %edx
	jg	.L64
	ret
	.p2align 4,,10
	.p2align 3
.L65:
	vxorps	%xmm0, %xmm0, %xmm0
.L56:
	ret
	.p2align 4,,10
	.p2align 3
.L70:
	vzeroupper
	ret
	.p2align 4,,10
	.p2align 3
.L66:
	xorl	%edi, %edi
	xorl	%eax, %eax
	vxorps	%xmm0, %xmm0, %xmm0
	jmp	.L58
	.cfi_endproc
.LFE7285:
	.size	_Z8dot_simdPKfS0_i, .-_Z8dot_simdPKfS0_i
	.p2align 4
	.globl	_Z9dot_simd2PKfS0_i
	.type	_Z9dot_simd2PKfS0_i, @function
_Z9dot_simd2PKfS0_i:
.LFB7286:
	.cfi_startproc
	endbr64
	subq	$40, %rsp
	.cfi_def_cfa_offset 48
	xorl	%ecx, %ecx
	movq	%fs:40, %rax
	movq	%rax, 24(%rsp)
	xorl	%eax, %eax
	movl	%edx, 16(%rsp)
	xorl	%edx, %edx
	movq	%rsi, 8(%rsp)
	movq	%rsp, %rsi
	movq	%rdi, (%rsp)
	leaq	_Z9dot_simd2PKfS0_i._omp_fn.0(%rip), %rdi
	movl	$0x00000000, 20(%rsp)
	call	GOMP_parallel@PLT
	vmovss	20(%rsp), %xmm0
	movq	24(%rsp), %rax
	subq	%fs:40, %rax
	jne	.L74
	addq	$40, %rsp
	.cfi_remember_state
	.cfi_def_cfa_offset 8
	ret
.L74:
	.cfi_restore_state
	call	__stack_chk_fail@PLT
	.cfi_endproc
.LFE7286:
	.size	_Z9dot_simd2PKfS0_i, .-_Z9dot_simd2PKfS0_i
	.p2align 4
	.globl	_Z9dot_simd3PKfS0_i
	.type	_Z9dot_simd3PKfS0_i, @function
_Z9dot_simd3PKfS0_i:
.LFB7287:
	.cfi_startproc
	endbr64
	subq	$40, %rsp
	.cfi_def_cfa_offset 48
	xorl	%ecx, %ecx
	movq	%fs:40, %rax
	movq	%rax, 24(%rsp)
	xorl	%eax, %eax
	movl	%edx, 16(%rsp)
	xorl	%edx, %edx
	movq	%rsi, 8(%rsp)
	movq	%rsp, %rsi
	movq	%rdi, (%rsp)
	leaq	_Z9dot_simd3PKfS0_i._omp_fn.0(%rip), %rdi
	movl	$0x00000000, 20(%rsp)
	call	GOMP_parallel@PLT
	vmovss	20(%rsp), %xmm0
	movq	24(%rsp), %rax
	subq	%fs:40, %rax
	jne	.L78
	addq	$40, %rsp
	.cfi_remember_state
	.cfi_def_cfa_offset 8
	ret
.L78:
	.cfi_restore_state
	call	__stack_chk_fail@PLT
	.cfi_endproc
.LFE7287:
	.size	_Z9dot_simd3PKfS0_i, .-_Z9dot_simd3PKfS0_i
	.p2align 4
	.globl	_Z4prodPKfi
	.type	_Z4prodPKfi, @function
_Z4prodPKfi:
.LFB7288:
	.cfi_startproc
	endbr64
	subq	$40, %rsp
	.cfi_def_cfa_offset 48
	xorl	%ecx, %ecx
	xorl	%edx, %edx
	movq	%fs:40, %rax
	movq	%rax, 24(%rsp)
	xorl	%eax, %eax
	movl	%esi, 8(%rsp)
	movq	%rsp, %rsi
	movq	%rdi, (%rsp)
	leaq	_Z4prodPKfi._omp_fn.0(%rip), %rdi
	movl	$0x3f800000, 12(%rsp)
	call	GOMP_parallel@PLT
	vmovss	12(%rsp), %xmm0
	movq	24(%rsp), %rax
	subq	%fs:40, %rax
	jne	.L82
	addq	$40, %rsp
	.cfi_remember_state
	.cfi_def_cfa_offset 8
	ret
.L82:
	.cfi_restore_state
	call	__stack_chk_fail@PLT
	.cfi_endproc
.LFE7288:
	.size	_Z4prodPKfi, .-_Z4prodPKfi
	.p2align 4
	.globl	_Z14horizontal_sumDv8_f
	.type	_Z14horizontal_sumDv8_f, @function
_Z14horizontal_sumDv8_f:
.LFB7289:
	.cfi_startproc
	endbr64
	vhaddps	%ymm0, %ymm0, %ymm1
	vmovaps	%xmm1, %xmm0
	vextractf128	$0x1, %ymm1, %xmm1
	vaddps	%xmm1, %xmm0, %xmm0
	vhaddps	%xmm0, %xmm0, %xmm0
	vhaddps	%xmm0, %xmm0, %xmm0
	ret
	.cfi_endproc
.LFE7289:
	.size	_Z14horizontal_sumDv8_f, .-_Z14horizontal_sumDv8_f
	.p2align 4
	.globl	_Z14dot_avx2manualPKfS0_i
	.type	_Z14dot_avx2manualPKfS0_i, @function
_Z14dot_avx2manualPKfS0_i:
.LFB7290:
	.cfi_startproc
	endbr64
	testl	%edx, %edx
	jle	.L87
	xorl	%eax, %eax
	vxorps	%xmm0, %xmm0, %xmm0
	.p2align 4,,10
	.p2align 3
.L86:
	vmovups	(%rsi,%rax,4), %ymm2
	vmulps	(%rdi,%rax,4), %ymm2, %ymm1
	addq	$8, %rax
	vaddps	%ymm1, %ymm0, %ymm0
	cmpl	%eax, %edx
	jg	.L86
.L85:
	vhaddps	%ymm0, %ymm0, %ymm0
	vmovaps	%xmm0, %xmm1
	vextractf128	$0x1, %ymm0, %xmm0
	vaddps	%xmm0, %xmm1, %xmm0
	vhaddps	%xmm0, %xmm0, %xmm0
	vhaddps	%xmm0, %xmm0, %xmm0
	vzeroupper
	ret
	.p2align 4,,10
	.p2align 3
.L87:
	vxorps	%xmm0, %xmm0, %xmm0
	jmp	.L85
	.cfi_endproc
.LFE7290:
	.size	_Z14dot_avx2manualPKfS0_i, .-_Z14dot_avx2manualPKfS0_i
	.section	.text.startup,"ax",@progbits
	.p2align 4
	.type	_GLOBAL__sub_I__Z8dot_simdPKfS0_i, @function
_GLOBAL__sub_I__Z8dot_simdPKfS0_i:
.LFB7772:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	leaq	_ZStL8__ioinit(%rip), %rbp
	movq	%rbp, %rdi
	call	_ZNSt8ios_base4InitC1Ev@PLT
	movq	_ZNSt8ios_base4InitD1Ev@GOTPCREL(%rip), %rdi
	movq	%rbp, %rsi
	popq	%rbp
	.cfi_def_cfa_offset 8
	leaq	__dso_handle(%rip), %rdx
	jmp	__cxa_atexit@PLT
	.cfi_endproc
.LFE7772:
	.size	_GLOBAL__sub_I__Z8dot_simdPKfS0_i, .-_GLOBAL__sub_I__Z8dot_simdPKfS0_i
	.section	.init_array,"aw"
	.align 8
	.quad	_GLOBAL__sub_I__Z8dot_simdPKfS0_i
	.local	_ZStL8__ioinit
	.comm	_ZStL8__ioinit,1,1
	.section	.rodata.cst4,"aM",@progbits,4
	.align 4
.LC1:
	.long	1065353216
	.hidden	__dso_handle
	.ident	"GCC: (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0"
	.section	.note.GNU-stack,"",@progbits
	.section	.note.gnu.property,"a"
	.align 8
	.long	1f - 0f
	.long	4f - 1f
	.long	5
0:
	.string	"GNU"
1:
	.align 8
	.long	0xc0000002
	.long	3f - 2f
2:
	.long	0x3
3:
	.align 8
4:
