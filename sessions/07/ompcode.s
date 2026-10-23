	.file	"ompcode.cpp"
	.text
	.p2align 4
	.globl	_Z14compute_serialPKdS0_Pd
	.type	_Z14compute_serialPKdS0_Pd, @function
_Z14compute_serialPKdS0_Pd:
.LFB3971:
	.cfi_startproc
	endbr64
	movq	%rdi, %rcx
	movq	%rdx, %rax
	leaq	8(%rsi), %rdi
	subq	%rdi, %rax
	cmpq	$16, %rax
	jbe	.L5
	leaq	8(%rcx), %rdi
	movq	%rdx, %rax
	subq	%rdi, %rax
	cmpq	$16, %rax
	jbe	.L5
	vmovapd	.LC0(%rip), %ymm2
	vmovapd	.LC1(%rip), %ymm1
	xorl	%eax, %eax
	.p2align 4,,10
	.p2align 3
.L3:
	vmulpd	(%rsi,%rax), %ymm2, %ymm0
	vfmadd231pd	(%rcx,%rax), %ymm1, %ymm0
	vmovupd	%ymm0, (%rdx,%rax)
	addq	$32, %rax
	cmpq	$128000000, %rax
	jne	.L3
	vzeroupper
	ret
.L5:
	vmovsd	.LC2(%rip), %xmm2
	vmovsd	.LC3(%rip), %xmm1
	xorl	%eax, %eax
	.p2align 4,,10
	.p2align 3
.L2:
	vmulsd	(%rsi,%rax,8), %xmm2, %xmm0
	vfmadd231sd	(%rcx,%rax,8), %xmm1, %xmm0
	vmovsd	%xmm0, (%rdx,%rax,8)
	addq	$1, %rax
	cmpq	$16000000, %rax
	jne	.L2
	ret
	.cfi_endproc
.LFE3971:
	.size	_Z14compute_serialPKdS0_Pd, .-_Z14compute_serialPKdS0_Pd
	.p2align 4
	.globl	_Z12compute_simdPKdS0_Pd
	.type	_Z12compute_simdPKdS0_Pd, @function
_Z12compute_simdPKdS0_Pd:
.LFB3974:
	.cfi_startproc
	endbr64
	vmovapd	.LC0(%rip), %ymm2
	vmovapd	.LC1(%rip), %ymm1
	xorl	%eax, %eax
	.p2align 4,,10
	.p2align 3
.L17:
	vmulpd	(%rsi,%rax), %ymm2, %ymm0
	vfmadd231pd	(%rdi,%rax), %ymm1, %ymm0
	vmovupd	%ymm0, (%rdx,%rax)
	addq	$32, %rax
	cmpq	$128000000, %rax
	jne	.L17
	vzeroupper
	ret
	.cfi_endproc
.LFE3974:
	.size	_Z12compute_simdPKdS0_Pd, .-_Z12compute_simdPKdS0_Pd
	.p2align 4
	.type	_Z10initializeRSt6vectorIdSaIdEES2_._omp_fn.0, @function
_Z10initializeRSt6vectorIdSaIdEES2_._omp_fn.0:
.LFB4735:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rdi, %rbp
	pushq	%rbx
	.cfi_def_cfa_offset 24
	.cfi_offset 3, -24
	subq	$8, %rsp
	.cfi_def_cfa_offset 32
	call	omp_get_num_threads@PLT
	movslq	%eax, %rbx
	call	omp_get_thread_num@PLT
	xorl	%edx, %edx
	movslq	%eax, %rcx
	movl	$16000000, %eax
	divq	%rbx
	cmpq	%rdx, %rcx
	jb	.L20
.L23:
	imulq	%rax, %rcx
	addq	%rdx, %rcx
	leaq	(%rax,%rcx), %rsi
	cmpq	%rsi, %rcx
	jnb	.L25
	movq	0(%rbp), %rax
	vmovsd	.LC4(%rip), %xmm5
	vxorps	%xmm1, %xmm1, %xmm1
	movabsq	$2361183241434822607, %rdi
	vmovsd	.LC5(%rip), %xmm4
	vmovsd	.LC6(%rip), %xmm3
	movq	(%rax), %r9
	movq	8(%rbp), %rax
	vmovsd	.LC7(%rip), %xmm2
	movq	(%rax), %r8
	.p2align 4,,10
	.p2align 3
.L22:
	movq	%rcx, %rdx
	shrq	$3, %rdx
	movq	%rdx, %rax
	mulq	%rdi
	movq	%rcx, %rax
	shrq	$4, %rdx
	imulq	$1000, %rdx, %rdx
	subq	%rdx, %rax
	movq	%rcx, %rdx
	shrq	$2, %rdx
	vcvtsi2sdq	%rax, %xmm1, %xmm0
	movq	%rdx, %rax
	mulq	%rdi
	movq	%rcx, %rax
	vfmadd132sd	%xmm5, %xmm4, %xmm0
	shrq	$4, %rdx
	imulq	$500, %rdx, %rdx
	subq	%rdx, %rax
	vmovsd	%xmm0, (%r9,%rcx,8)
	vcvtsi2sdq	%rax, %xmm1, %xmm0
	vfnmadd132sd	%xmm3, %xmm2, %xmm0
	vmovsd	%xmm0, (%r8,%rcx,8)
	addq	$1, %rcx
	cmpq	%rcx, %rsi
	jne	.L22
.L25:
	addq	$8, %rsp
	.cfi_remember_state
	.cfi_def_cfa_offset 24
	popq	%rbx
	.cfi_def_cfa_offset 16
	popq	%rbp
	.cfi_def_cfa_offset 8
	ret
	.p2align 4,,10
	.p2align 3
.L20:
	.cfi_restore_state
	addq	$1, %rax
	xorl	%edx, %edx
	jmp	.L23
	.cfi_endproc
.LFE4735:
	.size	_Z10initializeRSt6vectorIdSaIdEES2_._omp_fn.0, .-_Z10initializeRSt6vectorIdSaIdEES2_._omp_fn.0
	.p2align 4
	.globl	_Z16compute_parallelPKdS0_Pd
	.type	_Z16compute_parallelPKdS0_Pd, @function
_Z16compute_parallelPKdS0_Pd:
.LFB3972:
	.cfi_startproc
	endbr64
	subq	$40, %rsp
	.cfi_def_cfa_offset 48
	vmovq	%rdi, %xmm1
	xorl	%ecx, %ecx
	movq	%fs:40, %rax
	movq	%rax, 24(%rsp)
	xorl	%eax, %eax
	vpinsrq	$1, %rsi, %xmm1, %xmm0
	movq	%rdx, 16(%rsp)
	movq	%rsp, %rsi
	xorl	%edx, %edx
	leaq	_Z16compute_parallelPKdS0_Pd._omp_fn.0(%rip), %rdi
	vmovdqa	%xmm0, (%rsp)
	call	GOMP_parallel@PLT
	movq	24(%rsp), %rax
	subq	%fs:40, %rax
	jne	.L31
	addq	$40, %rsp
	.cfi_remember_state
	.cfi_def_cfa_offset 8
	ret
.L31:
	.cfi_restore_state
	call	__stack_chk_fail@PLT
	.cfi_endproc
.LFE3972:
	.size	_Z16compute_parallelPKdS0_Pd, .-_Z16compute_parallelPKdS0_Pd
	.p2align 4
	.globl	_Z21compute_parallel_simdPKdS0_Pd
	.type	_Z21compute_parallel_simdPKdS0_Pd, @function
_Z21compute_parallel_simdPKdS0_Pd:
.LFB3975:
	.cfi_startproc
	endbr64
	subq	$40, %rsp
	.cfi_def_cfa_offset 48
	vmovq	%rdi, %xmm1
	xorl	%ecx, %ecx
	movq	%fs:40, %rax
	movq	%rax, 24(%rsp)
	xorl	%eax, %eax
	vpinsrq	$1, %rsi, %xmm1, %xmm0
	movq	%rdx, 16(%rsp)
	movq	%rsp, %rsi
	xorl	%edx, %edx
	leaq	_Z21compute_parallel_simdPKdS0_Pd._omp_fn.0(%rip), %rdi
	vmovdqa	%xmm0, (%rsp)
	call	GOMP_parallel@PLT
	movq	24(%rsp), %rax
	subq	%fs:40, %rax
	jne	.L36
	addq	$40, %rsp
	.cfi_remember_state
	.cfi_def_cfa_offset 8
	ret
.L36:
	.cfi_restore_state
	call	__stack_chk_fail@PLT
	.cfi_endproc
.LFE3975:
	.size	_Z21compute_parallel_simdPKdS0_Pd, .-_Z21compute_parallel_simdPKdS0_Pd
	.p2align 4
	.type	_Z16compute_parallelPKdS0_Pd._omp_fn.0, @function
_Z16compute_parallelPKdS0_Pd._omp_fn.0:
.LFB4736:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	pushq	%r12
	.cfi_offset 12, -24
	movq	%rdi, %r12
	pushq	%rbx
	andq	$-32, %rsp
	.cfi_offset 3, -32
	call	omp_get_num_threads@PLT
	movslq	%eax, %rbx
	call	omp_get_thread_num@PLT
	xorl	%edx, %edx
	movslq	%eax, %rcx
	movl	$16000000, %eax
	divq	%rbx
	cmpq	%rdx, %rcx
	jb	.L38
.L47:
	imulq	%rax, %rcx
	addq	%rcx, %rdx
	leaq	(%rax,%rdx), %rcx
	cmpq	%rcx, %rdx
	jnb	.L61
	movq	16(%r12), %r9
	leaq	0(,%rdx,8), %r8
	movq	(%r12), %rsi
	leaq	8(%r8), %r11
	movq	8(%r12), %rdi
	leaq	(%r9,%r8), %r10
	leaq	(%rsi,%r11), %r12
	movq	%r10, %rbx
	subq	%r12, %rbx
	movq	%r10, %r12
	cmpq	$16, %rbx
	seta	%bl
	addq	%rdi, %r11
	subq	%r11, %r12
	cmpq	$16, %r12
	seta	%r11b
	testb	%r11b, %bl
	je	.L63
	cmpq	$1, %rax
	je	.L63
	leaq	-1(%rax), %rcx
	cmpq	$2, %rcx
	jbe	.L48
	movq	%rax, %r11
	vmovapd	.LC0(%rip), %ymm2
	vmovapd	.LC1(%rip), %ymm1
	xorl	%ecx, %ecx
	shrq	$2, %r11
	leaq	(%rsi,%r8), %rbx
	addq	%rdi, %r8
	salq	$5, %r11
	.p2align 4,,10
	.p2align 3
.L42:
	vmulpd	(%r8,%rcx), %ymm2, %ymm0
	vfmadd231pd	(%rbx,%rcx), %ymm1, %ymm0
	vmovupd	%ymm0, (%r10,%rcx)
	addq	$32, %rcx
	cmpq	%rcx, %r11
	jne	.L42
	movq	%rax, %r8
	andq	$-4, %r8
	addq	%r8, %rdx
	movq	%rdx, %rcx
	cmpq	%r8, %rax
	je	.L59
	subq	%r8, %rax
	cmpq	$1, %rax
	je	.L64
	vzeroupper
.L41:
	vmovapd	.LC8(%rip), %xmm0
	vmulpd	(%rdi,%rdx,8), %xmm0, %xmm0
	vmovupd	(%rsi,%rdx,8), %xmm4
	vfmadd231pd	.LC9(%rip), %xmm4, %xmm0
	vmovupd	%xmm0, (%r9,%rdx,8)
	movq	%rax, %rdx
	andq	$-2, %rdx
	addq	%rdx, %rcx
	cmpq	%rdx, %rax
	je	.L61
.L44:
	vmovsd	.LC2(%rip), %xmm0
	vmulsd	(%rdi,%rcx,8), %xmm0, %xmm0
	vmovsd	(%rsi,%rcx,8), %xmm3
	vfmadd231sd	.LC3(%rip), %xmm3, %xmm0
	vmovsd	%xmm0, (%r9,%rcx,8)
.L61:
	leaq	-16(%rbp), %rsp
	popq	%rbx
	popq	%r12
	popq	%rbp
	.cfi_remember_state
	.cfi_def_cfa 7, 8
	ret
	.p2align 4,,10
	.p2align 3
.L38:
	.cfi_restore_state
	addq	$1, %rax
	xorl	%edx, %edx
	jmp	.L47
	.p2align 4,,10
	.p2align 3
.L63:
	vmovsd	.LC2(%rip), %xmm2
	vmovsd	.LC3(%rip), %xmm1
	.p2align 4,,10
	.p2align 3
.L53:
	vmulsd	(%rdi,%rdx,8), %xmm2, %xmm0
	vfmadd231sd	(%rsi,%rdx,8), %xmm1, %xmm0
	vmovsd	%xmm0, (%r9,%rdx,8)
	addq	$1, %rdx
	cmpq	%rdx, %rcx
	jne	.L53
	jmp	.L61
	.p2align 4,,10
	.p2align 3
.L59:
	vzeroupper
	jmp	.L61
.L48:
	movq	%rdx, %rcx
	jmp	.L41
.L64:
	vzeroupper
	jmp	.L44
	.cfi_endproc
.LFE4736:
	.size	_Z16compute_parallelPKdS0_Pd._omp_fn.0, .-_Z16compute_parallelPKdS0_Pd._omp_fn.0
	.p2align 4
	.type	_Z15compute_dynamicPKdS0_Pd._omp_fn.0, @function
_Z15compute_dynamicPKdS0_Pd._omp_fn.0:
.LFB4737:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	pushq	%r15
	pushq	%r14
	pushq	%r13
	pushq	%r12
	pushq	%rbx
	andq	$-32, %rsp
	subq	$64, %rsp
	.cfi_offset 15, -24
	.cfi_offset 14, -32
	.cfi_offset 13, -40
	.cfi_offset 12, -48
	.cfi_offset 3, -56
	movq	16(%rdi), %r15
	movq	8(%rdi), %r13
	movq	%fs:40, %rax
	movq	%rax, 56(%rsp)
	xorl	%eax, %eax
	leaq	48(%rsp), %rbx
	leaq	40(%rsp), %r12
	movq	(%rdi), %r14
	movq	%rbx, %rsi
	movq	%r12, %rdi
	call	GOMP_loop_nonmonotonic_dynamic_next@PLT
	testb	%al, %al
	je	.L66
	vmovapd	.LC0(%rip), %ymm3
	vmovapd	.LC1(%rip), %ymm2
	.p2align 4,,10
	.p2align 3
.L74:
	movq	40(%rsp), %rax
	movq	48(%rsp), %rdi
	leaq	0(,%rax,8), %rdx
	movq	%rdi, %r8
	leaq	1(%rax), %rcx
	leaq	(%r15,%rdx), %rsi
	leaq	8(%rdx), %r9
	subq	%rax, %r8
	leaq	0(%r13,%r9), %r11
	movq	%rsi, %r10
	subq	%r11, %r10
	cmpq	$16, %r10
	seta	31(%rsp)
	cmpq	$1, %r8
	setne	%r10b
	cmpq	%rcx, %rdi
	setnb	%r11b
	andl	%r11d, %r10d
	testb	%r10b, 31(%rsp)
	je	.L93
	addq	%r14, %r9
	movq	%rsi, %r10
	subq	%r9, %r10
	cmpq	$16, %r10
	jbe	.L93
	cmpq	%rcx, %rdi
	movl	$1, %edi
	cmovnb	%r8, %rdi
	subq	$1, %r8
	cmpq	$2, %r8
	jbe	.L76
	movq	%rdi, %r8
	leaq	(%r14,%rdx), %r9
	xorl	%ecx, %ecx
	addq	%r13, %rdx
	shrq	$2, %r8
	salq	$5, %r8
	.p2align 4,,10
	.p2align 3
.L69:
	vmulpd	(%rdx,%rcx), %ymm3, %ymm0
	vfmadd231pd	(%r9,%rcx), %ymm2, %ymm0
	vmovupd	%ymm0, (%rsi,%rcx)
	addq	$32, %rcx
	cmpq	%rcx, %r8
	jne	.L69
	movq	%rdi, %rcx
	andq	$-4, %rcx
	leaq	(%rax,%rcx), %rdx
	cmpq	%rcx, %rdi
	je	.L73
.L68:
	subq	%rcx, %rdi
	cmpq	$1, %rdi
	je	.L71
	addq	%rcx, %rax
	vmovupd	0(%r13,%rax,8), %xmm6
	vmulpd	.LC8(%rip), %xmm6, %xmm0
	vmovupd	(%r14,%rax,8), %xmm7
	vfmadd231pd	.LC9(%rip), %xmm7, %xmm0
	vmovupd	%xmm0, (%r15,%rax,8)
	movq	%rdi, %rax
	andq	$-2, %rax
	addq	%rax, %rdx
	cmpq	%rax, %rdi
	je	.L73
.L71:
	vmovsd	.LC2(%rip), %xmm0
	vmulsd	0(%r13,%rdx,8), %xmm0, %xmm0
	vmovsd	(%r14,%rdx,8), %xmm5
	vfmadd231sd	.LC3(%rip), %xmm5, %xmm0
	vmovsd	%xmm0, (%r15,%rdx,8)
.L73:
	movq	%rbx, %rsi
	movq	%r12, %rdi
	vzeroupper
	call	GOMP_loop_nonmonotonic_dynamic_next@PLT
	vmovapd	.LC0(%rip), %ymm3
	vmovapd	.LC1(%rip), %ymm2
	testb	%al, %al
	jne	.L74
	vzeroupper
.L66:
	call	GOMP_loop_end_nowait@PLT
	movq	56(%rsp), %rax
	subq	%fs:40, %rax
	jne	.L94
	leaq	-40(%rbp), %rsp
	popq	%rbx
	popq	%r12
	popq	%r13
	popq	%r14
	popq	%r15
	popq	%rbp
	.cfi_remember_state
	.cfi_def_cfa 7, 8
	ret
	.p2align 4,,10
	.p2align 3
.L93:
	.cfi_restore_state
	vmovsd	.LC2(%rip), %xmm0
	vmovsd	.LC3(%rip), %xmm4
	jmp	.L89
	.p2align 4,,10
	.p2align 3
.L95:
	addq	$1, %rcx
.L89:
	vmulsd	0(%r13,%rax,8), %xmm0, %xmm1
	vfmadd231sd	(%r14,%rax,8), %xmm4, %xmm1
	vmovsd	%xmm1, (%r15,%rax,8)
	movq	%rcx, %rax
	cmpq	%rcx, %rdi
	ja	.L95
	jmp	.L73
.L76:
	movq	%rax, %rdx
	xorl	%ecx, %ecx
	jmp	.L68
.L94:
	call	__stack_chk_fail@PLT
	.cfi_endproc
.LFE4737:
	.size	_Z15compute_dynamicPKdS0_Pd._omp_fn.0, .-_Z15compute_dynamicPKdS0_Pd._omp_fn.0
	.p2align 4
	.globl	_Z15compute_dynamicPKdS0_Pd
	.type	_Z15compute_dynamicPKdS0_Pd, @function
_Z15compute_dynamicPKdS0_Pd:
.LFB3973:
	.cfi_startproc
	endbr64
	vmovq	%rdi, %xmm1
	subq	$40, %rsp
	.cfi_def_cfa_offset 48
	movl	$1, %r9d
	xorl	%ecx, %ecx
	movq	%fs:40, %rax
	movq	%rax, 24(%rsp)
	xorl	%eax, %eax
	vpinsrq	$1, %rsi, %xmm1, %xmm0
	movq	%rdx, 16(%rsp)
	xorl	%edx, %edx
	vmovdqa	%xmm0, (%rsp)
	movq	%rsp, %rsi
	movl	$16000000, %r8d
	leaq	_Z15compute_dynamicPKdS0_Pd._omp_fn.0(%rip), %rdi
	pushq	$0
	.cfi_def_cfa_offset 56
	pushq	$1
	.cfi_def_cfa_offset 64
	call	GOMP_parallel_loop_nonmonotonic_dynamic@PLT
	popq	%rax
	.cfi_def_cfa_offset 56
	popq	%rdx
	.cfi_def_cfa_offset 48
	movq	24(%rsp), %rax
	subq	%fs:40, %rax
	jne	.L100
	addq	$40, %rsp
	.cfi_remember_state
	.cfi_def_cfa_offset 8
	ret
.L100:
	.cfi_restore_state
	call	__stack_chk_fail@PLT
	.cfi_endproc
.LFE3973:
	.size	_Z15compute_dynamicPKdS0_Pd, .-_Z15compute_dynamicPKdS0_Pd
	.p2align 4
	.type	_Z21compute_parallel_simdPKdS0_Pd._omp_fn.0, @function
_Z21compute_parallel_simdPKdS0_Pd._omp_fn.0:
.LFB4738:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	pushq	%r12
	.cfi_offset 12, -24
	movq	%rdi, %r12
	pushq	%rbx
	andq	$-32, %rsp
	.cfi_offset 3, -32
	call	omp_get_num_threads@PLT
	movslq	%eax, %rbx
	call	omp_get_thread_num@PLT
	xorl	%edx, %edx
	movslq	%eax, %rcx
	movl	$16000000, %eax
	divq	%rbx
	cmpq	%rdx, %rcx
	jb	.L102
.L109:
	imulq	%rax, %rcx
	addq	%rcx, %rdx
	leaq	(%rax,%rdx), %rcx
	cmpq	%rcx, %rdx
	jnb	.L116
	leaq	-1(%rax), %rcx
	movq	16(%r12), %r8
	movq	8(%r12), %r10
	movq	(%r12), %r9
	cmpq	$2, %rcx
	jbe	.L110
	movq	%rax, %rdi
	leaq	0(,%rdx,8), %rsi
	xorl	%ecx, %ecx
	vmovapd	.LC0(%rip), %ymm2
	shrq	$2, %rdi
	vmovapd	.LC1(%rip), %ymm1
	leaq	(%r9,%rsi), %rbx
	leaq	(%r10,%rsi), %r11
	salq	$5, %rdi
	addq	%r8, %rsi
	.p2align 4,,10
	.p2align 3
.L105:
	vmulpd	(%r11,%rcx), %ymm2, %ymm0
	vfmadd231pd	(%rbx,%rcx), %ymm1, %ymm0
	vmovupd	%ymm0, (%rsi,%rcx)
	addq	$32, %rcx
	cmpq	%rdi, %rcx
	jne	.L105
	movq	%rax, %rsi
	andq	$-4, %rsi
	leaq	(%rdx,%rsi), %rcx
	cmpq	%rsi, %rax
	je	.L118
	vzeroupper
.L104:
	subq	%rsi, %rax
	cmpq	$1, %rax
	je	.L107
	addq	%rsi, %rdx
	vmovapd	.LC8(%rip), %xmm0
	vmulpd	(%r10,%rdx,8), %xmm0, %xmm0
	vmovupd	(%r9,%rdx,8), %xmm4
	vfmadd231pd	.LC9(%rip), %xmm4, %xmm0
	vmovupd	%xmm0, (%r8,%rdx,8)
	movq	%rax, %rdx
	andq	$-2, %rdx
	addq	%rdx, %rcx
	cmpq	%rdx, %rax
	je	.L116
.L107:
	vmovsd	.LC2(%rip), %xmm0
	vmulsd	(%r10,%rcx,8), %xmm0, %xmm0
	vmovsd	(%r9,%rcx,8), %xmm3
	vfmadd231sd	.LC3(%rip), %xmm3, %xmm0
	vmovsd	%xmm0, (%r8,%rcx,8)
.L116:
	leaq	-16(%rbp), %rsp
	popq	%rbx
	popq	%r12
	popq	%rbp
	.cfi_remember_state
	.cfi_def_cfa 7, 8
	ret
	.p2align 4,,10
	.p2align 3
.L102:
	.cfi_restore_state
	addq	$1, %rax
	xorl	%edx, %edx
	jmp	.L109
	.p2align 4,,10
	.p2align 3
.L110:
	movq	%rdx, %rcx
	xorl	%esi, %esi
	jmp	.L104
	.p2align 4,,10
	.p2align 3
.L118:
	vzeroupper
	leaq	-16(%rbp), %rsp
	popq	%rbx
	popq	%r12
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE4738:
	.size	_Z21compute_parallel_simdPKdS0_Pd._omp_fn.0, .-_Z21compute_parallel_simdPKdS0_Pd._omp_fn.0
	.p2align 4
	.globl	_Z10initializeRSt6vectorIdSaIdEES2_
	.type	_Z10initializeRSt6vectorIdSaIdEES2_, @function
_Z10initializeRSt6vectorIdSaIdEES2_:
.LFB3970:
	.cfi_startproc
	endbr64
	subq	$40, %rsp
	.cfi_def_cfa_offset 48
	vmovq	%rdi, %xmm1
	xorl	%ecx, %ecx
	xorl	%edx, %edx
	movq	%fs:40, %rax
	movq	%rax, 24(%rsp)
	xorl	%eax, %eax
	vpinsrq	$1, %rsi, %xmm1, %xmm0
	leaq	_Z10initializeRSt6vectorIdSaIdEES2_._omp_fn.0(%rip), %rdi
	movq	%rsp, %rsi
	vmovdqa	%xmm0, (%rsp)
	call	GOMP_parallel@PLT
	movq	24(%rsp), %rax
	subq	%fs:40, %rax
	jne	.L123
	addq	$40, %rsp
	.cfi_remember_state
	.cfi_def_cfa_offset 8
	ret
.L123:
	.cfi_restore_state
	call	__stack_chk_fail@PLT
	.cfi_endproc
.LFE3970:
	.size	_Z10initializeRSt6vectorIdSaIdEES2_, .-_Z10initializeRSt6vectorIdSaIdEES2_
	.section	.rodata._ZNSt6vectorIdSaIdEEC2EmRKS0_.str1.8,"aMS",@progbits,1
	.align 8
.LC10:
	.string	"cannot create std::vector larger than max_size()"
	.section	.text._ZNSt6vectorIdSaIdEEC2EmRKS0_,"axG",@progbits,_ZNSt6vectorIdSaIdEEC5EmRKS0_,comdat
	.align 2
	.p2align 4
	.weak	_ZNSt6vectorIdSaIdEEC2EmRKS0_
	.type	_ZNSt6vectorIdSaIdEEC2EmRKS0_, @function
_ZNSt6vectorIdSaIdEEC2EmRKS0_:
.LFB4325:
	.cfi_startproc
	endbr64
	pushq	%r13
	.cfi_def_cfa_offset 16
	.cfi_offset 13, -16
	movq	%rsi, %rax
	pushq	%r12
	.cfi_def_cfa_offset 24
	.cfi_offset 12, -24
	pushq	%rbp
	.cfi_def_cfa_offset 32
	.cfi_offset 6, -32
	pushq	%rbx
	.cfi_def_cfa_offset 40
	.cfi_offset 3, -40
	subq	$8, %rsp
	.cfi_def_cfa_offset 48
	shrq	$60, %rax
	jne	.L132
	vpxor	%xmm0, %xmm0, %xmm0
	movq	$0, 16(%rdi)
	movq	%rdi, %rbx
	movq	%rsi, %rbp
	vmovdqu	%xmm0, (%rdi)
	leaq	0(,%rsi,8), %r13
	testq	%rsi, %rsi
	je	.L126
	movq	%r13, %rdi
	call	_Znwm@PLT
	vmovq	%rax, %xmm1
	leaq	(%rax,%r13), %r12
	movq	$0x000000000, (%rax)
	leaq	8(%rax), %rdi
	vpunpcklqdq	%xmm1, %xmm1, %xmm0
	movq	%r12, 16(%rbx)
	vmovdqu	%xmm0, (%rbx)
	cmpq	$1, %rbp
	je	.L128
	cmpq	%rdi, %r12
	je	.L127
	leaq	-8(%r13), %rdx
	xorl	%esi, %esi
	call	memset@PLT
.L127:
	movq	%r12, 8(%rbx)
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
.L126:
	.cfi_restore_state
	xorl	%r12d, %r12d
	movq	$0, (%rdi)
	movq	$0, 16(%rdi)
	movq	%r12, 8(%rbx)
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
.L128:
	.cfi_restore_state
	movq	%rdi, %r12
	movq	%r12, 8(%rbx)
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
.L132:
	.cfi_restore_state
	leaq	.LC10(%rip), %rdi
	call	_ZSt20__throw_length_errorPKc@PLT
	.cfi_endproc
.LFE4325:
	.size	_ZNSt6vectorIdSaIdEEC2EmRKS0_, .-_ZNSt6vectorIdSaIdEEC2EmRKS0_
	.weak	_ZNSt6vectorIdSaIdEEC1EmRKS0_
	.set	_ZNSt6vectorIdSaIdEEC1EmRKS0_,_ZNSt6vectorIdSaIdEEC2EmRKS0_
	.section	.text._Z9benchmarkIPFvPKdS1_PdEEdT_S1_S1_S2_,"axG",@progbits,_Z9benchmarkIPFvPKdS1_PdEEdT_S1_S1_S2_,comdat
	.p2align 4
	.weak	_Z9benchmarkIPFvPKdS1_PdEEdT_S1_S1_S2_
	.type	_Z9benchmarkIPFvPKdS1_PdEEdT_S1_S1_S2_, @function
_Z9benchmarkIPFvPKdS1_PdEEdT_S1_S1_S2_:
.LFB4331:
	.cfi_startproc
	endbr64
	pushq	%r15
	.cfi_def_cfa_offset 16
	.cfi_offset 15, -16
	pushq	%r14
	.cfi_def_cfa_offset 24
	.cfi_offset 14, -24
	movq	%rcx, %r14
	pushq	%r13
	.cfi_def_cfa_offset 32
	.cfi_offset 13, -32
	movq	%rdx, %r13
	movq	%rcx, %rdx
	pushq	%r12
	.cfi_def_cfa_offset 40
	.cfi_offset 12, -40
	movq	%rsi, %r12
	movq	%r13, %rsi
	pushq	%rbp
	.cfi_def_cfa_offset 48
	.cfi_offset 6, -48
	movq	%rdi, %rbp
	movq	%r12, %rdi
	pushq	%rbx
	.cfi_def_cfa_offset 56
	.cfi_offset 3, -56
	movl	$8, %ebx
	subq	$8, %rsp
	.cfi_def_cfa_offset 64
	call	*%rbp
	call	_ZNSt6chrono3_V212steady_clock3nowEv@PLT
	movq	%rax, %r15
	.p2align 4,,10
	.p2align 3
.L134:
	movq	%r14, %rdx
	movq	%r13, %rsi
	movq	%r12, %rdi
	call	*%rbp
	subl	$1, %ebx
	jne	.L134
	call	_ZNSt6chrono3_V212steady_clock3nowEv@PLT
	vxorps	%xmm0, %xmm0, %xmm0
	subq	%r15, %rax
	vcvtsi2sdq	%rax, %xmm0, %xmm0
	vdivsd	.LC12(%rip), %xmm0, %xmm0
	vmulsd	.LC13(%rip), %xmm0, %xmm0
	addq	$8, %rsp
	.cfi_def_cfa_offset 56
	popq	%rbx
	.cfi_def_cfa_offset 48
	popq	%rbp
	.cfi_def_cfa_offset 40
	popq	%r12
	.cfi_def_cfa_offset 32
	popq	%r13
	.cfi_def_cfa_offset 24
	popq	%r14
	.cfi_def_cfa_offset 16
	popq	%r15
	.cfi_def_cfa_offset 8
	ret
	.cfi_endproc
.LFE4331:
	.size	_Z9benchmarkIPFvPKdS1_PdEEdT_S1_S1_S2_, .-_Z9benchmarkIPFvPKdS1_PdEEdT_S1_S1_S2_
	.section	.rodata.str1.1,"aMS",@progbits,1
.LC14:
	.string	"Elements: %'llu\n"
.LC15:
	.string	"Threads:  %d\n\n"
.LC16:
	.string	"Speedup"
.LC17:
	.string	"Time (s)"
.LC18:
	.string	"Method"
.LC19:
	.string	"%-22s %12s %12s\n"
.LC20:
	.string	"Serial"
.LC21:
	.string	"%-22s %12.6f %12.2fx\n"
.LC22:
	.string	"Parallel static"
.LC23:
	.string	"Dynamic, chunk 1"
.LC24:
	.string	"SIMD only"
.LC25:
	.string	"Parallel + SIMD"
.LC26:
	.string	"\nChecksum: %.9f\n"
	.section	.text.unlikely,"ax",@progbits
.LCOLDB27:
	.section	.text.startup,"ax",@progbits
.LHOTB27:
	.p2align 4
	.globl	main
	.type	main, @function
main:
.LFB3977:
	.cfi_startproc
	.cfi_personality 0x9b,DW.ref.__gxx_personality_v0
	.cfi_lsda 0x1b,.LLSDA3977
	endbr64
	pushq	%r14
	.cfi_def_cfa_offset 16
	.cfi_offset 14, -16
	pushq	%r13
	.cfi_def_cfa_offset 24
	.cfi_offset 13, -24
	pushq	%r12
	.cfi_def_cfa_offset 32
	.cfi_offset 12, -32
	pushq	%rbp
	.cfi_def_cfa_offset 40
	.cfi_offset 6, -40
	movl	%edi, %ebp
	pushq	%rbx
	.cfi_def_cfa_offset 48
	.cfi_offset 3, -48
	movq	%rsi, %rbx
	subq	$160, %rsp
	.cfi_def_cfa_offset 208
	movq	%fs:40, %rax
	movq	%rax, 152(%rsp)
	xorl	%eax, %eax
	leaq	64(%rsp), %r14
	leaq	96(%rsp), %r13
	vmovq	%r14, %xmm6
	vpinsrq	$1, %r13, %xmm6, %xmm5
	vmovdqa	%xmm5, (%rsp)
	call	omp_get_max_threads@PLT
	movl	%eax, %r12d
	cmpl	$1, %ebp
	jle	.L138
	movq	8(%rbx), %rdi
	movl	$10, %edx
	xorl	%esi, %esi
	call	strtol@PLT
	movl	$1, %edx
	testl	%eax, %eax
	cmovg	%eax, %edx
	movl	%edx, %r12d
.L138:
	movl	%r12d, %edi
	leaq	128(%rsp), %rbp
	call	omp_set_num_threads@PLT
	movq	%rbp, %rdx
	movl	$16000000, %esi
	movq	%r14, %rdi
.LEHB0:
	call	_ZNSt6vectorIdSaIdEEC1EmRKS0_
.LEHE0:
	movq	%rbp, %rdx
	movl	$16000000, %esi
	movq	%r13, %rdi
.LEHB1:
	call	_ZNSt6vectorIdSaIdEEC1EmRKS0_
.LEHE1:
	leaq	48(%rsp), %r13
	movl	$16000000, %esi
	movq	%rbp, %rdi
	movq	%r13, %rdx
.LEHB2:
	call	_ZNSt6vectorIdSaIdEEC1EmRKS0_
.LEHE2:
	vmovdqa	(%rsp), %xmm7
	xorl	%ecx, %ecx
	xorl	%edx, %edx
	movq	%r13, %rsi
	leaq	_Z10initializeRSt6vectorIdSaIdEES2_._omp_fn.0(%rip), %rdi
	vmovdqa	%xmm7, 48(%rsp)
	call	GOMP_parallel@PLT
	movq	96(%rsp), %rdx
	movq	128(%rsp), %rbp
	leaq	_Z14compute_serialPKdS0_Pd(%rip), %rdi
	movq	64(%rsp), %rsi
	movq	%rbp, %rcx
.LEHB3:
	call	_Z9benchmarkIPFvPKdS1_PdEEdT_S1_S1_S2_
	movq	96(%rsp), %rdx
	movq	64(%rsp), %rsi
	movq	%rbp, %rcx
	leaq	_Z16compute_parallelPKdS0_Pd(%rip), %rdi
	vmovsd	%xmm0, (%rsp)
	call	_Z9benchmarkIPFvPKdS1_PdEEdT_S1_S1_S2_
	movq	96(%rsp), %rdx
	movq	64(%rsp), %rsi
	movq	%rbp, %rcx
	leaq	_Z15compute_dynamicPKdS0_Pd(%rip), %rdi
	vmovsd	%xmm0, 16(%rsp)
	call	_Z9benchmarkIPFvPKdS1_PdEEdT_S1_S1_S2_
	movq	96(%rsp), %rdx
	movq	64(%rsp), %rsi
	movq	%rbp, %rcx
	leaq	_Z12compute_simdPKdS0_Pd(%rip), %rdi
	vmovsd	%xmm0, 24(%rsp)
	call	_Z9benchmarkIPFvPKdS1_PdEEdT_S1_S1_S2_
	movq	96(%rsp), %rdx
	movq	64(%rsp), %rsi
	movq	%rbp, %rcx
	leaq	_Z21compute_parallel_simdPKdS0_Pd(%rip), %rdi
	vmovsd	%xmm0, 32(%rsp)
	call	_Z9benchmarkIPFvPKdS1_PdEEdT_S1_S1_S2_
	vxorpd	%xmm1, %xmm1, %xmm1
	vmovsd	%xmm0, 40(%rsp)
	movq	%rbp, %rax
	leaq	128000000(%rbp), %rdx
	vmovq	%xmm1, %rbx
	.p2align 4,,10
	.p2align 3
.L139:
	vmovq	%rbx, %xmm3
	vaddsd	(%rax), %xmm3, %xmm2
	addq	$8192, %rax
	vmovq	%xmm2, %rbx
	cmpq	%rdx, %rax
	jne	.L139
	movl	$16000000, %edx
	leaq	.LC14(%rip), %rsi
	movl	$1, %edi
	xorl	%eax, %eax
	call	__printf_chk@PLT
	movl	%r12d, %edx
	leaq	.LC15(%rip), %rsi
	movl	$1, %edi
	xorl	%eax, %eax
	call	__printf_chk@PLT
	leaq	.LC16(%rip), %r8
	leaq	.LC17(%rip), %rcx
	xorl	%eax, %eax
	movl	$1, %edi
	leaq	.LC18(%rip), %rdx
	leaq	.LC19(%rip), %rsi
	call	__printf_chk@PLT
	vmovsd	(%rsp), %xmm0
	leaq	.LC21(%rip), %r12
	vmovsd	.LC5(%rip), %xmm1
	leaq	.LC20(%rip), %rdx
	movq	%r12, %rsi
	movl	$1, %edi
	movl	$2, %eax
	call	__printf_chk@PLT
	vmovsd	16(%rsp), %xmm5
	vmovsd	(%rsp), %xmm4
	movq	%r12, %rsi
	movl	$1, %edi
	leaq	.LC22(%rip), %rdx
	movl	$2, %eax
	vdivsd	%xmm5, %xmm4, %xmm1
	vmovsd	%xmm5, %xmm5, %xmm0
	call	__printf_chk@PLT
	vmovsd	24(%rsp), %xmm6
	vmovsd	(%rsp), %xmm4
	movq	%r12, %rsi
	movl	$1, %edi
	leaq	.LC23(%rip), %rdx
	movl	$2, %eax
	vdivsd	%xmm6, %xmm4, %xmm1
	vmovsd	%xmm6, %xmm6, %xmm0
	call	__printf_chk@PLT
	vmovsd	32(%rsp), %xmm7
	vmovsd	(%rsp), %xmm4
	movq	%r12, %rsi
	movl	$1, %edi
	leaq	.LC24(%rip), %rdx
	movl	$2, %eax
	vdivsd	%xmm7, %xmm4, %xmm1
	vmovsd	%xmm7, %xmm7, %xmm0
	call	__printf_chk@PLT
	vmovsd	(%rsp), %xmm4
	movq	%r12, %rsi
	movl	$1, %edi
	movl	$2, %eax
	vmovsd	40(%rsp), %xmm0
	leaq	.LC25(%rip), %rdx
	vdivsd	%xmm0, %xmm4, %xmm1
	call	__printf_chk@PLT
	vmovq	%rbx, %xmm0
	movl	$1, %edi
	movl	$1, %eax
	leaq	.LC26(%rip), %rsi
	call	__printf_chk@PLT
.LEHE3:
	testq	%rbp, %rbp
	je	.L140
	movq	144(%rsp), %rsi
	movq	%rbp, %rdi
	subq	%rbp, %rsi
	call	_ZdlPvm@PLT
.L140:
	movq	96(%rsp), %rdi
	testq	%rdi, %rdi
	je	.L141
	movq	112(%rsp), %rsi
	subq	%rdi, %rsi
	call	_ZdlPvm@PLT
.L141:
	movq	64(%rsp), %rdi
	testq	%rdi, %rdi
	je	.L142
	movq	80(%rsp), %rsi
	subq	%rdi, %rsi
	call	_ZdlPvm@PLT
.L142:
	movq	152(%rsp), %rax
	subq	%fs:40, %rax
	jne	.L176
	addq	$160, %rsp
	.cfi_remember_state
	.cfi_def_cfa_offset 48
	xorl	%eax, %eax
	popq	%rbx
	.cfi_def_cfa_offset 40
	popq	%rbp
	.cfi_def_cfa_offset 32
	popq	%r12
	.cfi_def_cfa_offset 24
	popq	%r13
	.cfi_def_cfa_offset 16
	popq	%r14
	.cfi_def_cfa_offset 8
	ret
.L176:
	.cfi_restore_state
	call	__stack_chk_fail@PLT
.L152:
	endbr64
	movq	%rax, %rbx
	jmp	.L143
.L151:
	endbr64
	movq	%rax, %rbp
	vzeroupper
	jmp	.L145
.L150:
	endbr64
	movq	%rax, %rbp
	vzeroupper
	jmp	.L147
	.globl	__gxx_personality_v0
	.section	.gcc_except_table,"a",@progbits
.LLSDA3977:
	.byte	0xff
	.byte	0xff
	.byte	0x1
	.uleb128 .LLSDACSE3977-.LLSDACSB3977
.LLSDACSB3977:
	.uleb128 .LEHB0-.LFB3977
	.uleb128 .LEHE0-.LEHB0
	.uleb128 0
	.uleb128 0
	.uleb128 .LEHB1-.LFB3977
	.uleb128 .LEHE1-.LEHB1
	.uleb128 .L150-.LFB3977
	.uleb128 0
	.uleb128 .LEHB2-.LFB3977
	.uleb128 .LEHE2-.LEHB2
	.uleb128 .L151-.LFB3977
	.uleb128 0
	.uleb128 .LEHB3-.LFB3977
	.uleb128 .LEHE3-.LEHB3
	.uleb128 .L152-.LFB3977
	.uleb128 0
.LLSDACSE3977:
	.section	.text.startup
	.cfi_endproc
	.section	.text.unlikely
	.cfi_startproc
	.cfi_personality 0x9b,DW.ref.__gxx_personality_v0
	.cfi_lsda 0x1b,.LLSDAC3977
	.type	main.cold, @function
main.cold:
.LFSB3977:
.L143:
	.cfi_def_cfa_offset 208
	.cfi_offset 3, -48
	.cfi_offset 6, -40
	.cfi_offset 12, -32
	.cfi_offset 13, -24
	.cfi_offset 14, -16
	movq	144(%rsp), %rsi
	subq	%rbp, %rsi
	testq	%rbp, %rbp
	je	.L173
	movq	%rbp, %rdi
	vzeroupper
	call	_ZdlPvm@PLT
.L144:
	movq	%rbx, %rbp
.L145:
	movq	96(%rsp), %rdi
	movq	112(%rsp), %rsi
	subq	%rdi, %rsi
	testq	%rdi, %rdi
	je	.L147
	call	_ZdlPvm@PLT
.L147:
	movq	64(%rsp), %rdi
	movq	80(%rsp), %rsi
	subq	%rdi, %rsi
	testq	%rdi, %rdi
	je	.L148
	call	_ZdlPvm@PLT
.L148:
	movq	%rbp, %rdi
.LEHB4:
	call	_Unwind_Resume@PLT
.LEHE4:
.L173:
	vzeroupper
	jmp	.L144
	.cfi_endproc
.LFE3977:
	.section	.gcc_except_table
.LLSDAC3977:
	.byte	0xff
	.byte	0xff
	.byte	0x1
	.uleb128 .LLSDACSEC3977-.LLSDACSBC3977
.LLSDACSBC3977:
	.uleb128 .LEHB4-.LCOLDB27
	.uleb128 .LEHE4-.LEHB4
	.uleb128 0
	.uleb128 0
.LLSDACSEC3977:
	.section	.text.unlikely
	.section	.text.startup
	.size	main, .-main
	.section	.text.unlikely
	.size	main.cold, .-main.cold
.LCOLDE27:
	.section	.text.startup
.LHOTE27:
	.p2align 4
	.type	_GLOBAL__sub_I__Z10initializeRSt6vectorIdSaIdEES2_, @function
_GLOBAL__sub_I__Z10initializeRSt6vectorIdSaIdEES2_:
.LFB4734:
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
.LFE4734:
	.size	_GLOBAL__sub_I__Z10initializeRSt6vectorIdSaIdEES2_, .-_GLOBAL__sub_I__Z10initializeRSt6vectorIdSaIdEES2_
	.section	.init_array,"aw"
	.align 8
	.quad	_GLOBAL__sub_I__Z10initializeRSt6vectorIdSaIdEES2_
	.local	_ZStL8__ioinit
	.comm	_ZStL8__ioinit,1,1
	.section	.rodata.cst32,"aM",@progbits,32
	.align 32
.LC0:
	.long	0
	.long	1072168960
	.long	0
	.long	1072168960
	.long	0
	.long	1072168960
	.long	0
	.long	1072168960
	.align 32
.LC1:
	.long	0
	.long	1072955392
	.long	0
	.long	1072955392
	.long	0
	.long	1072955392
	.long	0
	.long	1072955392
	.set	.LC2,.LC0
	.set	.LC3,.LC1
	.section	.rodata.cst8,"aM",@progbits,8
	.align 8
.LC4:
	.long	-755914244
	.long	1062232653
	.align 8
.LC5:
	.long	0
	.long	1072693248
	.align 8
.LC6:
	.long	-755914244
	.long	1063281229
	.align 8
.LC7:
	.long	0
	.long	1073741824
	.set	.LC8,.LC0
	.set	.LC9,.LC1
	.align 8
.LC12:
	.long	0
	.long	1104006501
	.align 8
.LC13:
	.long	0
	.long	1069547520
	.hidden	DW.ref.__gxx_personality_v0
	.weak	DW.ref.__gxx_personality_v0
	.section	.data.rel.local.DW.ref.__gxx_personality_v0,"awG",@progbits,DW.ref.__gxx_personality_v0,comdat
	.align 8
	.type	DW.ref.__gxx_personality_v0, @object
	.size	DW.ref.__gxx_personality_v0, 8
DW.ref.__gxx_personality_v0:
	.quad	__gxx_personality_v0
	.hidden	__dso_handle
	.ident	"GCC: (Ubuntu 11.4.0-1ubuntu1~22.04.3) 11.4.0"
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
