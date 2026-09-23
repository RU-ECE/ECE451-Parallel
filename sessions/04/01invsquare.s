	.file	"01invsquare.cpp"
	.text
	.local	_ZStL8__ioinit
	.comm	_ZStL8__ioinit,1,1
	.section	.text._ZSt12setprecisioni,"axG",@progbits,_ZSt12setprecisioni,comdat
	.weak	_ZSt12setprecisioni
	.type	_ZSt12setprecisioni, @function
_ZSt12setprecisioni:
.LFB6512:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	movl	%edi, -4(%rbp)
	movl	-4(%rbp), %eax
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE6512:
	.size	_ZSt12setprecisioni, .-_ZSt12setprecisioni
	.text
	.globl	_Z3summm
	.type	_Z3summm, @function
_Z3summm:
.LFB6530:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	movq	%rdi, -24(%rbp)
	movq	%rsi, -32(%rbp)
	vxorpd	%xmm0, %xmm0, %xmm0
	vmovsd	%xmm0, -16(%rbp)
	movq	-24(%rbp), %rax
	movq	%rax, -8(%rbp)
	jmp	.L4
.L7:
	movq	-8(%rbp), %rax
	imulq	%rax, %rax
	testq	%rax, %rax
	js	.L5
	vcvtsi2sdq	%rax, %xmm0, %xmm0
	jmp	.L6
.L5:
	movq	%rax, %rdx
	shrq	%rdx
	andl	$1, %eax
	orq	%rax, %rdx
	vcvtsi2sdq	%rdx, %xmm0, %xmm0
	vaddsd	%xmm0, %xmm0, %xmm0
.L6:
	vmovsd	.LC1(%rip), %xmm1
	vdivsd	%xmm0, %xmm1, %xmm0
	vmovsd	-16(%rbp), %xmm1
	vaddsd	%xmm0, %xmm1, %xmm0
	vmovsd	%xmm0, -16(%rbp)
	addq	$1, -8(%rbp)
.L4:
	movq	-8(%rbp), %rax
	cmpq	-32(%rbp), %rax
	jbe	.L7
	vmovsd	-16(%rbp), %xmm0
	vmovq	%xmm0, %rax
	vmovq	%rax, %xmm0
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE6530:
	.size	_Z3summm, .-_Z3summm
	.globl	_Z7sum_revmm
	.type	_Z7sum_revmm, @function
_Z7sum_revmm:
.LFB6531:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	movq	%rdi, -24(%rbp)
	movq	%rsi, -32(%rbp)
	vxorpd	%xmm0, %xmm0, %xmm0
	vmovsd	%xmm0, -16(%rbp)
	movq	-32(%rbp), %rax
	movq	%rax, -8(%rbp)
	jmp	.L10
.L13:
	movq	-8(%rbp), %rax
	imulq	%rax, %rax
	testq	%rax, %rax
	js	.L11
	vcvtsi2sdq	%rax, %xmm0, %xmm0
	jmp	.L12
.L11:
	movq	%rax, %rdx
	shrq	%rdx
	andl	$1, %eax
	orq	%rax, %rdx
	vcvtsi2sdq	%rdx, %xmm0, %xmm0
	vaddsd	%xmm0, %xmm0, %xmm0
.L12:
	vmovsd	.LC1(%rip), %xmm1
	vdivsd	%xmm0, %xmm1, %xmm0
	vmovsd	-16(%rbp), %xmm1
	vaddsd	%xmm0, %xmm1, %xmm0
	vmovsd	%xmm0, -16(%rbp)
	subq	$1, -8(%rbp)
.L10:
	movq	-8(%rbp), %rax
	cmpq	-24(%rbp), %rax
	jnb	.L13
	vmovsd	-16(%rbp), %xmm0
	vmovq	%xmm0, %rax
	vmovq	%rax, %xmm0
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE6531:
	.size	_Z7sum_revmm, .-_Z7sum_revmm
	.globl	_Z7sum_avxmm
	.type	_Z7sum_avxmm, @function
_Z7sum_avxmm:
.LFB6532:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	andq	$-32, %rsp
	subq	$704, %rsp
	movq	%rdi, 24(%rsp)
	movq	%rsi, 16(%rsp)
	movq	%fs:40, %rax
	movq	%rax, 696(%rsp)
	xorl	%eax, %eax
	vxorpd	%xmm0, %xmm0, %xmm0
	vmovapd	%ymm0, 96(%rsp)
	vmovsd	.LC1(%rip), %xmm0
	vmovsd	%xmm0, 592(%rsp)
	vmovsd	.LC1(%rip), %xmm0
	vmovsd	%xmm0, 600(%rsp)
	vmovsd	.LC1(%rip), %xmm0
	vmovsd	%xmm0, 608(%rsp)
	vmovsd	.LC1(%rip), %xmm0
	vmovsd	%xmm0, 616(%rsp)
	vmovsd	.LC1(%rip), %xmm0
	vmovsd	%xmm0, 624(%rsp)
	vmovsd	.LC2(%rip), %xmm0
	vmovsd	%xmm0, 632(%rsp)
	vmovsd	.LC3(%rip), %xmm0
	vmovsd	%xmm0, 640(%rsp)
	vmovsd	.LC4(%rip), %xmm0
	vmovsd	%xmm0, 648(%rsp)
	leaq	592(%rsp), %rax
	movq	%rax, 80(%rsp)
	movq	80(%rsp), %rax
	vmovupd	(%rax), %ymm0
	vmovapd	%ymm0, 160(%rsp)
	vmovsd	.LC4(%rip), %xmm0
	vmovsd	%xmm0, 72(%rsp)
	vbroadcastsd	72(%rsp), %ymm0
	vmovapd	%ymm0, 192(%rsp)
	leaq	624(%rsp), %rax
	movq	%rax, 64(%rsp)
	movq	64(%rsp), %rax
	vmovupd	(%rax), %ymm0
	vmovapd	%ymm0, 128(%rsp)
	movq	24(%rsp), %rax
	movq	%rax, 48(%rsp)
	jmp	.L20
.L25:
	vmovapd	128(%rsp), %ymm0
	vmovapd	%ymm0, 480(%rsp)
	vmovapd	128(%rsp), %ymm0
	vmovapd	%ymm0, 512(%rsp)
	vmovapd	480(%rsp), %ymm0
	vmulpd	512(%rsp), %ymm0, %ymm0
	vmovapd	%ymm0, 224(%rsp)
	vmovapd	160(%rsp), %ymm0
	vmovapd	%ymm0, 416(%rsp)
	vmovapd	224(%rsp), %ymm0
	vmovapd	%ymm0, 448(%rsp)
	vmovapd	416(%rsp), %ymm0
	vdivpd	448(%rsp), %ymm0, %ymm0
	vmovapd	%ymm0, 256(%rsp)
	vmovapd	96(%rsp), %ymm0
	vmovapd	%ymm0, 352(%rsp)
	vmovapd	256(%rsp), %ymm0
	vmovapd	%ymm0, 384(%rsp)
	vmovapd	352(%rsp), %ymm0
	vaddpd	384(%rsp), %ymm0, %ymm0
	vmovapd	%ymm0, 96(%rsp)
	vmovapd	128(%rsp), %ymm0
	vmovapd	%ymm0, 288(%rsp)
	vmovapd	192(%rsp), %ymm0
	vmovapd	%ymm0, 320(%rsp)
	vmovapd	288(%rsp), %ymm0
	vaddpd	320(%rsp), %ymm0, %ymm0
	vmovapd	%ymm0, 128(%rsp)
	addq	$4, 48(%rsp)
.L20:
	movq	48(%rsp), %rax
	cmpq	16(%rsp), %rax
	jbe	.L25
	leaq	656(%rsp), %rax
	movq	%rax, 88(%rsp)
	vmovapd	96(%rsp), %ymm0
	vmovapd	%ymm0, 544(%rsp)
	vmovapd	544(%rsp), %ymm0
	movq	88(%rsp), %rax
	vmovupd	%ymm0, (%rax)
	nop
	vxorpd	%xmm0, %xmm0, %xmm0
	vmovsd	%xmm0, 56(%rsp)
	movl	$0, 44(%rsp)
	jmp	.L26
.L27:
	movl	44(%rsp), %eax
	cltq
	vmovsd	656(%rsp,%rax,8), %xmm0
	vmovsd	56(%rsp), %xmm1
	vaddsd	%xmm0, %xmm1, %xmm0
	vmovsd	%xmm0, 56(%rsp)
	addl	$1, 44(%rsp)
.L26:
	cmpl	$3, 44(%rsp)
	jle	.L27
	vmovsd	56(%rsp), %xmm0
	vmovq	%xmm0, %rax
	movq	696(%rsp), %rdx
	subq	%fs:40, %rdx
	je	.L29
	call	__stack_chk_fail@PLT
.L29:
	vmovq	%rax, %xmm0
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE6532:
	.size	_Z7sum_avxmm, .-_Z7sum_avxmm
	.globl	_Z11sum_avx_revmm
	.type	_Z11sum_avx_revmm, @function
_Z11sum_avx_revmm:
.LFB6533:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	andq	$-32, %rsp
	subq	$704, %rsp
	movq	%rdi, 24(%rsp)
	movq	%rsi, 16(%rsp)
	movq	%fs:40, %rax
	movq	%rax, 696(%rsp)
	xorl	%eax, %eax
	vxorpd	%xmm0, %xmm0, %xmm0
	vmovapd	%ymm0, 96(%rsp)
	vmovsd	.LC1(%rip), %xmm0
	vmovsd	%xmm0, 592(%rsp)
	vmovsd	.LC1(%rip), %xmm0
	vmovsd	%xmm0, 600(%rsp)
	vmovsd	.LC1(%rip), %xmm0
	vmovsd	%xmm0, 608(%rsp)
	vmovsd	.LC1(%rip), %xmm0
	vmovsd	%xmm0, 616(%rsp)
	movq	16(%rsp), %rax
	testq	%rax, %rax
	js	.L32
	vcvtsi2sdq	%rax, %xmm0, %xmm0
	jmp	.L33
.L32:
	movq	%rax, %rdx
	shrq	%rdx
	andl	$1, %eax
	orq	%rax, %rdx
	vcvtsi2sdq	%rdx, %xmm0, %xmm0
	vaddsd	%xmm0, %xmm0, %xmm0
.L33:
	vmovsd	%xmm0, 624(%rsp)
	movq	16(%rsp), %rax
	subq	$1, %rax
	testq	%rax, %rax
	js	.L34
	vcvtsi2sdq	%rax, %xmm0, %xmm0
	jmp	.L35
.L34:
	movq	%rax, %rdx
	shrq	%rdx
	andl	$1, %eax
	orq	%rax, %rdx
	vcvtsi2sdq	%rdx, %xmm0, %xmm0
	vaddsd	%xmm0, %xmm0, %xmm0
.L35:
	vmovsd	%xmm0, 632(%rsp)
	movq	16(%rsp), %rax
	subq	$2, %rax
	testq	%rax, %rax
	js	.L36
	vcvtsi2sdq	%rax, %xmm0, %xmm0
	jmp	.L37
.L36:
	movq	%rax, %rdx
	shrq	%rdx
	andl	$1, %eax
	orq	%rax, %rdx
	vcvtsi2sdq	%rdx, %xmm0, %xmm0
	vaddsd	%xmm0, %xmm0, %xmm0
.L37:
	vmovsd	%xmm0, 640(%rsp)
	movq	16(%rsp), %rax
	subq	$3, %rax
	testq	%rax, %rax
	js	.L38
	vcvtsi2sdq	%rax, %xmm0, %xmm0
	jmp	.L39
.L38:
	movq	%rax, %rdx
	shrq	%rdx
	andl	$1, %eax
	orq	%rax, %rdx
	vcvtsi2sdq	%rdx, %xmm0, %xmm0
	vaddsd	%xmm0, %xmm0, %xmm0
.L39:
	vmovsd	%xmm0, 648(%rsp)
	leaq	592(%rsp), %rax
	movq	%rax, 80(%rsp)
	movq	80(%rsp), %rax
	vmovupd	(%rax), %ymm0
	vmovapd	%ymm0, 160(%rsp)
	vmovsd	.LC4(%rip), %xmm0
	vmovsd	%xmm0, 72(%rsp)
	vbroadcastsd	72(%rsp), %ymm0
	vmovapd	%ymm0, 192(%rsp)
	leaq	624(%rsp), %rax
	movq	%rax, 64(%rsp)
	movq	64(%rsp), %rax
	vmovupd	(%rax), %ymm0
	vmovapd	%ymm0, 128(%rsp)
	movq	16(%rsp), %rax
	movq	%rax, 48(%rsp)
	jmp	.L43
.L48:
	vmovapd	128(%rsp), %ymm0
	vmovapd	%ymm0, 480(%rsp)
	vmovapd	128(%rsp), %ymm0
	vmovapd	%ymm0, 512(%rsp)
	vmovapd	480(%rsp), %ymm0
	vmulpd	512(%rsp), %ymm0, %ymm0
	vmovapd	%ymm0, 224(%rsp)
	vmovapd	160(%rsp), %ymm0
	vmovapd	%ymm0, 416(%rsp)
	vmovapd	224(%rsp), %ymm0
	vmovapd	%ymm0, 448(%rsp)
	vmovapd	416(%rsp), %ymm0
	vdivpd	448(%rsp), %ymm0, %ymm0
	vmovapd	%ymm0, 256(%rsp)
	vmovapd	96(%rsp), %ymm0
	vmovapd	%ymm0, 352(%rsp)
	vmovapd	256(%rsp), %ymm0
	vmovapd	%ymm0, 384(%rsp)
	vmovapd	352(%rsp), %ymm0
	vaddpd	384(%rsp), %ymm0, %ymm0
	vmovapd	%ymm0, 96(%rsp)
	vmovapd	128(%rsp), %ymm0
	vmovapd	%ymm0, 288(%rsp)
	vmovapd	192(%rsp), %ymm0
	vmovapd	%ymm0, 320(%rsp)
	vmovapd	288(%rsp), %ymm0
	vsubpd	320(%rsp), %ymm0, %ymm0
	vmovapd	%ymm0, 128(%rsp)
	subq	$4, 48(%rsp)
.L43:
	movq	48(%rsp), %rax
	cmpq	24(%rsp), %rax
	jnb	.L48
	leaq	656(%rsp), %rax
	movq	%rax, 88(%rsp)
	vmovapd	96(%rsp), %ymm0
	vmovapd	%ymm0, 544(%rsp)
	vmovapd	544(%rsp), %ymm0
	movq	88(%rsp), %rax
	vmovupd	%ymm0, (%rax)
	nop
	vxorpd	%xmm0, %xmm0, %xmm0
	vmovsd	%xmm0, 56(%rsp)
	movl	$0, 44(%rsp)
	jmp	.L49
.L50:
	movl	44(%rsp), %eax
	cltq
	vmovsd	656(%rsp,%rax,8), %xmm0
	vmovsd	56(%rsp), %xmm1
	vaddsd	%xmm0, %xmm1, %xmm0
	vmovsd	%xmm0, 56(%rsp)
	addl	$1, 44(%rsp)
.L49:
	cmpl	$3, 44(%rsp)
	jle	.L50
	vmovsd	56(%rsp), %xmm0
	vmovq	%xmm0, %rax
	movq	696(%rsp), %rdx
	subq	%fs:40, %rdx
	je	.L52
	call	__stack_chk_fail@PLT
.L52:
	vmovq	%rax, %xmm0
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE6533:
	.size	_Z11sum_avx_revmm, .-_Z11sum_avx_revmm
	.globl	main
	.type	main, @function
main:
.LFB6534:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	pushq	%rbx
	subq	$24, %rsp
	.cfi_offset 3, -24
	movq	$800000, -24(%rbp)
	movl	$15, %edi
	call	_ZSt12setprecisioni
	movl	%eax, %esi
	leaq	_ZSt4cout(%rip), %rax
	movq	%rax, %rdi
	call	_ZStlsIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_St13_Setprecision@PLT
	movq	%rax, %rbx
	movl	$800000, %esi
	movl	$1, %edi
	call	_Z3summm
	vmovq	%xmm0, %rax
	vmovq	%rax, %xmm0
	movq	%rbx, %rdi
	call	_ZNSolsEd@PLT
	movq	_ZSt4endlIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_@GOTPCREL(%rip), %rdx
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	_ZNSolsEPFRSoS_E@PLT
	movl	$15, %edi
	call	_ZSt12setprecisioni
	movl	%eax, %esi
	leaq	_ZSt4cout(%rip), %rax
	movq	%rax, %rdi
	call	_ZStlsIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_St13_Setprecision@PLT
	movq	%rax, %rbx
	movl	$800000, %esi
	movl	$1, %edi
	call	_Z7sum_revmm
	vmovq	%xmm0, %rax
	vmovq	%rax, %xmm0
	movq	%rbx, %rdi
	call	_ZNSolsEd@PLT
	movq	_ZSt4endlIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_@GOTPCREL(%rip), %rdx
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	_ZNSolsEPFRSoS_E@PLT
	movl	$15, %edi
	call	_ZSt12setprecisioni
	movl	%eax, %esi
	leaq	_ZSt4cout(%rip), %rax
	movq	%rax, %rdi
	call	_ZStlsIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_St13_Setprecision@PLT
	movq	%rax, %rbx
	movl	$800000, %esi
	movl	$1, %edi
	call	_Z7sum_avxmm
	vmovq	%xmm0, %rax
	vmovq	%rax, %xmm0
	movq	%rbx, %rdi
	call	_ZNSolsEd@PLT
	movq	_ZSt4endlIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_@GOTPCREL(%rip), %rdx
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	_ZNSolsEPFRSoS_E@PLT
	movl	$15, %edi
	call	_ZSt12setprecisioni
	movl	%eax, %esi
	leaq	_ZSt4cout(%rip), %rax
	movq	%rax, %rdi
	call	_ZStlsIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_St13_Setprecision@PLT
	movq	%rax, %rbx
	movl	$800000, %esi
	movl	$1, %edi
	call	_Z11sum_avx_revmm
	vmovq	%xmm0, %rax
	vmovq	%rax, %xmm0
	movq	%rbx, %rdi
	call	_ZNSolsEd@PLT
	movq	_ZSt4endlIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_@GOTPCREL(%rip), %rdx
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	_ZNSolsEPFRSoS_E@PLT
	movl	$0, %eax
	movq	-8(%rbp), %rbx
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE6534:
	.size	main, .-main
	.type	_Z41__static_initialization_and_destruction_0ii, @function
_Z41__static_initialization_and_destruction_0ii:
.LFB7050:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$16, %rsp
	movl	%edi, -4(%rbp)
	movl	%esi, -8(%rbp)
	cmpl	$1, -4(%rbp)
	jne	.L57
	cmpl	$65535, -8(%rbp)
	jne	.L57
	leaq	_ZStL8__ioinit(%rip), %rax
	movq	%rax, %rdi
	call	_ZNSt8ios_base4InitC1Ev@PLT
	leaq	__dso_handle(%rip), %rax
	movq	%rax, %rdx
	leaq	_ZStL8__ioinit(%rip), %rax
	movq	%rax, %rsi
	movq	_ZNSt8ios_base4InitD1Ev@GOTPCREL(%rip), %rax
	movq	%rax, %rdi
	call	__cxa_atexit@PLT
.L57:
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE7050:
	.size	_Z41__static_initialization_and_destruction_0ii, .-_Z41__static_initialization_and_destruction_0ii
	.type	_GLOBAL__sub_I__Z3summm, @function
_GLOBAL__sub_I__Z3summm:
.LFB7051:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	movl	$65535, %esi
	movl	$1, %edi
	call	_Z41__static_initialization_and_destruction_0ii
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE7051:
	.size	_GLOBAL__sub_I__Z3summm, .-_GLOBAL__sub_I__Z3summm
	.section	.init_array,"aw"
	.align 8
	.quad	_GLOBAL__sub_I__Z3summm
	.section	.rodata
	.align 8
.LC1:
	.long	0
	.long	1072693248
	.align 8
.LC2:
	.long	0
	.long	1073741824
	.align 8
.LC3:
	.long	0
	.long	1074266112
	.align 8
.LC4:
	.long	0
	.long	1074790400
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
