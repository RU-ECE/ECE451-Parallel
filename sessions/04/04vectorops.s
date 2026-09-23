	.file	"04vectorops.cpp"
	.text
	.p2align 4
	.globl	_Z2f1mm
	.type	_Z2f1mm, @function
_Z2f1mm:
.LFB7285:
	.cfi_startproc
	endbr64
	leaq	(%rdi,%rsi), %rax
	ret
	.cfi_endproc
.LFE7285:
	.size	_Z2f1mm, .-_Z2f1mm
	.p2align 4
	.globl	_Z2f2mm
	.type	_Z2f2mm, @function
_Z2f2mm:
.LFB7286:
	.cfi_startproc
	endbr64
	movq	%rdi, %rax
	subq	%rsi, %rax
	ret
	.cfi_endproc
.LFE7286:
	.size	_Z2f2mm, .-_Z2f2mm
	.p2align 4
	.globl	_Z2f3mm
	.type	_Z2f3mm, @function
_Z2f3mm:
.LFB7287:
	.cfi_startproc
	endbr64
	movq	%rdi, %rax
	imulq	%rsi, %rax
	ret
	.cfi_endproc
.LFE7287:
	.size	_Z2f3mm, .-_Z2f3mm
	.p2align 4
	.globl	_Z2f4mm
	.type	_Z2f4mm, @function
_Z2f4mm:
.LFB7288:
	.cfi_startproc
	endbr64
	movq	%rdi, %rax
	xorl	%edx, %edx
	divq	%rsi
	ret
	.cfi_endproc
.LFE7288:
	.size	_Z2f4mm, .-_Z2f4mm
	.p2align 4
	.globl	_Z2f1dd
	.type	_Z2f1dd, @function
_Z2f1dd:
.LFB7289:
	.cfi_startproc
	endbr64
	vaddsd	%xmm1, %xmm0, %xmm0
	ret
	.cfi_endproc
.LFE7289:
	.size	_Z2f1dd, .-_Z2f1dd
	.p2align 4
	.globl	_Z2f2dd
	.type	_Z2f2dd, @function
_Z2f2dd:
.LFB7290:
	.cfi_startproc
	endbr64
	vsubsd	%xmm1, %xmm0, %xmm0
	ret
	.cfi_endproc
.LFE7290:
	.size	_Z2f2dd, .-_Z2f2dd
	.p2align 4
	.globl	_Z2f3dd
	.type	_Z2f3dd, @function
_Z2f3dd:
.LFB7291:
	.cfi_startproc
	endbr64
	vmulsd	%xmm1, %xmm0, %xmm0
	ret
	.cfi_endproc
.LFE7291:
	.size	_Z2f3dd, .-_Z2f3dd
	.p2align 4
	.globl	_Z2f4dd
	.type	_Z2f4dd, @function
_Z2f4dd:
.LFB7292:
	.cfi_startproc
	endbr64
	vdivsd	%xmm1, %xmm0, %xmm0
	ret
	.cfi_endproc
.LFE7292:
	.size	_Z2f4dd, .-_Z2f4dd
	.p2align 4
	.globl	_Z2f5Dv4_dS_
	.type	_Z2f5Dv4_dS_, @function
_Z2f5Dv4_dS_:
.LFB7293:
	.cfi_startproc
	endbr64
	vaddpd	%ymm1, %ymm0, %ymm0
	ret
	.cfi_endproc
.LFE7293:
	.size	_Z2f5Dv4_dS_, .-_Z2f5Dv4_dS_
	.p2align 4
	.globl	_Z2f6Dv8_fS_
	.type	_Z2f6Dv8_fS_, @function
_Z2f6Dv8_fS_:
.LFB7294:
	.cfi_startproc
	endbr64
	vaddps	%ymm1, %ymm0, %ymm0
	ret
	.cfi_endproc
.LFE7294:
	.size	_Z2f6Dv8_fS_, .-_Z2f6Dv8_fS_
	.p2align 4
	.globl	_Z2f7Dv8_fS_
	.type	_Z2f7Dv8_fS_, @function
_Z2f7Dv8_fS_:
.LFB7295:
	.cfi_startproc
	endbr64
	vsubps	%ymm1, %ymm0, %ymm0
	ret
	.cfi_endproc
.LFE7295:
	.size	_Z2f7Dv8_fS_, .-_Z2f7Dv8_fS_
	.section	.text.startup,"ax",@progbits
	.p2align 4
	.type	_GLOBAL__sub_I__Z2f1mm, @function
_GLOBAL__sub_I__Z2f1mm:
.LFB7777:
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
.LFE7777:
	.size	_GLOBAL__sub_I__Z2f1mm, .-_GLOBAL__sub_I__Z2f1mm
	.section	.init_array,"aw"
	.align 8
	.quad	_GLOBAL__sub_I__Z2f1mm
	.local	_ZStL8__ioinit
	.comm	_ZStL8__ioinit,1,1
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
