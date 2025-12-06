main._omp_fn.0:
        pushq   %rbp
        movq    %rsp, %rbp
        pushq   %r14
        pushq   %r13
        pushq   %r12
        movq    %rdi, %r12
        pushq   %rbx
        andq    $-32, %rsp
        call    omp_get_num_threads
        movl    %eax, %ebx
        call    omp_get_thread_num
        xorl    %edx, %edx
        movl    %eax, %esi
        movl    $16, %eax
        idivl   %ebx
        cmpl    %edx, %esi
        jl      .L2
.L9:
        imull   %eax, %esi
        leal    (%rdx,%rsi), %ebx
        leal    (%rax,%rbx), %r11d
        cmpl    %r11d, %ebx
        jge     .L16
        leal    -1(%rax), %ecx
        movq    16(%r12), %r8
        movq    8(%r12), %r9
        movq    (%r12), %r10
        cmpl    $6, %ecx
        jbe     .L10
        movslq  %esi, %rcx
        movslq  %edx, %rdi
        movl    %eax, %r12d
        addq    %rcx, %rdi
        shrl    $3, %r12d
        xorl    %ecx, %ecx
        salq    $2, %rdi
        salq    $5, %r12
        leaq    (%r10,%rdi), %r14
        leaq    (%r9,%rdi), %r13
        addq    %r8, %rdi
.L5:
        vmovups (%r14,%rcx), %ymm1
        vaddps  0(%r13,%rcx), %ymm1, %ymm0
        vmovups %ymm0, (%rdi,%rcx)
        addq    $32, %rcx
        cmpq    %rcx, %r12
        jne     .L5
        movl    %eax, %ecx
        andl    $-8, %ecx
        addl    %ecx, %ebx
        cmpl    %ecx, %eax
        je      .L19
        vzeroupper
.L4:
        subl    %ecx, %eax
        leal    -1(%rax), %edi
        cmpl    $2, %edi
        jbe     .L7
        movslq  %edx, %rdx
        movslq  %esi, %rsi
        addq    %rsi, %rdx
        addq    %rcx, %rdx
        vmovups (%r10,%rdx,4), %xmm2
        vaddps  (%r9,%rdx,4), %xmm2, %xmm0
        vmovups %xmm0, (%r8,%rdx,4)
        movl    %eax, %edx
        andl    $-4, %edx
        addl    %edx, %ebx
        testb   $3, %al
        je      .L16
.L7:
        movslq  %ebx, %rax
        vmovss  (%r10,%rax,4), %xmm0
        vaddss  (%r9,%rax,4), %xmm0, %xmm0
        leaq    0(,%rax,4), %rdx
        vmovss  %xmm0, (%r8,%rax,4)
        leal    1(%rbx), %eax
        cmpl    %eax, %r11d
        jle     .L16
        vmovss  4(%r9,%rdx), %xmm0
        vaddss  4(%r10,%rdx), %xmm0, %xmm0
        leal    2(%rbx), %eax
        vmovss  %xmm0, 4(%r8,%rdx)
        cmpl    %eax, %r11d
        jle     .L16
        vmovss  8(%r10,%rdx), %xmm0
        vaddss  8(%r9,%rdx), %xmm0, %xmm0
        vmovss  %xmm0, 8(%r8,%rdx)
.L16:
        leaq    -32(%rbp), %rsp
        popq    %rbx
        popq    %r12
        popq    %r13
        popq    %r14
        popq    %rbp
        ret
.L2:
        addl    $1, %eax
        xorl    %edx, %edx
        jmp     .L9
.L10:
        xorl    %ecx, %ecx
        jmp     .L4
.L19:
        vzeroupper
        jmp     .L16
main:
        pushq   %rbp
        movl    $64, %edi
        movq    %rsp, %rbp
        pushq   %r15
        pushq   %r14
        pushq   %r13
        pushq   %r12
        pushq   %r10
        pushq   %rbx
        subq    $32, %rsp
        call    operator new[](unsigned long)
        movl    $64, %edi
        movq    %rax, %r15
        call    operator new[](unsigned long)
        movl    $64, %edi
        movq    %rax, %r14
        call    operator new[](unsigned long)
        movq    %r14, -72(%rbp)
        xorl    %ecx, %ecx
        xorl    %edx, %edx
        movq    %rax, -64(%rbp)
        movq    %rax, %r13
        leaq    -80(%rbp), %rsi
        movl    $main._omp_fn.0, %edi
        vmovaps .LC0(%rip), %ymm0
        movq    %r15, -80(%rbp)
        movq    %r13, %rbx
        leaq    64(%r13), %r12
        vmovups %ymm0, (%r15)
        vmovups %ymm0, (%r14)
        vmovaps .LC1(%rip), %ymm0
        vmovups %ymm0, 32(%r15)
        vmovups %ymm0, 32(%r14)
        vzeroupper
        call    GOMP_parallel
.L24:
        vxorpd  %xmm1, %xmm1, %xmm1
        movl    $std::cout, %edi
        vcvtss2sd       (%rbx), %xmm1, %xmm0
        call    std::ostream& std::ostream::_M_insert<double>(double)
        movb    $32, -80(%rbp)
        movq    %rax, %rdi
        movq    (%rax), %rax
        movq    -24(%rax), %rax
        cmpq    $0, 16(%rdi,%rax)
        je      .L21
        movl    $1, %edx
        leaq    -80(%rbp), %rsi
        addq    $4, %rbx
        call    std::basic_ostream<char, std::char_traits<char>>& std::__ostream_insert<char, std::char_traits<char>>(std::basic_ostream<char, std::char_traits<char>>&, char const*, long)
        cmpq    %rbx, %r12
        jne     .L24
.L23:
        movq    std::cout(%rip), %rax
        movq    -24(%rax), %rax
        movq    std::cout+240(%rax), %rbx
        testq   %rbx, %rbx
        je      .L29
        cmpb    $0, 56(%rbx)
        je      .L26
        movsbl  67(%rbx), %esi
.L27:
        movl    $std::cout, %edi
        call    std::ostream::put(char)
        movq    %rax, %rdi
        call    std::ostream::flush()
        movq    %r15, %rdi
        call    operator delete[](void*)
        movq    %r14, %rdi
        call    operator delete[](void*)
        movq    %r13, %rdi
        call    operator delete[](void*)
        addq    $32, %rsp
        xorl    %eax, %eax
        popq    %rbx
        popq    %r10
        popq    %r12
        popq    %r13
        popq    %r14
        popq    %r15
        popq    %rbp
        ret
.L21:
        movl    $32, %esi
        addq    $4, %rbx
        call    std::ostream::put(char)
        cmpq    %r12, %rbx
        jne     .L24
        jmp     .L23
.L26:
        movq    %rbx, %rdi
        call    std::ctype<char>::_M_widen_init() const
        movq    (%rbx), %rax
        movl    $10, %esi
        movq    %rbx, %rdi
        call    *48(%rax)
        movsbl  %al, %esi
        jmp     .L27
.L29:
        call    std::__throw_bad_cast()
.LC0:
        .long   0
        .long   1065353216
        .long   1073741824
        .long   1077936128
        .long   1082130432
        .long   1084227584
        .long   1086324736
        .long   1088421888
.LC1:
        .long   1090519040
        .long   1091567616
        .long   1092616192
        .long   1093664768
        .long   1094713344
        .long   1095761920
        .long   1096810496
        .long   1097859072