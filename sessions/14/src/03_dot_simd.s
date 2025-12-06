dot_simd2(float const*, float const*, int) (._omp_fn.0):
        pushq   %rbp
        movq    %rsp, %rbp
        pushq   %r14
        pushq   %r13
        pushq   %r12
        pushq   %rbx
        movq    %rdi, %rbx
        movq    8(%rdi), %r13
        movq    (%rdi), %r14
        andq    $-32, %rsp
        call    omp_get_num_threads
        movl    %eax, %r12d
        call    omp_get_thread_num
        movl    %eax, %esi
        movl    16(%rbx), %eax
        cltd
        idivl   %r12d
        cmpl    %edx, %esi
        jl      .L2
.L11:
        imull   %eax, %esi
        vxorps  %xmm0, %xmm0, %xmm0
        leal    (%rdx,%rsi), %r8d
        leal    (%rax,%r8), %r9d
        cmpl    %r9d, %r8d
        jge     .L3
        leal    -1(%rax), %ecx
        cmpl    $6, %ecx
        jbe     .L13
        movslq  %esi, %rcx
        movslq  %edx, %rdi
        movl    %eax, %r10d
        addq    %rcx, %rdi
        shrl    $3, %r10d
        xorl    %ecx, %ecx
        salq    $2, %rdi
        salq    $5, %r10
        leaq    (%r14,%rdi), %r11
        addq    %r13, %rdi
.L5:
        vmovups (%r11,%rcx), %ymm4
        vmulps  (%rdi,%rcx), %ymm4, %ymm1
        addq    $32, %rcx
        vaddss  %xmm1, %xmm0, %xmm0
        vshufps $85, %xmm1, %xmm1, %xmm3
        vshufps $255, %xmm1, %xmm1, %xmm2
        vaddss  %xmm3, %xmm0, %xmm0
        vunpckhps       %xmm1, %xmm1, %xmm3
        vextractf128    $0x1, %ymm1, %xmm1
        vaddss  %xmm3, %xmm0, %xmm0
        vaddss  %xmm2, %xmm0, %xmm0
        vshufps $85, %xmm1, %xmm1, %xmm2
        vaddss  %xmm1, %xmm0, %xmm0
        vaddss  %xmm2, %xmm0, %xmm0
        vunpckhps       %xmm1, %xmm1, %xmm2
        vshufps $255, %xmm1, %xmm1, %xmm1
        vaddss  %xmm2, %xmm0, %xmm0
        vaddss  %xmm1, %xmm0, %xmm0
        cmpq    %rcx, %r10
        jne     .L5
        movl    %eax, %ecx
        andl    $-8, %ecx
        addl    %ecx, %r8d
        cmpl    %eax, %ecx
        je      .L24
        vzeroupper
.L4:
        subl    %ecx, %eax
        leal    -1(%rax), %edi
        cmpl    $2, %edi
        jbe     .L8
        movslq  %edx, %rdx
        movslq  %esi, %rsi
        addq    %rsi, %rdx
        addq    %rcx, %rdx
        vmovups (%r14,%rdx,4), %xmm7
        vmulps  0(%r13,%rdx,4), %xmm7, %xmm1
        movl    %eax, %edx
        andl    $-4, %edx
        addl    %edx, %r8d
        vaddss  %xmm1, %xmm0, %xmm0
        vshufps $85, %xmm1, %xmm1, %xmm2
        vaddss  %xmm2, %xmm0, %xmm0
        vunpckhps       %xmm1, %xmm1, %xmm2
        vshufps $255, %xmm1, %xmm1, %xmm1
        vaddss  %xmm2, %xmm0, %xmm0
        vaddss  %xmm1, %xmm0, %xmm0
        testb   $3, %al
        je      .L3
.L8:
        movslq  %r8d, %rdx
        vmovss  (%r14,%rdx,4), %xmm7
        leaq    0(,%rdx,4), %rax
        vfmadd231ss     0(%r13,%rdx,4), %xmm7, %xmm0
        leal    1(%r8), %edx
        cmpl    %edx, %r9d
        jle     .L3
        addl    $2, %r8d
        vmovss  4(%r14,%rax), %xmm5
        vfmadd231ss     4(%r13,%rax), %xmm5, %xmm0
        cmpl    %r8d, %r9d
        jle     .L3
        vmovss  8(%r14,%rax), %xmm6
        vfmadd231ss     8(%r13,%rax), %xmm6, %xmm0
.L3:
        movl    20(%rbx), %edx
        leaq    20(%rbx), %rcx
.L10:
        vmovd   %edx, %xmm6
        movl    %edx, %eax
        vaddss  %xmm6, %xmm0, %xmm5
        vmovd   %xmm5, %esi
        lock cmpxchgl   %esi, (%rcx)
        jne     .L25
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
        jmp     .L11
.L13:
        xorl    %ecx, %ecx
        jmp     .L4
.L24:
        vzeroupper
        jmp     .L3
.L25:
        movl    %eax, %edx
        jmp     .L10
dot_simd3(float const*, float const*, int) (._omp_fn.0):
        pushq   %rbp
        movq    %rsp, %rbp
        pushq   %r12
        pushq   %rbx
        movq    %rdi, %rbx
        andq    $-32, %rsp
        subq    $32, %rsp
        call    omp_get_num_threads
        movl    %eax, %r12d
        call    omp_get_thread_num
        movl    %eax, %ecx
        movl    16(%rbx), %eax
        cltd
        idivl   %r12d
        cmpl    %edx, %ecx
        jl      .L27
.L35:
        movl    %ecx, %esi
        vxorps  %xmm0, %xmm0, %xmm0
        imull   %eax, %esi
        leal    (%rdx,%rsi), %edi
        leal    (%rax,%rdi), %r8d
        cmpl    %r8d, %edi
        jge     .L28
        vpxor   %xmm0, %xmm0, %xmm0
        leal    -1(%rax), %ecx
        movq    8(%rbx), %r9
        movq    (%rbx), %r10
        vmovdqa %xmm0, (%rsp)
        vmovdqa %xmm0, 16(%rsp)
        cmpl    $6, %ecx
        jbe     .L32
        movslq  %edx, %rcx
        movslq  %esi, %rsi
        vxorps  %xmm0, %xmm0, %xmm0
        xorl    %edx, %edx
        addq    %rsi, %rcx
        movl    %eax, %esi
        salq    $2, %rcx
        shrl    $3, %esi
        leaq    (%r10,%rcx), %r11
        salq    $5, %rsi
        addq    %r9, %rcx
.L30:
        vmovups (%r11,%rdx), %ymm1
        vfmadd231ps     (%rcx,%rdx), %ymm1, %ymm0
        addq    $32, %rdx
        cmpq    %rdx, %rsi
        jne     .L30
        movl    %eax, %edx
        vmovaps %ymm0, (%rsp)
        andl    $-8, %edx
        addl    %edx, %edi
        cmpl    %edx, %eax
        je      .L42
        vzeroupper
.L32:
        movslq  %edi, %rdx
        vmovss  (%rsp), %xmm4
        vmovss  (%r10,%rdx,4), %xmm0
        vfmadd132ss     (%r9,%rdx,4), %xmm4, %xmm0
        leaq    0(,%rdx,4), %rax
        leal    1(%rdi), %edx
        vmovss  %xmm0, (%rsp)
        cmpl    %edx, %r8d
        jle     .L31
        vmovss  4(%r10,%rax), %xmm5
        vfmadd231ss     4(%r9,%rax), %xmm5, %xmm0
        leal    2(%rdi), %edx
        vmovss  %xmm0, (%rsp)
        cmpl    %edx, %r8d
        jle     .L31
        vmovss  8(%r10,%rax), %xmm6
        vfmadd231ss     8(%r9,%rax), %xmm6, %xmm0
        leal    3(%rdi), %edx
        vmovss  %xmm0, (%rsp)
        cmpl    %edx, %r8d
        jle     .L31
        vmovss  12(%r10,%rax), %xmm7
        vfmadd231ss     12(%r9,%rax), %xmm7, %xmm0
        leal    4(%rdi), %edx
        vmovss  %xmm0, (%rsp)
        cmpl    %edx, %r8d
        jle     .L31
        vmovss  16(%r10,%rax), %xmm7
        vfmadd231ss     16(%r9,%rax), %xmm7, %xmm0
        leal    5(%rdi), %edx
        vmovss  %xmm0, (%rsp)
        cmpl    %edx, %r8d
        jle     .L31
        vmovss  20(%r10,%rax), %xmm6
        vfmadd231ss     20(%r9,%rax), %xmm6, %xmm0
        addl    $6, %edi
        vmovss  %xmm0, (%rsp)
        cmpl    %edi, %r8d
        jle     .L31
        vmovss  24(%r10,%rax), %xmm5
        vfmadd231ss     24(%r9,%rax), %xmm5, %xmm0
        vmovss  %xmm0, (%rsp)
.L31:
        vxorps  %xmm0, %xmm0, %xmm0
        vaddss  (%rsp), %xmm0, %xmm0
        vaddss  4(%rsp), %xmm0, %xmm0
        vaddss  8(%rsp), %xmm0, %xmm0
        vaddss  12(%rsp), %xmm0, %xmm0
        vaddss  16(%rsp), %xmm0, %xmm0
        vaddss  20(%rsp), %xmm0, %xmm0
        vaddss  24(%rsp), %xmm0, %xmm0
        vaddss  28(%rsp), %xmm0, %xmm0
.L28:
        movl    20(%rbx), %edx
        leaq    20(%rbx), %rcx
.L34:
        vmovd   %edx, %xmm3
        movl    %edx, %eax
        vaddss  %xmm3, %xmm0, %xmm2
        vmovd   %xmm2, %esi
        lock cmpxchgl   %esi, (%rcx)
        jne     .L45
        leaq    -16(%rbp), %rsp
        popq    %rbx
        popq    %r12
        popq    %rbp
        ret
.L27:
        addl    $1, %eax
        xorl    %edx, %edx
        jmp     .L35
.L42:
        vzeroupper
        jmp     .L31
.L45:
        movl    %eax, %edx
        jmp     .L34
prod(float const*, int) (._omp_fn.0):
        pushq   %r12
        movq    (%rdi), %r12
        pushq   %rbp
        pushq   %rbx
        movq    %rdi, %rbx
        call    omp_get_num_threads
        movl    %eax, %ebp
        call    omp_get_thread_num
        movl    %eax, %ecx
        movl    8(%rbx), %eax
        cltd
        idivl   %ebp
        cmpl    %edx, %ecx
        jl      .L47
.L51:
        imull   %eax, %ecx
        vmovss  .LC2(%rip), %xmm0
        addl    %ecx, %edx
        leal    (%rax,%rdx), %ecx
        cmpl    %ecx, %edx
        jge     .L48
        movslq  %edx, %rdx
        movl    %eax, %eax
        addq    %rdx, %rax
        leaq    (%r12,%rdx,4), %rcx
        leaq    (%r12,%rax,4), %rax
        movq    %rax, %rdx
        subq    %rcx, %rdx
        andl    $4, %edx
        je      .L49
        vmulss  (%rcx), %xmm0, %xmm0
        addq    $4, %rcx
        cmpq    %rcx, %rax
        je      .L48
.L49:
        vmulss  (%rcx), %xmm0, %xmm0
        addq    $8, %rcx
        vmulss  -4(%rcx), %xmm0, %xmm0
        cmpq    %rcx, %rax
        jne     .L49
.L48:
        movl    12(%rbx), %edx
        leaq    12(%rbx), %rcx
.L50:
        vmovd   %edx, %xmm2
        movl    %edx, %eax
        vmulss  %xmm2, %xmm0, %xmm1
        vmovd   %xmm1, %esi
        lock cmpxchgl   %esi, (%rcx)
        jne     .L63
        popq    %rbx
        popq    %rbp
        popq    %r12
        ret
.L47:
        addl    $1, %eax
        xorl    %edx, %edx
        jmp     .L51
.L63:
        movl    %eax, %edx
        jmp     .L50
dot_simd(float const*, float const*, int):
        movq    %rsi, %rcx
        testl   %edx, %edx
        jle     .L72
        leal    -1(%rdx), %eax
        cmpl    $6, %eax
        jbe     .L73
        movl    %edx, %esi
        xorl    %eax, %eax
        vxorps  %xmm0, %xmm0, %xmm0
        shrl    $3, %esi
        salq    $5, %rsi
.L67:
        vmovups (%rdi,%rax), %ymm4
        vmulps  (%rcx,%rax), %ymm4, %ymm1
        addq    $32, %rax
        vaddss  %xmm1, %xmm0, %xmm0
        vshufps $85, %xmm1, %xmm1, %xmm3
        vshufps $255, %xmm1, %xmm1, %xmm2
        vaddss  %xmm3, %xmm0, %xmm0
        vunpckhps       %xmm1, %xmm1, %xmm3
        vextractf128    $0x1, %ymm1, %xmm1
        vaddss  %xmm3, %xmm0, %xmm0
        vaddss  %xmm2, %xmm0, %xmm0
        vshufps $85, %xmm1, %xmm1, %xmm2
        vaddss  %xmm1, %xmm0, %xmm0
        vaddss  %xmm2, %xmm0, %xmm0
        vunpckhps       %xmm1, %xmm1, %xmm2
        vshufps $255, %xmm1, %xmm1, %xmm1
        vaddss  %xmm2, %xmm0, %xmm0
        vaddss  %xmm1, %xmm0, %xmm0
        cmpq    %rax, %rsi
        jne     .L67
        movl    %edx, %eax
        andl    $-8, %eax
        movl    %eax, %esi
        cmpl    %eax, %edx
        je      .L79
        vzeroupper
.L66:
        movl    %edx, %r8d
        subl    %esi, %r8d
        leal    -1(%r8), %r9d
        cmpl    $2, %r9d
        jbe     .L70
        vmovups (%rdi,%rsi,4), %xmm5
        vmulps  (%rcx,%rsi,4), %xmm5, %xmm1
        vaddss  %xmm1, %xmm0, %xmm0
        vshufps $85, %xmm1, %xmm1, %xmm2
        vaddss  %xmm2, %xmm0, %xmm0
        vunpckhps       %xmm1, %xmm1, %xmm2
        vshufps $255, %xmm1, %xmm1, %xmm1
        vaddss  %xmm2, %xmm0, %xmm0
        vaddss  %xmm1, %xmm0, %xmm0
        testb   $3, %r8b
        je      .L64
        andl    $-4, %r8d
        addl    %r8d, %eax
.L70:
        movslq  %eax, %r8
        vmovss  (%rdi,%r8,4), %xmm6
        leaq    0(,%r8,4), %rsi
        vfmadd231ss     (%rcx,%r8,4), %xmm6, %xmm0
        leal    1(%rax), %r8d
        cmpl    %r8d, %edx
        jle     .L64
        addl    $2, %eax
        vmovss  4(%rdi,%rsi), %xmm7
        vfmadd231ss     4(%rcx,%rsi), %xmm7, %xmm0
        cmpl    %eax, %edx
        jle     .L64
        vmovss  8(%rdi,%rsi), %xmm7
        vfmadd231ss     8(%rcx,%rsi), %xmm7, %xmm0
        ret
.L72:
        vxorps  %xmm0, %xmm0, %xmm0
.L64:
        ret
.L79:
        vzeroupper
        ret
.L73:
        xorl    %esi, %esi
        xorl    %eax, %eax
        vxorps  %xmm0, %xmm0, %xmm0
        jmp     .L66
dot_simd2(float const*, float const*, int):
        subq    $40, %rsp
        xorl    %ecx, %ecx
        movl    %edx, 16(%rsp)
        xorl    %edx, %edx
        movq    %rsi, 8(%rsp)
        movq    %rsp, %rsi
        movq    %rdi, (%rsp)
        movl    $dot_simd2(float const*, float const*, int) (._omp_fn.0), %edi
        movl    $0x00000000, 20(%rsp)
        call    GOMP_parallel
        vmovss  20(%rsp), %xmm0
        addq    $40, %rsp
        ret
dot_simd3(float const*, float const*, int):
        subq    $40, %rsp
        xorl    %ecx, %ecx
        movl    %edx, 16(%rsp)
        xorl    %edx, %edx
        movq    %rsi, 8(%rsp)
        movq    %rsp, %rsi
        movq    %rdi, (%rsp)
        movl    $dot_simd3(float const*, float const*, int) (._omp_fn.0), %edi
        movl    $0x00000000, 20(%rsp)
        call    GOMP_parallel
        vmovss  20(%rsp), %xmm0
        addq    $40, %rsp
        ret
prod(float const*, int):
        subq    $24, %rsp
        xorl    %ecx, %ecx
        xorl    %edx, %edx
        movl    %esi, 8(%rsp)
        movq    %rsp, %rsi
        movq    %rdi, (%rsp)
        movl    $prod(float const*, int) (._omp_fn.0), %edi
        movl    $0x3f800000, 12(%rsp)
        call    GOMP_parallel
        vmovss  12(%rsp), %xmm0
        addq    $24, %rsp
        ret
horizontal_sum(float vector[8]):
        vhaddps %ymm0, %ymm0, %ymm1
        vmovaps %xmm1, %xmm0
        vextractf128    $0x1, %ymm1, %xmm1
        vaddps  %xmm1, %xmm0, %xmm0
        vhaddps %xmm0, %xmm0, %xmm0
        vhaddps %xmm0, %xmm0, %xmm0
        ret
dot_avx2manual(float const*, float const*, int):
        testl   %edx, %edx
        jle     .L90
        xorl    %eax, %eax
        vxorps  %xmm0, %xmm0, %xmm0
.L89:
        vmovups (%rsi,%rax,4), %ymm2
        vfmadd231ps     (%rdi,%rax,4), %ymm2, %ymm0
        addq    $8, %rax
        cmpl    %eax, %edx
        jg      .L89
.L88:
        vhaddps %ymm0, %ymm0, %ymm0
        vmovaps %xmm0, %xmm1
        vextractf128    $0x1, %ymm0, %xmm0
        vaddps  %xmm0, %xmm1, %xmm0
        vhaddps %xmm0, %xmm0, %xmm0
        vhaddps %xmm0, %xmm0, %xmm0
        vzeroupper
        ret
.L90:
        vxorps  %xmm0, %xmm0, %xmm0
        jmp     .L88
.LC2:
        .long   1065353216