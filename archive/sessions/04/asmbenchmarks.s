    .text

    # %rdi = pointer to array
    # %rsi = number of 64-bit elements
    .globl  testSequential64
testSequential64:
    test    %rsi, %rsi          # n == 0?
    jz      .Ldone1

.Lloop1:
    mov     (%rdi), %rax        # load 8 bytes
    add     $8, %rdi            # advance pointer
    dec     %rsi                # n--
    jnz     .Lloop1             # loop while n != 0

.Ldone1:
    ret

    .globl  testSequential64b
testSequential64b:
    test    %rsi, %rsi          # n == 0?
    jz      .Ldone2

.Lloop2:
    mov     (%rdi), %rax
    add     $8, %rdi
    dec     %rsi
    jnz     .Lloop2

.Ldone2:
    ret
