    .text

    .globl  f
    # rdi = address of the array
    # rsi = number of elements (each 8 bytes)
f:
    xor     %rax, %rax         # sum = 0

    test    %rsi, %rsi         # if count <= 0, return 0
    jle     .done

.loop:
    mov     (%rdi), %rdx       # load element (assume 64-bit)
    add     %rdx, %rax         # sum += element
    add     $8, %rdi           # advance to next element
    dec     %rsi               # count--
    jg      .loop              # if count > 0, keep going

.done:
    ret
