    .text

    .globl  main
main:
    mov     $12, %rax
    mov     $15, %rsi
    add     %rsi, %rax      # rax = 12 + 15 = 27
    ret
