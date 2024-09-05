
# %rsi = array %rdi = n
    .globl testSequential64
testSequential64:
loop:
    mov   (%rdi), %rax
    add   $8, %rdi
    sub   $1, %rsi
    cmp   $0, %rsi
    jz    loop 
    ret

    .globl testSequential64b
testSequential64b:
loop2:
    mov   (%rdi), %rax
    add   $8, %rdi
    sub   $1, %rsi
    jz    loop2 
    ret

