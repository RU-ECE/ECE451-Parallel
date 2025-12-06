    .text

    .globl  read_1byte_sequentially
    .globl  read_1byte_sequentially_b
    .globl  read_8byte_sequentially
    .globl  read_8byte_skip

# %rdi = address of array
# %rsi = length of the array (n bytes)
read_1byte_sequentially:
    movb    (%rdi), %al          # move the byte into a register
    add     $1, %rdi             # advance to next location
    sub     $1, %rsi             # count down
    jg      read_1byte_sequentially
    ret

# %rdi = address of array
# %rsi = length of the array (n bytes)
read_1byte_sequentially_b:
    mov     %rdi, %rdx           # save starting address
    add     %rsi, %rdx           # compute ending address = start + n
.loop1:
    movb    (%rdi), %al          # move the byte into a register
    add     $1, %rdi             # advance to next location
    cmp     %rdi, %rdx
    jne     .loop1
    ret

# %rdi = address of array
# %rsi = length of the array (n) in 64-bit words
read_8byte_sequentially:
    movq    (%rdi), %rax         # load 8 bytes into a register
    add     $8, %rdi             # advance to next 64-bit element
    sub     $1, %rsi             # count down
    jg      read_8byte_sequentially
    ret

# %rdi = address of array
# %rsi = number of elements per pass (n/skip) in 64-bit words
# %rdx = number to skip (stride in elements)
read_8byte_skip:
    mov     %rdx, %r8            # r8  = skip (also outer-loop count)
    mov     %rdi, %rcx           # rcx = starting address (base)
    mov     %rsi, %r9            # r9  = elements per inner pass
    shl     $3, %rdx             # rdx = skip * 8 (stride in bytes)

.outer_loop:
    mov     %r9, %rsi            # reset inner counter
    mov     %rcx, %rdi           # reset pointer to current base

.inner_loop:
    movq    (%rdi), %rax         # load 8 bytes into a register
    add     %rdx, %rdi           # advance by skip elements
    sub     $1, %rsi             # count down
    jg      .inner_loop

    add     $8, %rcx             # base++ (move to next element)
    sub     $1, %r8              # outer-loop count down
    jg      .outer_loop
    ret
