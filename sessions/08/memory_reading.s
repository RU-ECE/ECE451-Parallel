    .globl read_1byte_sequentially
    .globl read_1byte_sequentially_b
    .globl read_8byte_sequentially
    .globl read_8byte_skip

# on ARM this would be
# x0 = address of array
# x1 = length of the array (n)


# %rdi = address of array
# %rsi = length of the array (n)
read_1byte_sequentially:
    movb (%rdi), %al  # move the byte into a register
    add  $1, %rdi     # advance to next location
    sub  $1, %rsi     # count down
    jg   read_1byte_sequentially
    ret

# %rdi = address of array
# %rsi = length of the array (n)
read_1byte_sequentially_b:
    mov  %rdi, %rdx # save starting address
    add  %rsi, %rdx # compute ending address
loop:
    movb (%rdi), %al  # move the byte into a register
    add  $1, %rdi     # advance to next location
    cmp  %rdi, %rdx
    jne  loop
    ret

# %rdi = address of array
# %rsi = length of the array (n) in 64-bit words (1/8 of bytes)
read_8byte_sequentially:
    movq (%rdi), %rax  # load 8 bytes register
    add  $8, %rdi     # advance to next location
    sub  $1, %rsi     # count down
    jg   read_1byte_sequentially
    ret

# %rdi = address of array
# %rsi = the number of times to loop through the array (n/skip) in 64-bit words (1/8 of bytes)
# %rdx = number to skip
read_8byte_skip:
    mov %rdx, %r8    # r8 count down how many times around the outer loop
    mov  %rdi, %rcx  # rcx stores the starting address
    mov  %rsi, %rbx   # rbx stores the number of elements
    shl  $3, %rdx # rdx = rdx * 8 calculate in bytes (skip)
outer_loop:
    mov %rbx, %rsi   # each time, we start with n elements
    mov %rcx, %rdi   # point to start of array
loop2: 
    movq (%rdi), %rax  # load 8 bytes register
    add  %rdx, %rdi     # advance to next location
    sub  $1, %rsi     # count down
    jg   loop2
    add  $8, %rcx     # advance the array to next location   
    sub  $1, %r8     # count down
    jg   outer_loop 
    ret
