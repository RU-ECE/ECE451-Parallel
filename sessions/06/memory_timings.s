    .text

    .globl  read_one
    .globl  read_memory_scalar
    .globl  read_memory_sse
    .globl  read_one_avx
    .globl  read_memory_avx
    .globl  read_memory_sse_unaligned
    .globl  read_memory_avx_unaligned
    .globl  read_memory_every2
    .globl  read_memory_everyk
    .globl  write_one
    .globl  write_one_avx
    .globl  write_memory_scalar
    .globl  write_memory_sse
    .globl  write_memory_avx

# Read one location repeatedly
# rdi = pointer, rsi = number of reads
read_one:
    mov     (%rdi), %rax         # read one 64-bit memory
    sub     $1, %rsi             # count down
    jg      read_one             # keep going until all reads have been done
    ret

# [0] [1] [2] [3] ... [n-1]
# Read 8 bytes (64 bits) at a time
# rdi = pointer, rsi = number of 64-bit locations
read_memory_scalar:
    mov     (%rdi), %rax         # read one 64-bit memory
    add     $8, %rdi             # advance to next location
    sub     $1, %rsi             # count down
    jg      read_memory_scalar   # keep going until all n locations have been read
    ret

# Read all even words, then all odd words
# rdi = pointer, rsi = number of 64-bit locations
read_memory_every2:
    test    %rsi, %rsi
    jle     .done_every2

    mov     %rdi, %r8            # r8 = base pointer

    # --- even indices: 0, 2, 4, ... ---
    mov     %rsi, %rcx           # rcx = n
    add     $1, %rcx             # rcx = n + 1
    shr     $1, %rcx             # rcx = (n + 1) / 2 = number of even indices
    mov     %r8, %rdx            # rdx = pointer for evens

.even_loop:
    mov     (%rdx), %rax
    add     $16, %rdx            # skip one word -> +16 bytes
    dec     %rcx
    jg      .even_loop

    # --- odd indices: 1, 3, 5, ... ---
    mov     %rsi, %rcx           # rcx = n
    shr     $1, %rcx             # rcx = n / 2 = number of odd indices
    jz      .done_every2         # no odds if n <= 1

    lea     8(%r8), %rdx         # rdx = base + 8 = index 1

.odd_loop:
    mov     (%rdx), %rax
    add     $16, %rdx
    dec     %rcx
    jg      .odd_loop

.done_every2:
    ret

# Read words skipping k, then go back and fill in the missing ones
# for k=4, 0,4,8,12,... 1,5,9,13,... 2,6,10,14,... 3,7,11,15,...
# rdi = pointer to memory, rsi = number of locations, rdx = number to skip (k)
read_memory_everyk:
    mov     %rdx, %rcx           # rcx = k (number to skip)
    mov     %rsi, %rax           # dividend n = rsi -> rax
    xor     %rdx, %rdx           # clear rdx for div
    div     %rcx                 # rax = n / k, rdx = n % k
    mov     %rax, %r10           # r10 = n / k
    mov     %rcx, %r8            # r8  = k
    shl     $3, %r8              # r8  = k * 8 bytes
    mov     %rcx, %r11           # r11 = k (number of passes)

.outer_loop:
    mov     %rdi, %rdx           # rdx = starting address for this pass
    mov     %r10, %r9            # r9  = n / k for this pass

.inner_loop:
    mov     (%rdx), %rax         # read one 64-bit memory
    add     %r8, %rdx            # advance by k elements
    sub     $1, %r9              # count down
    jg      .inner_loop          # keep going until all n/k locations have been read

    add     $8, %rdi             # move start to next element
    sub     $1, %r11             # one pass done
    jg      .outer_loop          # continue until all passes are done

    ret


# Read 16 bytes (128 bits) at a time (aligned)
# rdi = pointer, rsi = number of 64-bit locations (must be multiple of 2)
read_memory_sse:
    # note: this is aligned data. movdqu allows unaligned access but is slower.
    movdqa  (%rdi), %xmm0        # read 16 bytes (128 bits)
    add     $16, %rdi            # advance to next location
    sub     $2, %rsi             # count down (2 x 64-bit words)
    jg      read_memory_sse      # keep going until all n locations have been read
    ret

# Read one location repeatedly using AVX (aligned)
# rdi = pointer, rsi = number of 64-bit locations (must be multiple of 4)
read_one_avx:
    # note: this instruction is aligned. vmovdqu allows unaligned but is slower.
    vmovdqa (%rdi), %ymm0        # read 32 bytes (256 bits)
    sub     $4, %rsi             # count down (4 x 64-bit words)
    jg      read_one_avx         # keep going until all reads have been done
    ret

# Read 32 bytes (256 bits) at a time (aligned)
# rdi = pointer, rsi = number of 64-bit locations (must be multiple of 4)
read_memory_avx:
    # note: this instruction is aligned. vmovdqu allows unaligned but is slower.
    vmovdqa (%rdi), %ymm0        # read 32 bytes (256 bits)
    add     $32, %rdi            # advance to next location
    sub     $4, %rsi             # count down (4 x 64-bit words)
    jg      read_memory_avx      # keep going until all n locations have been read
    ret

# Test whether the unaligned SSE instruction is slower on aligned data
# rdi = pointer, rsi = number of 64-bit locations (must be multiple of 2)
read_memory_sse_unaligned:
    movdqu  (%rdi), %xmm0        # read 16 bytes (128 bits)
    add     $16, %rdi            # advance to next location
    sub     $2, %rsi             # count down
    jg      read_memory_sse_unaligned
    ret

# Test whether the unaligned AVX instruction is slower on aligned data
# rdi = pointer, rsi = number of 64-bit locations (must be multiple of 4)
read_memory_avx_unaligned:
    vmovdqu (%rdi), %ymm0        # read 32 bytes (256 bits)
    add     $32, %rdi            # advance to next location
    sub     $4, %rsi             # count down
    jg      read_memory_avx_unaligned
    ret


# Write repeatedly to one 64-bit memory location
# rdi = pointer, rsi = number of writes
write_one:
    mov     $1, %rax
1:
    mov     %rax, (%rdi)
    sub     $1, %rsi
    jg      1b
    ret

# Write repeatedly to one 32-byte location using AVX
# rdi = pointer, rsi = number of vector writes
write_one_avx:
    mov     $1, %rax
    vmovd   %eax, %xmm0
    vbroadcastsd %xmm0, %ymm0
1:
    vmovdqa %ymm0, (%rdi)
    sub     $1, %rsi
    jg      1b
    ret

# Write 8 bytes (64 bits) at a time
# rdi = pointer, rsi = number of 64-bit locations
write_memory_scalar:
    mov     $1, %rax
1:
    mov     %rax, (%rdi)
    add     $8, %rdi
    sub     $1, %rsi
    jg      1b
    ret

# Write 16 bytes (128 bits) at a time using SSE
# rdi = pointer, rsi = number of 64-bit locations (must be multiple of 2)
write_memory_sse:
    mov     $1, %rax
    movq    %rax, %xmm0
    pinsrq  $1, %rax, %xmm0      # fill upper 64 bits too
1:
    movdqa  %xmm0, (%rdi)
    add     $16, %rdi
    sub     $2, %rsi
    jg      1b
    ret

# Write 32 bytes (256 bits) at a time using AVX
# rdi = pointer, rsi = number of 64-bit locations (must be multiple of 4)
write_memory_avx:
    mov     $1, %rax
    vmovq   %rax, %xmm0
    vpbroadcastq %xmm0, %ymm0
1:
    vmovdqa %ymm0, (%rdi)
    add     $32, %rdi
    sub     $4, %rsi
    jg      1b
    ret
