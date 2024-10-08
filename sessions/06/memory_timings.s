    .global read_one
    .global read_memory_scalar
    .global read_memory_sse
    .global read_one_avx
    .global read_memory_avx
    .global read_one_avx
    .global read_memory_sse_unaligned
    .global read_memory_avx_unaligned
    .global read_memory_every2
    .global read_memory_everyk
    .global write_one
    .global write_one_avx
    .global write_memory_scalar
    .global write_memory_sse
    .global write_memory_avx

# Read one location repeatedly
read_one:
        mov  (%rdi), %rax        # read one 64-bit memory
        sub  $1, %rsi            # count down
        jg   read_memory_scalar  # keep going until all n locations have been read
        ret

// [0] [1] [2] [3] ... [n-1]
# Read 8 bytes (64 bits) at a time
read_memory_scalar:
        mov  (%rdi), %rax        # read one 64-bit memory
        add  $8, %rdi      # advance to next location
        sub  $1, %rsi            # count down
        jg   read_memory_scalar  # keep going until all n locations have been read
        ret

# Read all even words, then odd
read_memory_every2:
        mov %rdi, %r8            # save rdi so we can restore it to do odd locations
        mov  (%rdi), %rax        # read one 64-bit memory
        add  $16, %rdi      # advance to next location
        sub  $2, %rsi            # count down
        jg   read_memory_scalar  # keep going until all n locations have been read
        add  $8, %r8
1:
        mov  (%r8), %rax        # read one 64-bit memory
        add  $16, %r8           # advance to next location
        sub  $2, %rsi           # count down
        jg   1b                 # keep going until all n locations have been read
        ret
// DDR4 RAM wants to read 8 sequential locations in a burst
// DDR5 RAM wants to read 16 sequential locations
// DDR5-46-45-45
// 46+1+1+1+1+1+1+1+1+1+1+1+1+1+1+1
// if your computer has 2 banks, perhaps double that..
// +1+1+1+1+1+1+1+1+1+1+1+1+1+1+1+1

# Read words skipping k, then go back and fill in the missing ones
# for k=4, 0, 4, 8, 12, ... 1, 5, 9, 13, ..., 2, 6, 10, 14, ..., 3, 7, 11, 15, ...
# rdi = pointer to memory, rsi = number of locations, rdx = number to skip
read_memory_everyk:
        mov %rdx, %rcx          # rcx = number to skip
        mov %rsi, %rax          # Move dividend (rsi) to rax
        xor %rdx, %rdx          # Clear rdx for the remainder
        div %rcx                # Divide rax by rcx, result in rax, remainder in rdx
        mov %rax, %r10          # Move the result (quotient) to r10
        mov %rcx, %r8           # rcx will remain the number to skip each time
        shl $3, %r8             # r8 = number of bytes to advance each time
        mov %rcx, %r11          # r11 = the number of times to scan through the array

.outer_loop:
        mov %rdi, %rdx          # rdx = starting address of the array each time
        mov %r10, %r9           # r9 = n / skip each time
.inner_loop:
        mov  (%rdx), %rax       # read one 64-bit memory
        add  %r8, %rdx          # advance to next location
        sub  $1, %r9            # count down
        jg   .inner_loop        # keep going until all n/k locations have been read

        add  $8, %rdi           # move to next starting position
        sub  $1, %r11           # count down for each pass through the array
        jg   .outer_loop        # continue until all passes are done

.done:
        ret


# Read 16 bytes (128 bits) at a time
read_memory_sse:
# note: this is aligned data. There is also movdqu which would allow unaligned access but is slower
        movdqa (%rdi), %xmm0 # read 16 bytes (128 bits)
        add  $16, %rdi       # advance to next location
        sub  $2, %rsi        #count down
        jg   read_memory_sse  #keep going until all n locations have been read
        ret

# Read one location repeatedly using AVX
read_one_avx:
# note this instruction is aligned. There is also vmovdqu which would allow unaligned access but is slower
        vmovdqa (%rdi), %ymm0 # read 32 bytes (256 bits)
        sub  $4, %rsi        # count down
        jg   read_memory_avx  #keep going until all n locations have been read
        ret

# Read 32 bytes (256 bits) at a time
read_memory_avx:
# note this instruction is aligned. There is also vmovdqu which would allow unaligned access but is slower
        vmovdqa (%rdi), %ymm0 # read 32 bytes (256 bits)
        add  $32, %rdi       # advance to next location
        sub  $4, %rsi        # count down
        jg   read_memory_avx  #keep going until all n locations have been read
        ret

# Test whether the unaligned instruction is slower on aligned data
read_memory_sse_unaligned:
        movdqu (%rdi), %xmm0 # read 16 bytes (128 bits)
        add  $16, %rdi       # advance to next location
        sub  $2, %rsi        #count down
        jg   read_memory_sse  #keep going until all n locations have been read
        ret

# Test whether the unaligned instruction is slower on aligned data
read_memory_avx_unaligned:
# note this instruction is aligned. There is also vmovdqu which would allow unaligned access but is slower
        vmovdqu (%rdi), %ymm0 # read 32 bytes (256 bits)
        add  $32, %rdi       # advance to next location
        sub  $4, %rsi        # count down
        jg   read_memory_avx  #keep going until all n locations have been read
        ret

# Write repeatedly to one 64-bit memory location
write_one:
        mov $1, %rax
1:
        mov  %rax, (%rdi)
        sub  $1, %rsi
        jg   1b
        ret

# Write repeatedly to one 64-bit memory location
write_one_avx:
        mov $1, %rax
        vmovd %eax, %xmm0
        vbroadcastsd %xmm0, %ymm0
1:
        vmovdqa %ymm0, (%rdi)
        sub  $1, %rsi
        jg   1b
        ret


# Write 8 bytes (64 bits) at a time
write_memory_scalar:
        mov $1, %rax
1:
        mov  %rax, (%rdi)
        add  $8, %rdi
        sub  $1, %rsi
        jg   1b
        ret

# Write 16 bytes (128 bits) at a time
write_memory_sse:
        mov $1, %rax
        movq %rax, %xmm0
        pinsrq $1, %rax, %xmm0
1:
        movdqa %xmm0, (%rdi)
        mov  %rax, (%rdi)
        add  $16, %rdi
        sub  $2, %rsi
        jg   1b
        ret

# Write 32 bytes (256 bits) at a time
write_memory_avx:
        mov $1, %rax
        vmovq %rax, %xmm0
        vpbroadcastq %xmm0, %ymm0
1:
        vmovdqa %ymm0, (%rdi)
        add  $32, %rdi
        sub  $4, %rsi
        jg   1b
        ret
