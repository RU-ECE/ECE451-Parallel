# Rules of Memory Access

1. When reading from RAM we always bring in a cache line
assuming: 2 banks of memory, DDR5 (burst 16)
If you read memory location x (x MOD 64 == 0)
  actually read x, x+8, x+16, x+24 ... = 8*32 = 256 bytes

  a. to read an arbitrary location:
    RAS + CAS + 15 = 46+45+15 = 
    
  b. to read in the current row:
     CAS + 15 = 45+15 = 60

READ FROM LOCATION RDI

1:
    mov (%r8), %rax   READ FROM CACHE L1
    sub $1, %rdi
    jg 1b


L1 cache = 4 clocks

2. SEQ going BACKWARDS THROUGH MEMORY
READ FROM LOCATION a[rdi/8]
    mov (%r8, %rdi, 8), %rax   READ FROM CACHE L1
    sub $1, %rdi
    jg 1b

CAS + 15 = 60 reads in 32 elements (2 banks, 16 burst each) 
60 / 32 = 1.875 clocks per element

RAS + CAS + 16 = 106

3. SEQ FORWARDS
    mov (%r8), %rax   READ FROM CACHE L1
    add $8, %r8
    sub $1, %rdi
    jg 1b


YOU CAN NEVER USE CACHE IF YOU DID NOT READ FROM IT BEFORE

4. SKIP 2
    mov (%r8), %rax   READ FROM CACHE L1
    add $16, %r8
    sub $1, %rdi
    jg 1b

    CAS + 15 = 60 clocks read 32 elements
    60 / 16 = 3.75 clocks per element


    4b. SKIP 4
    mov (%r8), %rax   READ FROM CACHE L1
    add $32, %r8
    sub $1, %rdi
    jg 1b

    CAS + 15 = 60 clocks read 32 elements
    60 / 8 ~= 8 clocks per element

5. Skipping 128
    mov (%r8), %rax   READ FROM CACHE L1
    add $128, %r8
    sub $1, %rdi
    jg 1b

    CAS + 15 = 60 clocks read 32 elements
    60 / 2 ~= 30 clocks per element

5b. Skipping 256
    mov (%r8), %rax   READ FROM CACHE L1
    add $256, %r8
    sub $1, %rdi
    jg 1b

    CAS + 15 = 60 clocks read 32 elements
    60 / 1 ~= 30 clocks per element


6b.. Skipping 8192
    mov (%r8), %rax   READ FROM CACHE L1
    add $256, %r8
    sub $1, %rdi
    jg 1b

    ras+CAS + 15 = 106 clocks read 32 elements
    106 / 1 ~= 106 clocks per element
