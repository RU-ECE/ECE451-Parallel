# Assembler Overview

## What is a thread? What context?

Every thread needs:
- program counter (rip) (instruction pointer)
- stack pointer (rsp) 
- for debugging (rbp)
- registers (16) rax, rbx, ...
- xmm0, xmm1, ...
- zmm0, zmm31 x 512 bits (32 byte) 32*32 = 1024

## Intel Assembler

- 16 integer registers
  rax, rbx, rcx, rdx, rsi, rdi, rbp, rsp, r8, r9, r10, r11, r12, r13, r14, r15
  (can't use rsp, rbp only if you aren't debugging so really 14)
- 1300+ instructions
- instruction length varies from 2 to 9 bytes
- Intel always stayed ahead with their obscene architecture by being ahead in chip manufacturing
- floating point
  - SSE, AVX:xmm0-xmm15 (vector 128 bit registers, you can just use the low half)
  - AVX2:ymm0-ymm15 (vector 256 bit registers, the low half of each one is xmm?)
  - AVX512:zmm0-zmm31 (vector 512 bit registers, the low half of each one is ymm?)
## CISC vs RISC

- CISC: Complex Instruction Set Computer
  - complex to decode
  - instructions have different formats
  - Divide only works with certain registers
- RISC: Reduced Instruction Set Computer
  - instructions are highly regular (fixed length)
  - more registers
  - every instruction can read/write from any register
- both CISC and RISC have added specialty instructions
  - hardware assisted instructions: video encoding, compression, encryption
  - low precision floating point for neural networks
  - branch prediction

CISC:
- more instructions
- less registers
- more complex instructions
- less register instructions
- more complex instructions


## ARM AARCH64

- 31 integer registers X0-X30  X31 is zero
