# Assembly Overview

## What is a thread? What context?

Every thread needs at least:

- Program counter (`RIP`, instruction pointer)
- Stack pointer (`RSP`)
- Frame/base pointer (`RBP`, mainly for debugging / stack walking)
- General-purpose registers (16 total): `RAX`, `RBX`, ..., `R15`
- SIMD/vector registers, e.g.:
	- `XMM0` to `XMM15` (128-bit)
	- `YMM0` to `YMM15` (256-bit; low 128 bits alias the corresponding `XMM` register)
	- `ZMM0` to `ZMM31` (512-bit; low 256 bits alias `YMM`, low 128 bits alias `XMM`)

## Intel Assembler

- Sixteen general-purpose integer registers:
	- `RAX`, `RBX`, `RCX`, `RDX`, `RSI`, `RDI`, `RBP`, `RSP`,
	  `R8`, `R9`, `R10`, `R11`, `R12`, `R13`, `R14`, `R15`
	- However, `RSP` (stack pointer) and `RBP` (frame pointer) are typically reserved for stack management and
	  debugging, so only fourteen are usually available for general-purpose use.
- 1300+ instructions
- Instruction length varies from 2 to 9 bytes
- Historically, Intel has stayed competitive with this complex architecture by strong chip manufacturing.
- Floating-point / SIMD:
	- SSE, AVX: `XMM0` to `XMM15` (128-bit vector registers)
	- AVX2: `YMM0` to `YMM15` (256-bit vector registers; low 128 bits are the `XMM` registers)
	- AVX-512: `ZMM0` to `ZMM31` (512-bit vector registers; low 256 bits are the `YMM` registers)

## CISC vs RISC

- **CISC**: Complex Instruction Set Computer
	- Complex to decode
	- Instructions have different formats and lengths
	- Some instructions only work with specific registers (e.g., divide)
- **RISC**: Reduced Instruction Set Computer
	- Instructions are highly regular (often fixed length)
	- More registers
	- Every instruction can generally read/write any register

- Both CISC and RISC have added many specialty instructions:
	- Hardware-assisted instructions: video encoding, compression, encryption
	- Low-precision floating point for neural networks
	- Branch prediction and other speculative features

### CISC (summary characteristics)

- More instructions
- Fewer registers (historically)
- More complex instructions
- More complex decoding logic

## ARM AArch64

- 31 general-purpose integer registers: `X0`–`X30`
- `X31` is used as the zero register (and also as the stack pointer in some encodings)
