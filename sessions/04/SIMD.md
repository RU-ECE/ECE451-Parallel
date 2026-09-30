# Single Instruction, MUltiple Data (SIMD) Vector Instructions

Intel AVX instructions
  SSE
  AVX
  AVX2
  AVX512

  each one more power than the previous

today: more efficient vector architectures

Whenever you use AVX512 instructions, your CPU slows down

INtel integer registers
4004 1971 2300 transistors 4-bit
8080 1974?                 8-bit  A
80286  16 bit                ax bx cx dx
80386  32 bit               eax ebx ecd edx esi edi ebp esp
80486 Pentium?  64-bit      rax, rbx, rcx, rdx, rsi, rdi, rbp, rsp,  r8, r9, .. r15     


arm 32-bit R0 .. r31
aarch64 64-bit   X0..x31   (x31 = 0)

transistors 5ps-10 switching      fastest computer 5GHz = 200ps clock time


Vector SIMD
SSE: xmm0.. xmm15   (128-bit) (2 double) (4 float) (8 16-bit integers)  (16 8-bit numbers)

AVX: ymm0 .. ymm15  (256-bit)  4 double   8 float
AVX2 : more instructions

AVX512: zmm0 ..zmm31  (512-bit)

P and E cores

