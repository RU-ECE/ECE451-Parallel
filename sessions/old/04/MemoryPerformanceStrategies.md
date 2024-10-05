# Memory Performance Strategies

What can we do in architecture to improve the situation?

1. Every core should have its own local memory
   1. HUGE problem: how to manage this resource?
   2. State. If we do share it, how do you save it out?
      1. context switching
      2. PC = rip
      3. SP
      4. registers: rax, rbx, rcx, .... r15
      5. vector registers xmm0, xmm1, xmm15
      6. avx2: ymm0, ...ymm15
      7. avx512 zmm0, zmm31
   3. Have different CPU "sizes"
   4. 