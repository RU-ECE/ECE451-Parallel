# What would parallel AVX sort code look like?

1. load as many values as you can into registers
   1. Peloton guys used 8 registers
   2. If you use AVX512 32 512-bit registers, more credit
      1. 16 registers, 47 swaps
      2. 47*3 swaps + transpose 120 = 270 instructions sort 256
   3. sort vertically
   4. transpose
   5. Now you have registers, each of which is sorted
   6. merge