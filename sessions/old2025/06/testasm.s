// rax, rbx, rcx, rdx, rsi, rdi, rbp, rsp, r8, r9, r10, r11, r12, r13, r14, r15
//                                x    x

    .global f
    ; rdi = address of the array  rsi = number of element
f:
      mov  $0, %rax
loop:
      mov (%rdi), %rbx  // load from each element in memory
      add %rbx, %rax  // rax = rax + rbx
      add $8, %rdi      // advance to next memory location
      sub $1, %rsi      //count down
      cmp $0, %rsi      // compare
      jg  loop          // jump if greater to do it again