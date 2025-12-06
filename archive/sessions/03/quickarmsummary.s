    @ x0..x31
    @ BUT x31 = 0
    @
    @ RIGHT TO LEFT!!!!!
    @ Neon (old system, used on ARM on laptop)
    @
    @ don't remember the name of the new vector spec

    mov     x0, #128          // x0 = 128
    vmov.f64 v0, [x0]         // load memory location 128

    load    x1, [x0]
    load    x2, [x0, 32]

    store   x1, [x0, 32]

    add     x0, x1, x2        // x0 = x1 + x2
    sub     x0, x1, x2        // x0 = x1 - x2
    mul     x0, x1, x2        // x0 = x1 * x2
    fma     x0, x1, x2        // x0 = x0 + x1*x2
