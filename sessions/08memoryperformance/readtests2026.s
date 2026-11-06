	.globl read1, readfromreg

	#rdi = location
	#rsi = number of times to access
read1:
	endbr64
1:	
	mov (%rdi), %rax
	sub $1, %rsi
	jne 1b
        ret

readfromreg:
	endbr64	
	mov  (%rdi), %rbx
1:	
	mov %rbx, %rax
	sub $1, %rsi
	jne 1b
	ret
