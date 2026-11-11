	.globl main
main:
	mov $1000000000, %rcx # 1 billion
	call loop2

	ret

loop:
	xor %rax, %rax  # rax = 0
1:	
	add $1, %rax    #
	cmp %rcx, %rax
	jne 1b
	ret

loop2:
	xor %rax, %rax  # rax = 0
1:	
	add $1, %rax    #
	imul %rdx, %r9   # nonsense multiply
	cmp %rcx, %rax
	jne 1b
	ret
	
	


