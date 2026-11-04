	.globl sequential
	# rdi points to memory
	# rsi the number of locations
	# assume this memory is ALIGNED
	#
	# 0:  0  1  2  3  4  5  6  7 
	# 8:  8  9 10 11 12 13 14 15
	# 16: 16 ...
	# 24:
	# 32:
	# 40:
	# 48: 48 49 50 51 52 53 54 55
	# 56: 56 57 58 59 60 61 62 63
	# 64:
	
sequential:
1:	
	mov (%rdi), %rax  #read in 64 bit number
	add $8, %rdi
	sub $1, %rsi      #countdown
	jne 1b
	ret

	.globl backward
backward:
	mov %rsi, %rcx
	shl $3, %rcx    # len <<= 3 ;
	sub $8, %rcx    # rsi will point to x[len-1]
	add %rcx, %rdi  # now point to the end?
1:	
	mov (%rdi), %rax  #read in 64 bit number
	sub $8, %rdi
	sub $1, %rsi      #countdown
	jne 1b
	ret 

	.globl skip2
skip2:
	mov %rsi, %rcx
	mov %rdi, %rdx  # save the number so we have it after first loop
1:	
	mov (%rsi), %rax  #read in 64 bit number
	add $16, %rsi
	sub $2, %rdi      #countdown
	jne 1b
	
	mov %rcx, %rsi # start over, and DO IT AGAIN
	mov %rdx, %rdi # len=original length
1:	
	mov (%rsi), %rax  #read in 64 bit number
	add $16, %rsi
	sub $2, %rdi      #countdown
	jne 1b
	
	ret

	
