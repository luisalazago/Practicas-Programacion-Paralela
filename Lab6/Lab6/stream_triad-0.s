	.file	"stream_triad.c"
	.local	a
	.comm	a,134217728,32
	.local	b
	.comm	b,134217728,32
	.local	c
	.comm	c,134217728,32
	.section	.rodata
	.align 8
.LC5:
	.string	"\033[1;32mAverage runtime is %lf msecs\n\033[0m"
	.text
	.globl	main
	.type	main, @function
main:
.LFB0:
	.cfi_startproc
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$64, %rsp
	movl	%edi, -52(%rbp)
	movq	%rsi, -64(%rbp)
	movabsq	$4613937818241073152, %rax
	movq	%rax, -24(%rbp)
	movl	$0, %eax
	movq	%rax, -32(%rbp)
	movl	$0, -44(%rbp)
	jmp	.L2
.L3:
	movl	-44(%rbp), %eax
	movslq	%eax, %rdx
	movabsq	$4607182418800017408, %rax
	movq	%rax, a(,%rdx,8)
	movl	-44(%rbp), %eax
	movslq	%eax, %rdx
	movabsq	$4611686018427387904, %rax
	movq	%rax, b(,%rdx,8)
	addl	$1, -44(%rbp)
.L2:
	cmpl	$16777215, -44(%rbp)
	jle	.L3
	movl	$0, -40(%rbp)
	jmp	.L4
.L7:
	leaq	-16(%rbp), %rax
	movq	%rax, %rdi
	call	cpu_timer_start
	movl	$0, -36(%rbp)
	jmp	.L5
.L6:
	movl	-36(%rbp), %eax
	cltq
	vmovsd	a(,%rax,8), %xmm1
	movl	-36(%rbp), %eax
	cltq
	vmovsd	b(,%rax,8), %xmm0
	vmulsd	-24(%rbp), %xmm0, %xmm0
	vaddsd	%xmm0, %xmm1, %xmm0
	movl	-36(%rbp), %eax
	cltq
	vmovsd	%xmm0, c(,%rax,8)
	addl	$1, -36(%rbp)
.L5:
	cmpl	$16777215, -36(%rbp)
	jle	.L6
	movq	-16(%rbp), %rdx
	movq	-8(%rbp), %rax
	movq	%rdx, %rdi
	movq	%rax, %rsi
	call	cpu_timer_stop
	vmovsd	-32(%rbp), %xmm1
	vaddsd	%xmm0, %xmm1, %xmm0
	vmovsd	%xmm0, -32(%rbp)
	movq	c+16(%rip), %rax
	movq	%rax, c+8(%rip)
	addl	$1, -40(%rbp)
.L4:
	cmpl	$15, -40(%rbp)
	jle	.L7
	vmovsd	-32(%rbp), %xmm0
	vmovsd	.LC4(%rip), %xmm1
	vdivsd	%xmm1, %xmm0, %xmm0
	movl	$.LC5, %edi
	movl	$1, %eax
	call	printf
	movl	$0, %eax
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE0:
	.size	main, .-main
	.section	.rodata
	.align 8
.LC4:
	.long	0
	.long	1076887552
	.ident	"GCC: (Ubuntu 4.8.4-2ubuntu1~14.04.4) 4.8.4"
	.section	.note.GNU-stack,"",@progbits
