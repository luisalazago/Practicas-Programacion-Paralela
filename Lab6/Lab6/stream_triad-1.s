	.file	"stream_triad.c"
	.section	.rodata.str1.8,"aMS",@progbits,1
	.align 8
.LC5:
	.string	"\033[1;32mAverage runtime is %lf msecs\n\033[0m"
	.text
	.globl	main
	.type	main, @function
main:
.LFB24:
	.cfi_startproc
	movl	$0, %eax
	vmovapd	.LC1(%rip), %ymm1
	vmovapd	.LC2(%rip), %ymm0
.L3:
	vmovapd	%ymm1, a(%rax)
	vmovapd	%ymm0, b(%rax)
	addq	$32, %rax
	cmpq	$134217728, %rax
	jne	.L3
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	pushq	%rbx
	andq	$-32, %rsp
	subq	$32, %rsp
	.cfi_offset 3, -24
	movl	$16, %ebx
	vxorpd	%xmm4, %xmm4, %xmm4
	vmovsd	%xmm4, 8(%rsp)
.L7:
	leaq	16(%rsp), %rdi
	call	cpu_timer_start
	movl	$0, %eax
.L5:
	vmovapd	.LC3(%rip), %ymm2
	vmulpd	b(%rax), %ymm2, %ymm0
	vaddpd	a(%rax), %ymm0, %ymm0
	vmovapd	%ymm0, c(%rax)
	addq	$32, %rax
	cmpq	$134217728, %rax
	jne	.L5
	movq	16(%rsp), %rdi
	movq	24(%rsp), %rsi
	call	cpu_timer_stop
	vaddsd	8(%rsp), %xmm0, %xmm3
	vmovsd	%xmm3, 8(%rsp)
	vmovsd	c+16(%rip), %xmm0
	vmovsd	%xmm0, c+8(%rip)
	subl	$1, %ebx
	jne	.L7
	vmulsd	.LC4(%rip), %xmm3, %xmm0
	movl	$.LC5, %esi
	movl	$1, %edi
	movl	$1, %eax
	call	__printf_chk
	movl	$0, %eax
	movq	-8(%rbp), %rbx
	leave
	.cfi_restore 6
	.cfi_restore 3
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE24:
	.size	main, .-main
	.local	c
	.comm	c,134217728,32
	.local	b
	.comm	b,134217728,32
	.local	a
	.comm	a,134217728,32
	.section	.rodata.cst32,"aM",@progbits,32
	.align 32
.LC1:
	.long	0
	.long	1072693248
	.long	0
	.long	1072693248
	.long	0
	.long	1072693248
	.long	0
	.long	1072693248
	.align 32
.LC2:
	.long	0
	.long	1073741824
	.long	0
	.long	1073741824
	.long	0
	.long	1073741824
	.long	0
	.long	1073741824
	.align 32
.LC3:
	.long	0
	.long	1074266112
	.long	0
	.long	1074266112
	.long	0
	.long	1074266112
	.long	0
	.long	1074266112
	.section	.rodata.cst8,"aM",@progbits,8
	.align 8
.LC4:
	.long	0
	.long	1068498944
	.ident	"GCC: (Ubuntu 4.8.4-2ubuntu1~14.04.4) 4.8.4"
	.section	.note.GNU-stack,"",@progbits
