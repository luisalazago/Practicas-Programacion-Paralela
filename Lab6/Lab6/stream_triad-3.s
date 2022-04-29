	.file	"stream_triad.c"
	.section	.rodata.str1.8,"aMS",@progbits,1
	.align 8
.LC5:
	.string	"\033[1;32mAverage runtime is %lf msecs\n\033[0m"
	.section	.text.startup,"ax",@progbits
	.p2align 4,,15
	.globl	main
	.type	main, @function
main:
.LFB24:
	.cfi_startproc
	vmovapd	.LC1(%rip), %ymm1
	xorl	%eax, %eax
	vmovapd	.LC2(%rip), %ymm0
	.p2align 4,,10
	.p2align 3
.L3:
	vmovapd	%ymm1, a(%rax)
	addq	$32, %rax
	vmovapd	%ymm0, b-32(%rax)
	cmpq	$134217728, %rax
	jne	.L3
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	vxorpd	%xmm3, %xmm3, %xmm3
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	pushq	%rbx
	.cfi_offset 3, -24
	movl	$16, %ebx
	andq	$-32, %rsp
	subq	$64, %rsp
	vmovsd	%xmm3, 40(%rsp)
	vmovapd	.LC3(%rip), %ymm1
	.p2align 4,,10
	.p2align 3
.L7:
	leaq	48(%rsp), %rdi
	vmovapd	%ymm1, (%rsp)
	vzeroupper
	call	cpu_timer_start
	vmovapd	(%rsp), %ymm1
	xorl	%eax, %eax
	.p2align 4,,10
	.p2align 3
.L5:
	vmovapd	b(%rax), %ymm0
	addq	$32, %rax
	vfmadd213pd	a-32(%rax), %ymm1, %ymm0
	vmovapd	%ymm0, c-32(%rax)
	cmpq	$134217728, %rax
	jne	.L5
	movq	48(%rsp), %rdi
	movq	56(%rsp), %rsi
	vmovapd	%ymm1, (%rsp)
	vzeroupper
	call	cpu_timer_stop
	vaddsd	40(%rsp), %xmm0, %xmm2
	vmovsd	c+16(%rip), %xmm0
	subl	$1, %ebx
	vmovapd	(%rsp), %ymm1
	vmovsd	%xmm0, c+8(%rip)
	vmovsd	%xmm2, 40(%rsp)
	jne	.L7
	vmulsd	.LC4(%rip), %xmm2, %xmm0
	movl	$.LC5, %esi
	movl	$1, %edi
	movl	$1, %eax
	vzeroupper
	call	__printf_chk
	xorl	%eax, %eax
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
