	.text
	.amdgcn_target "amdgcn-amd-amdhsa--gfx1102"
	.section	.text._Z20vector_square_kernelIfEvPT_PKS0_y,#alloc,#execinstr
	.protected	_Z20vector_square_kernelIfEvPT_PKS0_y ; -- Begin function _Z20vector_square_kernelIfEvPT_PKS0_y
	.globl	_Z20vector_square_kernelIfEvPT_PKS0_y
	.p2align	8
	.type	_Z20vector_square_kernelIfEvPT_PKS0_y,@function
_Z20vector_square_kernelIfEvPT_PKS0_y:  ; @_Z20vector_square_kernelIfEvPT_PKS0_y
; %bb.0:
	s_load_b32 s4, s[0:1], 0x4
	s_load_b64 s[8:9], s[2:3], 0x10
	s_waitcnt lgkmcnt(0)
	s_and_b32 s4, s4, 0xffff
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_mad_u64_u32 v[3:4], null, s15, s4, v[0:1]
	v_mov_b32_e32 v1, v3
	v_mov_b32_e32 v2, 0
	s_mov_b32 s4, exec_lo
	v_cmpx_gt_u64_e64 s[8:9], v[1:2]
	s_cbranch_execz .LBB0_3
; %bb.1:
	s_load_b32 s10, s[0:1], 0xc
	s_load_b128 s[4:7], s[2:3], 0x0
	v_lshlrev_b64 v[3:4], 2, v[1:2]
	s_mov_b32 s11, 0
	s_waitcnt lgkmcnt(0)
	s_lshl_b64 s[2:3], s[10:11], 2
	.p2align	6
.LBB0_2:                                ; =>This Inner Loop Header: Depth=1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_u32 v5, vcc_lo, s6, v3
	v_add_co_ci_u32_e32 v6, vcc_lo, s7, v4, vcc_lo
	v_add_co_u32 v1, vcc_lo, v1, s10
	v_add_co_ci_u32_e32 v2, vcc_lo, 0, v2, vcc_lo
	global_load_b32 v0, v[5:6], off
	v_add_co_u32 v5, vcc_lo, s4, v3
	v_add_co_ci_u32_e32 v6, vcc_lo, s5, v4, vcc_lo
	v_cmp_le_u64_e32 vcc_lo, s[8:9], v[1:2]
	v_add_co_u32 v3, s0, v3, s2
	s_delay_alu instid0(VALU_DEP_1)
	v_add_co_ci_u32_e64 v4, s0, s3, v4, s0
	s_or_b32 s11, vcc_lo, s11
	s_waitcnt vmcnt(0)
	v_mul_f32_e32 v0, v0, v0
	global_store_b32 v[5:6], v0, off
	s_and_not1_b32 exec_lo, exec_lo, s11
	s_cbranch_execnz .LBB0_2
.LBB0_3:
	s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)
	s_endpgm
	.section	.rodata,#alloc
	.p2align	6, 0x0
	.amdhsa_kernel _Z20vector_square_kernelIfEvPT_PKS0_y
		.amdhsa_group_segment_fixed_size 0
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 24
		.amdhsa_user_sgpr_count 15
		.amdhsa_user_sgpr_dispatch_ptr 1
		.amdhsa_user_sgpr_queue_ptr 0
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_user_sgpr_dispatch_id 0
		.amdhsa_user_sgpr_private_segment_size 0
		.amdhsa_wavefront_size32 1
		.amdhsa_enable_private_segment 0
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_system_sgpr_workgroup_id_y 0
		.amdhsa_system_sgpr_workgroup_id_z 0
		.amdhsa_system_sgpr_workgroup_info 0
		.amdhsa_system_vgpr_workitem_id 0
		.amdhsa_next_free_vgpr 7
		.amdhsa_next_free_sgpr 16
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_float_denorm_mode_16_64 3
		.amdhsa_dx10_clamp 1
		.amdhsa_ieee_mode 1
		.amdhsa_fp16_overflow 0
		.amdhsa_workgroup_processor_mode 1
		.amdhsa_memory_ordered 1
		.amdhsa_forward_progress 0
		.amdhsa_shared_vgpr_count 0
		.amdhsa_exception_fp_ieee_invalid_op 0
		.amdhsa_exception_fp_denorm_src 0
		.amdhsa_exception_fp_ieee_div_zero 0
		.amdhsa_exception_fp_ieee_overflow 0
		.amdhsa_exception_fp_ieee_underflow 0
		.amdhsa_exception_fp_ieee_inexact 0
		.amdhsa_exception_int_div_zero 0
	.end_amdhsa_kernel
	.section	.text._Z20vector_square_kernelIfEvPT_PKS0_y,#alloc,#execinstr
.Lfunc_end0:
	.size	_Z20vector_square_kernelIfEvPT_PKS0_y, .Lfunc_end0-_Z20vector_square_kernelIfEvPT_PKS0_y
                                        ; -- End function
	.section	.AMDGPU.csdata
; Kernel info:
; codeLenInByte = 228
; NumSgprs: 18
; NumVgprs: 7
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 2
; VGPRBlocks: 0
; NumSGPRsForWavesPerEU: 18
; NumVGPRsForWavesPerEU: 7
; Occupancy: 16
; WaveLimiterHint : 0
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 15
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 0
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 0
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 0
	.text
	.p2alignl 7, 3214868480
	.fill 96, 4, 3214868480
	.ident	"AMD clang version 15.0.0 (https://github.com/RadeonOpenCompute/llvm-project roc-5.3.0 22362 3cf23f77f8208174a2ee7c616f4be23674d7b081)"
	.section	".note.GNU-stack"
	.addrsig
	.amdgpu_metadata
---
amdhsa.kernels:
  - .args:
      - .address_space:  global
        .offset:         0
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .offset:         8
        .size:           8
        .value_kind:     global_buffer
      - .offset:         16
        .size:           8
        .value_kind:     by_value
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 24
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 1024
    .name:           _Z20vector_square_kernelIfEvPT_PKS0_y
    .private_segment_fixed_size: 0
    .sgpr_count:     18
    .sgpr_spill_count: 0
    .symbol:         _Z20vector_square_kernelIfEvPT_PKS0_y.kd
    .vgpr_count:     7
    .vgpr_spill_count: 0
    .wavefront_size: 32
amdhsa.target:   amdgcn-amd-amdhsa--gfx1102
amdhsa.version:
  - 1
  - 1
...

	.end_amdgpu_metadata
