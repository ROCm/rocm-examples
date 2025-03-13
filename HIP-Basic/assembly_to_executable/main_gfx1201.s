	.text
	.amdgcn_target "amdgcn-amd-amdhsa--gfx1201"
	.section	.text._Z20vector_square_kernelIfEvPT_PKS0_y,"axG",@progbits,_Z20vector_square_kernelIfEvPT_PKS0_y,comdat
	.protected	_Z20vector_square_kernelIfEvPT_PKS0_y ; -- Begin function _Z20vector_square_kernelIfEvPT_PKS0_y
	.globl	_Z20vector_square_kernelIfEvPT_PKS0_y
	.p2align	8
	.type	_Z20vector_square_kernelIfEvPT_PKS0_y,@function
_Z20vector_square_kernelIfEvPT_PKS0_y:  ; @_Z20vector_square_kernelIfEvPT_PKS0_y
; %bb.0:
	s_clause 0x1
	s_load_b32 s4, s[0:1], 0x24
	s_load_b64 s[2:3], s[0:1], 0x10
	s_wait_kmcnt 0x0
	s_and_b32 s8, s4, 0xffff
	s_mov_b32 s4, exec_lo
	v_mad_co_u64_u32 v[0:1], null, ttmp9, s8, v[0:1]
	v_mov_b32_e32 v1, 0
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_u64_e64 s[2:3], v[0:1]
	s_cbranch_execz .LBB0_3
; %bb.1:
	s_add_nc_u64 s[4:5], s[0:1], 24
	v_lshlrev_b64_e32 v[2:3], 2, v[0:1]
	s_load_b32 s9, s[4:5], 0x0
	s_load_b128 s[4:7], s[0:1], 0x0
	s_wait_kmcnt 0x0
	s_mul_i32 s8, s9, s8
	s_mov_b32 s9, 0
	s_delay_alu instid0(SALU_CYCLE_1)
	s_lshl_b64 s[10:11], s[8:9], 2
.LBB0_2:                                ; =>This Inner Loop Header: Depth=1
	v_add_co_u32 v4, vcc_lo, s6, v2
	v_add_co_ci_u32_e32 v5, vcc_lo, s7, v3, vcc_lo
	v_add_co_u32 v0, vcc_lo, v0, s8
	v_add_co_ci_u32_e32 v1, vcc_lo, 0, v1, vcc_lo
	global_load_b32 v6, v[4:5], off
	v_add_co_u32 v4, vcc_lo, s4, v2
	v_add_co_ci_u32_e32 v5, vcc_lo, s5, v3, vcc_lo
	v_cmp_le_u64_e32 vcc_lo, s[2:3], v[0:1]
	v_add_co_u32 v2, s0, v2, s10
	s_delay_alu instid0(VALU_DEP_1)
	v_add_co_ci_u32_e64 v3, s0, s11, v3, s0
	s_or_b32 s9, vcc_lo, s9
	s_wait_loadcnt 0x0
	v_mul_f32_e32 v6, v6, v6
	global_store_b32 v[4:5], v6, off
	s_and_not1_b32 exec_lo, exec_lo, s9
	s_cbranch_execnz .LBB0_2
.LBB0_3:
	s_nop 0
	s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel _Z20vector_square_kernelIfEvPT_PKS0_y
		.amdhsa_group_segment_fixed_size 0
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 280
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_dispatch_ptr 0
		.amdhsa_user_sgpr_queue_ptr 0
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_user_sgpr_dispatch_id 0
		.amdhsa_user_sgpr_private_segment_size 0
		.amdhsa_wavefront_size32 1
		.amdhsa_uses_dynamic_stack 0
		.amdhsa_enable_private_segment 0
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_system_sgpr_workgroup_id_y 0
		.amdhsa_system_sgpr_workgroup_id_z 0
		.amdhsa_system_sgpr_workgroup_info 0
		.amdhsa_system_vgpr_workitem_id 0
		.amdhsa_next_free_vgpr 7
		.amdhsa_next_free_sgpr 12
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_float_denorm_mode_16_64 3
		.amdhsa_fp16_overflow 0
		.amdhsa_workgroup_processor_mode 1
		.amdhsa_memory_ordered 1
		.amdhsa_forward_progress 0
		.amdhsa_round_robin_scheduling 0
		.amdhsa_exception_fp_ieee_invalid_op 0
		.amdhsa_exception_fp_denorm_src 0
		.amdhsa_exception_fp_ieee_div_zero 0
		.amdhsa_exception_fp_ieee_overflow 0
		.amdhsa_exception_fp_ieee_underflow 0
		.amdhsa_exception_fp_ieee_inexact 0
		.amdhsa_exception_int_div_zero 0
	.end_amdhsa_kernel
	.section	.text._Z20vector_square_kernelIfEvPT_PKS0_y,"axG",@progbits,_Z20vector_square_kernelIfEvPT_PKS0_y,comdat
.Lfunc_end0:
	.size	_Z20vector_square_kernelIfEvPT_PKS0_y, .Lfunc_end0-_Z20vector_square_kernelIfEvPT_PKS0_y
                                        ; -- End function
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 224
; NumSgprs: 14
; NumVgprs: 7
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 1
; VGPRBlocks: 0
; NumSGPRsForWavesPerEU: 14
; NumVGPRsForWavesPerEU: 7
; Occupancy: 16
; WaveLimiterHint : 0
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 2
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 0
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 0
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 0
	.text
	.p2alignl 7, 3214868480
	.fill 96, 4, 3214868480
	.type	__hip_cuid_f4fad75352be2d4a,@object ; @__hip_cuid_f4fad75352be2d4a
	.section	.bss,"aw",@nobits
	.globl	__hip_cuid_f4fad75352be2d4a
__hip_cuid_f4fad75352be2d4a:
	.byte	0                               ; 0x0
	.size	__hip_cuid_f4fad75352be2d4a, 1

	.ident	"AMD clang version 18.0.0git (https://github.com/RadeonOpenCompute/llvm-project roc-6.3.3 25012 e5bf7e55c91490b07c49d8960fa7983d864936c4)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.addrsig_sym __hip_cuid_f4fad75352be2d4a
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
      - .offset:         24
        .size:           4
        .value_kind:     hidden_block_count_x
      - .offset:         28
        .size:           4
        .value_kind:     hidden_block_count_y
      - .offset:         32
        .size:           4
        .value_kind:     hidden_block_count_z
      - .offset:         36
        .size:           2
        .value_kind:     hidden_group_size_x
      - .offset:         38
        .size:           2
        .value_kind:     hidden_group_size_y
      - .offset:         40
        .size:           2
        .value_kind:     hidden_group_size_z
      - .offset:         42
        .size:           2
        .value_kind:     hidden_remainder_x
      - .offset:         44
        .size:           2
        .value_kind:     hidden_remainder_y
      - .offset:         46
        .size:           2
        .value_kind:     hidden_remainder_z
      - .offset:         64
        .size:           8
        .value_kind:     hidden_global_offset_x
      - .offset:         72
        .size:           8
        .value_kind:     hidden_global_offset_y
      - .offset:         80
        .size:           8
        .value_kind:     hidden_global_offset_z
      - .offset:         88
        .size:           2
        .value_kind:     hidden_grid_dims
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 280
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 1024
    .name:           _Z20vector_square_kernelIfEvPT_PKS0_y
    .private_segment_fixed_size: 0
    .sgpr_count:     14
    .sgpr_spill_count: 0
    .symbol:         _Z20vector_square_kernelIfEvPT_PKS0_y.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     7
    .vgpr_spill_count: 0
    .wavefront_size: 32
    .workgroup_processor_mode: 1
amdhsa.target:   amdgcn-amd-amdhsa--gfx1201
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata
