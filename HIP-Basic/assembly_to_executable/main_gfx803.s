	.text
	.amdgcn_target "amdgcn-amd-amdhsa--gfx803"
	.section	.text._Z20vector_square_kernelIfEvPT_PKS0_y,#alloc,#execinstr
	.protected	_Z20vector_square_kernelIfEvPT_PKS0_y ; -- Begin function _Z20vector_square_kernelIfEvPT_PKS0_y
	.globl	_Z20vector_square_kernelIfEvPT_PKS0_y
	.p2align	8
	.type	_Z20vector_square_kernelIfEvPT_PKS0_y,@function
_Z20vector_square_kernelIfEvPT_PKS0_y:  ; @_Z20vector_square_kernelIfEvPT_PKS0_y
; %bb.0:
	s_load_dword s0, s[4:5], 0x4
	s_load_dwordx2 s[10:11], s[6:7], 0x10
	v_mov_b32_e32 v1, 0
	s_waitcnt lgkmcnt(0)
	s_and_b32 s0, s0, 0xffff
	s_mul_i32 s8, s8, s0
	v_add_u32_e32 v0, vcc, s8, v0
	v_cmp_gt_u64_e32 vcc, s[10:11], v[0:1]
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_3
; %bb.1:
	s_load_dword s8, s[4:5], 0xc
	s_load_dwordx4 s[4:7], s[6:7], 0x0
	s_mov_b32 s9, 0
	v_lshlrev_b64 v[2:3], 2, v[0:1]
	s_mov_b64 s[14:15], 0
	s_waitcnt lgkmcnt(0)
	s_lshl_b64 s[12:13], s[8:9], 2
.LBB0_2:                                ; =>This Inner Loop Header: Depth=1
	v_mov_b32_e32 v5, s7
	v_add_u32_e32 v4, vcc, s6, v2
	v_addc_u32_e32 v5, vcc, v5, v3, vcc
	flat_load_dword v6, v[4:5]
	v_mov_b32_e32 v5, s5
	v_mov_b32_e32 v7, s9
	v_add_u32_e32 v0, vcc, s8, v0
	v_mov_b32_e32 v8, s13
	v_add_u32_e64 v4, s[0:1], s4, v2
	v_add_u32_e64 v2, s[2:3], s12, v2
	v_addc_u32_e64 v5, s[0:1], v5, v3, s[0:1]
	v_addc_u32_e32 v1, vcc, v1, v7, vcc
	v_addc_u32_e64 v3, vcc, v3, v8, s[2:3]
	v_cmp_le_u64_e32 vcc, s[10:11], v[0:1]
	s_or_b64 s[14:15], vcc, s[14:15]
	s_waitcnt vmcnt(0)
	v_mul_f32_e32 v6, v6, v6
	flat_store_dword v[4:5], v6
	s_andn2_b64 exec, exec, s[14:15]
	s_cbranch_execnz .LBB0_2
.LBB0_3:
	s_endpgm
	.section	.rodata,#alloc
	.p2align	6
	.amdhsa_kernel _Z20vector_square_kernelIfEvPT_PKS0_y
		.amdhsa_group_segment_fixed_size 0
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 24
		.amdhsa_user_sgpr_count 8
		.amdhsa_user_sgpr_private_segment_buffer 1
		.amdhsa_user_sgpr_dispatch_ptr 1
		.amdhsa_user_sgpr_queue_ptr 0
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_user_sgpr_dispatch_id 0
		.amdhsa_user_sgpr_flat_scratch_init 0
		.amdhsa_user_sgpr_private_segment_size 0
		.amdhsa_system_sgpr_private_segment_wavefront_offset 0
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_system_sgpr_workgroup_id_y 0
		.amdhsa_system_sgpr_workgroup_id_z 0
		.amdhsa_system_sgpr_workgroup_info 0
		.amdhsa_system_vgpr_workitem_id 0
		.amdhsa_next_free_vgpr 9
		.amdhsa_next_free_sgpr 16
		.amdhsa_reserve_flat_scratch 0
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 0
		.amdhsa_float_denorm_mode_16_64 3
		.amdhsa_dx10_clamp 1
		.amdhsa_ieee_mode 1
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
; codeLenInByte = 216
; NumSgprs: 18
; NumVgprs: 9
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 192
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 2
; VGPRBlocks: 2
; NumSGPRsForWavesPerEU: 18
; NumVGPRsForWavesPerEU: 9
; Occupancy: 10
; WaveLimiterHint : 1
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 8
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 0
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 0
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 0
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
    .vgpr_count:     9
    .vgpr_spill_count: 0
    .wavefront_size: 64
amdhsa.target:   amdgcn-amd-amdhsa--gfx803
amdhsa.version:
  - 1
  - 1
...

	.end_amdgpu_metadata
