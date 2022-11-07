	.text
	.amdgcn_target "amdgcn-amd-amdhsa--gfx1030"
	.protected	_Z20vector_square_kernelIfEvPT_PKS0_x ; -- Begin function _Z20vector_square_kernelIfEvPT_PKS0_x
	.globl	_Z20vector_square_kernelIfEvPT_PKS0_x
	.p2align	8
	.type	_Z20vector_square_kernelIfEvPT_PKS0_x,@function
_Z20vector_square_kernelIfEvPT_PKS0_x:  ; @_Z20vector_square_kernelIfEvPT_PKS0_x
; %bb.0:
	s_load_dword s0, s[4:5], 0x4
	s_load_dwordx2 s[2:3], s[6:7], 0x10
	v_mov_b32_e32 v1, 0
	s_waitcnt lgkmcnt(0)
	s_and_b32 s0, s0, 0xffff
	s_mul_i32 s8, s8, s0
	v_add_nc_u32_e32 v0, s8, v0
	v_cmp_gt_u64_e32 vcc_lo, s[2:3], v[0:1]
	s_and_saveexec_b32 s0, vcc_lo
	s_cbranch_execz BB0_3
; %bb.1:
	s_load_dword s8, s[4:5], 0xc
	s_load_dwordx4 s[4:7], s[6:7], 0x0
	v_lshlrev_b64 v[2:3], 2, v[0:1]
	s_mov_b32 s9, 0
	s_mov_b32 s1, s9
	s_waitcnt lgkmcnt(0)
	s_lshl_b64 s[10:11], s[8:9], 2
	.p2align	6
BB0_2:                                  ; =>This Inner Loop Header: Depth=1
	v_add_co_u32 v4, vcc_lo, s6, v2
	v_add_co_ci_u32_e32 v5, vcc_lo, s7, v3, vcc_lo
	v_add_co_u32 v0, vcc_lo, v0, s8
	v_add_co_ci_u32_e32 v1, vcc_lo, s9, v1, vcc_lo
	global_load_dword v6, v[4:5], off
	v_add_co_u32 v4, vcc_lo, s4, v2
	v_add_co_ci_u32_e32 v5, vcc_lo, s5, v3, vcc_lo
	v_cmp_le_u64_e32 vcc_lo, s[2:3], v[0:1]
	v_add_co_u32 v2, s0, v2, s10
	v_add_co_ci_u32_e64 v3, s0, s11, v3, s0
	s_or_b32 s1, vcc_lo, s1
	s_waitcnt vmcnt(0)
	v_mul_f32_e32 v6, v6, v6
	global_store_dword v[4:5], v6, off
	s_andn2_b32 exec_lo, exec_lo, s1
	s_cbranch_execnz BB0_2
BB0_3:
	s_endpgm
	.section	.rodata,#alloc
	.p2align	6
	.amdhsa_kernel _Z20vector_square_kernelIfEvPT_PKS0_x
		.amdhsa_group_segment_fixed_size 0
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 80
		.amdhsa_user_sgpr_private_segment_buffer 1
		.amdhsa_user_sgpr_dispatch_ptr 1
		.amdhsa_user_sgpr_queue_ptr 0
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_user_sgpr_dispatch_id 0
		.amdhsa_user_sgpr_flat_scratch_init 0
		.amdhsa_user_sgpr_private_segment_size 0
		.amdhsa_wavefront_size32 1
		.amdhsa_system_sgpr_private_segment_wavefront_offset 0
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_system_sgpr_workgroup_id_y 0
		.amdhsa_system_sgpr_workgroup_id_z 0
		.amdhsa_system_sgpr_workgroup_info 0
		.amdhsa_system_vgpr_workitem_id 0
		.amdhsa_next_free_vgpr 7
		.amdhsa_next_free_sgpr 12
		.amdhsa_reserve_flat_scratch 0
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
		.amdhsa_exception_fp_ieee_invalid_op 0
		.amdhsa_exception_fp_denorm_src 0
		.amdhsa_exception_fp_ieee_div_zero 0
		.amdhsa_exception_fp_ieee_overflow 0
		.amdhsa_exception_fp_ieee_underflow 0
		.amdhsa_exception_fp_ieee_inexact 0
		.amdhsa_exception_int_div_zero 0
	.end_amdhsa_kernel
	.text
.Lfunc_end0:
	.size	_Z20vector_square_kernelIfEvPT_PKS0_x, .Lfunc_end0-_Z20vector_square_kernelIfEvPT_PKS0_x
                                        ; -- End function
	.section	.AMDGPU.csdata
; Kernel info:
; codeLenInByte = 188
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
; WaveLimiterHint : 1
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 8
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 0
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 0
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 0
	.text
	.p2alignl 6, 3214868480
	.fill 48, 4, 3214868480
	.protected	_ZN17__HIP_CoordinatesI14__HIP_BlockIdxE1xE ; @_ZN17__HIP_CoordinatesI14__HIP_BlockIdxE1xE
	.type	_ZN17__HIP_CoordinatesI14__HIP_BlockIdxE1xE,@object
	.section	.rodata._ZN17__HIP_CoordinatesI14__HIP_BlockIdxE1xE,#alloc
	.weak	_ZN17__HIP_CoordinatesI14__HIP_BlockIdxE1xE
_ZN17__HIP_CoordinatesI14__HIP_BlockIdxE1xE:
	.zero	1
	.size	_ZN17__HIP_CoordinatesI14__HIP_BlockIdxE1xE, 1

	.protected	_ZN17__HIP_CoordinatesI14__HIP_BlockDimE1xE ; @_ZN17__HIP_CoordinatesI14__HIP_BlockDimE1xE
	.type	_ZN17__HIP_CoordinatesI14__HIP_BlockDimE1xE,@object
	.section	.rodata._ZN17__HIP_CoordinatesI14__HIP_BlockDimE1xE,#alloc
	.weak	_ZN17__HIP_CoordinatesI14__HIP_BlockDimE1xE
_ZN17__HIP_CoordinatesI14__HIP_BlockDimE1xE:
	.zero	1
	.size	_ZN17__HIP_CoordinatesI14__HIP_BlockDimE1xE, 1

	.protected	_ZN17__HIP_CoordinatesI15__HIP_ThreadIdxE1xE ; @_ZN17__HIP_CoordinatesI15__HIP_ThreadIdxE1xE
	.type	_ZN17__HIP_CoordinatesI15__HIP_ThreadIdxE1xE,@object
	.section	.rodata._ZN17__HIP_CoordinatesI15__HIP_ThreadIdxE1xE,#alloc
	.weak	_ZN17__HIP_CoordinatesI15__HIP_ThreadIdxE1xE
_ZN17__HIP_CoordinatesI15__HIP_ThreadIdxE1xE:
	.zero	1
	.size	_ZN17__HIP_CoordinatesI15__HIP_ThreadIdxE1xE, 1

	.protected	_ZN17__HIP_CoordinatesI13__HIP_GridDimE1xE ; @_ZN17__HIP_CoordinatesI13__HIP_GridDimE1xE
	.type	_ZN17__HIP_CoordinatesI13__HIP_GridDimE1xE,@object
	.section	.rodata._ZN17__HIP_CoordinatesI13__HIP_GridDimE1xE,#alloc
	.weak	_ZN17__HIP_CoordinatesI13__HIP_GridDimE1xE
_ZN17__HIP_CoordinatesI13__HIP_GridDimE1xE:
	.zero	1
	.size	_ZN17__HIP_CoordinatesI13__HIP_GridDimE1xE, 1

	.ident	"AMD clang version 14.0.0 (https://github.com/RadeonOpenCompute/llvm-project roc-5.0.0 22051 235b6880e2e515507478181ec11a20c1ec87945b)"
	.section	".note.GNU-stack"
	.addrsig
	.addrsig_sym _ZN17__HIP_CoordinatesI14__HIP_BlockIdxE1xE
	.addrsig_sym _ZN17__HIP_CoordinatesI14__HIP_BlockDimE1xE
	.addrsig_sym _ZN17__HIP_CoordinatesI15__HIP_ThreadIdxE1xE
	.addrsig_sym _ZN17__HIP_CoordinatesI13__HIP_GridDimE1xE
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
        .size:           8
        .value_kind:     hidden_global_offset_x
      - .offset:         32
        .size:           8
        .value_kind:     hidden_global_offset_y
      - .offset:         40
        .size:           8
        .value_kind:     hidden_global_offset_z
      - .address_space:  global
        .offset:         48
        .size:           8
        .value_kind:     hidden_none
      - .address_space:  global
        .offset:         56
        .size:           8
        .value_kind:     hidden_none
      - .address_space:  global
        .offset:         64
        .size:           8
        .value_kind:     hidden_none
      - .address_space:  global
        .offset:         72
        .size:           8
        .value_kind:     hidden_multigrid_sync_arg
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 80
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 1024
    .name:           _Z20vector_square_kernelIfEvPT_PKS0_x
    .private_segment_fixed_size: 0
    .sgpr_count:     14
    .sgpr_spill_count: 0
    .symbol:         _Z20vector_square_kernelIfEvPT_PKS0_x.kd
    .vgpr_count:     7
    .vgpr_spill_count: 0
    .wavefront_size: 32
amdhsa.target:   amdgcn-amd-amdhsa--gfx1030
amdhsa.version:
  - 1
  - 1
...

	.end_amdgpu_metadata
