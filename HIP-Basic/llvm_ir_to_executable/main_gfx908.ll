; ModuleID = 'main_gfx908.bc'
source_filename = "./main.hip"
target datalayout = "e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7"
target triple = "amdgcn-amd-amdhsa"

%"struct.__HIP_Coordinates<__HIP_BlockIdx>::__X" = type { i8 }
%"struct.__HIP_Coordinates<__HIP_BlockDim>::__X" = type { i8 }
%"struct.__HIP_Coordinates<__HIP_ThreadIdx>::__X" = type { i8 }
%"struct.__HIP_Coordinates<__HIP_GridDim>::__X" = type { i8 }

$_ZN17__HIP_CoordinatesI14__HIP_BlockIdxE1xE = comdat any

$_ZN17__HIP_CoordinatesI14__HIP_BlockDimE1xE = comdat any

$_ZN17__HIP_CoordinatesI15__HIP_ThreadIdxE1xE = comdat any

$_ZN17__HIP_CoordinatesI13__HIP_GridDimE1xE = comdat any

@_ZN17__HIP_CoordinatesI14__HIP_BlockIdxE1xE = weak protected addrspace(4) externally_initialized constant %"struct.__HIP_Coordinates<__HIP_BlockIdx>::__X" undef, comdat, align 1
@_ZN17__HIP_CoordinatesI14__HIP_BlockDimE1xE = weak protected addrspace(4) externally_initialized constant %"struct.__HIP_Coordinates<__HIP_BlockDim>::__X" undef, comdat, align 1
@_ZN17__HIP_CoordinatesI15__HIP_ThreadIdxE1xE = weak protected addrspace(4) externally_initialized constant %"struct.__HIP_Coordinates<__HIP_ThreadIdx>::__X" undef, comdat, align 1
@_ZN17__HIP_CoordinatesI13__HIP_GridDimE1xE = weak protected addrspace(4) externally_initialized constant %"struct.__HIP_Coordinates<__HIP_GridDim>::__X" undef, comdat, align 1
@llvm.compiler.used = appending addrspace(1) global [4 x i8*] [i8* addrspacecast (i8 addrspace(4)* getelementptr inbounds (%"struct.__HIP_Coordinates<__HIP_GridDim>::__X", %"struct.__HIP_Coordinates<__HIP_GridDim>::__X" addrspace(4)* @_ZN17__HIP_CoordinatesI13__HIP_GridDimE1xE, i32 0, i32 0) to i8*), i8* addrspacecast (i8 addrspace(4)* getelementptr inbounds (%"struct.__HIP_Coordinates<__HIP_BlockDim>::__X", %"struct.__HIP_Coordinates<__HIP_BlockDim>::__X" addrspace(4)* @_ZN17__HIP_CoordinatesI14__HIP_BlockDimE1xE, i32 0, i32 0) to i8*), i8* addrspacecast (i8 addrspace(4)* getelementptr inbounds (%"struct.__HIP_Coordinates<__HIP_BlockIdx>::__X", %"struct.__HIP_Coordinates<__HIP_BlockIdx>::__X" addrspace(4)* @_ZN17__HIP_CoordinatesI14__HIP_BlockIdxE1xE, i32 0, i32 0) to i8*), i8* addrspacecast (i8 addrspace(4)* getelementptr inbounds (%"struct.__HIP_Coordinates<__HIP_ThreadIdx>::__X", %"struct.__HIP_Coordinates<__HIP_ThreadIdx>::__X" addrspace(4)* @_ZN17__HIP_CoordinatesI15__HIP_ThreadIdxE1xE, i32 0, i32 0) to i8*)], section "llvm.metadata"

; Function Attrs: mustprogress nofree norecurse nosync nounwind
define protected amdgpu_kernel void @_Z20vector_square_kernelIfEvPT_PKS0_x(float addrspace(1)* nocapture %0, float addrspace(1)* nocapture readonly %1, i64 %2) local_unnamed_addr #0 {
  %4 = tail call i32 @llvm.amdgcn.workgroup.id.x() #2
  %5 = tail call align 4 dereferenceable(64) i8 addrspace(4)* @llvm.amdgcn.dispatch.ptr() #2
  %6 = getelementptr inbounds i8, i8 addrspace(4)* %5, i64 12
  %7 = bitcast i8 addrspace(4)* %6 to i32 addrspace(4)*
  %8 = load i32, i32 addrspace(4)* %7, align 4, !tbaa !4
  %9 = getelementptr i8, i8 addrspace(4)* %5, i64 4
  %10 = bitcast i8 addrspace(4)* %9 to i16 addrspace(4)*
  %11 = load i16, i16 addrspace(4)* %10, align 4, !range !13, !invariant.load !14
  %12 = zext i16 %11 to i32
  %13 = mul i32 %4, %12
  %14 = tail call i32 @llvm.amdgcn.workitem.id.x() #2, !range !15
  %15 = add i32 %13, %14
  %16 = zext i32 %15 to i64
  %17 = zext i32 %8 to i64
  %18 = icmp ult i64 %16, %2
  br i1 %18, label %20, label %19

19:                                               ; preds = %20, %3
  ret void

20:                                               ; preds = %3, %20
  %21 = phi i64 [ %26, %20 ], [ %16, %3 ]
  %22 = getelementptr inbounds float, float addrspace(1)* %1, i64 %21
  %23 = load float, float addrspace(1)* %22, align 4, !tbaa !16
  %24 = fmul contract float %23, %23
  %25 = getelementptr inbounds float, float addrspace(1)* %0, i64 %21
  store float %24, float addrspace(1)* %25, align 4, !tbaa !16
  %26 = add i64 %21, %17
  %27 = icmp ult i64 %26, %2
  br i1 %27, label %20, label %19, !llvm.loop !20
}

; Function Attrs: nounwind readnone speculatable willreturn
declare align 4 i8 addrspace(4)* @llvm.amdgcn.dispatch.ptr() #1

; Function Attrs: nounwind readnone speculatable willreturn
declare i32 @llvm.amdgcn.workgroup.id.x() #1

; Function Attrs: nounwind readnone speculatable willreturn
declare i32 @llvm.amdgcn.workitem.id.x() #1

attributes #0 = { mustprogress nofree norecurse nosync nounwind "amdgpu-flat-work-group-size"="1,1024" "amdgpu-implicitarg-num-bytes"="56" "frame-pointer"="none" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="gfx908" "target-features"="+16-bit-insts,+ci-insts,+dl-insts,+dot1-insts,+dot2-insts,+dot3-insts,+dot4-insts,+dot5-insts,+dot6-insts,+dot7-insts,+dpp,+flat-address-space,+gfx8-insts,+gfx9-insts,+mai-insts,+s-memrealtime,+s-memtime-inst" "uniform-work-group-size"="true" }
attributes #1 = { nounwind readnone speculatable willreturn }
attributes #2 = { nounwind }

!llvm.module.flags = !{!0, !1}
!opencl.ocl.version = !{!2}
!llvm.ident = !{!3}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 7, !"PIC Level", i32 1}
!2 = !{i32 2, i32 0}
!3 = !{!"AMD clang version 14.0.0 (https://github.com/RadeonOpenCompute/llvm-project roc-5.0.0 22051 235b6880e2e515507478181ec11a20c1ec87945b)"}
!4 = !{!5, !9, i64 12}
!5 = !{!"hsa_kernel_dispatch_packet_s", !6, i64 0, !6, i64 2, !6, i64 4, !6, i64 6, !6, i64 8, !6, i64 10, !9, i64 12, !9, i64 16, !9, i64 20, !9, i64 24, !9, i64 28, !10, i64 32, !11, i64 40, !10, i64 48, !12, i64 56}
!6 = !{!"short", !7, i64 0}
!7 = !{!"omnipotent char", !8, i64 0}
!8 = !{!"Simple C/C++ TBAA"}
!9 = !{!"int", !7, i64 0}
!10 = !{!"long", !7, i64 0}
!11 = !{!"any pointer", !7, i64 0}
!12 = !{!"hsa_signal_s", !10, i64 0}
!13 = !{i16 1, i16 1025}
!14 = !{}
!15 = !{i32 0, i32 1024}
!16 = !{!17, !17, i64 0}
!17 = !{!"float", !18, i64 0}
!18 = !{!"omnipotent char", !19, i64 0}
!19 = !{!"Simple C++ TBAA"}
!20 = distinct !{!20, !21}
!21 = !{!"llvm.loop.mustprogress"}
