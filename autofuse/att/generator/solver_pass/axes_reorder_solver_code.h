/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ATT_AXES_REORDER_SOLVER_CODE_H_
#define ATT_AXES_REORDER_SOLVER_CODE_H_

#include <string>
#include <cstdint>

namespace att {

// ============================================================================
// Section 1: Core Data Structures
// ============================================================================

std::string GenConstraintType();
std::string GenStructDef();
std::string GenVariable();
std::string GenTilingVariable();
std::string GenConstraint();
std::string GenReorderSolverInputDebugString();
std::string GenAxesReorderSolverInput();

// ============================================================================
// Section 2: Class Definition
// ============================================================================

std::string GenDestructor();
std::string GenAxesReorderSolverDefaultPrivateFuncDefine();
// Helper functions for GenAxesReorderSolverPrivateFuncDefine
std::string GenEqualOrderBasicFuncDeclarations(bool enable_equal_order_tiling);
std::string GenLocalBufTilingFuncDeclarations(bool enable_equal_order_tiling);
std::string GenUtilityFuncDeclarations();
std::string GenEqualPriorityAxesFuncDeclarations(bool enable_equal_order_tiling);
std::string GenDualThresholdFuncDeclarations(bool enable_equal_order_tiling);
std::string GenThreePhaseFrameworkDeclarations(bool enable_equal_order_tiling);
std::string GenProtectedFuncDeclarations();
std::string GenAxesReorderSolverPrivateFuncDefine(bool enable_equal_order_tiling = false);
std::string GenAxesReorderSolver(bool enable_equal_order_tiling = false);

// ============================================================================
// Section 3: Core Implementation
// ============================================================================

std::string GenInitAllVars();
std::string GenCommonPartOfSatisfyCons();
std::string GenSatisfyConsByRatio();
std::string GenSatisfyCons();
std::string GenTuneNoTailVar();
std::string GenCopyVars();
std::string GenWorkLoadBalance();
std::string GenApplyPromptAlign();

// ============================================================================
// Section 4: Multicore Tiling
// ============================================================================

std::string GenMutiCoreTilingCore();
std::string GenOptimizeMCVariable();
std::string GenMulticoreTiling();

// ============================================================================
// Section 5: Local Buffer Tiling
// ============================================================================

std::string GenInitLocalMCVars();
std::string GenMcRelatedNaiveTiling();
std::string GenNaiveLocalBufTilingImpl();
std::string GenNaiveLocalBufTilingWithEqualOrderImpl();
std::string GenNaiveLocalBufTiling(bool enable_equal_order_tiling = false);
std::string GenBinaryLocalBufTilingCore();
std::string GenBinaryLoadBufTilingEqualOrder();
std::string GenBinaryLocalBufTiling(bool enable_equal_order_tiling = false);
std::string GenLocalBufTiling(bool enable_equal_order_tiling = false);

// ============================================================================
// Section 6: Equal Priority Axes Support
// ============================================================================

std::string GenBinarySearchWithAlignment();
std::string GenDecreaseUntilSatisfied();
std::string GenShrinkBoundaryUntilSatisfied();
std::string GenTuneNoTail();
std::string GenIdentifyEqualPriorityAxes();
std::string GenBinarySearchEqualPriorityAxes();
std::string GenIterativeSolveEqualPriorityAxes();
std::string GenSolveEqualPriorityAxesWithDualThreshold();
// Helper functions for GenBinarySearchEqualPriorityAxesWithDualThreshold
std::string GenBinarySearchDualThresholdSignature();
std::string GenBinarySearchDualThresholdInitialization();
std::string GenBinarySearchDualThresholdLoop();
std::string GenBinarySearchDualThresholdPostProcess();
std::string GenBinarySearchEqualPriorityAxesWithDualThreshold();
std::string GenProcessNonMCAxes();
std::string GenProcessSingleMCAxis();
std::string GenProcessDualMCAxes();
std::string GenTuneDualMCAxesScenario1();
std::string GenTuneDualMCAxesScenario2();
std::string GenFinalizeEqualPriorityAxes();

// ============================================================================
// Section 7: PGO and Auto-tuning
// ============================================================================

std::string GenPgoSolverGenerateAllTilingDataHead();
std::string GenPgoSolverGenerateAllTilingDataBody();
std::string GenPgoSolverGenerateAllTilingDataTail();
std::string GenPgoSolverGenerateAllTilingData();
std::string GenGetTiling(bool enable_equal_order_tiling = false);
std::string GenGetMaxBlockDimTiling(bool enable_equal_order_tiling = false);
std::string GenFindNextUpperBlockDim();
std::string GenFindNextLowerBlockDim();
std::string GenSaveInputTilingVars();
std::string GenRestoreInputTilingVars();
// Helper functions for GenFindBetterSolutionByLowerBlockDim
std::string GenFindBetterSolutionSignature();
std::string GenFindBetterSolutionSignatureParam(bool enable_equal_order_tiling);
std::string GenFindBetterSolutionVariables();
std::string GenFindBetterSolutionLoop();
std::string GenFindBetterSolutionConditionals(bool enable_equal_order_tiling);
std::string GenFindBetterSolutionCleanup();
std::string GenFindBetterSolutionByLowerBlockDim(bool enable_equal_order_tiling = false);
std::string GenFindBetterSolutionByUpperBlockDim(bool enable_equal_order_tiling = false);
std::string GenAutoTuningInputCheck();
std::string GenAutoTuningFindUpperLowerBlockDim();
std::string GenAutoTuningFindUpperLowerBlockDimNoEqual();
std::string GenAutoTuningFindBetteSolutionEqual();
std::string GenAutoTuningFindBetteSolutionNoEqual();
std::string GenAutoTuningFindBetteSolution(const bool enable_equal_order_tiling);
std::string GenAutoTuning(bool enable_equal_order_tiling = false);

// ============================================================================
// Section 8: Utility Functions
// ============================================================================

std::string GenIsSatisfyCons();
std::string GenBinaryFindUpperBoundSatisfiedUBLimit();
std::string GenBinaryFindLowerBoundSatisfiedUBThresholdCond();
std::string GenBinaryFindLowerBoundSatisfiedCoreNum();
std::string GenBinaryFindLowerBoundSatisfiedCoreNum_Advanced();

// ============================================================================
// Section 8.1: Math Utility Functions (for equal order tiling)
// ============================================================================

std::string GenGcd();
std::string GenLcm();
std::string GenAlignToUpperBound();
std::string GenDualAxesInfo();

// ============================================================================
// Section 8.2: Binary Search Helper Functions
// ============================================================================

std::string GenBinarySearchSetup();
std::string GenBinarySearchLoop();
std::string GenBinarySearchRestore();

// ============================================================================
// Section 8.3: Three-Phase Algorithm Framework
// ============================================================================

std::string GenInitializeDualAxesInfo();
std::string GenExecuteStep1_FindUBLimit();
std::string GenExecuteStep2_FindUBThreshold();
std::string GenExecuteStep3_FindCoreNumTileSize();
std::string GenProcessDualMCAxesOrchestration();

// ============================================================================
// Section 8.4: NaiveTiling Helper Functions
// ============================================================================

std::string GenProcessMCAxisNaive();
std::string GenProcessSingleAxisNaive();

// ============================================================================
// Section 9: Main Entry Points
// ============================================================================

std::string GenWorkloadBalancePrepare();
std::string GenObjDrivenOptimize(bool enable_equal_order_tiling = false);
std::string GenAxesReorderRun(bool enable_equal_order_tiling = false);
std::string GenEmptyTensorCheck();
// Helper functions for GetAxesSolverSolverFunc
std::string GenCoreSolverFunctions();
std::string GenBinarySearchFunctions();
std::string GenEqualOrderSolverFunctions(bool enable_equal_order_tiling);
std::string GenLocalBufferTilingFunctions(bool enable_equal_order_tiling);
std::string GenMainSolverFunctions(bool enable_equal_order_tiling);
// The Reduce balance switch is aggregated across all models that share this solver definition.
std::string GetAxesSolverSolverHead(bool enable_equal_order_tiling = false);
std::string GetAxesSolverSolverFunc(bool enable_equal_order_tiling = false);
std::string GetAxesSolverPgoSolverHead(int64_t pgo_step_max);
std::string GetAxesSolverPgoSolverFunc();

// Global constants
extern const std::string AXES_SOLVER_CODE_HEAD;
extern const std::string AXES_SOLVER_CODE_FUNC;
extern const std::string AXES_SOLVER_PGO_CODE_FUNC;

}  // namespace att

#endif  // ATT_AXES_REORDER_SOLVER_CODE_H_
