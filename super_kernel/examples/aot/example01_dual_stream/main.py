#!/usr/bin/python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""
双流 + Super Kernel + NPU Event 手动控制边样例

对比 super_kernel_optimize 开启(SK) 与关闭(Non-SK) 场景下的输出一致性，
对比 dq_res_2 结果。

特性:
1. Stream 1 中间算子完成后记录 NPU event
2. Stream 2 等待 NPU event(建立控制边，而非数据依赖)
3. 纯 aclnn 算子调用
运行方式: 通过仓库根目录 build.sh 运行
"""

import torch
import torch.nn as nn
import torch_npu
import numpy as np
import os
import sys


# ============================================================================
class DualStreamModel(nn.Module):
    """
    双流模型: 使用 NPU event 手动控制边

    Stream 1 (主流):
        quant_matmul → grouped_matmul → [record event]
        → swiglu_quant → dynamic_quant

    Stream 2 (从流):
        [wait event] → quant_matmul → add_rms_norm
        → dynamic_quant → [record event]
    """

    def forward(
        self,
        x1_1,
        x2_1,
        scale_1,
        offset_1,
        bias_1,
        pertoken_scale_1,
        smooth_scales_1,
        x1_2,
        x2_2,
        scale_2,
        offset_2,
        bias_2,
        pertoken_scale_2,
        arn2_x2,
        arn2_gamma,
        smooth_scales_2,
        swiglu_quant_scale_1,
        gmm1_weight_1,
        gmm1_bias_1,
        stream1,
        stream2,
        event1,
        event2,
    ):
        # ========== Stream 1 (主流) ==========
        event1.record()
        with torch.npu.stream(stream1):
            # torch.npu.super_kernel_scope_begin("sk0")
            event1.wait(stream1)

            qm_out_1 = torch_npu.npu_quant_matmul(
                x1_1,
                x2_1,
                scale_1,
                offset=offset_1,
                bias=bias_1,
                pertoken_scale=pertoken_scale_1,
                output_dtype=torch.bfloat16,
            )
            gmm_out_1 = torch_npu.npu_grouped_matmul(
                x=[qm_out_1],
                weight=gmm1_weight_1,
                bias=gmm1_bias_1,
                split_item=0,
                group_list=None,
                group_type=-1,
                output_dtype=torch.bfloat16,
            )
            # 通知 stream2: stream1 中间算子已完成
            event1.record()

            swiglu_res_1 = torch_npu.npu_dequant_swiglu_quant(
                gmm_out_1[0], quant_scale=swiglu_quant_scale_1, quant_mode=1
            )
            event2.record()

            dq_res_1 = torch_npu.npu_dynamic_quant(
                gmm_out_1[0], smooth_scales=smooth_scales_1
            )
            # torch.npu.super_kernel_scope_end("sk0")

        # ========== Stream 2 (从流) - 等待 stream1 事件 ==========
        with torch.npu.stream(stream2):
            # torch.npu.super_kernel_scope_begin("sk1")
            event1.wait(stream2)

            qm_out_2 = torch_npu.npu_quant_matmul(
                x1_2,
                x2_2,
                scale_2,
                offset=offset_2,
                bias=bias_2,
                pertoken_scale=pertoken_scale_2,
                output_dtype=torch.bfloat16,
            )
            arn2_out, _, _ = torch_npu.npu_add_rms_norm(
                qm_out_2, arn2_x2, arn2_gamma, epsilon=1e-5
            )
            event2.wait(stream2)

            dq_res_2 = torch_npu.npu_dynamic_quant(
                arn2_out, smooth_scales=smooth_scales_2
            )
            # torch.npu.super_kernel_scope_end("sk1")

        return dq_res_1, dq_res_2, swiglu_res_1


# ============================================================================
# 数据准备
# ============================================================================


def prepare_data(seed=1236):
    """准备双流模型所需的所有输入数据"""
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Stream 1 参数
    m1, k1, n1 = 864, 7168, 4096
    x1_1 = torch.randint(-10, 10, (m1, k1), dtype=torch.int8)
    x2_1 = torch_npu.npu_format_cast(
        torch.randint(-10, 10, (n1, k1), dtype=torch.int8)
        .npu()
        .transpose(1, 0)
        .contiguous(),
        29,
    )
    scale_1 = torch.randn((n1,), dtype=torch.float32)
    pertoken_scale_1 = torch.randn((m1,), dtype=torch.float32)
    smooth_scales_1 = torch.randn((200,), dtype=torch.bfloat16)
    swiglu_quant_scale_1 = torch.randn((1, 100), dtype=torch.float32)
    gmm1_weight_1 = [torch.rand(n1, 200, dtype=torch.bfloat16).npu()]
    gmm1_bias_1 = [torch.rand(200, dtype=torch.float32).npu()]

    # Stream 2 参数
    m2, k2, n2 = 864, 7168, 4096
    x1_2 = torch.randint(-10, 10, (m2, k2), dtype=torch.int8)
    x2_2 = torch_npu.npu_format_cast(
        torch.randint(-10, 10, (n2, k2), dtype=torch.int8)
        .npu()
        .transpose(1, 0)
        .contiguous(),
        29,
    )
    scale_2 = torch.randn((n2,), dtype=torch.float32)
    pertoken_scale_2 = torch.randn((m2,), dtype=torch.float32)
    arn2_x2 = torch.rand(m2, n2, dtype=torch.bfloat16).npu()
    arn2_gamma = torch.rand(n2, dtype=torch.bfloat16).npu()
    smooth_scales_2 = torch.randn((n2,), dtype=torch.bfloat16).npu()

    # NPU streams/events
    stream1 = torch.npu.Stream()
    stream2 = torch.npu.Stream()
    event1 = torch.npu.Event()
    event2 = torch.npu.Event()

    return dict(
        x1_1=x1_1.npu(),
        x2_1=x2_1,
        scale_1=scale_1.npu(),
        offset_1=None,
        bias_1=None,
        pertoken_scale_1=pertoken_scale_1.npu(),
        smooth_scales_1=smooth_scales_1.npu(),
        swiglu_quant_scale_1=swiglu_quant_scale_1.npu(),
        x1_2=x1_2.npu(),
        x2_2=x2_2,
        scale_2=scale_2.npu(),
        offset_2=None,
        bias_2=None,
        pertoken_scale_2=pertoken_scale_2.npu(),
        arn2_x2=arn2_x2,
        arn2_gamma=arn2_gamma,
        smooth_scales_2=smooth_scales_2,
        gmm1_weight_1=gmm1_weight_1,
        gmm1_bias_1=gmm1_bias_1,
        stream1=stream1,
        stream2=stream2,
        event1=event1,
        event2=event2,
    )


# ============================================================================
# 编译 & 运行
# ============================================================================


def build_compile_options(enable_sk=True):
    """构建 npugraph_ex 编译选项, enable_sk 控制 super kernel 开关"""
    options = {}
    if enable_sk:
        options.update(
            {
                "static_kernel_compile": True,
                "super_kernel_optimize": True,
                "super_kernel_optimize_options": {"auto_op_parallel": 1},
            }
        )
    return options


def run_model(enable_sk, data):
    """运行模型并返回 dq_res_2, 异常时打印错误并退出"""
    tag = "SK" if enable_sk else "Non-SK"
    print(f"\n{'=' * 60}")
    print(f"  运行 {tag} 版本 (super_kernel_optimize={enable_sk})")
    print(f"{'=' * 60}")

    try:
        model = torch.compile(
            DualStreamModel(),
            backend="npugraph_ex",
            options=build_compile_options(enable_sk),
            dynamic=False,
        )
        _, dq_res_2, _ = model(**data)
        torch_npu.npu.synchronize()
    except Exception as e:
        print(f"[ERROR] {tag} 版本运行失败: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        sys.exit(2)
    print(f"{tag} 版本运行完成")
    return dq_res_2


# ============================================================================
# 精度对比
# ============================================================================


def compare_results(dq_res_2_non_sk, dq_res_2_sk, atol=1e-3, rtol=1e-3):
    """对比 Non-SK 与 SK 版本的 dq_res_2 输出"""
    print(f"\n{'=' * 60}")
    print("  精度对比 (dq_res_2)")
    print(f"{'=' * 60}")

    if len(dq_res_2_non_sk) != len(dq_res_2_sk):
        print(f"  输出数量不一致: Non-SK={len(dq_res_2_non_sk)}, SK={len(dq_res_2_sk)}")
        return False

    all_outputs_close = True
    for output_idx, (non_sk, sk) in enumerate(zip(dq_res_2_non_sk, dq_res_2_sk)):
        non_sk_np = non_sk.cpu().float().numpy()
        sk_np = sk.cpu().float().numpy()
        abs_diff = np.abs(non_sk_np - sk_np)
        max_diff = np.max(abs_diff)
        mean_diff = np.mean(abs_diff)
        output_close = np.allclose(non_sk_np, sk_np, atol=atol, rtol=rtol)
        all_outputs_close = all_outputs_close and output_close

        print(f"  output[{output_idx}] 最大绝对误差: {max_diff:.6e}")
        print(f"  output[{output_idx}] 平均绝对误差: {mean_diff:.6e}")
        print(
            f"  output[{output_idx}] allclose(atol={atol}, rtol={rtol}): "
            f"{'PASS' if output_close else 'FAIL'}"
        )

    return all_outputs_close


# ============================================================================
# 主入口
# ============================================================================

if __name__ == "__main__":
    # 设备初始化
    device_id = os.getenv("NPU_DEVICE_ID") or os.getenv("ASCEND_DEVICE_ID") or "0"
    torch_npu.npu.set_device(f"npu:{device_id}")
    torch_npu.npu.set_op_timeout_ms(10000)

    print("=" * 60)
    print("  双流 + Super Kernel + NPU Event 对比样例")
    print("=" * 60)
    print("Stream 1: quant_matmul → grouped_matmul → [record]")
    print("          → swiglu_quant → dynamic_quant")
    print("Stream 2: [wait] → quant_matmul → add_rms_norm")
    print("          → dynamic_quant → [record]")
    print("对比项:   dq_res_2 (SK vs Non-SK)")
    print("=" * 60)

    try:
        # 准备数据
        data = prepare_data()

        # 运行 SK 版本
        dq_res_2_sk = run_model(enable_sk=True, data=data)

        # 准备数据
        data = prepare_data()

        # 运行 Non-SK 基线版本
        dq_res_2_non_sk = run_model(enable_sk=False, data=data)

        # 精度对比
        passed = compare_results(dq_res_2_non_sk, dq_res_2_sk)
    except Exception as e:
        print(f"\n[ERROR] 测试异常退出: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        sys.exit(2)

    if passed:
        print("\n测试通过: SK 与 Non-SK 输出一致!")
        sys.exit(0)

    print("\n测试失败: SK 与 Non-SK 输出存在差异!")
    sys.exit(1)
