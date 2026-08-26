# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
"""Preflight checks for the real codegen CTest entry point."""

import logging
import os
from pathlib import Path


def main():
    jit = os.environ.get("AUTOFUSE_DEVICE_JIT")
    if not jit:
        logging.error("AUTOFUSE_DEVICE_JIT is required")
        return 1
    path = Path(jit)
    if not path.is_file() or not os.access(path, os.X_OK):
        logging.error("AUTOFUSE_DEVICE_JIT must be a regular executable file")
        return 1
    try:
        import autofuse.pyautofuse  # noqa: F401
    except ImportError as error:
        logging.error("autofuse.pyautofuse is unavailable: %s", error)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
