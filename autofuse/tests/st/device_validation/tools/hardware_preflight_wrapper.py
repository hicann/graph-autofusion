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
"""Validate hardware-only prerequisites before invoking argparse or the runner."""

import logging
import os
import re
import subprocess
import sys
import ctypes
import ctypes.util
from pathlib import Path


def _device_count():
    candidates = []
    ascend_home = os.environ.get("ASCEND_HOME_PATH")
    if ascend_home:
        candidates.append(Path(ascend_home) / "lib64" / "libascendcl.so")
    library = ctypes.util.find_library("ascendcl")
    if library:
        candidates.append(Path(library))
    candidates.append(Path("libascendcl.so"))
    try:
        acl = next(ctypes.CDLL(str(path)) for path in candidates)
        acl.aclInit.argtypes = [ctypes.c_char_p]
        acl.aclInit.restype = ctypes.c_uint32
        if acl.aclInit(None) != 0:
            return None
        try:
            acl.aclFinalize.argtypes = []
            acl.aclFinalize.restype = ctypes.c_uint32
            acl.aclrtGetDeviceCount.argtypes = [ctypes.POINTER(ctypes.c_uint32)]
            acl.aclrtGetDeviceCount.restype = ctypes.c_uint32
            count = ctypes.c_uint32()
            status = acl.aclrtGetDeviceCount(ctypes.byref(count))
        finally:
            finalize_status = acl.aclFinalize()
        return int(count.value) if status == 0 and finalize_status == 0 else None
    except (OSError, AttributeError):
        return None


def main():
    device_id = os.environ.get("DEVICE_VALIDATION_DEVICE_ID")
    if device_id is None or not re.fullmatch(r"[0-9]+", device_id):
        logging.error(
            "hardware_precondition_missing: DEVICE_VALIDATION_DEVICE_ID must be a non-negative integer"
        )
        return 3
    count = _device_count()
    if count is None:
        logging.error(
            "hardware_precondition_missing: unable to query CANN device count"
        )
        return 3
    if int(device_id) >= count:
        logging.error(
            f"hardware_precondition_invalid: device id {device_id} is outside device count {count}"
        )
        return 3
    return subprocess.call(
        [
            sys.executable,
            "-m",
            "device_validation.tools.run_device_validation",
            *sys.argv[1:],
            "--device",
            device_id,
        ],
        env=os.environ.copy(),
    )


if __name__ == "__main__":
    raise SystemExit(main())
