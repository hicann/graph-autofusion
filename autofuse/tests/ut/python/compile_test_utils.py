# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import importlib.util
import os
import sys
import types
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[4]
PYTHON_DIR = os.fspath(BASE_DIR / "autofuse/compiler/python")


@dataclass
class LoadedCompileModule:
    module: object
    duration_records: list = field(default_factory=list)
    duration_reports: list = field(default_factory=list)

    def __getattr__(self, name):
        return getattr(self.module, name)


@contextmanager
def load_compile_module(
    module_name, module_path, extra_autofuse_attrs=None, extra_modules=None
):
    duration_records = []
    duration_reports = []
    original_modules = {}
    extra_modules = extra_modules or {}

    class FakeUtils:
        @staticmethod
        def duration_record(labels, start, duration):
            duration_records.append((labels, start, duration))

        @staticmethod
        def report_durations():
            duration_reports.append(True)

    pyautofuse_module = types.ModuleType("autofuse.pyautofuse")
    pyautofuse_module.ascir = types.SimpleNamespace(utils=FakeUtils)
    autofuse_module = types.ModuleType("autofuse")
    autofuse_module.__path__ = [PYTHON_DIR]
    autofuse_module.pyautofuse = pyautofuse_module
    if extra_autofuse_attrs:
        for name, value in extra_autofuse_attrs.items():
            setattr(autofuse_module, name, value)

    modules_to_patch = {
        "autofuse": autofuse_module,
        "autofuse.pyautofuse": pyautofuse_module,
        **extra_modules,
    }
    for name, value in modules_to_patch.items():
        original_modules[name] = sys.modules.get(name)
        sys.modules[name] = value

    try:
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        module = importlib.util.module_from_spec(spec)
        if spec is not None and spec.loader is not None:
            spec.loader.exec_module(module)
        yield LoadedCompileModule(module, duration_records, duration_reports)
    finally:
        for name, original_module in original_modules.items():
            if original_module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = original_module
