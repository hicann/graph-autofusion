/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef __PYASCIR_TYPES_H__
#define __PYASCIR_TYPES_H__

#include <Python.h>
#include <structmember.h>

#include "ascendc_ir.h"
#include "graph/symbolizer/symbolic.h"
#include "schedule_result.h"

namespace pyascir {
constexpr int kPythonSuccess = 1;
constexpr int kPythonFail = 0;
constexpr int kPythonError = -1;
class SizeExpr {
 public:
  struct Object {
    PyObject_HEAD af::Expression *expression;
  };

  static PyTypeObject type;
  static PyMemberDef members[];
  static PyGetSetDef getseters[];
  static PyNumberMethods NumberMethods;

  static void Dealloc(PyObject *self);
  static PyObject *New(PyTypeObject *type, PyObject *args, PyObject *kwargs);
  static int Init(PyObject *self_pyobject, PyObject *args, PyObject *kwargs);
  static int Init(PyObject *self_pyobject, const af::Expression &expr);
  static PyObject *FromSizeExpr(const af::Expression &expr);
  static const af::Expression AsSizeExpr(PyObject *obj);
  static PyObject *Add(PyObject *self, PyObject *args);
  static PyObject *Mul(PyObject *self, PyObject *args);
  static PyObject *Div(PyObject *self, PyObject *args);
  static PyObject *Sub(PyObject *self, PyObject *args);
  static PyObject *Negate(PyObject *self);
  static PyObject *Pow(PyObject *self, PyObject *args, PyObject *modulo);
  static PyObject *Remainder(PyObject *self, PyObject *args);
  static PyObject *FloorDiv(PyObject *self, PyObject *args);
  static PyObject *Compare(PyObject *self, PyObject *other, int op);
};

class Axis {
 public:
  struct Object {
    PyObject_HEAD

        int id;
    PyObject *size;
    PyObject *name;
    PyObject *type;
  };

  static PyTypeObject type;
  static PyMemberDef members[];

  static void Dealloc(PyObject *self);
  static PyObject *New(PyTypeObject *type, PyObject *args, PyObject *kwargs);
  static int Init(PyObject *self_pyobject, int id, const af::Expression &size, const char *name, af::Axis::Type type);
};

class Operator {
 public:
  struct Object {
    PyObject_HEAD

        PyObject *name;
    PyObject *type;
    af::Operator *op;
  };

  static PyTypeObject type;
  static PyMemberDef members[];
  static PyMethodDef methods[];

  static void Dealloc(PyObject *self_pyobject);
  static PyObject *New(PyTypeObject *type, PyObject *args, PyObject *kwargs);
  static int Init(PyObject *self_pyobject, PyObject *args, PyObject *kwargs);
};

class HintGraph {
 public:
  struct Object {
    PyObject_HEAD

        PyObject *name;
    af::AscGraph *graph;
  };

  static PyTypeObject type;
  static PyMemberDef members[];
  static PyMethodDef methods[];

  static void Dealloc(PyObject *self_pyobject);
  static PyObject *New(PyTypeObject *type, PyObject *args, PyObject *kwargs);
  static int Init(PyObject *self_pyobject, PyObject *args, PyObject *kwargs);
  static PyObject *CreateSize(PyObject *self_pyobject, PyObject *args);
  static PyObject *CreateAxis(PyObject *self_pyobject, PyObject *args, PyObject *kwargs);
  static PyObject *SetAxisMap(PyObject *self_pyobject, PyObject *args);
  static PyObject *FromGraph(af::AscGraph *graph);
  static PyObject *InferDtype(PyObject *self_pyobject);
  static PyObject *GetInputNum(PyObject *self_pyobject);
  static PyObject *GetOutputNum(PyObject *self_pyobject);
  static PyObject *GetName(PyObject *self_pyobject);
  static PyObject *SetName(PyObject *self_pyobject, PyObject *args);
  static PyObject *GetSched(PyObject *self_pyobject, void *closure);
  static PyGetSetDef getsetters[];
  static bool ProcessSingleNode(const af::AscNodePtr &node);
};

class HintComputeGraph {
 public:
  struct Object {
    PyObject_HEAD

        af::ComputeGraphPtr compute_graph;
  };

  static PyTypeObject type;
  static PyMemberDef members[];
  static PyMethodDef methods[];

  static void Dealloc(PyObject *self_pyobject);
  static PyObject *New(PyTypeObject *type, PyObject *args, PyObject *kwargs);
  static int Init(PyObject *self_pyobject, PyObject *args, PyObject *kwargs);
  static PyObject *GetInfo(PyObject *self_pyobject);
  static PyObject *GetName(PyObject *self_pyobject);
};

class FusedGraph {
 public:
  struct Object {
    PyObject_HEAD af::ComputeGraphPtr graph;
  };

  static PyTypeObject type;
  static PyMemberDef members[];
  static PyMethodDef methods[];

  static void Dealloc(PyObject *self_pyobject);
  static PyObject *New(PyTypeObject *type, PyObject *args, PyObject *kwargs);
  static int Init(PyObject *self_pyobject, PyObject *args, PyObject *kwargs);
};

class ShapeInfo {
 public:
  struct Object {
    PyObject_HEAD std::map<std::string, std::string> shape_info;
  };

  static PyTypeObject type;
  static PyMemberDef members[];
  static PyMethodDef methods[];

  static void Dealloc(PyObject *self_pyobject);
  static PyObject *New(PyTypeObject *type, PyObject *args, PyObject *kwargs);
  static int Init(PyObject *self_pyobject, PyObject *args, PyObject *kwargs);
};

class FusedScheduledResult {
 public:
  struct Object {
    PyObject_HEAD ascir::FusedScheduledResult fused_schedule_result;
  };

  static PyTypeObject type;
  static PyMemberDef members[];
  static PyMethodDef methods[];

  static void Dealloc(PyObject *self_pyobject);
  static PyObject *New(PyTypeObject *type, PyObject *args, PyObject *kwargs);
  static int32_t Init(PyObject *self_pyobject, PyObject *args, PyObject *kwargs);
  static PyObject *GetInputNum(PyObject *self_pyobject);
  static PyObject *GetOutputNum(PyObject *self_pyobject);
  static PyObject *GetName(PyObject *self_pyobject);
  static PyObject *IsCubeType(PyObject *self_pyobject);
  static PyObject *GetCubeAttributes(PyObject *self_pyobject);
  static PyObject *IsConvType(PyObject *self_pyobject);
};

#define SET_DICT_LONG(dict, key, value)          \
  do {                                           \
    PyObject *tmp = PyLong_FromLong(value);      \
    PY_ASSERT_NOTNULL(tmp, key " is not ready"); \
    PyDict_SetItemString(dict, key, tmp);        \
    Py_DECREF(tmp);                              \
  } while (0)

class PyObjectGuard {
  PyObject *obj;

 public:
  explicit PyObjectGuard(PyObject *o) : obj(o) {}
  ~PyObjectGuard() {
    Py_XDECREF(obj);
  }
  PyObject *get() const {
    return obj;
  }
  PyObjectGuard(const PyObjectGuard &) = delete;
  PyObjectGuard &operator=(const PyObjectGuard &) = delete;
};
}  // namespace pyascir

#endif
