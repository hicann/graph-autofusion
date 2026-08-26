/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "inprocess_profiler.h"

#include <gtest/gtest.h>

#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace device_validation {
namespace {
struct Call {
  std::string name;
  int32_t device_id = -1;
  std::string output_dir;
};

struct FakeState {
  std::vector<Call> calls;
  bool destroyed = false;
};

class FakeProfiler final : public DeviceProfiler {
 public:
  explicit FakeProfiler(std::shared_ptr<FakeState> state) : state_(std::move(state)) {}

  ~FakeProfiler() override {
    state_->destroyed = true;
  }

  Status Start(int32_t device_id, const std::string &output_dir) override {
    state_->calls.push_back({"start", device_id, output_dir});
    if (start_status != Status::kOk) {
      last_error_ = start_error;
      return start_status;
    }
    last_error_.clear();
    return Status::kOk;
  }

  Status Stop() override {
    state_->calls.push_back({"stop", -1, ""});
    if (stop_status != Status::kOk) {
      last_error_ = stop_error;
      return stop_status;
    }
    last_error_.clear();
    return Status::kOk;
  }

  const std::string &last_error() const override {
    return last_error_;
  }

  Status start_status = Status::kOk;
  Status stop_status = Status::kOk;
  std::string start_error;
  std::string stop_error;

 private:
  std::string last_error_;
  std::shared_ptr<FakeState> state_;
};
}  // namespace

class InProcessProfilerTest : public ::testing::Test {
 protected:
  InProcessProfilerTest() {
    auto impl = std::make_unique<FakeProfiler>(impl_state_);
    impl_ = impl.get();
    profiler_ = std::make_unique<InProcessProfiler>(std::move(impl));
  }

  std::shared_ptr<FakeState> impl_state_ = std::make_shared<FakeState>();
  FakeProfiler *impl_ = nullptr;
  std::unique_ptr<InProcessProfiler> profiler_;
};

TEST_F(InProcessProfilerTest, StartForwardsDeviceAndDirectoryAndMarksStarted) {
  ASSERT_EQ(profiler_->Start(3, "/tmp/collect"), Status::kOk);
  ASSERT_TRUE(profiler_->started());
  ASSERT_EQ(impl_state_->calls.size(), 1U);
  ASSERT_EQ(impl_state_->calls[0].name, "start");
  ASSERT_EQ(impl_state_->calls[0].device_id, 3);
  ASSERT_EQ(impl_state_->calls[0].output_dir, "/tmp/collect");
}

TEST_F(InProcessProfilerTest, StartFailureNeverMarksStartedAndSurfacesError) {
  impl_->start_status = Status::kRuntimeError;
  impl_->start_error = "aclprofInit failed";
  ASSERT_EQ(profiler_->Start(0, "/tmp/collect"), Status::kRuntimeError);
  ASSERT_FALSE(profiler_->started());
  ASSERT_EQ(profiler_->last_error(), "aclprofInit failed");
  ASSERT_EQ(impl_state_->calls.size(), 1U);
}

TEST_F(InProcessProfilerTest, StartFailureUsesDefaultErrorWhenImplIsSilent) {
  impl_->start_status = Status::kRuntimeError;
  ASSERT_EQ(profiler_->Start(0, "/tmp/collect"), Status::kRuntimeError);
  ASSERT_EQ(profiler_->last_error(), "in-process profiler start failed");
}

TEST_F(InProcessProfilerTest, DoubleStartIsRejectedWithoutCallingImpl) {
  ASSERT_EQ(profiler_->Start(0, "/tmp/collect"), Status::kOk);
  ASSERT_EQ(profiler_->Start(0, "/tmp/collect"), Status::kInvalidArgument);
  ASSERT_EQ(impl_state_->calls.size(), 1U);
  ASSERT_EQ(profiler_->last_error(), "in-process profiler is already started");
}

TEST_F(InProcessProfilerTest, StopBeforeStartIsIdempotentNoOp) {
  ASSERT_EQ(profiler_->Stop(), Status::kOk);
  ASSERT_EQ(impl_state_->calls.size(), 0U);
}

TEST_F(InProcessProfilerTest, StopCallsImplOnceAndStopsDoubleStop) {
  ASSERT_EQ(profiler_->Start(0, "/tmp/collect"), Status::kOk);
  ASSERT_EQ(profiler_->Stop(), Status::kOk);
  ASSERT_EQ(profiler_->Stop(), Status::kOk);
  ASSERT_FALSE(profiler_->started());
  ASSERT_EQ(impl_state_->calls.size(), 2U);
  ASSERT_EQ(impl_state_->calls[1].name, "stop");
}

TEST_F(InProcessProfilerTest, StopFailureIsSurfacedAndDoesNotMarkStarted) {
  ASSERT_EQ(profiler_->Start(0, "/tmp/collect"), Status::kOk);
  impl_->stop_status = Status::kRuntimeError;
  impl_->stop_error = "aclprofStop failed";
  ASSERT_EQ(profiler_->Stop(), Status::kRuntimeError);
  ASSERT_FALSE(profiler_->started());
  ASSERT_EQ(profiler_->last_error(), "aclprofStop failed");
}

TEST_F(InProcessProfilerTest, NullImplIsReportedAsUnavailable) {
  InProcessProfiler profiler;
  ASSERT_EQ(profiler.Start(0, "/tmp/collect"), Status::kRuntimeError);
  ASSERT_EQ(profiler.last_error(), "in-process profiler is unavailable");
  ASSERT_EQ(profiler.Stop(), Status::kRuntimeError);
}

TEST_F(InProcessProfilerTest, DestructorStopsAnUnstoppedProfiler) {
  ASSERT_EQ(profiler_->Start(0, "/tmp/collect"), Status::kOk);
  ASSERT_EQ(profiler_->Stop(), Status::kOk);
  auto scoped_state = std::make_shared<FakeState>();
  {
    InProcessProfiler scoped(std::make_unique<FakeProfiler>(scoped_state));
    ASSERT_EQ(scoped.Start(1, "/tmp/other"), Status::kOk);
    scoped_state->calls.clear();
  }
  EXPECT_TRUE(scoped_state->destroyed);
  ASSERT_EQ(scoped_state->calls.size(), 1U);
  EXPECT_EQ(scoped_state->calls[0].name, "stop");
}

TEST_F(InProcessProfilerTest, MoveTransfersImplAndMovedFromDoesNotStop) {
  auto moved_state = std::make_shared<FakeState>();
  InProcessProfiler source(std::make_unique<FakeProfiler>(moved_state));
  ASSERT_EQ(source.Start(0, "/tmp/collect"), Status::kOk);
  InProcessProfiler target(std::move(source));
  ASSERT_TRUE(target.started());
  ASSERT_FALSE(source.started());
  target.Stop();
  ASSERT_EQ(moved_state->calls.size(), 2U);
  ASSERT_EQ(moved_state->calls[1].name, "stop");
}
}  // namespace device_validation
