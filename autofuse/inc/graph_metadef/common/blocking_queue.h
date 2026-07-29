/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef INC_COMMON_BLOCKING_QUEUE_H_
#define INC_COMMON_BLOCKING_QUEUE_H_

#include <condition_variable>
#include <list>
#include <mutex>

namespace af {
constexpr uint32_t kDefaultMaxQueueSize = 2048U;
constexpr int32_t kDefaultWaitTimeoutInSec = 600;

template <typename T>
class BlockingQueue {
 public:
  explicit BlockingQueue(const uint32_t max_size = kDefaultMaxQueueSize) : max_size_(max_size) {}

  ~BlockingQueue() = default;

  bool Pop(T &item, const int32_t time_out = INT32_MAX) {
    std::unique_lock<std::mutex> lock(mutex_);

    while (!empty_cond_.wait_for(lock, std::chrono::seconds(time_out),
                                 [this]() -> bool { return (!queue_.empty()) || (is_stoped_); })) {
      is_stuck_ = true;
      return false;
    }

    if (is_stoped_) {
      return false;
    }

    item = std::move(queue_.front());
    queue_.pop_front();

    full_cond_.notify_one();

    return true;
  }

  bool Pop(T &item, bool &is_stuck) {
    const auto ret = Pop(item, kDefaultWaitTimeoutInSec);
    is_stuck = is_stuck_;
    return ret;
  }

  bool Push(const T &item, const bool is_wait = true) {
    std::unique_lock<std::mutex> lock(mutex_);

    while ((queue_.size() >= max_size_) && (!is_stoped_)) {
      if (!is_wait) {
        return false;
      }
      full_cond_.wait(lock);
    }

    if (is_stoped_) {
      return false;
    }

    queue_.push_back(item);

    empty_cond_.notify_one();

    return true;
  }

  bool Push(T &&item, const bool is_wait = true) {
    std::unique_lock<std::mutex> lock(mutex_);

    while ((queue_.size() >= max_size_) && (!is_stoped_)) {
      if (!is_wait) {
        return false;
      }
      full_cond_.wait(lock);
    }

    if (is_stoped_) {
      return false;
    }

    (void)queue_.emplace_back(std::move(item));

    empty_cond_.notify_one();

    return true;
  }

  void Stop() {
    {
      const std::unique_lock<std::mutex> lock(mutex_);
      is_stoped_ = true;
    }

    full_cond_.notify_all();
    empty_cond_.notify_all();
  }

  void Restart() {
    const std::unique_lock<std::mutex> lock(mutex_);
    is_stoped_ = false;
  }

  // if the queue is stopped ,need call this function to release the unprocessed items
  std::list<T> GetRemainItems() {
    const std::unique_lock<std::mutex> lock(mutex_);

    if (!is_stoped_) {
      return std::list<T>();
    }

    return queue_;
  }

  bool IsFull() {
    const std::unique_lock<std::mutex> lock(mutex_);
    return queue_.size() >= max_size_;
  }

  void Clear() {
    const std::unique_lock<std::mutex> lock(mutex_);
    queue_.clear();
  }

  void SetMaxSize(const uint32_t size) {
    const std::unique_lock<std::mutex> lock(mutex_);
    if (size == 0U) {
      max_size_ = kDefaultMaxQueueSize;
      return;
    }
    max_size_ = size;
  }

  uint32_t Size() {
    const std::unique_lock<std::mutex> lock(mutex_);
    return static_cast<uint32_t>(queue_.size());
  }

 private:
  std::list<T> queue_;
  std::mutex mutex_;
  std::condition_variable empty_cond_;
  std::condition_variable full_cond_;
  uint32_t max_size_;

  bool is_stoped_{false};
  bool is_stuck_{false};
};
}  // namespace af

#endif  // INC_COMMON_BLOCKING_QUEUE_H_
