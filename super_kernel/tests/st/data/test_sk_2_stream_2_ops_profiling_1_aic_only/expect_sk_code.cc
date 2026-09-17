
#if 1
#include "kernel_operator.h"

__BLOCK_LOCAL__ __inline__ uint32_t g_profiling_task_id;
__BLOCK_LOCAL__ __inline__ __gm__ uint8_t* g_profiling_base_addr;
__BLOCK_LOCAL__ __inline__ __gm__ uint8_t* g_profiling_working_addr;
__BLOCK_LOCAL__ __inline__ __gm__ uint8_t* g_profiling_max_addr;
__BLOCK_LOCAL__ __inline__ bool g_profiling_off;
__BLOCK_LOCAL__ __inline__ uint32_t g_percore_size;
constexpr uint64_t PROFILING_MAGIC_NUMBER = 0xbdca8756;
constexpr uint32_t PROFILING_WORKINF_PTR_OFFSET = 8;
constexpr uint32_t PROFILING_MAX_PTR_OFFSET = 16;
constexpr uint32_t ONE_PROFILING_HEAD_SIZE = 16;
constexpr uint32_t ONE_PROFILING_DATA_SIZE = 16;
__aicore__ inline bool ProfilingAreaIsValid()
{
    return (*((__gm__ uint64_t*)g_profiling_base_addr) == PROFILING_MAGIC_NUMBER) &&
        ((*((__gm__ uint64_t*)g_profiling_working_addr)) < (*((__gm__ uint64_t*)g_profiling_max_addr)));
}

__aicore__ inline uint8_t GetProfilingBlockIdx()
{
    if ASCEND_IS_AIV {
        return get_block_idx() * get_subblockdim() + get_subblockid();
    } else {
        return get_block_idx() + 50;
    }
}

__aicore__ inline void RecordProfiling()
{
    if (g_profiling_off) {
        return;
    }
    uint8_t blockIdx = GetProfilingBlockIdx();
    uint64_t workAddr = *((__gm__ uint64_t*)g_profiling_working_addr);
    *((__gm__ uint64_t*)workAddr) = ((uint64_t)g_profiling_task_id << 32) | (((uint64_t)blockIdx) << 8) | 0xff;
    *((__gm__ uint64_t*)workAddr + 1) = static_cast<uint64_t>(AscendC::GetSystemCycle());
    dcci((__gm__ uint64_t*)workAddr, 0, 2);
    *((__gm__ uint64_t*)g_profiling_working_addr) += ONE_PROFILING_DATA_SIZE;
    if (!ProfilingAreaIsValid()) {
        g_profiling_off = true;
    }
    dcci((__gm__ uint64_t*)g_profiling_working_addr, 0, 2);
}

__aicore__ inline void RecordProfiling(uint32_t index, uint8_t profilingType, bool startFlag)
{
    if (g_profiling_off) {
        return;
    }
    uint8_t blockIdx = GetProfilingBlockIdx();
    uint64_t workAddr = *((__gm__ uint64_t*)g_profiling_working_addr);
    if (startFlag) {
        *((__gm__ uint64_t*)workAddr) = ((uint64_t)index << 32) | (((uint64_t)profilingType & 0xf) << 8) | 0x0;
    } else {
        *((__gm__ uint64_t*)workAddr) =
            ((uint64_t)index << 32) | (1 << 12) | (((uint64_t)profilingType & 0xf) << 8) | 0x0;
    }
    *((__gm__ uint64_t*)workAddr + 1) = static_cast<uint64_t>(AscendC::GetSystemCycle());
    dcci((__gm__ uint64_t*)workAddr, 0, 2);
    *((__gm__ uint64_t*)g_profiling_working_addr) += ONE_PROFILING_DATA_SIZE;
    if (!ProfilingAreaIsValid()) {
        g_profiling_off = true;
    }
    dcci((__gm__ uint64_t*)g_profiling_working_addr, 0, 2);
}

__aicore__ inline void InitProfiling(uint32_t taskId, GM_ADDR profilingPtr)
{
    g_profiling_off = false;
    uint8_t blockIdx = GetProfilingBlockIdx();
    g_percore_size = *((__gm__ uint32_t*)(profilingPtr + 12));
    g_profiling_base_addr = profilingPtr + 64 + blockIdx * g_percore_size;
    g_profiling_working_addr = g_profiling_base_addr + PROFILING_WORKINF_PTR_OFFSET;
    g_profiling_max_addr = g_profiling_base_addr + PROFILING_MAX_PTR_OFFSET;
    if (!ProfilingAreaIsValid()) {
        g_profiling_off = true;
        return;
    }
    g_profiling_task_id = taskId;
    RecordProfiling();
}

template<bool aic_flag>
__aicore__ inline void NotifyFunc(GM_ADDR notify_lock_addr)
{
    if constexpr (aic_flag) {
        if (get_block_idx() == 0) {
            __gm__ uint64_t* notifyLock = reinterpret_cast<__gm__ uint64_t*>(notify_lock_addr);
            *notifyLock = 1;
            dcci(notifyLock, 0, 2);
        }
    } else {
        if (AscendC::GetBlockIdx() == 0) {
            __gm__ uint64_t* notifyLock = reinterpret_cast<__gm__ uint64_t*>(notify_lock_addr);
            *notifyLock = 1;
            dcci(notifyLock, 0, 2);
        }
    }
}


template<bool aic_flag>
__aicore__ inline void WaitFunc(GM_ADDR wait_lock_addr)
{
    if constexpr (aic_flag) {
        __gm__ volatile uint64_t* waitLock = reinterpret_cast<__gm__ uint64_t*>(wait_lock_addr);
        if (get_block_idx() == 0) {
            dcci(waitLock, 0, 2);
            while(*waitLock != 1) {
                dcci(waitLock, 0, 2);
            }
        }
    } else {
        __gm__ volatile uint64_t* waitLock = reinterpret_cast<__gm__ uint64_t*>(wait_lock_addr);
        if (AscendC::GetBlockIdx() == 0) {
            dcci(waitLock, 0, 2);
            while(*waitLock != 1) {
                dcci(waitLock, 0, 2);
            }
        }
    }
}

extern "C"  __aicore__ void is_inf__kernel0_middle(uint64_t args_offset);

extern "C"  __aicore__ void is_inf__kernel0_middle_split1(uint64_t args_offset);

extern "C"  __aicore__ void is_inf__kernel0_middle_split2(uint64_t args_offset);

extern "C"  __aicore__ void is_inf__kernel0_middle_split3(uint64_t args_offset);

extern "C"  __aicore__ void pows__kernel0_middle(uint64_t args_offset);

extern "C"  __aicore__ void pows__kernel0_middle_split1(uint64_t args_offset);

extern "C"  __aicore__ void pows__kernel0_middle_split2(uint64_t args_offset);

extern "C"  __aicore__ void pows__kernel0_middle_split3(uint64_t args_offset);

__aicore__ inline void auto_gen_te_superkernel_2_stream_2_ops_kernel_aic(void) {
    GM_ADDR *param_base = (GM_ADDR *)get_para_base();
    //begin func call of sub operator is_inf__kernel0
    if ASCEND_IS_AIC {
      if (get_block_idx() < 32) {
        uint8_t coreid = (uint8_t)get_coreid();
        if ((coreid % 4) == 0) {
          preload((const void *)is_inf__kernel0_middle, 2);
        } else if ((coreid % 4) == 1) {
          preload((const void *)is_inf__kernel0_middle_split1, 2);
        } else if ((coreid % 4) == 2) {
          preload((const void *)is_inf__kernel0_middle_split2, 2);
        } else {
          preload((const void *)is_inf__kernel0_middle_split3, 2);
        }

      }

    }

    if ASCEND_IS_AIC {
      if (get_block_idx() < 1) {
        uint8_t coreid = (uint8_t)get_coreid();
        if ((coreid % 4) == 0) {
          preload((const void *)pows__kernel0_middle, 5);
        } else if ((coreid % 4) == 1) {
          preload((const void *)pows__kernel0_middle_split1, 5);
        } else if ((coreid % 4) == 2) {
          preload((const void *)pows__kernel0_middle_split2, 5);
        } else {
          preload((const void *)pows__kernel0_middle_split3, 5);
        }

      }

    }

    RecordProfiling(1, 0x8, true);
    if ASCEND_IS_AIC {
      if (get_block_idx() < 32) {
        uint8_t coreid = (uint8_t)get_coreid();
        if ((coreid % 4) == 0) {
          is_inf__kernel0_middle(1);
        } else if ((coreid % 4) == 1) {
          is_inf__kernel0_middle_split1(1);
        } else if ((coreid % 4) == 2) {
          is_inf__kernel0_middle_split2(1);
        } else {
          is_inf__kernel0_middle_split3(1);
        }

      }

    }

    RecordProfiling(1, 0x8, false);
    //begin func call of sub operator pows__kernel0
    // insert pipe all for ops
       pipe_barrier(PIPE_ALL);
    RecordProfiling(2, 0x8, true);
    if ASCEND_IS_AIC {
      if (get_block_idx() < 1) {
        uint8_t coreid = (uint8_t)get_coreid();
        if ((coreid % 4) == 0) {
          pows__kernel0_middle(4);
        } else if ((coreid % 4) == 1) {
          pows__kernel0_middle_split1(4);
        } else if ((coreid % 4) == 2) {
          pows__kernel0_middle_split2(4);
        } else {
          pows__kernel0_middle_split3(4);
        }

      }

    }

    RecordProfiling(2, 0x8, false);
}

extern "C"  __global__ __attribute__((aligned(512))) __aicore__ void auto_gen_te_superkernel_2_stream_2_ops_kernel(void) {
    GM_ADDR *param_base = (GM_ADDR *)get_para_base();
    GM_ADDR profilingPtr = param_base[8];
    uint32_t taskId = *((__gm__ uint32_t*)(get_para_base() + 8 * 9));
    InitProfiling(taskId, profilingPtr);
    GM_ADDR ffts_addr = param_base[0];
    if (ffts_addr != nullptr) {
        set_ffts_base_addr((uint64_t)ffts_addr);
    }

    RecordProfiling(0, 0, true);
    if ASCEND_IS_AIC {
        auto_gen_te_superkernel_2_stream_2_ops_kernel_aic();
    }
    RecordProfiling(0, 0, false);
}

#if TILING_KEY_VAR == 0UL
static const struct FunLevelMixCoreType te_superkernel_2_stream_2_ops_mix_aic_section __attribute__ ((used, section (".ascend.meta.te_superkernel_2_stream_2_ops_mix_aic"))) = { {{F_TYPE_KTYPE, sizeof(unsigned int)}, K_TYPE_MIX_AIC_MAIN},    {{F_TYPE_MIX_TASK_RATION, sizeof(unsigned int)}, 1, 0} };
#endif
#endif
