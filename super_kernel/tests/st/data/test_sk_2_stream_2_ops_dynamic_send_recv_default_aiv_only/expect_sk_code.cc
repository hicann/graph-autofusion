
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


// begin implement of dynamic op is_inf__kernel0
static __aicore__ void switch_func_of_is_inf__kernel0(GM_ADDR __ac_dynamic_tiling_key_0, GM_ADDR __ac_dynamic_block_num_0, GM_ADDR __ac_wait_lock_0, uint64_t& aiv_func_addr, uint64_t& aiv_func_addr_split1, uint64_t& aiv_func_addr_split2, uint64_t& aiv_func_addr_split3, uint64_t& aic_func_addr, uint64_t& aic_func_addr_split1, uint64_t& aic_func_addr_split2, uint64_t& aic_func_addr_split3, uint64_t& dy_block_num) {
    __gm__ uint64_t* tilingKeyAddr = reinterpret_cast<__gm__ uint64_t*>(__ac_dynamic_tiling_key_0);
    __gm__ uint64_t* blockNumAddr = reinterpret_cast<__gm__ uint64_t*>(__ac_dynamic_block_num_0);
    __gm__ volatile uint64_t* lockAddr = reinterpret_cast<__gm__ uint64_t*>(__ac_wait_lock_0);
    dcci(lockAddr, 0, 2);
    while(*lockAddr != 1) {
        dcci(lockAddr, 0, 2);
    }

    if (*tilingKeyAddr == 2) {

    aiv_func_addr = (uint64_t)(is_inf__kernel0_middle);
        aiv_func_addr_split1 = (uint64_t)(is_inf__kernel0_middle_split1);
        aiv_func_addr_split2 = (uint64_t)(is_inf__kernel0_middle_split2);
        aiv_func_addr_split3 = (uint64_t)(is_inf__kernel0_middle_split3);
    dy_block_num = ((uint64_t)0) << 32 | (*blockNumAddr);

}
    return;
}

__aicore__ inline void call_func_of_is_inf__kernel0(uint64_t args_offset, const uint64_t dy_aiv_func_ptr, const uint64_t dy_aiv_func_ptr_split1, const uint64_t dy_aiv_func_ptr_split2, const uint64_t dy_aiv_func_ptr_split3, const uint64_t dy_aic_func_ptr, const uint64_t dy_aic_func_ptr_split1, const uint64_t dy_aic_func_ptr_split2, const uint64_t dy_aic_func_ptr_split3, const uint64_t dy_block_num) {
    uint64_t kernelType = dy_block_num  >> 32;
    uint64_t numBlocks = 48;
    g_super_kernel_dynamic_block_num = dy_block_num & 0xFFFFFFFF;
    using FuncType = void (*)(uint64_t args_offset);
    
    FuncType aiv_ptr = (FuncType)(dy_aiv_func_ptr);
    FuncType aic_ptr = (FuncType)(dy_aic_func_ptr);
    
    FuncType aiv_ptr_split1 = (FuncType)(dy_aiv_func_ptr_split1);
    FuncType aic_ptr_split1 = (FuncType)(dy_aic_func_ptr_split1);
    
    FuncType aiv_ptr_split2 = (FuncType)(dy_aiv_func_ptr_split2);
    FuncType aic_ptr_split2 = (FuncType)(dy_aic_func_ptr_split2);
    
    FuncType aiv_ptr_split3 = (FuncType)(dy_aiv_func_ptr_split3);
    FuncType aic_ptr_split3 = (FuncType)(dy_aic_func_ptr_split3);
    
    if (kernelType == 6 || kernelType == 7) {
        if (get_block_idx() < numBlocks) {
            uint8_t coreid = get_coreid();
            if ASCEND_IS_AIC {
                if ((coreid % 4) == 0) {
 
                aic_ptr(args_offset);
            } else if ((coreid % 4) == 1) {
            aic_ptr_split1(args_offset);
            } else if ((coreid % 4) == 2) {
            aic_ptr_split2(args_offset);
            } else {
 
 
                aic_ptr_split3(args_offset);
            }
            } else {
                if ((coreid % 4) == 0) {
 
                aiv_ptr(args_offset);
            } else if ((coreid % 4) == 1) {
            aiv_ptr_split1(args_offset);
            } else if ((coreid % 4) == 2) {
            aiv_ptr_split2(args_offset);
            } else {
 
 
                aiv_ptr_split3(args_offset);
            }
            }
        }
    } else if(kernelType == 0 || kernelType == 4) {
        if (AscendC::GetBlockIdx() < numBlocks) {
            uint8_t coreid = get_coreid();
            if ASCEND_IS_AIV{
                if ((coreid % 4) == 0) {
 
                aiv_ptr(args_offset);
            } else if ((coreid % 4) == 1) {
            aiv_ptr_split1(args_offset);
            } else if ((coreid % 4) == 2) {
            aiv_ptr_split2(args_offset);
            } else {
 
 
                aiv_ptr_split3(args_offset);
            }
            }
        }
    } else if (kernelType == 1 || kernelType == 5) {
        if (get_block_idx() < numBlocks) {
            uint8_t coreid = get_coreid();
            if ASCEND_IS_AIC {
                if ((coreid % 4) == 0) {
 
                aic_ptr(args_offset);
            } else if ((coreid % 4) == 1) {
            aic_ptr_split1(args_offset);
            } else if ((coreid % 4) == 2) {
            aic_ptr_split2(args_offset);
            } else {
 
 
                aic_ptr_split3(args_offset);
            }
            }
        }
    }
}

extern "C"  __aicore__ void is_finite__kernel0_middle(uint64_t args_offset);

extern "C"  __aicore__ void is_finite__kernel0_middle_split1(uint64_t args_offset);

extern "C"  __aicore__ void is_finite__kernel0_middle_split2(uint64_t args_offset);

extern "C"  __aicore__ void is_finite__kernel0_middle_split3(uint64_t args_offset);

__aicore__ inline void auto_gen_te_superkernel_2_stream_2_ops_kernel_aiv(void) {
    GM_ADDR *param_base = (GM_ADDR *)get_para_base();
    uint64_t aiv_func_addr = 0;
    uint64_t aic_func_addr = 0;
    uint64_t dy_blockNum = 0;
    uint64_t aiv_func_addr_split1 = 0;
    uint64_t aic_func_addr_split1 = 0;
    uint64_t aiv_func_addr_split2 = 0;
    uint64_t aic_func_addr_split2 = 0;
    uint64_t aiv_func_addr_split3 = 0;
    uint64_t aic_func_addr_split3 = 0;
    //begin func call of sub operator is_inf__kernel0

    switch_func_of_is_inf__kernel0(param_base[4], param_base[5], param_base[6], aiv_func_addr, aiv_func_addr_split1, aiv_func_addr_split2, aiv_func_addr_split3, aic_func_addr, aic_func_addr_split1, aic_func_addr_split2, aic_func_addr_split3, dy_blockNum);

    if ASCEND_IS_AIV {
      if (AscendC::GetBlockIdx() < 48) {
        uint8_t coreid = (uint8_t)get_coreid();
        if ((coreid % 4) == 0) {
          preload((const void *)aiv_func_addr, 8);
        } else if ((coreid % 4) == 1) {
          preload((const void *)aiv_func_addr_split1, 8);
        } else if ((coreid % 4) == 2) {
          preload((const void *)aiv_func_addr_split2, 8);
        } else {
          preload((const void *)aiv_func_addr_split3, 8);
        }

      }

    }


    if ASCEND_IS_AIV {
      if (AscendC::GetBlockIdx() < 32) {
        uint8_t coreid = (uint8_t)get_coreid();
        if ((coreid % 4) == 0) {
          preload((const void *)is_finite__kernel0_middle, 2);
        } else if ((coreid % 4) == 1) {
          preload((const void *)is_finite__kernel0_middle_split1, 2);
        } else if ((coreid % 4) == 2) {
          preload((const void *)is_finite__kernel0_middle_split2, 2);
        } else {
          preload((const void *)is_finite__kernel0_middle_split3, 2);
        }

      }

    }

    RecordProfiling(1, 0x8, true);
        AscendC::SyncAll<false>(); // reason: double stream need syncall to wait switch func
    call_func_of_is_inf__kernel0(1, aiv_func_addr, aiv_func_addr_split1, aiv_func_addr_split2, aiv_func_addr_split3, aic_func_addr, aic_func_addr_split1, aic_func_addr_split2, aic_func_addr_split3, dy_blockNum);
    if ASCEND_IS_AIV {

        if (AscendC::GetBlockIdx() == 0) {
            __gm__ volatile uint64_t* lockAddr = reinterpret_cast<__gm__ uint64_t*>(param_base[6]);
            *lockAddr = 0;
            dcci(lockAddr, 0, 2);
        }
    }
    RecordProfiling(1, 0x8, false);
    //begin func call of sub operator is_finite__kernel0
    // Rule 1 : sync all aiv must be insert behind each aiv sub operator, when has real send info
    // sync all V->V kernel_name:is_inf__kernel0, send_info:{}
    ffts_cross_core_sync(PIPE_MTE3, AscendC::GetffstMsg(0x0, AscendC::SYNC_AIV_ONLY_ALL));
    wait_flag_dev(AscendC::SYNC_AIV_ONLY_ALL);

    if ASCEND_IS_AIV {
        // kernel=is_inf__kernel0, ev=100, param_offset=7
    RecordProfiling(100, 0x4, true);
        NotifyFunc<false>(param_base[7]);
    RecordProfiling(100, 0x4, false);
    }
    if ASCEND_IS_AIV {
        // kernel=is_finite__kernel0, ev=101, param_offset=11
    RecordProfiling(101, 12, true);
        WaitFunc<false>(param_base[11]);
    RecordProfiling(101, 12, false);
    }
// two stream when has wait event, add sync by current operator kernel type
// reason3: for continues notify/wait event
ffts_cross_core_sync(PIPE_MTE3, AscendC::GetffstMsg(0x0, AscendC::SYNC_AIV_ONLY_ALL));
wait_flag_dev(AscendC::SYNC_AIV_ONLY_ALL);

    RecordProfiling(2, 0x8, true);
    if ASCEND_IS_AIV {
      if (AscendC::GetBlockIdx() < 32) {
        uint8_t coreid = (uint8_t)get_coreid();
        if ((coreid % 4) == 0) {
          is_finite__kernel0_middle(8);
        } else if ((coreid % 4) == 1) {
          is_finite__kernel0_middle_split1(8);
        } else if ((coreid % 4) == 2) {
          is_finite__kernel0_middle_split2(8);
        } else {
          is_finite__kernel0_middle_split3(8);
        }

      }

    }

    RecordProfiling(2, 0x8, false);
}

extern "C"  __global__ __attribute__((aligned(512))) __aicore__ void auto_gen_te_superkernel_2_stream_2_ops_kernel(void) {
    GM_ADDR *param_base = (GM_ADDR *)get_para_base();
    GM_ADDR workspace = param_base[12];
    #if defined ASCENDC_DUMP || defined ASCENDC_TIME_STAMP_ON
    constexpr uint32_t ASCENDC_DUMP_SIZE = 0;
    AscendC::InitDump(false, workspace + 0, ASCENDC_DUMP_SIZE);
    #endif
    GM_ADDR profilingPtr = param_base[13];
    uint32_t taskId = *((__gm__ uint32_t*)(get_para_base() + 8 * 14));
    InitProfiling(taskId, profilingPtr);
    GM_ADDR ffts_addr = param_base[0];
    if (ffts_addr != nullptr) {
        set_ffts_base_addr((uint64_t)ffts_addr);
    }

    RecordProfiling(0, 0, true);
    if ASCEND_IS_AIV {
        auto_gen_te_superkernel_2_stream_2_ops_kernel_aiv();
    }
    if ASCEND_IS_AIV {
        if (AscendC::GetBlockIdx() == 0) {
            *(reinterpret_cast<__gm__ uint64_t*>(param_base[11])) = 0;
        }
    }
    RecordProfiling(0, 0, false);
}

#if TILING_KEY_VAR == 0UL
static const struct FunLevelMixCoreType te_superkernel_2_stream_2_ops_mix_aiv_section __attribute__ ((used, section (".ascend.meta.te_superkernel_2_stream_2_ops_mix_aiv"))) = { {{F_TYPE_KTYPE, sizeof(unsigned int)}, K_TYPE_MIX_AIV_MAIN},    {{F_TYPE_MIX_TASK_RATION, sizeof(unsigned int)}, 0, 1} };
#endif
#endif
