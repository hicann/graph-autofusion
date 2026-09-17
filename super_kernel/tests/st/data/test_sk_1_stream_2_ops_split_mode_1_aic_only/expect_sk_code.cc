
#if 1
#include "kernel_operator.h"
#if (defined(__DAV_CUBE__) && __NPU_ARCH__ == 2201)
    __gm__ struct OpSystemRunCfg g_opSystemRunCfg = {0};
#else
    extern __gm__ struct OpSystemRunCfg g_opSystemRunCfg;
#endif


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

extern "C"  __aicore__ void is_finite__kernel0_middle(uint64_t args_offset);

extern "C"  __global__ __attribute__((aligned(512))) __aicore__ void auto_gen_te_superkernel_1_stream_2_ops_kernel(void) {
    GM_ADDR *param_base = (GM_ADDR *)get_para_base();
    GM_ADDR ffts_addr = param_base[0];
    if (ffts_addr != nullptr) {
        set_ffts_base_addr((uint64_t)ffts_addr);
    }

    //begin func call of sub operator is_inf__kernel0
    if ASCEND_IS_AIC {
      if (get_block_idx() < 32) {
          preload((const void *)is_inf__kernel0_middle, 2);
      }

    }

    if ASCEND_IS_AIC {
      if (get_block_idx() < 32) {
          preload((const void *)is_finite__kernel0_middle, 2);
      }

    }

    if ASCEND_IS_AIC {
      if (get_block_idx() < 32) {
          is_inf__kernel0_middle(1);
      }

    }

    //begin func call of sub operator is_finite__kernel0
    // begin inter ops barrier
    if ASCEND_IS_AIC {
        if (true) {
            AscendC::SetNextTaskStart();
        }
    }

    if ASCEND_IS_AIV {
        if (true) {
            // AIC only, no complement early start set flag.
        }
    }

    g_super_kernel_early_start_config = 0;
    if ASCEND_IS_AIC {
        if (true) {
            AscendC::WaitPreTaskEndImpl<0>();
        }
    }

    if ASCEND_IS_AIV {
        if (true) {
            AscendC::WaitPreTaskEndImpl<0>();
        }
    }

    if ASCEND_IS_AIC {
      if (get_block_idx() < 32) {
          is_finite__kernel0_middle(4);
      }

    }

}

#if TILING_KEY_VAR == 0UL
static const struct FunLevelMixCoreType te_superkernel_1_stream_2_ops_mix_aic_section __attribute__ ((used, section (".ascend.meta.te_superkernel_1_stream_2_ops_mix_aic"))) = { {{F_TYPE_KTYPE, sizeof(unsigned int)}, K_TYPE_MIX_AIC_MAIN},    {{F_TYPE_MIX_TASK_RATION, sizeof(unsigned int)}, 1, 0} };
#endif
#endif
