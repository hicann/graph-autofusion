
#if 1
#include "kernel_operator.h"

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

__aicore__ inline void auto_gen_te_superkernel_2_stream_4_ops_kernel_aic(void) {
    GM_ADDR *param_base = (GM_ADDR *)get_para_base();
    //begin func call of sub operator pows__kernel0
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

    // Rule 3.2 : sync all v2c must be insert when recvinfo has v2c, kernel_name:pows__kernel0, send_info:{'is_inf__kernel0_2': 'vec:cub'}
    // receive sync of V->C;
    wait_flag_dev(AscendC::SYNC_AIV_FLAG);
    if ASCEND_IS_AIC {
      if (get_block_idx() < 1) {
        uint8_t coreid = (uint8_t)get_coreid();
        if ((coreid % 4) == 0) {
          pows__kernel0_middle(5);
        } else if ((coreid % 4) == 1) {
          pows__kernel0_middle_split1(5);
        } else if ((coreid % 4) == 2) {
          pows__kernel0_middle_split2(5);
        } else {
          pows__kernel0_middle_split3(5);
        }

      }

    }

    //begin func call of sub operator pows__kernel0
    // Rule 1 : sync all aic must be insert behind each aic sub operator, when has real send info
    // sync all C->C kernel_name:pows__kernel0, send_info:{'pows__kernel0_3': 'cub:cub'}
    ffts_cross_core_sync(PIPE_FIX, AscendC::GetffstMsg(0x0, AscendC::SYNC_AIC_FLAG));
    wait_flag_dev(AscendC::SYNC_AIC_FLAG);

    if ASCEND_IS_AIC {
      if (get_block_idx() < 1) {
        uint8_t coreid = (uint8_t)get_coreid();
        if ((coreid % 4) == 0) {
          pows__kernel0_middle(14);
        } else if ((coreid % 4) == 1) {
          pows__kernel0_middle_split1(14);
        } else if ((coreid % 4) == 2) {
          pows__kernel0_middle_split2(14);
        } else {
          pows__kernel0_middle_split3(14);
        }

      }

    }

}

__aicore__ inline void auto_gen_te_superkernel_2_stream_4_ops_kernel_aiv(void) {
    GM_ADDR *param_base = (GM_ADDR *)get_para_base();
    //begin func call of sub operator is_inf__kernel0
    if ASCEND_IS_AIV {
      if (AscendC::GetBlockIdx() < 32) {
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

    if ASCEND_IS_AIV {
      if (AscendC::GetBlockIdx() < 32) {
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

    if ASCEND_IS_AIV {
      if (AscendC::GetBlockIdx() < 32) {
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

    //begin func call of sub operator is_inf__kernel0
    // Rule 1 : sync all aiv must be insert behind each aiv sub operator, when has real send info
    // sync all V->V kernel_name:is_inf__kernel0, send_info:{'is_inf__kernel0_2': 'vec:vec'}
    ffts_cross_core_sync(PIPE_MTE3, AscendC::GetffstMsg(0x0, AscendC::SYNC_AIV_ONLY_ALL));
    wait_flag_dev(AscendC::SYNC_AIV_ONLY_ALL);

    if ASCEND_IS_AIV {
      if (AscendC::GetBlockIdx() < 32) {
        uint8_t coreid = (uint8_t)get_coreid();
        if ((coreid % 4) == 0) {
          is_inf__kernel0_middle(10);
        } else if ((coreid % 4) == 1) {
          is_inf__kernel0_middle_split1(10);
        } else if ((coreid % 4) == 2) {
          is_inf__kernel0_middle_split2(10);
        } else {
          is_inf__kernel0_middle_split3(10);
        }

      }

    }

    // Rule 1 : sync all aiv must be insert behind each aiv sub operator, when has real send info
    // sync all V->V kernel_name:is_inf__kernel0, send_info:{'pows__kernel0_1': 'vec:cub'}
    ffts_cross_core_sync(PIPE_MTE3, AscendC::GetffstMsg(0x0, AscendC::SYNC_AIV_ONLY_ALL));
    wait_flag_dev(AscendC::SYNC_AIV_ONLY_ALL);

    // Rule 3.1 : sync all v2c must be insert when sendinfo has v2c, kernel_name:is_inf__kernel0, send_info:{'pows__kernel0_1': 'vec:cub'}
    // send sync of V->C;
    ffts_cross_core_sync(PIPE_MTE3, AscendC::GetffstMsg(0x02, AscendC::SYNC_AIV_FLAG));

    if ASCEND_IS_AIV {
        // kernel=is_inf__kernel0, ev=201, param_offset=13
        NotifyFunc<false>(param_base[13]);
    }
}

extern "C"  __global__ __attribute__((aligned(512))) __aicore__ void auto_gen_te_superkernel_2_stream_4_ops_kernel(void) {
    GM_ADDR *param_base = (GM_ADDR *)get_para_base();
    GM_ADDR ffts_addr = param_base[0];
    if (ffts_addr != nullptr) {
        set_ffts_base_addr((uint64_t)ffts_addr);
    }

    if ASCEND_IS_AIC {
        auto_gen_te_superkernel_2_stream_4_ops_kernel_aic();
    }
    if ASCEND_IS_AIV {
        auto_gen_te_superkernel_2_stream_4_ops_kernel_aiv();
    }
    if ASCEND_IS_AIC {
        if (get_block_idx() == 0) {
            *(reinterpret_cast<__gm__ uint64_t*>(param_base[9])) = 0;
        }
    }
}

#if TILING_KEY_VAR == 0UL
#if defined(__DAV_C220_CUBE__)
static const struct FunLevelMixCoreType te_superkernel_2_stream_4_ops_mix_aic_section __attribute__ ((used, section (".ascend.meta.te_superkernel_2_stream_4_ops_mix_aic"))) = { {{F_TYPE_KTYPE, sizeof(unsigned int)}, K_TYPE_MIX_AIC_MAIN},    {{F_TYPE_MIX_TASK_RATION, sizeof(unsigned int)}, 1, 2} };
#endif
#endif

#if TILING_KEY_VAR == 0UL
#if defined(__DAV_C220_VEC__)
static const struct FunLevelMixCoreType te_superkernel_2_stream_4_ops_mix_aiv_section __attribute__ ((used, section (".ascend.meta.te_superkernel_2_stream_4_ops_mix_aiv"))) = { {{F_TYPE_KTYPE, sizeof(unsigned int)}, K_TYPE_MIX_AIC_MAIN},    {{F_TYPE_MIX_TASK_RATION, sizeof(unsigned int)}, 1, 2} };
#endif
#endif
#endif
