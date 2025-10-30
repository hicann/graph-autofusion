
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

extern "C"  __aicore__ void add_aiv(uint64_t args_offset);

extern "C"  __aicore__ void add_aiv_split1(uint64_t args_offset);

extern "C"  __aicore__ void add_aiv_split2(uint64_t args_offset);

extern "C"  __aicore__ void add_aiv_split3(uint64_t args_offset);

extern "C"  __global__ __attribute__((aligned(512))) __aicore__ void auto_gen_test_add_kernel(void) {
    GM_ADDR *param_base = (GM_ADDR *)get_para_base();
    GM_ADDR ffts_addr = param_base[0];
    if (ffts_addr != nullptr) {
        set_ffts_base_addr((uint64_t)ffts_addr);
    }

    //begin func call of sub operator te_op_add
    if ASCEND_IS_AIV {
      if (AscendC::GetBlockIdx() < 36) {
        uint8_t coreid = (uint8_t)get_coreid();
        if ((coreid % 4) == 0) {
          preload((const void *)add_aiv, 0);
        } else if ((coreid % 4) == 1) {
          preload((const void *)add_aiv_split1, 0);
        } else if ((coreid % 4) == 2) {
          preload((const void *)add_aiv_split2, 0);
        } else {
          preload((const void *)add_aiv_split3, 0);
        }

      }

    }

    if ASCEND_IS_AIV {
      if (AscendC::GetBlockIdx() < 36) {
        uint8_t coreid = (uint8_t)get_coreid();
        if ((coreid % 4) == 0) {
          add_aiv(1);
        } else if ((coreid % 4) == 1) {
          add_aiv_split1(1);
        } else if ((coreid % 4) == 2) {
          add_aiv_split2(1);
        } else {
          add_aiv_split3(1);
        }

      }

    }

}

