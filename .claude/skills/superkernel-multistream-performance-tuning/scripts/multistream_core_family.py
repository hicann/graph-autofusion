#!/usr/bin/env python3
"""Derive compute-resource families from Ascend profiler kernel row fields."""


CORE_FAMILIES = {"CUBE", "VECTOR", "MIX", "COMMUNICATION", "WAIT", "OTHER"}


def _nonnegative_integer(value, label, *, optional=False):
    if optional and value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        suffix = " or null" if optional else ""
        raise ValueError(f"{label} must be a non-negative integer{suffix}")
    return value


def classify_profile_identity(accelerator_core, block_num, mix_block_num):
    """Classify one profiler row using all three resource identity fields."""
    if not isinstance(accelerator_core, str) or not accelerator_core.strip():
        raise ValueError("accelerator_core must be a non-empty string")
    accelerator_core = accelerator_core.strip().upper()
    block_num = _nonnegative_integer(block_num, "block_num")
    mix_block_num = _nonnegative_integer(
        mix_block_num,
        "mix_block_num",
        optional=accelerator_core == "COMMUNICATION",
    )

    if accelerator_core == "AI_VECTOR_CORE":
        family = "VECTOR"
    elif accelerator_core == "AI_CORE":
        family = "CUBE"
    elif accelerator_core == "MIX_AIV":
        family = "VECTOR" if mix_block_num == 0 else "MIX"
    elif accelerator_core == "MIX_AIC":
        family = "CUBE" if mix_block_num == 0 else "MIX"
    elif accelerator_core == "COMMUNICATION":
        family = "COMMUNICATION"
    elif accelerator_core == "WAIT":
        family = "WAIT"
    else:
        family = "OTHER"

    return {
        "accelerator_core": accelerator_core,
        "block_num": block_num,
        "mix_block_num": mix_block_num,
        "core_family": family,
    }
