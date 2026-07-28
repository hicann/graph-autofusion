# Autofuse Use Case Demonstration

## Use Case Function:

Autofuse fuses mul + reducesum two operators.

## Execution Command

```bash
python3 af_mul_reducesum.py
```

## Expected Execution Result

In the profiling directory under the current directory, there are generated profiling files. In PROF_000001_timestampxx/mindstudio_profiler_output, you can open the op_summary_timestampxx.csv file to see execution operator details. At this point, there is only a kernel named autofused_mul_sum_topology_hash, indicating that mul+reducesum two operators have been fused into a single fusion operator.