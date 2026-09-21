# Profiler Usage Instructions

1. Encapsulate the GetTiling function of the tiling func from the main function, compile the main function, and obtain the executable file path.

2. Input parameters:
    --log_path: tiling log file path (required)
    --summary_path: on-board profiling path (can be default value, missing will not output real_cost information)

Example:
```bash
python3 profiler.py --log_path=./tiling.log
python3 profiler.py --log_path=./rec.log --summary_path=./summary.csv
```

3. Profiler outputs profiling information in print form and also outputs tables in output.csv form.
