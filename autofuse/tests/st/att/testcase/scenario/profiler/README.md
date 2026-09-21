# profiler 使用说明
1. 由main函数封装tiling func的GetTiling函数，编译main函数，获取可执行文件路径
2. 输入参数：
    --log_path: tiling log文件路径（必填）
    --summary_path: 上版profiling路径（可以是缺省值，缺失则不输出real_cost信息）
example:
    python3 profiler.py --log_path=./tiling.log
    python3 profiler.py --log_path=./rec.log --summary_path=./summary.csv
3. profiler以print形式输出profiling信息，也以output.csv形式输出表格
