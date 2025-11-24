#!/bin/bash

# 设置日志文件路径
LOG_FILE="error_log.log"

# 清空或创建日志文件
> "$LOG_FILE"

# 遍历reproducible_tests目录下的所有.py文件
for test_script in /root/PaddleAPITest/reproducible_tests/*.py; do
    echo "Running: $test_script" | tee -a "$LOG_FILE"
    python3 "$test_script" 2>> "$LOG_FILE"
    echo "----------------------------------------" | tee -a "$LOG_FILE"
done

echo "All tests completed. Errors logged to $LOG_FILE"