#!/bin/bash

# 要操作的目录，默认当前目录
TARGET_DIR=${1:-.}

# 查找目录下，所有非png文件并删除
find "$TARGET_DIR" -type f ! -name "*.png" -exec rm -f {} +
find "$TARGET_DIR" -type d -empty -delete

echo "已删除 $TARGET_DIR 下除png文件以外的所有文件。"

