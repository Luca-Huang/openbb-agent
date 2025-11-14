#!/usr/bin/env bash
# 设置 TuShare Token 并写入到当前 shell 环境中。
# 使用方法：
#   1. 填写你的 TuShare token 到下方 TUSHARE_TOKEN 变量；
#   2. 运行 `source ./setup_tushare_env.sh`；
#   3. 之后执行 Python/OpenBB 时即可读取到 $TUSHARE_TOKEN。

export TUSHARE_TOKEN="YOUR_TUSHARE_TOKEN_HERE"

echo "TUSHARE_TOKEN 已载入当前终端会话。"
