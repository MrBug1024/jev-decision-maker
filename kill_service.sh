#!/bin/bash
# kill_service.sh - 精确杀掉 0.0.0.0:8019 的 gunicorn 服务

APP_NAME="server:app"
PORT="8019"
WORKER_MODULE="uvicorn.workers.UvicornWorker"

echo "==> 正在查找目标进程..."

# 1. 找到监听该端口的 master 进程 PID
MASTER_PID=$(ss -lntp 2>/dev/null | awk -v p=":$PORT" '$4 ~ p {print $6}' \
    | grep -oP 'pid=\K[0-9]+' | head -n1)

if [ -z "$MASTER_PID" ]; then
    echo "端口 $PORT 上未找到监听进程，尝试通过命令行匹配..."
    MASTER_PID=$(pgrep -f "gunicorn.*${APP_NAME}" | head -n1)
fi

if [ -z "$MASTER_PID" ]; then
    echo "未找到目标服务，退出。"
    exit 0
fi

# 2. 通过 master PID 找到整个进程组（gunicorn 的 master + workers 是父子关系）
echo "==> 找到 master 进程 PID: $MASTER_PID"
echo "==> 进程树如下:"
pstree -ap "$MASTER_PID" 2>/dev/null || ps --ppid "$MASTER_PID" -o pid,ppid,cmd

# 3. 收集所有子进程 PID
CHILD_PIDS=$(pgrep -P "$MASTER_PID")

# 4. 先优雅停止 master（gunicorn 会通知 worker 退出）
echo "==> 发送 TERM 信号..."
kill -TERM "$MASTER_PID" 2>/dev/null

# 5. 等待最多 10 秒
for i in $(seq 1 10); do
    if ! kill -0 "$MASTER_PID" 2>/dev/null; then
        echo "==> 服务已优雅退出。"
        exit 0
    fi
    sleep 1
done

# 6. 还活着就强杀 master + 所有子进程
echo "==> 超时未退出，强制 kill -9..."
kill -9 "$MASTER_PID" 2>/dev/null
for pid in $CHILD_PIDS; do
    kill -9 "$pid" 2>/dev/null
done

# 7. 最终确认
sleep 1
if ss -lntp 2>/dev/null | grep -q ":$PORT "; then
    echo "警告：端口 $PORT 仍被占用！"
    ss -lntp | grep ":$PORT "
    exit 1
fi

echo "==> 全部清理完成。"