#!/bin/bash
# kill_jev_8019.sh —— 只杀 venv=jev-decision-maker 且 cwd=~/work/jev-decision-maker 的 8019 服务
set -u

PORT=8019
VENV_MARK="jev-decision-maker"                    # 出现在 /proc/PID/exe 路径中
PROJECT_DIR="$HOME/work/jev-decision-maker"       # 进程 cwd 必须在此目录下
APP_MARK="server:app"                             # 命令行标记（辅助校验）

# ---- 1. 优先从端口反查 ----
echo "==> 从端口 $PORT 反查监听进程..."
PIDS=$(sudo ss -lntp 2>/dev/null \
    | awk -v p=":${PORT}\$" '$4 ~ p {print $6}' \
    | grep -oP 'pid=\K[0-9]+' | sort -u)

# 端口没监听但进程可能还在（僵死），用 venv+cwd 兜底反查
if [ -z "$PIDS" ]; then
    echo "端口无监听，改用 venv+cwd 反查..."
    PIDS=$(pgrep -f "$APP_MARK" | while read p; do
        exe=$(readlink /proc/$p/exe 2>/dev/null)
        cwd=$(readlink /proc/$p/cwd 2>/dev/null)
        [[ "$exe" == *"$VENV_MARK"* && "$cwd" == "$PROJECT_DIR"* ]] && echo "$p"
    done | sort -u)
fi

[ -z "$PIDS" ] && { echo "未找到目标服务，退出。"; exit 0; }
echo "==> 候选 PID: $PIDS"

# ---- 2. 三重校验，任何一条不匹配就中止 ----
echo "==> 校验身份（exe / cwd / cmdline）..."
BAD=0
for pid in $PIDS; do
    exe=$(sudo readlink /proc/$pid/exe 2>/dev/null)
    cwd=$(sudo readlink /proc/$pid/cwd 2>/dev/null)
    cmd=$(sudo tr '\0' ' ' < /proc/$pid/cmdline 2>/dev/null)
    ok=1
    [[ "$exe" == *"$VENV_MARK"* ]]       || ok=0
    [[ "$cwd" == "$PROJECT_DIR"* ]]       || ok=0
    [[ "$cmd" == *"$APP_MARK"* ]]         || ok=0
    if [ $ok -eq 1 ]; then
        echo "  [OK] pid=$pid  exe=$exe  cwd=$cwd"
    else
        echo "  [!!] pid=$pid  exe=$exe  cwd=$cwd  cmd=$cmd"
        BAD=1
    fi
done
[ $BAD -eq 1 ] && { echo "存在不匹配进程，为安全起见中止。"; exit 1; }

# ---- 3. 找 master（PPID 不在候选集合里的那个） ----
MASTERS=""
for pid in $PIDS; do
    ppid=$(ps -o ppid= -p "$pid" | tr -d ' ')
    echo "$PIDS" | grep -qw "$ppid" || MASTERS="$MASTERS $pid"
done
echo "==> master PID:$MASTERS"

# ---- 4. 优雅停止 ----
echo "==> 发送 TERM ..."
sudo kill -TERM $MASTERS
for i in $(seq 1 15); do
    alive=0
    for pid in $MASTERS; do sudo kill -0 "$pid" 2>/dev/null && alive=1; done
    [ $alive -eq 0 ] && { echo "==> 已优雅退出。"; exit 0; }
    sleep 1
done

# ---- 5. 强杀 ----
echo "==> 超时，kill -9 ..."
sudo kill -9 $MASTERS
sleep 1
sudo ss -lntp | grep -q ":${PORT}[[:space:]]" \
    && { echo "端口仍被占用："; sudo ss -lntp | grep ":${PORT}[[:space:]]"; exit 1; } \
    || echo "==> 端口已释放，完成。"