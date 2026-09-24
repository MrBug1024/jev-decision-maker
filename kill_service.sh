#!/bin/bash
# 用: bash kill_service.sh   (不要加 sudo)
set -u

PORT=8020
VENV_MARK="omin-decision-maker"
PROJECT_DIR="$HOME/work/jev-decision-maker/jev_omni_server"
APP_MARK="open_jev_server.server:app"

# ---- 1. 端口反查（只看自己的进程；不是自己的在没 sudo 下也看不到，反而更安全）----
echo "==> 端口反查 $PORT ..."
PIDS=$(ss -lntp 2>/dev/null \
       | awk -v p=":${PORT}\$" '$4 ~ p {print $6}' \
       | grep -oP 'pid=\K[0-9]+' | sort -u)

# ---- 2. 端口没监听就按 venv+cwd 在自己名下反查 ----
if [ -z "$PIDS" ]; then
    echo "端口无监听，按 venv+cwd 在自己名下反查 ..."
    PIDS=$(pgrep -u "$USER" -f "$APP_MARK" 2>/dev/null | while read -r p; do
        exe=$(readlink /proc/$p/exe 2>/dev/null)
        cwd=$(readlink /proc/$p/cwd 2>/dev/null)
        [[ "$exe" == *"$VENV_MARK"* && "$cwd" == "$PROJECT_DIR"* ]] && echo "$p"
    done | sort -u)
fi

[ -z "$PIDS" ] && { echo "未找到目标，退出。"; exit 0; }
echo "==> 候选 PID: $(echo $PIDS | tr '\n' ' ')"

# ---- 3. 三重校验：exe / cwd / cmdline，任一不符就中止 ----
echo "==> 校验身份 ..."
BAD=0
for pid in $PIDS; do
    exe=$(readlink /proc/$pid/exe 2>/dev/null)
    cwd=$(readlink /proc/$pid/cwd 2>/dev/null)
    cmd=$(tr '\0' ' ' < /proc/$pid/cmdline 2>/dev/null)
    ok=1
    [[ "$exe" == *"$VENV_MARK"* ]] || ok=0
    [[ "$cwd" == "$PROJECT_DIR"*  ]] || ok=0
    [[ "$cmd" == *"$APP_MARK"*    ]] || ok=0
    if [ $ok -eq 1 ]; then
        echo "  [OK] pid=$pid exe=$exe cwd=$cwd"
    else
        echo "  [!!] pid=$pid exe=$exe cwd=$cwd cmd=$cmd"
        BAD=1
    fi
done
[ $BAD -eq 1 ] && { echo "有不匹配进程，中止，请人工排查。"; exit 1; }

# ---- 4. 找 master（PPID 不在候选集合里的那个）----
MASTERS=""
for pid in $PIDS; do
    ppid=$(ps -o ppid= -p "$pid" | tr -d ' ')
    echo "$PIDS" | grep -qw "$ppid" || MASTERS="$MASTERS $pid"
done
echo "==> master:$MASTERS"

# ---- 5. 优雅停止 ----
kill -TERM $MASTERS
for i in $(seq 1 15); do
    alive=0
    for pid in $MASTERS; do kill -0 "$pid" 2>/dev/null && alive=1; done
    [ $alive -eq 0 ] && { echo "==> 已优雅退出。"; exit 0; }
    sleep 1
done

# ---- 6. 强杀 ----
echo "==> 超时，kill -9 ..."
kill -9 $MASTERS
sleep 1
if ss -lntp 2>/dev/null | grep -q ":${PORT}[[:space:]]"; then
    echo "端口仍被占用:"; ss -lntp | grep ":${PORT}[[:space:]]"; exit 1
else
    echo "==> 端口已释放，完成。"
fi