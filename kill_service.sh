#!/usr/bin/env bash
# Stop only the two services owned by this repository.
# Usage: bash kill_service.sh [--dry-run]

set -Eeuo pipefail

if [[ "$(id -u)" -eq 0 ]]; then
    printf '%s\n' "Refusing to run as root. Run this script as the service owner." >&2
    exit 2
fi

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
CURRENT_UID="$(id -u)"
DRY_RUN=0

case "${1:-}" in
    "") ;;
    --dry-run) DRY_RUN=1 ;;
    *)
        printf 'Usage: %s [--dry-run]\n' "$(basename "$0")" >&2
        exit 2
        ;;
esac

SERVICE_DIRS=(
    "$PROJECT_ROOT/open_jev_server"
    "$PROJECT_ROOT/jev_omni_server"
)

read_cmdline() {
    local pid=$1
    tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null || true
}

process_uid() {
    local pid=$1
    awk '/^Uid:/{print $2; exit}' "/proc/$pid/status" 2>/dev/null || true
}

process_cwd() {
    local pid=$1
    readlink -f "/proc/$pid/cwd" 2>/dev/null || true
}

service_name_for_pid() {
    local pid=$1
    local cwd cmd service name

    [[ -r "/proc/$pid/cmdline" ]] || return 1
    [[ "$(process_uid "$pid")" == "$CURRENT_UID" ]] || return 1

    cwd="$(process_cwd "$pid")"
    cmd="$(read_cmdline "$pid")"
    [[ -n "$cwd" && -n "$cmd" ]] || return 1

    for service in "${SERVICE_DIRS[@]}"; do
        name="${service##*/}"

        # Direct execution from the service directory, for example:
        #   python server.py
        if [[ "$cwd" == "$service" ]] && {
            [[ "$cmd" == *"server.py"* ]] ||
            [[ "$cmd" == *"server:app"* ]] ||
            [[ "$cmd" == *"$name.server:app"* ]]
        }; then
            printf '%s\n' "$name"
            return 0
        fi

        # Module/absolute-path execution from the repository root, for example:
        #   python -m uvicorn jev_omni_server.server:app
        if [[ "$cwd" == "$PROJECT_ROOT" ]] && {
            [[ "$cmd" == *"$name.server:app"* ]] ||
            [[ "$cmd" == *"$service/server.py"* ]]
        }; then
            printf '%s\n' "$name"
            return 0
        fi
    done

    return 1
}

collect_targets() {
    local proc pid name

    for proc in /proc/[0-9]*; do
        pid="${proc##*/}"
        name="$(service_name_for_pid "$pid" || true)"
        [[ -n "$name" ]] && printf '%s|%s\n' "$pid" "$name"
    done | sort -t '|' -k1,1n -u
}

print_target() {
    local record=$1
    local pid=${record%%|*}
    local name=${record##*|}
    printf '  [%s] pid=%s cwd=%s cmd=%s\n' \
        "$name" "$pid" "$(process_cwd "$pid")" "$(read_cmdline "$pid")"
}

is_running_target() {
    local pid=$1
    [[ -d "/proc/$pid" ]] || return 1
    [[ "$(ps -o stat= -p "$pid" 2>/dev/null | tr -d ' ')" != Z* ]] || return 1
    service_name_for_pid "$pid" >/dev/null
}

mapfile -t targets < <(collect_targets)

if ((${#targets[@]} == 0)); then
    printf '%s\n' "No jev-decision-maker service process found. No other process was touched."
    exit 0
fi

if ((DRY_RUN)); then
    printf '%s\n' "Managed processes that would be stopped:"
    for record in "${targets[@]}"; do
        print_target "$record"
    done
    exit 0
fi

printf '%s\n' "Stopping managed jev-decision-maker services:"
for record in "${targets[@]}"; do
    print_target "$record"
done

for record in "${targets[@]}"; do
    pid=${record%%|*}
    if is_running_target "$pid"; then
        kill -TERM "$pid" 2>/dev/null || true
    fi
done

for _ in {1..20}; do
    mapfile -t current_targets < <(collect_targets)
    running=0

    # Catch a same-repository worker or supervisor child that appeared after
    # the first scan, while rechecking its identity before every signal.
    for record in "${current_targets[@]}"; do
        pid=${record%%|*}
        if is_running_target "$pid"; then
            running=1
            kill -TERM "$pid" 2>/dev/null || true
        fi
    done

    ((running == 0)) && {
        printf '%s\n' "Managed services stopped gracefully. No unrelated process was touched."
        exit 0
    }
    sleep 1
done

mapfile -t current_targets < <(collect_targets)
if ((${#current_targets[@]} > 0)); then
    printf '%s\n' "Graceful stop timed out; force-stopping only still-validated targets:" >&2
    for record in "${current_targets[@]}"; do
        print_target "$record" >&2
        pid=${record%%|*}
        is_running_target "$pid" && kill -KILL "$pid" 2>/dev/null || true
    done
fi

sleep 1
mapfile -t remaining < <(collect_targets)
if ((${#remaining[@]} > 0)); then
    printf '%s\n' "Some managed processes remain; no other process was killed:" >&2
    for record in "${remaining[@]}"; do
        print_target "$record" >&2
    done
    exit 1
fi

printf '%s\n' "Managed services stopped. No unrelated process was touched."
