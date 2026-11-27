#!/usr/bin/env bash
# Run a search command with a 64 MiB memory cap, including the static cache.
set -euo pipefail
if [[ $# -eq 0 ]]; then
    echo "usage: $0 COMMAND [ARG ...]" >&2
    exit 2
fi
limit=$((64 * 1024 * 1024))
if [[ -e /sys/fs/cgroup/cgroup.controllers ]]; then
    root=${CGROUP_MEMORY_ROOT:-/sys/fs/cgroup}
    mode=v2
else
    root=${CGROUP_MEMORY_ROOT:-/sys/fs/cgroup/memory}
    mode=v1
fi
group=$(mktemp -d "$root/aisaq-bge.XXXXXX")
cleanup() {
    if [[ $mode == v2 ]]; then
        cat "$group/memory.peak" "$group/memory.events" >&2
    else
        cat "$group/memory.max_usage_in_bytes" "$group/memory.failcnt" >&2
    fi
    rmdir "$group"
}
trap cleanup EXIT
if [[ $mode == v2 ]]; then
    echo "$limit" > "$group/memory.max"
    echo 0 > "$group/memory.swap.max"
else
    echo "$limit" > "$group/memory.limit_in_bytes"
    echo "$limit" > "$group/memory.memsw.limit_in_bytes"
    echo 0 > "$group/memory.swappiness"
fi
(
    echo "$BASHPID" > "$group/cgroup.procs"
    exec "$@"
)
