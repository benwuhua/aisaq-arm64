#!/usr/bin/env bash
set -euo pipefail

here=$(cd -- "$(dirname -- "$0")" && pwd)
repo=${KNOWHERE_ROOT:-$(cd -- "$here/../.." && pwd)}
lib=${KNOWHERE_LIB_DIR:?Set KNOWHERE_LIB_DIR to the directory containing libknowhere.so}
out=${BUILD_DIR:-"$here/build"}
compiler=${CXX:-g++}
mkdir -p "$out"

includes=(-I"$here/src" -I"$here/compat" -I"$repo/include" -I"$repo/src" -I"$repo/thirdparty/DiskANN/include")
IFS=: read -r -a extra <<< "${EXTRA_INCLUDE_DIRS:-}"
for directory in "${extra[@]}"; do
    includes+=("-I$directory")
done
arch=${AISAQ_MARCH:-armv8.2-a}
if [[ -z ${AISAQ_MARCH:-} ]] && grep -qw asimdhp /proc/cpuinfo && grep -qw asimddp /proc/cpuinfo; then
    arch=armv8.2-a+fp16+dotprod
fi
common=(-std=c++17 -O3 -fopenmp "-march=$arch")
uring=()
if [[ ${AISAQ_URING:-auto} != off ]] && command -v pkg-config >/dev/null && pkg-config --exists liburing; then
    read -r -a uring <<< "$(pkg-config --cflags --libs liburing)"
    common+=(-DAISAQ_HAVE_URING)
fi
link=(-L"$lib" -Wl,-rpath,"$lib" -Wl,-rpath-link,"$lib" -lknowhere)
"$compiler" "${common[@]}" "${includes[@]}" "$here/src/runner.cpp" "$here/src/scheduler.cpp" "${link[@]}" -laio "${uring[@]}" -o "$out/search"
"$compiler" "${common[@]}" "$here/src/pack.cpp" -o "$out/pack"
"$compiler" "${common[@]}" "$here/src/pq.cpp" "${link[@]}" -o "$out/pq"
