#!/usr/bin/env sh
set -eu

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
build_dir="${script_dir}/build"

if [ ! -f "${build_dir}/meson-private/build.dat" ]; then
  meson setup "${build_dir}" "${script_dir}"
fi

meson test --maxfail=1 -C "${build_dir}" --suite ops_reg:ops_reg
