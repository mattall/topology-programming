#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
switch_name="${YATES_OPAM_SWITCH:-yates}"
build_root="${YATES_BUILD_DIR:-${repo_root}/.cache/yates-build}"
frenetic_commit="fdce6cc51a438585e8c6f54e9d2c22de60f9a6c6"
frenetic_dir="${build_root}/frenetic-${frenetic_commit}"
frenetic_patch="${repo_root}/scripts/patches/frenetic-tcpip-checksum.patch"

require_command() {
    if ! command -v "$1" >/dev/null 2>&1; then
        printf "error: missing required command: %s\n" "$1" >&2
        exit 1
    fi
}

require_command git
require_command make
require_command opam

opam_version="$(opam --version)"
opam_major="${opam_version%%.*}"
opam_rest="${opam_version#*.}"
opam_minor="${opam_rest%%.*}"
if [ "$opam_major" -lt 2 ] || { [ "$opam_major" -eq 2 ] && [ "$opam_minor" -lt 2 ]; }; then
    printf "error: opam 2.2 or newer is required (found %s)\n" "$opam_version" >&2
    exit 1
fi

if ! command -v pkgconf >/dev/null 2>&1 && ! command -v pkg-config >/dev/null 2>&1; then
    printf "error: pkgconf is required (for Homebrew: brew install pkgconf)\n" >&2
    exit 1
fi

git -C "$repo_root" submodule update --init --recursive external/yates

if ! opam switch list --short | grep -Fxq "$switch_name"; then
    opam switch create "$switch_name" ocaml-base-compiler.4.12.0 -y
fi

mkdir -p "$build_root"
if [ ! -d "${frenetic_dir}/.git" ]; then
    git clone https://github.com/frenetic-lang/frenetic.git "$frenetic_dir"
    git -C "$frenetic_dir" checkout --detach "$frenetic_commit"
fi

if ! git -C "$frenetic_dir" merge-base --is-ancestor "$frenetic_commit" HEAD; then
    printf "error: unexpected Frenetic checkout at %s\n" "$frenetic_dir" >&2
    exit 1
fi

if ! grep -Fq "tcpip tcpip.checksum" "${frenetic_dir}/src/lib/kernel/dune"; then
    git -C "$frenetic_dir" apply "$frenetic_patch"
    git -C "$frenetic_dir" \
        -c user.name="Topology Programming Setup" \
        -c user.email="noreply@localhost" \
        commit -am "Restore tcpip checksum dependency for tcpip 7"
fi

opam pin add --switch="$switch_name" -n -y frenetic "$frenetic_dir"
opam install --switch="$switch_name" -y \
    core.v0.14.1 \
    async.v0.14.0 \
    ppx_jane.v0.14.0 \
    tcpip.7.1.2 \
    frenetic

opam exec --switch="$switch_name" -- make -C "${repo_root}/external/yates"
opam exec --switch="$switch_name" -- make -C "${repo_root}/external/yates" install

yates_bin="$(opam exec --switch="$switch_name" -- which yates)"
"$yates_bin" -help >/dev/null

printf "YATES installed successfully: %s\n" "$yates_bin"
printf "For this shell, run:\n  export YATES_BIN=%s\n" "$yates_bin"
