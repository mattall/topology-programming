#!/usr/bin/env bash
set -u

status=0
warnings=0
python_bin="${PYTHON_BIN:-}"

if [ -z "$python_bin" ]; then
    if [ -x ".venv/bin/python" ]; then
        python_bin=".venv/bin/python"
    else
        python_bin="python"
    fi
fi

check_cmd() {
    name="$1"
    hint="$2"

    if command -v "$name" >/dev/null 2>&1; then
        printf "ok: %s -> %s\n" "$name" "$(command -v "$name")"
    else
        printf "missing: %s\n  %s\n" "$name" "$hint"
        status=1
    fi
}

check_python_import() {
    module="$1"
    hint="$2"

    if "$python_bin" -c "import ${module}" >/dev/null 2>&1; then
        printf "ok: %s import %s\n" "$python_bin" "$module"
    else
        printf "missing: %s import %s\n  %s\n" "$python_bin" "$module" "$hint"
        status=1
    fi
}

check_optional_cmd() {
    name="$1"
    hint="$2"

    if command -v "$name" >/dev/null 2>&1; then
        printf "ok: %s -> %s\n" "$name" "$(command -v "$name")"
    else
        printf "optional: missing %s\n  %s\n" "$name" "$hint"
        warnings=1
    fi
}

check_optional_python_import() {
    module="$1"
    hint="$2"

    if "$python_bin" -c "import ${module}" >/dev/null 2>&1; then
        printf "ok: %s import %s\n" "$python_bin" "$module"
    else
        printf "optional: missing %s import %s\n  %s\n" "$python_bin" "$module" "$hint"
        warnings=1
    fi
}

printf "Checking topology-programming environment...\n"

if command -v "$python_bin" >/dev/null 2>&1 || [ -x "$python_bin" ]; then
    printf "ok: python -> %s\n" "$("$python_bin" -c 'import sys; print(sys.executable)')"
else
    printf "missing: python -> %s\n  Install Python 3.13 and activate the project environment.\n" "$python_bin"
    status=1
fi

if ! "$python_bin" -c 'import sys; raise SystemExit(sys.version_info[:2] < (3, 13))'; then
    printf "unsupported: %s\n  Use Python 3.13 or newer.\n" "$python_bin"
    status=1
fi

check_optional_cmd gurobi_cl "Required for Gurobi-backed topology optimization."
check_optional_python_import gurobipy "Required for Gurobi-backed topology optimization."
check_python_import highspy "Install highspy with pip install highspy."
check_python_import onset "Install the project and its dependencies with pip install -e ."
check_python_import networkx "Install project Python dependencies with pip install ."
check_python_import numpy "Install project Python dependencies with pip install ."
check_python_import scipy "Install project Python dependencies with pip install ."

if [ "$status" -eq 0 ]; then
    if [ "$warnings" -eq 0 ]; then
        printf "Environment check passed.\n"
    else
        printf "Environment check passed with optional components missing.\n"
    fi
else
    printf "Environment check failed.\n"
fi

exit "$status"
