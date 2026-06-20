#!/usr/bin/env bash
set -u

status=0
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

printf "Checking topology-programming environment...\n"

if command -v "$python_bin" >/dev/null 2>&1 || [ -x "$python_bin" ]; then
    printf "ok: python -> %s\n" "$("$python_bin" -c 'import sys; print(sys.executable)')"
else
    printf "missing: python -> %s\n  Install Python 3.8+ and activate the project environment.\n" "$python_bin"
    status=1
fi

if [ -n "${YATES_BIN:-}" ]; then
    if [ -x "$YATES_BIN" ]; then
        printf "ok: YATES_BIN -> %s\n" "$YATES_BIN"
    else
        printf "missing: YATES_BIN is set but not executable -> %s\n" "$YATES_BIN"
        status=1
    fi
else
    check_cmd yates "Install cornell-netlab/yates or set YATES_BIN=/absolute/path/to/yates."
fi

check_cmd gurobi_cl "Install Gurobi and ensure gurobi_cl is on PATH."
check_python_import gurobipy "Install gurobipy and ensure your Gurobi license is configured."
check_python_import networkx "Install project Python dependencies with pip install ."
check_python_import numpy "Install project Python dependencies with pip install ."

if [ "$status" -eq 0 ]; then
    printf "Environment check passed.\n"
else
    printf "Environment check failed.\n"
fi

exit "$status"
