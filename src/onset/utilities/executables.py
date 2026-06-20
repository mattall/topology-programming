import os
import shutil


def resolve_executable(env_var: str, default_name: str, install_hint: str) -> str:
    """Resolve an executable from an env var or PATH, raising a clear error."""
    configured_path = os.environ.get(env_var, "").strip()
    executable = configured_path or default_name

    if os.path.isabs(executable) or os.sep in executable:
        if os.path.isfile(executable) and os.access(executable, os.X_OK):
            return executable
    else:
        resolved = shutil.which(executable)
        if resolved:
            return resolved

    raise RuntimeError(
        f"Unable to find executable '{executable}'. Install it or set "
        f"{env_var}=/absolute/path/to/{default_name}. {install_hint}"
    )


def resolve_yates_executable() -> str:
    return resolve_executable(
        "YATES_BIN",
        "yates",
        "YATES is available at https://github.com/cornell-netlab/yates.",
    )

