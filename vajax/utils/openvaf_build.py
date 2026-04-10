"""OpenVAF-r binary discovery and build utilities."""

import logging
import os
import shutil
import subprocess
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Project root: vajax/utils/openvaf_build.py -> vajax root
_PROJECT_ROOT = Path(__file__).parent.parent.parent

_SEARCH_PATHS = [
    _PROJECT_ROOT / "vendor/OpenVAF/target/release/openvaf-r",
    Path.home() / "bin/openvaf-r",
    Path("/usr/local/bin/openvaf-r"),
]

_BUILD_SCRIPT = _PROJECT_ROOT / "scripts/build_openvaf.sh"


def ensure_openvaf(build_if_missing: bool = True) -> Optional[Path]:
    """Find openvaf-r binary, optionally building if not found.

    Search order:
    1. OPENVAF_DIR environment variable (set in CI)
    2. vendor/OpenVAF/target/release/openvaf-r (local build)
    3. ~/bin/openvaf-r, /usr/local/bin/openvaf-r
    4. PATH lookup

    Args:
        build_if_missing: If True and binary not found, attempt to build
            via scripts/build_openvaf.sh (requires LLVM 18+ and Rust).

    Returns:
        Path to openvaf-r binary, or None if not found (and not built).
    """
    # Check OPENVAF_DIR env var first (set in CI)
    if openvaf_dir := os.environ.get("OPENVAF_DIR"):
        path = Path(openvaf_dir) / "openvaf-r"
        if path.exists() and path.is_file():
            return path
        logger.warning("OPENVAF_DIR set to %s but openvaf-r not found there", openvaf_dir)

    # Check common locations
    for path in _SEARCH_PATHS:
        if path.exists() and path.is_file():
            logger.debug("Found openvaf-r at %s", path)
            return path

    # Check PATH
    path_bin = shutil.which("openvaf-r")
    if path_bin:
        logger.debug("Found openvaf-r on PATH: %s", path_bin)
        return Path(path_bin)

    if not build_if_missing:
        return None

    # Attempt to build
    if not _BUILD_SCRIPT.exists():
        logger.warning("Build script not found: %s", _BUILD_SCRIPT)
        return None

    logger.info("openvaf-r not found, building from source (requires LLVM 18+ and Rust)...")
    result = subprocess.run(
        ["bash", str(_BUILD_SCRIPT)],
        capture_output=True,
        text=True,
        timeout=600,
    )

    if result.returncode != 0:
        logger.error("openvaf-r build failed:\n%s", result.stderr[-2000:])
        return None

    # Check again after build
    for path in _SEARCH_PATHS:
        if path.exists() and path.is_file():
            logger.info("openvaf-r built successfully: %s", path)
            return path

    logger.error("openvaf-r build completed but binary not found in search paths")
    return None
