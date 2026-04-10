"""VACASK binary discovery and build utilities."""

import logging
import os
import subprocess
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Project root: vajax/utils/vacask_build.py -> vajax root
_PROJECT_ROOT = Path(__file__).parent.parent.parent

_SEARCH_PATHS = [
    _PROJECT_ROOT / "vendor/VACASK/build/simulator/vacask",
    _PROJECT_ROOT / "vendor/VACASK/build.VACASK/Release/simulator/vacask",
    _PROJECT_ROOT / "vendor/VACASK/build.VACASK/Debug/simulator/vacask",
    _PROJECT_ROOT / "build/vacask/simulator/vacask",  # Legacy local build path
    Path.home() / "bin/vacask",
    Path("/usr/local/bin/vacask"),
]

_BUILD_SCRIPT = _PROJECT_ROOT / "scripts/build_vacask.sh"


def ensure_vacask(build_if_missing: bool = True) -> Optional[Path]:
    """Find VACASK binary, optionally building if not found.

    Search order:
    1. VACASK_BIN environment variable
    2. vendor/VACASK/build/simulator/vacask (CI & build_vacask.sh output)
    3. vendor/VACASK/build.VACASK/{Release,Debug}/simulator/vacask
    4. build/vacask/simulator/vacask (legacy local build path)
    5. ~/bin/vacask, /usr/local/bin/vacask

    Args:
        build_if_missing: If True and binary not found, attempt to build
            via scripts/build_vacask.sh.

    Returns:
        Path to VACASK binary, or None if not found (and not built).
    """
    # Check environment variable first
    if env_path := os.environ.get("VACASK_BIN"):
        path = Path(env_path).resolve()
        if path.exists() and path.is_file():
            return path
        logger.warning("VACASK_BIN set to %s but file not found", env_path)

    # Check common locations
    for path in _SEARCH_PATHS:
        if path.exists() and path.is_file():
            logger.debug("Found VACASK at %s", path)
            return path

    if not build_if_missing:
        return None

    # Attempt to build
    if not _BUILD_SCRIPT.exists():
        logger.warning("Build script not found: %s", _BUILD_SCRIPT)
        return None

    logger.info("VACASK not found, building from source...")
    result = subprocess.run(
        ["bash", str(_BUILD_SCRIPT)],
        capture_output=True,
        text=True,
        timeout=600,
    )

    if result.returncode != 0:
        logger.error("VACASK build failed:\n%s", result.stderr[-2000:])
        return None

    # Check again after build
    for path in _SEARCH_PATHS:
        if path.exists() and path.is_file():
            logger.info("VACASK built successfully: %s", path)
            return path

    logger.error("VACASK build completed but binary not found in search paths")
    return None
