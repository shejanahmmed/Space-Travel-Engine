"""NASA/JPL SPK Planetary Ephemeris Loader and Cache Manager.

Manages downloading, local caching, and opening of NASA JPL's development
ephemeris SPK binary kernels (specifically DE440s, covering 1849 to 2150).

Data Source:
- NASA Jet Propulsion Laboratory (JPL) Planetary & Lunar Ephemerides DE440/DE441
  URL: https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets/de440s.bsp
- Reference: Park et al. (2021), "The JPL Planetary and Lunar Ephemerides DE440
  and DE441", The Astronomical Journal, 161:105.
"""

from __future__ import annotations
from pathlib import Path
from typing import Optional
import urllib.request
import shutil

from jplephem.spk import SPK

# Primary and authoritative fallback mirrors for DE440s (~32 MB)
DE440S_MIRRORS: tuple[str, ...] = (
    "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets/de440s.bsp",
    "https://spiftp.esac.esa.int/data/SPICE/generic_kernels/spk/planets/de440s.bsp",
)
DE440S_URL: str = DE440S_MIRRORS[0]

# Expected size of official de440s.bsp in bytes (32.7 MB, 14 segments)
DE440S_EXPECTED_SIZE_BYTES: int = 32_726_016


def get_default_ephemeris_path() -> Path:
    """Return the default local filesystem path for the DE440s kernel.

    Stored under data/ephemeris/ relative to the repository root.
    """
    # Project root is 3 levels up from src/relativistic_engine/ephemeris/
    project_root = Path(__file__).resolve().parent.parent.parent.parent
    data_dir = project_root / "data" / "ephemeris"
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir / "de440s.bsp"


def download_de440s_kernel(
    destination_path: Optional[Path] = None,
    force: bool = False,
) -> Path:
    """Download the NASA/JPL DE440s kernel to local disk with mirror fallbacks.

    Parameters
    ----------
    destination_path : Optional[Path], optional
        Target path on disk. If None, defaults to data/ephemeris/de440s.bsp.
    force : bool, optional
        If True, re-downloads even if the file exists (default: False).

    Returns
    -------
    Path
        Path to the downloaded and verified kernel file.

    Raises
    ------
    RuntimeError
        If the download fails across all authoritative mirrors.
    """
    dest = destination_path or get_default_ephemeris_path()

    if dest.exists() and not force:
        try:
            # Validate existing kernel integrity by opening SPK segments
            SPK.open(str(dest))
            return dest
        except Exception:
            pass

    dest.parent.mkdir(parents=True, exist_ok=True)
    temp_dest = dest.with_suffix(".tmp")

    last_error: Optional[Exception] = None

    for mirror_url in DE440S_MIRRORS:
        for attempt in range(1, 4):
            try:
                req = urllib.request.Request(
                    mirror_url,
                    headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) RelativisticEngine/1.0.0"},
                )
                with urllib.request.urlopen(req, timeout=90) as response, open(temp_dest, "wb") as out_file:
                    shutil.copyfileobj(response, out_file)

                # Validate downloaded kernel size and SPK structure
                if temp_dest.stat().st_size >= 30_000_000:
                    SPK.open(str(temp_dest))
                    temp_dest.replace(dest)
                    return dest
            except Exception as exc:
                last_error = exc
                if temp_dest.exists():
                    try:
                        temp_dest.unlink()
                    except Exception:
                        pass
                import time
                time.sleep(1.5 * attempt)

    raise RuntimeError(
        f"Failed to download DE440s kernel across all mirrors ({DE440S_MIRRORS}): {last_error}"
    ) from last_error


def load_jpl_ephemeris(
    kernel_path: Optional[Path] = None,
    auto_download: bool = True,
) -> SPK:
    """Load the NASA/JPL SPK ephemeris kernel.

    Parameters
    ----------
    kernel_path : Optional[Path], optional
        Path to the .bsp file. If None, defaults to data/ephemeris/de440s.bsp.
    auto_download : bool, optional
        If True, automatically downloads the kernel if not present locally (default: True).

    Returns
    -------
    SPK
        Active jplephem SPK kernel handle.

    Raises
    ------
    FileNotFoundError
        If the kernel is missing and auto_download is False.
    """
    path = kernel_path or get_default_ephemeris_path()

    if not path.exists():
        if auto_download:
            download_de440s_kernel(path)
        else:
            raise FileNotFoundError(
                f"JPL ephemeris kernel not found at {path}. Set auto_download=True to fetch."
            )

    return SPK.open(str(path))
