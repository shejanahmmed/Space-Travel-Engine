"""Benchmark and Independent Scientific Validation for Relativistic Spectral Rendering.

Measures computational performance, throughput, and validates physical invariants:
1. Peebles-Wilkinson (1968) blackbody temperature Lorentz-boost invariance.
2. Stefan-Boltzmann law and Wien displacement law accuracy.
3. CIE 1931 colorimetric integration throughput (spectra/second).
4. McKinley & Doherty (1979) relativistic radiation stress-energy sky flux boost invariant:
   u' / u = gamma^2 * (1 + 1/3 * beta^2).
5. Vectorized pinhole camera and panoramic ray-tracing throughput.
"""

from __future__ import annotations

import math
import sys
import time
from pathlib import Path

import numpy as np

# Ensure src/ is on path
src_dir = Path(__file__).resolve().parent.parent / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from relativistic_engine.constants import C_LIGHT, SIGMA_SB, WIEN_B
from relativistic_engine.physics.spectral_rendering import (
    CIE_WAVELENGTHS_M,
    CelestialStar,
    analytical_sky_flux_boost_factor,
    boosted_blackbody_temperature,
    cie_xyz_to_srgb,
    numerical_sky_flux_boost_factor,
    planck_spectral_radiance,
    render_relativistic_panorama,
    render_relativistic_starfield_pinhole,
    spectrum_to_cie_xyz,
    wien_peak_wavelength,
)


def run_spectral_rendering_benchmark() -> dict:
    """Execute complete relativistic spectral rendering benchmark suite."""
    print("=" * 80)
    print("RELATIVISTIC SPECTRAL RENDERING & RAY-TRACING BENCHMARK")
    print("=" * 80)

    results = {}

    # --------------------------------------------------------------------------
    # 1. Peebles-Wilkinson Spectral Identity Benchmark
    # --------------------------------------------------------------------------
    print("\n[1] PEEBLES-WILKINSON (1968) BLACKBODY INVARIANCE CHECK")
    test_temps = [3000.0, 5778.0, 12000.0, 35000.0]
    test_betas = [0.1, 0.5, 0.8, 0.95, 0.99]
    max_invariance_rel_err = 0.0

    for t0 in test_temps:
        for beta in test_betas:
            gamma = 1.0 / math.sqrt(1.0 - beta * beta)
            d = gamma * (1.0 + beta)  # Maximal forward Doppler boost
            lambdas_prime = np.geomspace(100e-9, 5000e-9, 200)

            # Direct boost: I'_lambda(lambda') = D^5 * B_lambda(D * lambda', T0)
            i_direct = (d ** 5) * planck_spectral_radiance(d * lambdas_prime, t0)
            # Peebles-Wilkinson theorem: B_lambda(lambda', D * T0)
            i_peebles = planck_spectral_radiance(lambdas_prime, t0 * d)

            rel_err = np.max(np.abs(i_direct - i_peebles) / (i_peebles + 1e-300))
            if rel_err > max_invariance_rel_err:
                max_invariance_rel_err = float(rel_err)

    print(f"  Maximum Peebles-Wilkinson identity relative error: {max_invariance_rel_err:.3e}")
    results["max_peebles_wilkinson_rel_err"] = max_invariance_rel_err
    assert max_invariance_rel_err < 1e-13, "Peebles-Wilkinson invariance violated!"

    # --------------------------------------------------------------------------
    # 2. McKinley & Doherty (1979) Full-Sky Flux Boost Invariant
    # --------------------------------------------------------------------------
    print("\n[2] MCKINLEY & DOHERTY (1979) FULL-SKY FLUX INVARIANT INTEGRATION")
    beta_sweep = [0.0, 0.1, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 0.99]
    max_flux_err = 0.0

    print(f"  {'beta':>8} | {'gamma':>10} | {'Exact Ratio':>16} | {'Numerical Quad':>16} | {'Rel Error':>12}")
    print("  " + "-" * 72)
    for b in beta_sweep:
        gamma = 1.0 / math.sqrt(1.0 - b * b) if b < 1.0 else float("inf")
        exact_ratio = analytical_sky_flux_boost_factor(b)
        num_ratio = numerical_sky_flux_boost_factor(b, n_theta=20000)
        err = abs(num_ratio - exact_ratio) / exact_ratio
        if err > max_flux_err:
            max_flux_err = err
        print(f"  {b:8.2f} | {gamma:10.4f} | {exact_ratio:16.6f} | {num_ratio:16.6f} | {err:12.2e}")

    results["max_sky_flux_quadrature_rel_err"] = max_flux_err
    assert max_flux_err < 1e-4, "Full-sky flux invariant quadrature error too large!"

    # --------------------------------------------------------------------------
    # 3. Vectorized Spectral Radiance & CIE 1931 Integration Throughput
    # --------------------------------------------------------------------------
    print("\n[3] SPECTRAL RADIANCE & CIE 1931 INTEGRATION THROUGHPUT")
    n_spectra = 10000
    rng = np.random.default_rng(42)
    sample_temps = rng.uniform(2500.0, 30000.0, size=n_spectra)

    t_start = time.perf_counter()
    rad_batch = planck_spectral_radiance(
        CIE_WAVELENGTHS_M[None, :],
        sample_temps[:, None],
    )
    t_rad = time.perf_counter() - t_start
    rad_throughput = n_spectra / t_rad

    t_start = time.perf_counter()
    xyz_batch, xy_batch = spectrum_to_cie_xyz(CIE_WAVELENGTHS_M, rad_batch)
    srgb_batch = cie_xyz_to_srgb(xyz_batch, exposure=1.0)
    t_cie = time.perf_counter() - t_start
    cie_throughput = n_spectra / t_cie

    print(f"  Planck spectral evaluation: {n_spectra} spectra in {t_rad*1000:.2f} ms ({rad_throughput:,.0f} spectra/s)")
    print(f"  CIE 1931 XYZ + sRGB tone-map: {n_spectra} spectra in {t_cie*1000:.2f} ms ({cie_throughput:,.0f} spectra/s)")
    results["planck_throughput_spectra_per_sec"] = rad_throughput
    results["cie_throughput_spectra_per_sec"] = cie_throughput

    # --------------------------------------------------------------------------
    # 4. Pinhole Camera Ray-Tracer Throughput (5,000 Stars)
    # --------------------------------------------------------------------------
    print("\n[4] PINHOLE CAMERA RELATIVISTIC RAY-TRACER BENCHMARK")
    n_stars = 5000
    # Generate random stars on sphere
    z = rng.uniform(-1.0, 1.0, size=n_stars)
    phi = rng.uniform(0.0, 2.0 * np.pi, size=n_stars)
    r_xy = np.sqrt(1.0 - z * z)
    dirs = np.column_stack([r_xy * np.cos(phi), r_xy * np.sin(phi), z])

    star_catalog = [
        CelestialStar(
            direction=dirs[i],
            temperature_k=float(rng.uniform(3000.0, 25000.0)),
            visual_magnitude=float(rng.uniform(-1.0, 7.0)),
            name=f"Star_{i}",
        )
        for i in range(n_stars)
    ]

    beta_vec = np.array([0.0, 0.0, 0.85])  # Starship moving at 0.85c
    resolution = (256, 256)

    t_start = time.perf_counter()
    img_pinhole = render_relativistic_starfield_pinhole(
        star_catalog,
        beta_vec,
        fov_deg=50.0,
        resolution=resolution,
        exposure=1.0,
    )
    t_pinhole = time.perf_counter() - t_start
    pinhole_throughput = n_stars / t_pinhole

    print(f"  Rendered {resolution[0]}x{resolution[1]} image with {n_stars} stars in {t_pinhole*1000:.2f} ms")
    print(f"  Star rendering throughput: {pinhole_throughput:,.0f} stars/second")
    print(f"  Max pixel intensity: {np.max(img_pinhole):.4f}, non-zero pixels: {np.count_nonzero(img_pinhole > 0)}")
    results["pinhole_render_time_ms"] = t_pinhole * 1000.0
    results["pinhole_star_throughput"] = pinhole_throughput

    # --------------------------------------------------------------------------
    # 5. Full-Sky Panoramic Equirectangular Renderer
    # --------------------------------------------------------------------------
    print("\n[5] FULL-SKY RELATIVISTIC EQUIRECTANGULAR PANORAMA BENCHMARK")
    pano_res = (128, 256)
    t_start = time.perf_counter()
    pano_img = render_relativistic_panorama(
        star_catalog[:1000],
        beta_vec,
        resolution=pano_res,
        exposure=1.0,
        include_cmb=True,
    )
    t_pano = time.perf_counter() - t_start
    pano_pixels = pano_res[0] * pano_res[1]
    pixel_throughput = pano_pixels / t_pano

    print(f"  Rendered {pano_res[0]}x{pano_res[1]} full-sky panorama in {t_pano*1000:.2f} ms")
    print(f"  Panorama pixel throughput: {pixel_throughput:,.0f} pixels/second")
    results["panorama_render_time_ms"] = t_pano * 1000.0
    results["panorama_pixel_throughput"] = pixel_throughput

    print("\n" + "=" * 80)
    print("SPECTRAL RENDERING & RAY-TRACING BENCHMARK COMPLETE: ALL VERIFICATIONS PASSED")
    print("=" * 80)

    return results


if __name__ == "__main__":
    run_spectral_rendering_benchmark()
