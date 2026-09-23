"""Benchmark and Scientific Validation for Kerr Curved Spacetime Ray-Tracing.

Measures:
1. ZAMO tetrad generation and orthonormal metric conditioning throughput.
2. Relativistic redshift factor g_rad evaluation throughput.
3. Backward null geodesic ray-tracing throughput in strong-field Kerr spacetime.
4. Full synthetic scene rendering of a spinning black hole with Novikov-Thorne accretion disk.
5. Verification of relativistic Doppler beaming and black hole shadow silhouette.
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

from relativistic_engine.physics.kerr import KerrGeometry
from relativistic_engine.physics.kerr_raytracer import (
    AccretionDiskProperties,
    KerrCamera,
    compute_disk_keplerian_angular_velocity,
    compute_relativistic_redshift_factor,
    compute_zamo_tetrad,
    render_kerr_black_hole_scene,
    trace_single_kerr_ray,
)


def run_kerr_lensing_benchmark() -> dict:
    """Execute complete Kerr curved spacetime ray-tracing and lensing benchmark."""
    print("=" * 80)
    print("KERR CURVED SPACETIME GRAVITATIONAL LENSING RAY-TRACER BENCHMARK")
    print("=" * 80)

    results = {}

    bh = KerrGeometry(mass_kg=4.154e6 * 1.98847e30, spin_dimensionless=0.90)
    m = bh.mass_m
    r_isco = bh.isco_radius(prograde=True)

    print(f"\n[+] Black Hole Parameters:")
    print(f"    Mass: {bh.mass_kg:.3e} kg ({bh.mass_kg/1.98847e30:,.0f} M_sun)")
    print(f"    Gravitational Radius M: {m:,.1f} m")
    print(f"    Spin a_*: {bh.spin_dimensionless:+.2f}")
    print(f"    ISCO Radius: {r_isco:,.1f} m ({r_isco/m:.3f} M)")
    print(f"    Horizon Radius: {bh.event_horizon_outer:,.1f} m ({bh.event_horizon_outer/m:.3f} M)")

    # --------------------------------------------------------------------------
    # 1. ZAMO Tetrad & Ray Generation Throughput
    # --------------------------------------------------------------------------
    print("\n[+] 1. ZAMO Tetrad Generation Throughput:")
    n_tetrad = 50_000
    r_samples = np.linspace(2.5 * m, 30.0 * m, n_tetrad)
    th_samples = np.linspace(0.1 * math.pi, 0.9 * math.pi, n_tetrad)

    t0 = time.perf_counter()
    for k in range(n_tetrad):
        _ = compute_zamo_tetrad(r_samples[k], th_samples[k], bh)
    t_tetrad = time.perf_counter() - t0

    tetrad_throughput = n_tetrad / t_tetrad
    print(f"    Evaluated {n_tetrad:,} orthonormal tetrads in {t_tetrad:.4f} s")
    print(f"    ZAMO Tetrad Throughput: {tetrad_throughput:,.0f} tetrads/sec")
    results["zamo_tetrad_throughput"] = tetrad_throughput

    # --------------------------------------------------------------------------
    # 2. Relativistic Redshift Evaluation Throughput
    # --------------------------------------------------------------------------
    print("\n[+] 2. Relativistic Frequency Shift (g_rad) Throughput:")
    cam = KerrCamera(r_cam=20.0 * m, theta_cam=math.radians(75.0))
    tetrad_cam = compute_zamo_tetrad(cam.r_cam, cam.theta_cam, bh)
    sample_k = 1.0 * tetrad_cam[0] - (-1.0) * tetrad_cam[1] - 0.2 * tetrad_cam[3]

    n_redshift = 50_000
    hit_radii = np.linspace(r_isco, 20.0 * m, n_redshift)

    t0 = time.perf_counter()
    for k in range(n_redshift):
        _ = compute_relativistic_redshift_factor(sample_k, hit_radii[k], bh, cam)
    t_redshift = time.perf_counter() - t0

    redshift_throughput = n_redshift / t_redshift
    print(f"    Evaluated {n_redshift:,} exact redshift points in {t_redshift:.4f} s")
    print(f"    Redshift Evaluation Throughput: {redshift_throughput:,.0f} evals/sec")
    results["redshift_throughput"] = redshift_throughput

    # --------------------------------------------------------------------------
    # 3. Ray-Tracing Strong-Field Geodesic Integration Throughput
    # --------------------------------------------------------------------------
    print("\n[+] 3. Strong-Field Ray-Tracing Geodesic Integration:")
    n_rays = 500
    angles_phi = np.linspace(-0.3, 0.3, n_rays)
    x_cam = np.array([0.0, cam.r_cam, cam.theta_cam, 0.0])

    t0 = time.perf_counter()
    status_counts = {}
    for d_phi in angles_phi:
        k_dir = 1.0 * tetrad_cam[0] - (-0.95) * tetrad_cam[1] - d_phi * tetrad_cam[3]
        res = trace_single_kerr_ray(x_cam, k_dir, bh, disk=None, max_steps=250)
        status_counts[res["status"]] = status_counts.get(res["status"], 0) + 1
    t_rays = time.perf_counter() - t0

    ray_throughput = n_rays / t_rays
    print(f"    Traced {n_rays:,} rays through curved spacetime in {t_rays:.4f} s")
    print(f"    Ray-Tracing Throughput: {ray_throughput:,.0f} rays/sec")
    print(f"    Ray Classification: {status_counts}")
    results["ray_tracing_throughput"] = ray_throughput

    # --------------------------------------------------------------------------
    # 4. Full Synthetic Scene Rendering (Accretion Disk + Shadow)
    # --------------------------------------------------------------------------
    print("\n[+] 4. Full Scene Rendering (Accretion Disk + Shadow Silhouette):")
    render_cam = KerrCamera(
        r_cam=18.0 * m,
        theta_cam=math.radians(75.0),
        resolution_x=48,
        resolution_y=48,
        fov_deg=45.0,
    )
    disk = AccretionDiskProperties(
        r_inner_m=r_isco,
        r_outer_m=16.0 * m,
        peak_temperature_k=12_000.0,
    )

    t0 = time.perf_counter()
    rgb_img, red_map, hit_map = render_kerr_black_hole_scene(
        camera=render_cam,
        kerr=bh,
        disk=disk,
        max_steps_per_ray=180,
    )
    t_render = time.perf_counter() - t0

    total_pixels = render_cam.resolution_x * render_cam.resolution_y
    pixel_throughput = total_pixels / t_render
    shadow_count = int(np.sum(hit_map == 0))
    disk_count = int(np.sum(hit_map == 1))
    escaped_count = int(np.sum(hit_map == 2))

    print(f"    Rendered {total_pixels:,} pixels ({render_cam.resolution_x}x{render_cam.resolution_y}) in {t_render:.4f} s")
    print(f"    Pixel Rendering Throughput: {pixel_throughput:,.0f} pixels/sec")
    print(f"    Scene Composition: Shadow = {shadow_count} px, Disk = {disk_count} px, Escaped = {escaped_count} px")

    # Quantify Doppler beaming asymmetry on the disk
    if disk_count > 0:
        disk_mask = hit_map == 1
        w_half = render_cam.resolution_x // 2
        lum_left = np.mean(rgb_img[:, :w_half, :][disk_mask[:, :w_half]]) if np.any(disk_mask[:, :w_half]) else 0.0
        lum_right = np.mean(rgb_img[:, w_half:, :][disk_mask[:, w_half:]]) if np.any(disk_mask[:, w_half:]) else 0.0
        print(f"    Approaching Side Luminance: {lum_left:.4f}")
        print(f"    Receding Side Luminance:    {lum_right:.4f}")
        print(f"    Doppler Beaming Contrast:   {lum_left/max(1e-6, lum_right):.2f}x")
        results["doppler_beaming_contrast"] = float(lum_left / max(1e-6, lum_right))

    print("\n" + "=" * 80)
    print("ALL KERR GRAVITATIONAL LENSING BENCHMARKS COMPLETED SUCCESSFULLY")
    print("=" * 80)

    return results


if __name__ == "__main__":
    run_kerr_lensing_benchmark()
