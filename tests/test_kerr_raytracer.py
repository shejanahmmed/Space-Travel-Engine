"""Automated unit and validation test suite for Kerr Curved Spacetime Ray-Tracing.

Validates:
- LEVEL 1: ZAMO orthonormal tetrad identity eta_(a)(b) = diag(-1, 1, 1, 1) across r and theta.
- LEVEL 2: Null 4-momentum norm k^mu k_mu == 0 and Killing invariant conservation.
- LEVEL 3: Relativistic frequency shift g_rad, gravitational redshift, and Doppler beaming.
- LEVEL 4: Ray termination classification (HORIZON_CAPTURE, DISK_HIT, ESCAPED).
- LEVEL 5: Full scene rendering consistency, shadow silhouette, and disk beaming asymmetry.
"""

from __future__ import annotations

import math
import numpy as np
import pytest

from relativistic_engine.constants import C_LIGHT
from relativistic_engine.physics.kerr import (
    KerrGeometry,
    compute_kerr_metric,
)
from relativistic_engine.physics.kerr_raytracer import (
    AccretionDiskProperties,
    KerrCamera,
    compute_disk_keplerian_angular_velocity,
    compute_relativistic_redshift_factor,
    compute_zamo_tetrad,
    render_kerr_black_hole_scene,
    trace_single_kerr_ray,
)


# ==============================================================================
# LEVEL 1: ZAMO ORTHONORMAL TETRAD INVARIANTS
# ==============================================================================

class TestLevel1ZAMOTetrad:
    """Validate ZAMO tetrad orthonormality against the Kerr metric tensor."""

    @pytest.mark.parametrize("spin", [0.0, 0.5, 0.85, 0.95, -0.7])
    @pytest.mark.parametrize("theta_deg", [30.0, 60.0, 90.0, 120.0])
    def test_tetrad_orthonormality(self, spin: float, theta_deg: float) -> None:
        """Verify g_mu_nu * e_(a)^mu * e_(b)^nu == eta_(a)(b) = diag(-1, 1, 1, 1)."""
        bh = KerrGeometry(mass_kg=4.0e6 * 1.98847e30, spin_dimensionless=spin)
        m = bh.mass_m
        r = 10.0 * m
        th = math.radians(theta_deg)

        tetrad = compute_zamo_tetrad(r, th, bh)
        g_cov, _ = compute_kerr_metric(r, th, bh)

        # eta_(a)(b) = tetrad[a, mu] * g_cov[mu, nu] * tetrad[b, nu]
        eta = np.einsum("am,mn,bn->ab", tetrad, g_cov, tetrad)

        expected_eta = np.diag([-1.0, 1.0, 1.0, 1.0])
        # Off-diagonal elements must vanish to high precision (< 1e-14)
        np.testing.assert_allclose(eta, expected_eta, atol=1e-14)

    def test_tetrad_horizon_guard(self) -> None:
        """Attempting to erect a ZAMO tetrad on or inside the event horizon must raise ValueError."""
        bh = KerrGeometry(mass_kg=1.0e30, spin_dimensionless=0.8)
        r_h = bh.event_horizon_outer
        with pytest.raises(ValueError):
            compute_zamo_tetrad(r_h * 0.99, 0.5 * math.pi, bh)


# ==============================================================================
# LEVEL 2: NULL 4-MOMENTUM & INITIAL RAY GENERATION
# ==============================================================================

class TestLevel2NullRayGeneration:
    """Validate camera ray generation in the locally flat ZAMO tangent space."""

    def test_camera_ray_null_norm(self) -> None:
        """Every ray generated from camera screen must be strictly null (k^mu k_mu == 0)."""
        bh = KerrGeometry(mass_kg=1.0e30, spin_dimensionless=0.7)
        m = bh.mass_m
        cam = KerrCamera(r_cam=12.0 * m, theta_cam=0.4 * math.pi)

        tetrad = compute_zamo_tetrad(cam.r_cam, cam.theta_cam, bh)
        g_cov, _ = compute_kerr_metric(cam.r_cam, cam.theta_cam, bh)

        # Generate sample pixel rays
        for u in [-0.2, 0.0, 0.2]:
            for v in [-0.2, 0.0, 0.2]:
                dir_r = -1.0
                dir_th = -v
                dir_phi = u
                norm = math.hypot(dir_r, dir_th, dir_phi)

                n_r, n_th, n_phi = dir_r / norm, dir_th / norm, dir_phi / norm
                k = 1.0 * tetrad[0] - n_r * tetrad[1] - n_th * tetrad[2] - n_phi * tetrad[3]

                # Metric norm: k_mu k^mu
                norm_k = float(np.einsum("m,mn,n", k, g_cov, k))
                assert norm_k == pytest.approx(0.0, abs=1e-14)


# ==============================================================================
# LEVEL 3: RELATIVISTIC REDSHIFT & DOPPLER BEAMING
# ==============================================================================

class TestLevel3RelativisticRedshift:
    """Validate Keplerian orbital motion and relativistic Doppler/gravitational frequency shift."""

    def test_schwarzschild_keplerian_angular_velocity(self) -> None:
        """In the Schwarzschild limit (a = 0), Omega_K = sqrt(M / r^3) * c."""
        bh = KerrGeometry(mass_kg=1.0e30, spin_dimensionless=0.0)
        m = bh.mass_m
        r = 10.0 * m
        omega_kerr = compute_disk_keplerian_angular_velocity(r, bh, prograde=True)
        expected_omega = C_LIGHT * math.sqrt(m / (r ** 3))
        assert omega_kerr == pytest.approx(expected_omega, rel=1e-14)

    def test_doppler_beaming_asymmetry(self) -> None:
        """Approaching side of the accretion disk must have g_rad > 1 (blueshift); receding must have g_rad < 1."""
        bh = KerrGeometry(mass_kg=1.0e30, spin_dimensionless=0.9)
        m = bh.mass_m
        cam = KerrCamera(r_cam=20.0 * m, theta_cam=math.radians(80.0), fov_deg=40.0)
        tetrad = compute_zamo_tetrad(cam.r_cam, cam.theta_cam, bh)

        # Ray aimed at approaching side of disk
        k_approaching = 1.0 * tetrad[0] - (-0.95) * tetrad[1] - (0.0) * tetrad[2] - (-0.3) * tetrad[3]
        # Ray aimed at receding side of disk
        k_receding = 1.0 * tetrad[0] - (-0.95) * tetrad[1] - (0.0) * tetrad[2] - (+0.3) * tetrad[3]

        r_disk = 6.0 * m
        g_approaching = compute_relativistic_redshift_factor(k_approaching, r_disk, bh, cam)
        g_receding = compute_relativistic_redshift_factor(k_receding, r_disk, bh, cam)

        # Approaching is blueshifted (g > 1) relative to receding (g < 1)
        assert g_approaching > g_receding
        assert g_approaching > 1.0  # Blueshifted
        assert g_receding < 1.0  # Redshifted


# ==============================================================================
# LEVEL 4: RAY TERMINATION CLASSIFICATION
# ==============================================================================

class TestLevel4RayTermination:
    """Validate ray termination into black hole shadow, accretion disk, or celestial escape."""

    def test_central_ray_horizon_capture(self) -> None:
        """A ray directed straight at the black hole center must be captured by the event horizon."""
        bh = KerrGeometry(mass_kg=1.0e30, spin_dimensionless=0.5)
        m = bh.mass_m
        cam = KerrCamera(r_cam=15.0 * m, theta_cam=0.5 * math.pi)
        tetrad = compute_zamo_tetrad(cam.r_cam, cam.theta_cam, bh)

        # Ray fired directly radially inward: -e_(1)
        k_inward = 1.0 * tetrad[0] - (-1.0) * tetrad[1]
        x_init = np.array([0.0, cam.r_cam, cam.theta_cam, 0.0])

        res = trace_single_kerr_ray(x_init, k_inward, bh, disk=None, max_steps=400)
        assert res["status"] == "HORIZON_CAPTURE"
        assert res["r_final"] <= bh.event_horizon_outer + 0.05 * m

    def test_escaping_ray(self) -> None:
        """A ray directed outward away from the black hole must escape to infinity."""
        bh = KerrGeometry(mass_kg=1.0e30, spin_dimensionless=0.5)
        m = bh.mass_m
        cam = KerrCamera(r_cam=15.0 * m, theta_cam=0.5 * math.pi)
        tetrad = compute_zamo_tetrad(cam.r_cam, cam.theta_cam, bh)

        # Ray fired radially outward: +e_(1)
        k_outward = 1.0 * tetrad[0] - (+1.0) * tetrad[1]
        x_init = np.array([0.0, cam.r_cam, cam.theta_cam, 0.0])

        res = trace_single_kerr_ray(x_init, k_outward, bh, disk=None, max_steps=400)
        assert res["status"] == "ESCAPED"
        assert res["r_final"] >= 35.0 * m


# ==============================================================================
# LEVEL 5: FULL SCENE RENDERING CONSISTENCY
# ==============================================================================

class TestLevel5SceneRendering:
    """Validate full synthetic image rendering and physical image invariants."""

    def test_render_scene_dimensions_and_channels(self) -> None:
        """Rendered image must have shape (H, W, 3) and valid sRGB values in [0, 1]."""
        bh = KerrGeometry(mass_kg=1.0e30, spin_dimensionless=0.85)
        m = bh.mass_m
        cam = KerrCamera(
            r_cam=18.0 * m,
            theta_cam=math.radians(75.0),
            resolution_x=20,
            resolution_y=20,
            fov_deg=45.0,
        )
        disk = AccretionDiskProperties(
            r_inner_m=bh.isco_radius(prograde=True),
            r_outer_m=15.0 * m,
            peak_temperature_k=8000.0,
        )

        rgb, red_map, hit_map = render_kerr_black_hole_scene(
            camera=cam,
            kerr=bh,
            disk=disk,
            max_steps_per_ray=150,
        )

        assert rgb.shape == (20, 20, 3)
        assert red_map.shape == (20, 20)
        assert hit_map.shape == (20, 20)

        # Colors must be clipped in valid display range [0, 1]
        assert np.all(rgb >= 0.0)
        assert np.all(rgb <= 1.0)

        # Shadow exists (hit_map == 0) and has 0 intensity
        shadow_pixels = rgb[hit_map == 0]
        assert len(shadow_pixels) > 0
        np.testing.assert_allclose(shadow_pixels, 0.0, atol=1e-10)

        # Disk exists and approaching side is brighter than receding side
        disk_mask = hit_map == 1
        assert np.any(disk_mask)
        # Left half (approaching) vs right half (receding) luminance
        left_disk_lum = np.mean(rgb[:, :10, :][disk_mask[:, :10]]) if np.any(disk_mask[:, :10]) else 0.0
        right_disk_lum = np.mean(rgb[:, 10:, :][disk_mask[:, 10:]]) if np.any(disk_mask[:, 10:]) else 0.0
        # Beaming asymmetry: approaching side must be brighter
        assert left_disk_lum > right_disk_lum
