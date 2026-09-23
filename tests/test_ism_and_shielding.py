"""Unit and validation tests for Relativistic ISM Ram Drag, Shielding & Deflection.

Validates:
- LEVEL 1: Analytical ram pressure and energy flux limits as beta -> 0 and beta -> 1.
- LEVEL 2: Bethe-Bloch minimum ionizing particle (MIP) behavior matching NIST PSTAR data.
- LEVEL 3: CSDA penetration range scaling across materials (H2, Be, C, Al, Fe, W, Pb).
- LEVEL 4: Active magnetic deflection shield rigidity, field integrals, and virial mass.
- LEVEL 5: Coupled trajectory integration showing monotonic drag deceleration over interstellar cruise.
"""

from __future__ import annotations

import math
import numpy as np
import pytest

from relativistic_engine.constants import (
    C_LIGHT,
    E_CHARGE,
    M_PROTON,
    SEC_PER_JULIAN_YEAR,
    SIGMA_SB,
)
from relativistic_engine.physics.ism import (
    ISMConditions,
    ISM_LOCAL_CLOUD,
    ISM_COLD_NEUTRAL,
    ISM_WARM_IONIZED,
    ISM_HOT_CORONAL,
    ISM_INTERGALACTIC,
    MaterialProperties,
    MATERIAL_LIQUID_HYDROGEN,
    MATERIAL_BERYLLIUM,
    MATERIAL_GRAPHITE,
    MATERIAL_ALUMINUM,
    MATERIAL_IRON,
    MATERIAL_TUNGSTEN,
    MATERIAL_LEAD,
    compute_relativistic_ram_pressure,
    compute_ism_drag_acceleration,
    compute_kinetic_energy_flux,
    compute_equilibrium_temperature,
    compute_bethe_bloch_stopping_power,
    compute_csda_range,
    compute_bremsstrahlung_losses,
    compute_magnetic_shield_requirements,
)
from relativistic_engine.numerical.trajectory import propagate_trajectory_3d


# ==============================================================================
# LEVEL 1: ANALYTICAL LIMITS & KINEMATIC INVARIANTS
# ==============================================================================

class TestLevel1ISMRamPressureAndFlux:
    """Test Level 1 analytical limits and exact relativistic scalings."""

    def test_ram_pressure_zero_at_rest(self) -> None:
        """Ram pressure must vanish identically at zero speed."""
        rho = ISM_LOCAL_CLOUD.total_mass_density
        p_ram = compute_relativistic_ram_pressure(0.0, rho)
        assert p_ram == pytest.approx(0.0, abs=1e-30)

    def test_newtonian_limit_at_low_velocity(self) -> None:
        """At low velocities (beta << 1), P_ram -> C_D * rho * v^2."""
        rho = ISM_LOCAL_CLOUD.total_mass_density
        beta = 1.0e-4  # ~30 km/s (typical planetary orbital speed)
        v = beta * C_LIGHT
        p_rel = compute_relativistic_ram_pressure(beta, rho, c_d=1.0)
        p_newt = rho * (v ** 2)
        # Relativistic correction is O(beta^2) ~ 1e-8
        assert p_rel == pytest.approx(p_newt, rel=1e-6)

    def test_ultrarelativistic_divergence(self) -> None:
        """Ram pressure must diverge as gamma^2 as beta -> 1."""
        rho = ISM_LOCAL_CLOUD.total_mass_density
        p1 = compute_relativistic_ram_pressure(0.9, rho)
        p2 = compute_relativistic_ram_pressure(0.99, rho)
        p3 = compute_relativistic_ram_pressure(0.999, rho)
        assert p3 > p2 > p1
        # gamma(0.999) approx 22.37, gamma^2 approx 500
        gamma_999 = 1.0 / math.sqrt(1.0 - 0.999 ** 2)
        expected_ratio = (gamma_999 * 0.999) ** 2 / ((1.0 / math.sqrt(1.0 - 0.9 ** 2)) * 0.9) ** 2
        assert (p3 / p1) == pytest.approx(expected_ratio, rel=1e-10)

    def test_kinetic_energy_flux_scaling(self) -> None:
        """Flux F_K = gamma * (gamma - 1) * beta * rho * c^3."""
        rho = 2.0e-21  # kg/m^3
        beta = 0.5
        gamma = 1.0 / math.sqrt(1.0 - 0.5 ** 2)
        expected_flux = gamma * (gamma - 1.0) * beta * rho * (C_LIGHT ** 3)
        actual_flux = compute_kinetic_energy_flux(beta, rho)
        assert actual_flux == pytest.approx(expected_flux, rel=1e-14)

    def test_stefan_boltzmann_equilibrium_temperature(self) -> None:
        """Hull equilibrium temperature must satisfy epsilon * sigma * T^4 = F_K."""
        rho = ISM_LOCAL_CLOUD.total_mass_density
        beta = 0.8
        emissivity = 0.85
        t_eq = compute_equilibrium_temperature(beta, rho, emissivity=emissivity)
        recalculated_flux = emissivity * SIGMA_SB * (t_eq ** 4)
        actual_flux = compute_kinetic_energy_flux(beta, rho)
        assert recalculated_flux == pytest.approx(actual_flux, rel=1e-12)

    def test_invalid_velocity_bounds(self) -> None:
        """Velocities outside [0, 1) must raise ValueError."""
        rho = ISM_LOCAL_CLOUD.total_mass_density
        with pytest.raises(ValueError):
            compute_relativistic_ram_pressure(1.0, rho)
        with pytest.raises(ValueError):
            compute_relativistic_ram_pressure(-0.1, rho)
        with pytest.raises(ValueError):
            compute_kinetic_energy_flux(1.05, rho)


# ==============================================================================
# LEVEL 2: BETHE-BLOCH STOPPING POWER & NIST COMPARISON
# ==============================================================================

class TestLevel2BetheBlochStoppingPower:
    """Validate Bethe-Bloch electronic stopping power against physical laws and NIST benchmarks."""

    def test_graphite_minimum_ionizing_particle(self) -> None:
        """Protons in Carbon (Graphite) must exhibit minimum ionization near beta*gamma ~ 3.5.

        According to the NIST PSTAR database and PDG, the minimum ionization for carbon
        is approximately 1.74 - 1.80 MeV * cm^2 / g.
        """
        # Test beta range around minimum ionizing (E_kin ~ 2-3 GeV, beta ~ 0.96)
        beta_mip = 0.96
        gamma_mip = 1.0 / math.sqrt(1.0 - beta_mip ** 2)
        assert 3.0 < (beta_mip * gamma_mip) < 4.0

        _, mass_stopping_pdg = compute_bethe_bloch_stopping_power(
            beta=beta_mip,
            material=MATERIAL_GRAPHITE,
            projectile_charge=1,
            projectile_mass_kg=M_PROTON,
        )
        # NIST PSTAR value for Carbon at 2.5 GeV: ~1.75 MeV * cm^2 / g
        assert 1.65 < mass_stopping_pdg < 1.85

    def test_aluminum_minimum_ionizing_particle(self) -> None:
        """Protons in Aluminum must have minimum mass stopping power ~1.6 MeV * cm^2 / g."""
        beta_mip = 0.96
        _, mass_stopping = compute_bethe_bloch_stopping_power(
            beta=beta_mip,
            material=MATERIAL_ALUMINUM,
        )
        # NIST PSTAR value for Aluminum at 2.5 GeV: ~1.62 MeV * cm^2 / g
        assert 1.50 < mass_stopping < 1.75

    def test_alpha_particle_charge_squared_scaling(self) -> None:
        """Stopping power must scale as projectile charge squared (z^2).

        An alpha particle (z=2, He-4) must experience 4x the stopping power
        of a proton (z=1) at the same velocity beta.
        """
        beta = 0.7
        m_alpha = 4.0 * M_PROTON
        lin_p, _ = compute_bethe_bloch_stopping_power(
            beta=beta,
            material=MATERIAL_BERYLLIUM,
            projectile_charge=1,
            projectile_mass_kg=M_PROTON,
        )
        lin_alpha, _ = compute_bethe_bloch_stopping_power(
            beta=beta,
            material=MATERIAL_BERYLLIUM,
            projectile_charge=2,
            projectile_mass_kg=m_alpha,
        )
        # z^2 ratio is 4. Minor difference from W_max projectile mass dependence is < 0.1%
        assert (lin_alpha / lin_p) == pytest.approx(4.0, rel=1e-3)

    def test_density_effect_suppresses_relativistic_rise(self) -> None:
        """Sternheimer density correction must suppress logarithmic rise at ultra-relativistic beta."""
        beta_ultra = 0.999  # gamma ~ 22.37
        _, stopping_with_density = compute_bethe_bloch_stopping_power(
            beta=beta_ultra,
            material=MATERIAL_GRAPHITE,
            include_density_effect=True,
        )
        _, stopping_no_density = compute_bethe_bloch_stopping_power(
            beta=beta_ultra,
            material=MATERIAL_GRAPHITE,
            include_density_effect=False,
        )
        # Density effect reduces energy loss due to dielectric polarization of the medium
        assert stopping_with_density < stopping_no_density


# ==============================================================================
# LEVEL 3: CSDA PENETRATION RANGE & BREMSSTRAHLUNG
# ==============================================================================

class TestLevel3CSDARangeAndBremsstrahlung:
    """Validate penetration depth integration and secondary Bremsstrahlung."""

    def test_range_monotonic_with_energy(self) -> None:
        """Penetration range must increase monotonically with initial kinetic energy."""
        r1 = compute_csda_range(0.3, MATERIAL_ALUMINUM)
        r2 = compute_csda_range(0.6, MATERIAL_ALUMINUM)
        r3 = compute_csda_range(0.85, MATERIAL_ALUMINUM)
        assert r3 > r2 > r1 > 0.0

    def test_dense_absorber_compactness(self) -> None:
        """Linear range in Lead (Pb, 11.35 g/cm^3) must be far smaller than in Beryllium (Be, 1.85 g/cm^3)."""
        beta = 0.5
        range_lead = compute_csda_range(beta, MATERIAL_LEAD)
        range_be = compute_csda_range(beta, MATERIAL_BERYLLIUM)
        assert range_lead < range_be
        # Lead density is ~6.1x higher, so linear range is significantly smaller
        assert (range_be / range_lead) > 3.0

    def test_proton_bremsstrahlung_subdominant(self) -> None:
        """Proton Bremsstrahlung must be negligible compared to electronic ionization at cruise speeds."""
        losses = compute_bremsstrahlung_losses(0.8, MATERIAL_GRAPHITE, projectile_mass_kg=M_PROTON)
        assert losses["ionization_to_radiation_ratio"] > 1.0e5


# ==============================================================================
# LEVEL 4: ACTIVE MAGNETIC DEFLECTION SHIELDING
# ==============================================================================

class TestLevel4ActiveMagneticDeflection:
    """Validate active magnetic shielding physics and virial mass bounds."""

    def test_magnetic_rigidity_formula(self) -> None:
        """Rigidity R = p / q = (gamma * m * beta * c) / e."""
        beta = 0.8
        gamma = 1.0 / math.sqrt(1.0 - 0.8 ** 2)
        expected_rigidity = (gamma * M_PROTON * beta * C_LIGHT) / E_CHARGE
        shield = compute_magnetic_shield_requirements(
            beta=beta,
            vehicle_radius_m=10.0,
            standoff_distance_m=50.0,
        )
        assert shield["magnetic_rigidity_t_m"] == pytest.approx(expected_rigidity, rel=1e-12)

    def test_deflection_angle_geometry(self) -> None:
        """Deflection angle must satisfy sin(theta) = R_v / L."""
        r_v = 15.0
        l_standoff = 60.0
        shield = compute_magnetic_shield_requirements(
            beta=0.7,
            vehicle_radius_m=r_v,
            standoff_distance_m=l_standoff,
        )
        expected_deg = math.degrees(math.asin(r_v / l_standoff))
        assert shield["deflection_angle_deg"] == pytest.approx(expected_deg, rel=1e-10)

    def test_virial_mass_positivity_and_scaling(self) -> None:
        """Stored magnetic energy and virial structural mass must be strictly positive and scale with B^2."""
        shield1 = compute_magnetic_shield_requirements(0.5, 10.0, 50.0)
        shield2 = compute_magnetic_shield_requirements(0.9, 10.0, 50.0)
        assert shield1["stored_magnetic_energy_j"] > 0.0
        assert shield1["minimum_virial_mass_kg"] > 0.0
        # Higher beta requires higher momentum deflection, thus higher field and virial mass
        assert shield2["minimum_virial_mass_kg"] > shield1["minimum_virial_mass_kg"]


# ==============================================================================
# LEVEL 5: TRAJECTORY INTEGRATION WITH ISM RAM DRAG
# ==============================================================================

class TestLevel5TrajectoryISMDragDeceleration:
    """Validate coupled 3D trajectory integration under continuous ISM ram drag."""

    def test_coasting_drag_deceleration(self) -> None:
        """A coasting relativistic starship through ISM must experience monotonic velocity decay."""
        # Initial cruise: 0.5 c along X axis in flat space
        v0 = np.array([0.5 * C_LIGHT, 0.0, 0.0])
        r0 = np.array([0.0, 0.0, 0.0])
        # Starship parameters: 1,000,000 kg, 100 m^2 frontal area
        m_ship = 1.0e6
        area = 100.0
        t_span = (0.0, 2.0 * SEC_PER_JULIAN_YEAR)

        # Dense interstellar cloud to make deceleration measurable over 2 years
        ism_dense = ISMConditions(
            name="Dense Nebula Test",
            n_h=1.0e10,  # 10,000 cm^-3
        )

        res_drag = propagate_trajectory_3d(
            r0=r0,
            v0=v0,
            t_span=t_span,
            spacecraft_mass_kg=m_ship,
            ism_conditions=ism_dense,
            ism_frontal_area=area,
            ism_drag_coefficient=1.0,
            rtol=1e-10,
            atol=1e-11,
        )

        res_vacuum = propagate_trajectory_3d(
            r0=r0,
            v0=v0,
            t_span=t_span,
            spacecraft_mass_kg=m_ship,
            rtol=1e-10,
            atol=1e-11,
        )

        v_final_drag = np.linalg.norm(res_drag.v[-1])
        v_final_vac = np.linalg.norm(res_vacuum.v[-1])

        # In vacuum, velocity is strictly conserved
        assert v_final_vac == pytest.approx(0.5 * C_LIGHT, rel=1e-10)
        # With ISM drag, final velocity must be strictly lower
        assert v_final_drag < v_final_vac
        # Drag force is antiparallel, so displacement is slightly less than vacuum
        assert res_drag.r[-1, 0] < res_vacuum.r[-1, 0]
