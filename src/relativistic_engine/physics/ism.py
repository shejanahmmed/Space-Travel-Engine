"""Relativistic Interstellar Medium (ISM) Ram Pressure, Shielding & Deflection.

This module models the physical interactions between a relativistic starship and
the diffuse interstellar medium (ISM), including:
1. Relativistic ram pressure and aerodynamic drag:
   P_ram = C_D * gamma^2 * beta^2 * rho_ISM * c^2
2. Kinetic energy influx and thermal radiative equilibrium:
   F_K = gamma * (gamma - 1) * beta * rho_ISM * c^3
   T_eq = (F_K / (epsilon * sigma))^(1/4)
3. Relativistic ion stopping power via the Bethe-Bloch formula:
   -dE/dx = 2 * pi * N_A * r_e^2 * m_e * c^2 * rho_mat * (Z/A) * (z^2/beta^2) * [...]
   with kinematic maximum energy transfer W_max and Sternheimer density corrections.
4. Continuous Slowing Down Approximation (CSDA) range (Bragg curve integration).
5. Relativistic Bremsstrahlung radiation losses and characteristic radiation lengths.
6. Active magnetic deflection shielding:
   Relativistic gyroradius r_L = (gamma * m * beta * c) / (|q| * B),
   required magnetic line integral int B_perp dl, stored field energy, and
   minimum virial containment mass.

All calculations use strict SI units (m, s, kg, J, T, N, Pa) unless explicitly noted.
References:
- Particle Data Group (PDG), "Passage of particles through matter" (2022).
- ICRU Report 49: Stopping Powers and Ranges for Protons and Alpha Particles.
- NIST PSTAR Database: Proton Stopping Powers and Ranges.
- Crawford, I. A. (2011), "Dispersion, deflection and deceleration of interstellar vehicles".
- Frisbee, R. H. (2009), "Limits of Interstellar Flight Technology".
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from relativistic_engine.constants import (
    ALPHA_FS,
    C_LIGHT,
    E_CHARGE,
    EPSILON_0,
    K_BOLTZMANN,
    M_ELECTRON,
    M_PROTON,
    MU_0,
    N_AVOGADRO,
    R_ELECTRON,
    SIGMA_SB,
)


@dataclass(frozen=True)
class ISMConditions:
    """Interstellar Medium environment parameter contract.

    Attributes:
        name: Environment designation.
        n_h: Total hydrogen atom number density in m^-3 (1 cm^-3 = 1e6 m^-3).
        ionized_fraction: Fraction of hydrogen that is ionized (0.0 to 1.0).
        he_fraction: Number abundance ratio n_He / n_H (typically ~0.08 to 0.10).
        dust_to_gas_mass_ratio: Mass ratio of refractory dust grains to total gas.
        temperature_k: Kinetic temperature of the gas in Kelvin.
    """
    name: str
    n_h: float
    ionized_fraction: float = 0.20
    he_fraction: float = 0.085
    dust_to_gas_mass_ratio: float = 0.010
    temperature_k: float = 7000.0

    @property
    def gas_mass_density(self) -> float:
        """Mean gas mass density in kg/m^3 including hydrogen and helium."""
        # Mean molecular weight contribution per hydrogen atom: m_H + (n_He/n_H)*m_He
        # where m_He approx 3.97 * m_p
        m_eff = M_PROTON * (1.0 + self.he_fraction * 3.9717)
        return self.n_h * m_eff

    @property
    def total_mass_density(self) -> float:
        """Total mass density in kg/m^3 including neutral/ionized gas and dust."""
        return self.gas_mass_density * (1.0 + self.dust_to_gas_mass_ratio)


# Canonical interstellar environments
ISM_LOCAL_CLOUD = ISMConditions(
    name="Local Interstellar Cloud (LIC)",
    n_h=1.0e5,  # 0.1 cm^-3
    ionized_fraction=0.25,
    he_fraction=0.085,
    dust_to_gas_mass_ratio=0.010,
    temperature_k=7000.0,
)

ISM_COLD_NEUTRAL = ISMConditions(
    name="Cold Neutral Medium (CNM)",
    n_h=3.0e7,  # 30 cm^-3
    ionized_fraction=0.001,
    he_fraction=0.085,
    dust_to_gas_mass_ratio=0.012,
    temperature_k=80.0,
)

ISM_WARM_IONIZED = ISMConditions(
    name="Warm Ionized Medium (WIM)",
    n_h=2.0e5,  # 0.2 cm^-3
    ionized_fraction=0.95,
    he_fraction=0.085,
    dust_to_gas_mass_ratio=0.008,
    temperature_k=8000.0,
)

ISM_HOT_CORONAL = ISMConditions(
    name="Hot Ionized Medium (HIM)",
    n_h=3.0e3,  # 0.003 cm^-3
    ionized_fraction=1.0,
    he_fraction=0.085,
    dust_to_gas_mass_ratio=0.002,
    temperature_k=1.0e6,
)

ISM_INTERGALACTIC = ISMConditions(
    name="Intergalactic Medium (IGM)",
    n_h=1.0,  # 1e-6 cm^-3
    ionized_fraction=1.0,
    he_fraction=0.085,
    dust_to_gas_mass_ratio=0.0001,
    temperature_k=1.0e5,
)


@dataclass(frozen=True)
class MaterialProperties:
    """Solid state material physical properties for stopping power and radiation attenuation.

    Attributes:
        name: Material name.
        atomic_number_z: Mean atomic number Z.
        atomic_mass_kg_mol: Atomic mass A in kg/mol (e.g. 0.012011 kg/mol for Carbon).
        density_kg_m3: Solid or liquid mass density in kg/m^3.
        mean_excitation_energy_j: Mean excitation potential I in Joules (eV * 1.602176634e-19).
        radiation_length_m: Characteristic radiation length X_0 in meters.
        sternheimer_cbar: Sternheimer density correction parameter C_bar.
        sternheimer_x0: Sternheimer lower threshold X_0 = log10(beta * gamma).
        sternheimer_x1: Sternheimer upper threshold X_1.
        sternheimer_a: Sternheimer power fit coefficient a.
        sternheimer_m: Sternheimer power fit exponent m.
    """
    name: str
    atomic_number_z: float
    atomic_mass_kg_mol: float
    density_kg_m3: float
    mean_excitation_energy_j: float
    radiation_length_m: float
    sternheimer_cbar: float = 0.0
    sternheimer_x0: float = 0.0
    sternheimer_x1: float = 0.0
    sternheimer_a: float = 0.0
    sternheimer_m: float = 3.0


# Standard shielding and structural materials (PDG / NIST values)
MATERIAL_LIQUID_HYDROGEN = MaterialProperties(
    name="Liquid Hydrogen (LH2)",
    atomic_number_z=1.0,
    atomic_mass_kg_mol=1.008e-3,
    density_kg_m3=70.85,
    mean_excitation_energy_j=21.8 * E_CHARGE,
    radiation_length_m=8.65,
    sternheimer_cbar=3.2632,
    sternheimer_x0=0.4759,
    sternheimer_x1=1.9215,
    sternheimer_a=0.13483,
    sternheimer_m=5.624,
)

MATERIAL_BERYLLIUM = MaterialProperties(
    name="Beryllium (Be)",
    atomic_number_z=4.0,
    atomic_mass_kg_mol=9.012182e-3,
    density_kg_m3=1848.0,
    mean_excitation_energy_j=63.7 * E_CHARGE,
    radiation_length_m=0.3528,
    sternheimer_cbar=2.7847,
    sternheimer_x0=0.0592,
    sternheimer_x1=1.6922,
    sternheimer_a=0.80392,
    sternheimer_m=2.4339,
)

MATERIAL_GRAPHITE = MaterialProperties(
    name="Graphite / Carbon (C)",
    atomic_number_z=6.0,
    atomic_mass_kg_mol=12.011e-3,
    density_kg_m3=2260.0,
    mean_excitation_energy_j=78.0 * E_CHARGE,
    radiation_length_m=0.1882,
    sternheimer_cbar=2.8680,
    sternheimer_x0=-0.0351,
    sternheimer_x1=2.486,
    sternheimer_a=0.2024,
    sternheimer_m=3.0,
)

MATERIAL_ALUMINUM = MaterialProperties(
    name="Aluminum (Al)",
    atomic_number_z=13.0,
    atomic_mass_kg_mol=26.981538e-3,
    density_kg_m3=2700.0,
    mean_excitation_energy_j=166.0 * E_CHARGE,
    radiation_length_m=0.08897,
    sternheimer_cbar=4.2395,
    sternheimer_x0=0.1708,
    sternheimer_x1=3.0127,
    sternheimer_a=0.08024,
    sternheimer_m=3.6345,
)

MATERIAL_IRON = MaterialProperties(
    name="Iron (Fe)",
    atomic_number_z=26.0,
    atomic_mass_kg_mol=55.845e-3,
    density_kg_m3=7874.0,
    mean_excitation_energy_j=286.0 * E_CHARGE,
    radiation_length_m=0.01757,
    sternheimer_cbar=4.2911,
    sternheimer_x0=-0.0012,
    sternheimer_x1=3.1531,
    sternheimer_a=0.1468,
    sternheimer_m=2.9632,
)

MATERIAL_TUNGSTEN = MaterialProperties(
    name="Tungsten (W)",
    atomic_number_z=74.0,
    atomic_mass_kg_mol=183.84e-3,
    density_kg_m3=19300.0,
    mean_excitation_energy_j=727.0 * E_CHARGE,
    radiation_length_m=0.003504,
    sternheimer_cbar=5.4059,
    sternheimer_x0=0.2167,
    sternheimer_x1=3.4960,
    sternheimer_a=0.1550,
    sternheimer_m=2.8436,
)

MATERIAL_LEAD = MaterialProperties(
    name="Lead (Pb)",
    atomic_number_z=82.0,
    atomic_mass_kg_mol=207.2e-3,
    density_kg_m3=11350.0,
    mean_excitation_energy_j=823.0 * E_CHARGE,
    radiation_length_m=0.005612,
    sternheimer_cbar=6.2018,
    sternheimer_x0=0.3776,
    sternheimer_x1=3.8073,
    sternheimer_a=0.0936,
    sternheimer_m=3.1608,
)


# ==============================================================================
# RELATIVISTIC RAM PRESSURE & AERODYNAMIC DRAG
# ==============================================================================

def compute_relativistic_ram_pressure(
    beta: Union[float, np.ndarray],
    rho_ism: float,
    c_d: float = 1.0,
) -> Union[float, np.ndarray]:
    """Calculate relativistic ram pressure in Pascals (N/m^2).

    In the ship rest frame, ISM particle density is Lorentz contracted by gamma,
    and each particle arrives with relativistic momentum gamma * m * beta * c.
    The resulting momentum flux per unit proper time and unit area is:
        P_ram = C_D * gamma^2 * beta^2 * rho_0 * c^2

    Args:
        beta: Velocity ratio v / c in [0, 1).
        rho_ism: Undisturbed ISM mass density in kg/m^3.
        c_d: Drag / momentum transfer coefficient (1.0 for inelastic absorption,
            2.0 for specular reflection, ~1.0-1.2 for diffuse scatter).

    Returns:
        Effective ram pressure in Pascals.
    """
    beta_arr = np.asarray(beta, dtype=np.float64)
    if np.any(beta_arr < 0.0) or np.any(beta_arr >= 1.0):
        raise ValueError("Beta must satisfy 0 <= beta < 1.")

    beta_sq = np.square(beta_arr)
    # gamma^2 * beta^2 = beta^2 / (1 - beta^2)
    gamma_sq_beta_sq = np.where(beta_sq < 1.0, beta_sq / (1.0 - beta_sq), 0.0)
    p_ram = c_d * gamma_sq_beta_sq * rho_ism * (C_LIGHT ** 2)

    return float(p_ram) if np.ndim(beta) == 0 else p_ram


def compute_ism_drag_acceleration(
    velocity_vector: np.ndarray,
    ship_mass: float,
    frontal_area: float,
    rho_ism: float,
    c_d: float = 1.0,
) -> np.ndarray:
    """Calculate coordinate deceleration 3-vector due to ISM ram drag.

    The drag 3-force in the ship rest frame is:
        F_drag' = C_D * gamma^2 * beta^2 * rho_ism * c^2 * A
    In the coordinate frame, the spatial acceleration is:
        a_drag = - F_drag' / (gamma * ship_mass) * v_hat
               = - (C_D * gamma * beta^2 * rho_ism * c^2 * A / ship_mass) * v_hat

    Args:
        velocity_vector: 3D coordinate velocity in m/s.
        ship_mass: Current vehicle mass in kg.
        frontal_area: Vehicle cross-sectional collision area in m^2.
        rho_ism: Undisturbed ISM mass density in kg/m^3.
        c_d: Aerodynamic drag coefficient.

    Returns:
        3D coordinate acceleration vector in m/s^2.
    """
    v_vec = np.asarray(velocity_vector, dtype=np.float64)
    v_norm = np.linalg.norm(v_vec)
    if v_norm == 0.0:
        return np.zeros(3, dtype=np.float64)

    beta = v_norm / C_LIGHT
    if beta >= 1.0:
        raise ValueError(f"Velocity exceeds or equals speed of light: beta = {beta}")

    gamma = 1.0 / math.sqrt(1.0 - beta ** 2)
    # Coordinate deceleration magnitude
    acc_mag = (c_d * gamma * (beta ** 2) * rho_ism * (C_LIGHT ** 2) * frontal_area) / ship_mass
    v_hat = v_vec / v_norm

    return -acc_mag * v_hat


def compute_kinetic_energy_flux(
    beta: Union[float, np.ndarray],
    rho_ism: float,
) -> Union[float, np.ndarray]:
    """Calculate relativistic kinetic energy influx in W/m^2 on the forward hull.

    Flux = n_p * v * T_K = (gamma * n_0) * (beta * c) * [(gamma - 1) * m_0 * c^2]
         = gamma * (gamma - 1) * beta * rho_ism * c^3

    Args:
        beta: Velocity ratio v / c in [0, 1).
        rho_ism: ISM mass density in kg/m^3.

    Returns:
        Kinetic energy flux in Watts per square meter.
    """
    beta_arr = np.asarray(beta, dtype=np.float64)
    if np.any(beta_arr < 0.0) or np.any(beta_arr >= 1.0):
        raise ValueError("Beta must satisfy 0 <= beta < 1.")

    beta_sq = np.square(beta_arr)
    gamma = 1.0 / np.sqrt(1.0 - beta_sq)
    flux = gamma * (gamma - 1.0) * beta_arr * rho_ism * (C_LIGHT ** 3)

    return float(flux) if np.ndim(beta) == 0 else flux


def compute_equilibrium_temperature(
    beta: Union[float, np.ndarray],
    rho_ism: float,
    emissivity: float = 0.90,
) -> Union[float, np.ndarray]:
    """Calculate steady-state radiative equilibrium temperature of the forward hull.

    Equates kinetic energy absorption to Stefan-Boltzmann thermal re-radiation:
        epsilon * sigma * T^4 = F_K
        T_eq = (F_K / (epsilon * sigma))^(1/4)

    Args:
        beta: Velocity ratio v / c in [0, 1).
        rho_ism: ISM mass density in kg/m^3.
        emissivity: Surface thermal emissivity (0.0 to 1.0).

    Returns:
        Equilibrium temperature in Kelvin.
    """
    flux = compute_kinetic_energy_flux(beta, rho_ism)
    flux_arr = np.asarray(flux, dtype=np.float64)
    t_eq = np.power(flux_arr / (emissivity * SIGMA_SB), 0.25)

    return float(t_eq) if np.ndim(beta) == 0 else t_eq


# ==============================================================================
# BETHE-BLOCH ION STOPPING POWER & PENETRATION RANGE
# ==============================================================================

# Bethe-Bloch prefactor: K = 4 * pi * N_A * r_e^2 * m_e * c^2 in J * m^2 / mol
K_BETHE_BLOCH_SI: float = (
    4.0 * math.pi * N_AVOGADRO * (R_ELECTRON ** 2) * M_ELECTRON * (C_LIGHT ** 2)
)  # ~ 4.91989e-18 J * m^2 / mol


def compute_bethe_bloch_stopping_power(
    beta: Union[float, np.ndarray],
    material: MaterialProperties,
    projectile_charge: int = 1,
    projectile_mass_kg: float = M_PROTON,
    include_density_effect: bool = True,
) -> Tuple[Union[float, np.ndarray], Union[float, np.ndarray]]:
    """Compute relativistic Bethe-Bloch stopping power for heavy charged particles.

    Evaluates the mean rate of energy loss -dE/dx through electronic ionization
    and excitation according to the standard PDG formulation:
        -dE/dx = rho_s * K * (Z/A) * (z^2 / beta^2) * [ 0.5 * ln(2 m_e c^2 beta^2 gamma^2 W_max / I^2)
                                                         - beta^2 - delta / 2 ]

    Args:
        beta: Projectile velocity ratio beta = v / c in (0.01, 1.0).
        material: Target absorber material properties.
        projectile_charge: Projectile charge state z in units of e (e.g. 1 for proton).
        projectile_mass_kg: Projectile rest mass in kg (default: proton).
        include_density_effect: If True, applies Sternheimer density effect correction delta.

    Returns:
        Tuple of:
        - Linear stopping power -dE/dx in J / m (or N).
        - Mass stopping power -(1/rho)*dE/dx in MeV * cm^2 / g (canonical nuclear physics unit).
    """
    beta_arr = np.asarray(beta, dtype=np.float64)
    if np.any(beta_arr <= 0.0) or np.any(beta_arr >= 1.0):
        raise ValueError("Beta must be strictly in range 0 < beta < 1.")

    beta_sq = np.square(beta_arr)
    gamma = 1.0 / np.sqrt(1.0 - beta_sq)
    gamma_sq = np.square(gamma)

    # Maximum kinematic energy transfer to a single free electron (PDG Eq. 34.2)
    s = M_ELECTRON / projectile_mass_kg
    m_e_c2 = M_ELECTRON * (C_LIGHT ** 2)
    w_max = (2.0 * m_e_c2 * beta_sq * gamma_sq) / (1.0 + 2.0 * gamma * s + s ** 2)

    # Logarithmic argument: (2 * m_e * c^2 * beta^2 * gamma^2 * W_max) / I^2
    i_sq = material.mean_excitation_energy_j ** 2
    log_arg = (2.0 * m_e_c2 * beta_sq * gamma_sq * w_max) / i_sq

    # Prevent unphysical negative stopping power at sub-MeV energies where Bethe-Bloch breaks down
    bracket = np.maximum(0.5 * np.log(log_arg) - beta_sq, 0.05)

    # Sternheimer density effect correction delta(beta * gamma)
    if include_density_effect:
        bg = beta_arr * gamma
        # Safe log10 with guard
        x = np.log10(np.maximum(bg, 1e-6))
        delta = np.zeros_like(x)

        idx_mid = (x >= material.sternheimer_x0) & (x < material.sternheimer_x1)
        idx_high = x >= material.sternheimer_x1

        # Intermediate regime: 2*ln(10)*X - C_bar + a*(X1 - X)^m
        ln10_2 = 2.0 * math.log(10.0)
        delta[idx_mid] = (
            ln10_2 * x[idx_mid]
            - material.sternheimer_cbar
            + material.sternheimer_a * np.power(material.sternheimer_x1 - x[idx_mid], material.sternheimer_m)
        )
        # Asymptotic ultra-relativistic Fermi plateau: 2*ln(10)*X - C_bar
        delta[idx_high] = ln10_2 * x[idx_high] - material.sternheimer_cbar

        delta = np.maximum(delta, 0.0)
        bracket = np.maximum(bracket - 0.5 * delta, 0.05)

    # Linear stopping power in SI units: J / m
    # K_SI * (Z / A_kg) has units (J * m^2 / mol) * (mol / kg) = J * m^2 / kg
    mass_factor = K_BETHE_BLOCH_SI * (material.atomic_number_z / material.atomic_mass_kg_mol)
    z_sq = projectile_charge ** 2

    # Mass stopping power in SI: J * m^2 / kg
    dE_dx_mass_si = mass_factor * (z_sq / beta_sq) * bracket
    # Linear stopping power in SI: J / m
    dE_dx_linear_si = material.density_kg_m3 * dE_dx_mass_si

    # Conversion factor from J * m^2 / kg to MeV * cm^2 / g:
    # 1 J * m^2 / kg = (1 / 1.602176634e-13 MeV) * (1e4 cm^2) / (1e3 g) = 6.241509e13 MeV * cm^2 / g
    conv_mev_cm2_g = 1.0e1 / (E_CHARGE * 1.0e6)
    dE_dx_mass_pdg = dE_dx_mass_si * conv_mev_cm2_g

    if np.ndim(beta) == 0:
        return float(dE_dx_linear_si), float(dE_dx_mass_pdg)
    return dE_dx_linear_si, dE_dx_mass_pdg


def compute_csda_range(
    beta_initial: float,
    material: MaterialProperties,
    projectile_charge: int = 1,
    projectile_mass_kg: float = M_PROTON,
    n_steps: int = 400,
) -> float:
    """Compute Continuous Slowing Down Approximation (CSDA) penetration depth in meters.

    Integrates the inverse stopping power over kinetic energy from initial energy down
    to threshold (beta = 0.05, ~1.2 MeV for protons):
        R = integral_E_min^E_0 ( -dE/dx )^(-1) dE

    Args:
        beta_initial: Initial projectile speed ratio v / c.
        material: Shielding material.
        projectile_charge: Projectile charge state z.
        projectile_mass_kg: Projectile rest mass in kg.
        n_steps: Number of Simpson quadrature sub-intervals.

    Returns:
        Total penetration range in meters of material.
    """
    if beta_initial <= 0.06 or beta_initial >= 1.0:
        raise ValueError("Initial beta must be in range 0.06 < beta < 1.0.")

    beta_min = 0.05
    gamma_init = 1.0 / math.sqrt(1.0 - beta_initial ** 2)
    e_k_init = (gamma_init - 1.0) * projectile_mass_kg * (C_LIGHT ** 2)

    gamma_min = 1.0 / math.sqrt(1.0 - beta_min ** 2)
    e_k_min = (gamma_min - 1.0) * projectile_mass_kg * (C_LIGHT ** 2)

    # Logarithmic spacing in kinetic energy to accurately resolve the Bragg peak
    e_k_grid = np.geomspace(e_k_min, e_k_init, n_steps)
    gamma_grid = 1.0 + e_k_grid / (projectile_mass_kg * (C_LIGHT ** 2))
    beta_grid = np.sqrt(1.0 - 1.0 / np.square(gamma_grid))

    # Evaluate linear stopping power in J/m
    dE_dx, _ = compute_bethe_bloch_stopping_power(
        beta=beta_grid,
        material=material,
        projectile_charge=projectile_charge,
        projectile_mass_kg=projectile_mass_kg,
    )

    # Integrand: 1 / (dE/dx) in m / J
    integrand = 1.0 / dE_dx
    # Composite trapezoidal quadrature
    range_m = float(np.trapezoid(integrand, e_k_grid))

    return range_m


def compute_bremsstrahlung_losses(
    beta: float,
    material: MaterialProperties,
    projectile_mass_kg: float = M_PROTON,
) -> Dict[str, float]:
    """Calculate radiative Bremsstrahlung energy loss rate and critical parameters.

    Radiation losses scale inversely with projectile mass squared:
        (dE/dx)_rad / (dE/dx)_rad,electron = (m_e / M)^2
    For protons, Bremsstrahlung becomes significant only at ultra-relativistic
    energies (E > 1 TeV).

    Args:
        beta: Projectile speed ratio v / c.
        material: Absorber material.
        projectile_mass_kg: Projectile mass in kg.

    Returns:
        Dictionary containing:
        - total_energy_j: Total kinetic energy of particle in J.
        - radiation_loss_rate_j_m: Radiative -dE/dx in J / m.
        - ionization_to_radiation_ratio: Ratio of ionization loss to radiation loss.
        - characteristic_radiation_length_m: Radiation length X_0 in meters.
    """
    gamma = 1.0 / math.sqrt(1.0 - beta ** 2)
    e_total = gamma * projectile_mass_kg * (C_LIGHT ** 2)

    # Mass scaling ratio squared relative to electron
    mass_ratio_sq = (M_ELECTRON / projectile_mass_kg) ** 2
    # Radiative energy loss rate: -dE/dx_rad = (E / X_0) * (m_e / M)^2
    rad_loss_rate = (e_total / material.radiation_length_m) * mass_ratio_sq

    # Electronic ionization loss rate
    ion_loss_rate, _ = compute_bethe_bloch_stopping_power(
        beta=beta,
        material=material,
        projectile_mass_kg=projectile_mass_kg,
    )

    return {
        "total_energy_j": e_total,
        "radiation_loss_rate_j_m": rad_loss_rate,
        "ionization_to_radiation_ratio": float(ion_loss_rate / rad_loss_rate) if rad_loss_rate > 0 else float("inf"),
        "characteristic_radiation_length_m": material.radiation_length_m,
    }


# ==============================================================================
# ACTIVE MAGNETIC DEFLECTION SHIELDING
# ==============================================================================

def compute_magnetic_shield_requirements(
    beta: float,
    vehicle_radius_m: float,
    standoff_distance_m: float,
    projectile_charge: int = 1,
    projectile_mass_kg: float = M_PROTON,
    coil_yield_stress_pa: float = 1.0e9,  # 1 GPa (high-strength carbon fiber / titanium)
    coil_density_kg_m3: float = 1800.0,   # Carbon fiber composite
) -> Dict[str, float]:
    """Calculate active superconducting magnetic deflection shield specifications.

    Deflects charged relativistic ISM ions before they strike the vehicle hull.
    The trajectory deflection angle theta must satisfy:
        sin(theta) >= vehicle_radius / standoff_distance
    The required transverse field-path integral is:
        int B_perp dl = (p / q) * sin(theta)
    where p = gamma * m * beta * c is the relativistic momentum.

    Structural containment mass is bounded from below by the Virial Theorem:
        M_virial >= (rho_structure / sigma_yield) * U_magnetic

    Args:
        beta: Vehicle cruise velocity ratio v / c.
        vehicle_radius_m: Cylindrical starship frontal radius R_v in meters.
        standoff_distance_m: Axial distance L over which deflection occurs.
        projectile_charge: Ion charge state z (e.g. 1 for proton).
        projectile_mass_kg: Ion mass in kg (default: proton).
        coil_yield_stress_pa: Maximum allowable structural tensile stress in Pa.
        coil_density_kg_m3: Structural support material density in kg/m^3.

    Returns:
        Dictionary of engineering shield specifications.
    """
    if beta <= 0.0 or beta >= 1.0:
        raise ValueError("Beta must be in range 0 < beta < 1.")

    gamma = 1.0 / math.sqrt(1.0 - beta ** 2)
    momentum = gamma * projectile_mass_kg * beta * C_LIGHT
    charge = abs(projectile_charge) * E_CHARGE

    # Magnetic rigidity: R = p / q in Tesla * meters
    rigidity_t_m = momentum / charge

    # Geometric deflection angle
    sin_theta = min(1.0, vehicle_radius_m / standoff_distance_m)
    deflection_angle_rad = math.asin(sin_theta)

    # Required transverse magnetic field integral: int B dl in T * m
    required_field_integral_t_m = rigidity_t_m * sin_theta
    # Characteristic average transverse magnetic field
    b_mean_t = required_field_integral_t_m / standoff_distance_m

    # Approximate deflection volume: cylinder of radius R_v and length L
    deflection_volume_m3 = math.pi * (vehicle_radius_m ** 2) * standoff_distance_m
    # Stored magnetic energy: U_B = (B^2 / (2 * mu_0)) * Volume
    u_magnetic_j = (b_mean_t ** 2 / (2.0 * MU_0)) * deflection_volume_m3

    # Virial structural support mass lower bound
    m_virial_kg = (coil_density_kg_m3 / coil_yield_stress_pa) * u_magnetic_j

    return {
        "beta": beta,
        "gamma": gamma,
        "kinetic_energy_gev": float((gamma - 1.0) * projectile_mass_kg * (C_LIGHT ** 2) / (1.0e9 * E_CHARGE)),
        "magnetic_rigidity_t_m": rigidity_t_m,
        "deflection_angle_deg": math.degrees(deflection_angle_rad),
        "required_field_integral_t_m": required_field_integral_t_m,
        "mean_magnetic_field_t": b_mean_t,
        "deflection_volume_m3": deflection_volume_m3,
        "stored_magnetic_energy_j": u_magnetic_j,
        "minimum_virial_mass_kg": m_virial_kg,
    }
