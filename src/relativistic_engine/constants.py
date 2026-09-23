"""Authoritative Physical and Astronomical Constants.

All constants are traceable to official standards:
- BIPM / SI Base Definitions
- CODATA 2018 recommended values
- IAU (International Astronomical Union) Resolutions
- IERS (International Earth Rotation and Reference Systems Service) Conventions

Units are strictly SI:
- Distance: meters (m)
- Time: seconds (s)
- Mass: kilograms (kg)
- Velocity: meters per second (m/s)
- Acceleration: meters per second squared (m/s^2)
- Gravitational Parameter (GM): m^3 / s^2
"""

import math

# ==============================================================================
# FUNDAMENTAL PHYSICAL CONSTANTS
# ==============================================================================

# Speed of light in vacuum (exact by SI definition, BIPM 1983)
# Unit: m / s
C_LIGHT: float = 299792458.0

# Standard acceleration due to Earth gravity (exact, 3rd CGPM 1901)
# Unit: m / s^2
G0: float = 9.80665

# Newtonian constant of gravitation (CODATA 2018)
# Value: 6.67430(15) x 10^-11 m^3 kg^-1 s^-2
# Relative standard uncertainty: 2.2 x 10^-5
# Unit: m^3 / (kg * s^2)
G_NEWTON: float = 6.67430e-11
G_NEWTON_UNCERTAINTY: float = 0.00015e-11

# Planck constant (exact by SI definition, 26th CGPM 2019)
# Unit: J * s
H_PLANCK: float = 6.62607015e-34

# Boltzmann constant (exact by SI definition, 26th CGPM 2019)
# Unit: J / K
K_BOLTZMANN: float = 1.380649e-23

# Stefan-Boltzmann constant (derived from exact SI 2019 fundamental constants):
# sigma = (2 * pi^5 * k_B^4) / (15 * c^2 * h^3)
# Unit: W / (m^2 * K^4)
SIGMA_SB: float = (2.0 * (math.pi ** 5) * (K_BOLTZMANN ** 4)) / (15.0 * (C_LIGHT ** 2) * (H_PLANCK ** 3))

# Wien displacement law constant for wavelength: b = h * c / (x * k_B) where x * e^x / (e^x - 1) = 5
# b approx 2.897771955e-3 m * K (CODATA 2018)
WIEN_B: float = 2.897771955185172e-3

# ==============================================================================
# ASTRONOMICAL CONSTANTS (IAU / IERS Standards)
# ==============================================================================

# Astronomical Unit (exact by IAU 2012 Resolution B2)
# Unit: m
AU: float = 149597870700.0

# Seconds per standard Julian day (exact)
# Unit: s
SEC_PER_DAY: float = 86400.0

# Days per Julian year (exact, IAU definition)
# Unit: days
DAYS_PER_JULIAN_YEAR: float = 365.25

# Seconds per Julian year (exact: 365.25 * 86400.0)
# Unit: s
SEC_PER_JULIAN_YEAR: float = DAYS_PER_JULIAN_YEAR * SEC_PER_DAY  # 31557600.0

# Light-Year (distance light travels in vacuum in one Julian year of 365.25 days, IAU)
# Exact: C_LIGHT * SEC_PER_JULIAN_YEAR = 9,460,730,472,580,800.0 m
# Unit: m
LIGHT_YEAR: float = C_LIGHT * SEC_PER_JULIAN_YEAR

# Parsec (exact definition from IAU 2015 Resolution B2: (180 * 3600 / pi) * AU)
# Unit: m
PARSEC: float = (648000.0 / math.pi) * AU  # ~ 3.0856775814913673e16 m

# ==============================================================================
# STANDARD GRAVITATIONAL PARAMETERS (GM)
# Source: IAU 2015 Resolution B3 / IERS Conventions (2010) / JPL DE440
# Using GM avoids uncertainty from the product G * M
# Unit: m^3 / s^2
# ==============================================================================

# Nominal solar gravitational parameter (IAU 2015 Resolution B3)
GM_SUN: float = 1.3271244004193938e20

# Mercurian gravitational parameter (JPL DE440)
GM_MERCURY: float = 2.203186855e13

# Venusian gravitational parameter (IAU 2015 / JPL DE440)
GM_VENUS: float = 3.24858592e14

# Terrestrial gravitational parameter (IERS 2010 / IAU 2015)
GM_EARTH: float = 3.986004418e14

# Lunar gravitational parameter (JPL DE440)
GM_MOON: float = 4.902800066e12

# Martian gravitational parameter (IAU 2015)
GM_MARS: float = 4.282837e13

# Jovian system gravitational parameter (IAU 2015)
GM_JUPITER: float = 1.26686534e17

# Saturnian system gravitational parameter (IAU 2015)
GM_SATURN: float = 3.7931187e16

# ==============================================================================
# IAU RELATIVISTIC TIME SCALE TRANSFORMATION CONSTANTS
# Source: IAU 2000 Resolution B1.9 & IAU 2006 Resolution 3
# ==============================================================================

# Defining rate difference between TCB (Barycentric Coordinate Time)
# and TDB (Barycentric Dynamical Time): d(TDB)/d(TCB) = 1 - L_B
# Value is exact by IAU 2006 Resolution 3 definition.
L_B: float = 1.550519768e-8

# Defining rate difference between TCG (Geocentric Coordinate Time)
# and TT (Terrestrial Time): d(TT)/d(TCG) = 1 - L_G
# Value is exact by IAU 2000 Resolution B1.9 definition.
L_G: float = 6.969290134e-10

# ==============================================================================
# ATOMIC, NUCLEAR & ELECTROMAGNETIC CONSTANTS (CODATA 2018 / SI 2019)
# ==============================================================================

# Elementary charge (exact by SI definition, 26th CGPM 2019)
# Unit: C
E_CHARGE: float = 1.602176634e-19

# Electron rest mass (CODATA 2018)
# Unit: kg
M_ELECTRON: float = 9.1093837015e-31

# Proton rest mass (CODATA 2018)
# Unit: kg
M_PROTON: float = 1.67262192369e-27

# Neutron rest mass (CODATA 2018)
# Unit: kg
M_NEUTRON: float = 1.67492749804e-27

# Avogadro constant (exact by SI definition, 26th CGPM 2019)
# Unit: mol^-1
N_AVOGADRO: float = 6.02214076e23

# Vacuum electric permittivity (CODATA 2018)
# Unit: F / m = C^2 / (N * m^2)
EPSILON_0: float = 8.8541878128e-12

# Vacuum magnetic permeability (CODATA 2018: mu_0 = 1 / (epsilon_0 * c^2))
# Unit: N / A^2 = H / m
MU_0: float = 1.0 / (EPSILON_0 * (C_LIGHT ** 2))

# Classical electron radius: r_e = e^2 / (4 * pi * epsilon_0 * m_e * c^2)
# Unit: m
R_ELECTRON: float = (E_CHARGE ** 2) / (4.0 * math.pi * EPSILON_0 * M_ELECTRON * (C_LIGHT ** 2))

# Fine structure constant (CODATA 2018)
ALPHA_FS: float = 7.2973525693e-3

# ==============================================================================
# NOMINAL PLANETARY EQUATORIAL RADII (IAU 2015 Resolution B3)
# Unit: m
# ==============================================================================
RADIUS_SUN: float = 6.957e8
RADIUS_EARTH: float = 6.3781366e6
RADIUS_MARS: float = 3.39619e6
RADIUS_JUPITER: float = 7.1492e7
RADIUS_SATURN: float = 6.0268e7

