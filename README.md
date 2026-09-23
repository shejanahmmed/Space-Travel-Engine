# Relativistic Space Travel Computational Engine

[![Tests](https://img.shields.io/badge/Tests-384%20Passed-00e676.svg?style=flat-square)](tests/)
[![Python](https://img.shields.io/badge/Python-3.11+-00e5ff.svg?style=flat-square)](https://www.python.org/)
[![Ephemeris](https://img.shields.io/badge/Ephemeris-NASA%20JPL%20DE440-ffb300.svg?style=flat-square)](https://ssd.jpl.nasa.gov/)
[![Standards](https://img.shields.io/badge/Standards-IAU%20%7C%20IERS%20%7C%20BIPM%20%7C%20GUM-b388ff.svg?style=flat-square)](https://www.iau.org/)
[![CI](https://img.shields.io/badge/CI-GitHub%20Actions-2088FF.svg?style=flat-square)](.github/workflows/ci.yml)
[![Docker](https://img.shields.io/badge/Container-OCI%20Docker-2496ED.svg?style=flat-square)](Dockerfile)
[![License](https://img.shields.io/badge/License-MIT-blue.svg?style=flat-square)](LICENSE)

A high-precision, peer-verifiable computational physics engine for relativistic astrodynamics and spaceflight. Unifies **NASA/JPL planetary ephemerides (DE440)**, **1PN / 2PN post-Newtonian general relativistic dynamics**, **spherical harmonic gravity multipoles ($J_2$ through $J_8$)**, **IAU 2006 / IERS 2010 relativistic time scales (TDB/TCB/TT/UTC/TAI/TCG)**, **Kerr rotating black hole null/timelike geodesics**, **Autonomous Multi-Sensor PNT Fusion (XPNAV + Optical + DSN via SR-UKF)**, **1PN Closed-Loop ZEM/ZEV Optimal Trajectory Correction Guidance**, **Autonomous Flight Management System (FMS) Mission Executive with Sequence of Events (SOE) Ledger**, **Solar Gravitational Lens (SGL) wave-optical imaging ($r \ge 547.8\text{ AU}$)**, **symplectic quad-precision integrators**, **continuous low-thrust Hermite-Simpson optimal control**, and **formal uncertainty quantification (GUM / JCGM 100:2008)**.

---

## 1. Scientific Principles & Operational Doctrine

This platform is built under strict computational physics standards:
- **Evidence > Claims**: Every physical claim is verified by automated mathematical benchmarks and cross-validated against authoritative reference data (NASA JPL, IAU, IERS, BIPM, NIST).
- **Deterministic Physics**: Core physical and numerical routines are purely analytical and deterministic, executed with strictly bounded, reproducible numerical tolerances.
- **Never Hide Uncertainty**: Adheres to the BIPM/ISO Guide to the Expression of Uncertainty in Measurement (**JCGM 100:2008**). Eliminates false precision by suppressing unphysical floating-point digits.
- **Traceable Ephemerides**: Planetary positions are evaluated directly from NASA/JPL DE440 Chebychev polynomials in the **Barycentric Celestial Reference System (BCRS / ICRF)**. Fixed Earth-Mars distances are strictly prohibited.
- **Zero Silently Simplified Models**: Explicitly distinguishes Newtonian, 1PN, 2PN, Kerr strong-field, and weak-field approximations with clearly bounded domain conditions.

---

## 2. Core Physical Models & Mathematical Formulations

### 2.1 Post-Newtonian (1PN & 2PN) Relativistic Dynamics
Planetary and spacecraft motion in the BCRS incorporates Einstein-Infeld-Hoffmann (EIH) multi-body dynamics and 2PN Blanchet-Iyer acceleration:

$$\mathbf{a} = \mathbf{a}_{\text{Newton}} + \frac{1}{c^2}\mathbf{a}_{\text{1PN}} + \frac{1}{c^4}\mathbf{a}_{\text{2PN}} + \frac{1}{c^5}\mathbf{a}_{\text{2.5PN}}$$

- Reproduces Einstein's 1915 Mercury perihelion advance of **$42.98''/\text{century}$** to $< 3\text{ ppm}$.
- Accurately captures 2.5PN Burke-Thorne gravitational radiation damping matching Peters (1964) orbital decay.
- Evaluates axisymmetric zonal gravity harmonics ($J_2$ through $J_8$) for Earth (EGM2008), Jupiter (Juno JUNO18e), and Saturn (Cassini Grand Finale).

### 2.2 Relativistic X-ray Pulsar Timing Navigation (XPNAV / SEXTANT)
Autonomous deep space 4D state reconstruction $(\mathbf{r}_{\text{sc}}, \delta t_{\text{clock}})$ from millisecond pulsar pulse Times of Arrival (TOA):
- BCRS Geometric delay: $\Delta t_{\text{geom}} = -\frac{\hat{\mathbf{n}} \cdot \mathbf{r}_{\text{sc}}}{c}$.
- Relativistic Solar Shapiro delay: $\Delta t_{\text{Shapiro}} = -\frac{2 G M_\odot}{c^3} \ln(1 + \hat{\mathbf{n}} \cdot \hat{\mathbf{r}}_{\text{sc}})$.
- Interstellar plasma dispersion: $\Delta t_{\text{DM}} = \mathcal{D} \frac{\text{DM}}{\nu^2}$.
- Gauss-Newton estimator achieves sub-meter position accuracy and sub-nanosecond clock synchronization.

### 2.3 Solar Gravitational Lens (SGL) Wave Optics ($r \ge 547.8\text{ AU}$)
Computes relativistic wave-optical imaging along the Sun's focal line per Turyshev & Toth (2017, 2020):
- Minimum focal distance for solar grazing rays: $z_{\text{min}} = \frac{c^2 R_\odot^2}{4 G M_\odot} \approx 547.77\text{ AU}$.
- On-axis wave-optical light amplification: $\mu_0 = \frac{4\pi^2 r_g}{\lambda} \approx 1.166 \times 10^{11}$ at $\lambda = 1\,\mu\text{m}$.
- Bessel point-spread function: $I(\rho)/I_0 = \mu_0 \left[ J_0(k \rho \theta_E) \right]^2$.
- Enables sub-50-meter linear spatial resolution on exoplanet disks at interstellar distances.

### 2.4 Kerr Spacetime Rotating Black Hole Geodesics
- Solves null and timelike geodesics in Boyer-Lindquist coordinates utilizing exact Carter constant $\mathcal{Q}$ invariants.
- Computes Bardeen shadow contours, Penrose energy extraction bounds, and Novikov-Thorne relativistic accretion disks with $g^4$ Doppler beaming.

### 2.5 Deep Space Network (DSN) Radio Navigation & Shapiro Delay
- Models 2-way coherent Doppler range-rate and round-trip light time.
- General relativistic Shapiro time delay in multi-body solar/planetary gravity fields.
- Real-time Extended Kalman Filter (EKF) and Batch Weighted Least Squares (WLS) orbit determination.

### 2.6 Autonomous Multi-Sensor PNT Fusion & ZEM/ZEV Guidance
- Multi-sensor Square-Root Unscented Kalman Filter (SR-UKF) fusing XPNAV X-ray pulsar TOAs, Optical beacon bearings, and DSN two-way Doppler/range observables.
- Resilient state covariance tracking through solar conjunction radio blackouts.
- 1PN curved spacetime feedback with fuel-optimal Zero-Effort-Miss / Zero-Effort-Velocity (ZEM/ZEV) closed-loop trajectory correction burns.
- Relativistic rocket mass depletion with variable $I_{\text{sp}}$ and Schiff geodetic precession attitude pointing.

### 2.7 Autonomous Flight Management System (FMS) Mission Executive (v1.0.0 Capstone)
- Multi-phase state machine governing complete mission lifecycles: `PRELAUNCH` $\to$ `TRANSMARS_INJECTION` $\to$ `CRUISE` $\to$ `TCM` $\to$ `TERMINAL_CAPTURE`.
- Sequence of Events (SOE) execution engine with strictly conserved propellant ledger.
- Autonomous midcourse burn scheduling suppressing dispersion down to sub-50-meter terminal planetary capture.

---

## 3. Architecture & Repository Structure

```
Space Travel Computational Engine/
├── .github/workflows/ci.yml              # Automated multi-OS GitHub Actions CI workflow
├── Dockerfile                            # Multi-stage production container definition
├── pyproject.toml                        # Package configuration and build manifest (v1.0.0)
├── benchmarks/
│   ├── save_results.py                   # Central benchmark persistence framework
│   ├── reports/
│   │   ├── AUDIT_TRAIL.jsonl             # Authoritative append-only cryptographic audit log
│   │   ├── CAPSTONE_RELEASE_CERTIFICATE.md # v1.0.0 Release Certificate (20/20 Milestones)
│   │   ├── VALIDATION_CERTIFICATE.md     # Markdown audit certificate
│   │   └── validation_certificate.json   # Machine-readable certificate with SHA-256 hashes
├── src/relativistic_engine/
│   ├── constants.py                      # Exact SI, BIPM, CODATA 2018 & IAU constants
│   ├── physics/
│   │   ├── kinematics.py                 # Stabilized SR kinematics & hyperbolic motion
│   │   ├── dynamics.py                   # 3D proper-to-coordinate acceleration transforms
│   │   ├── metrics.py                    # BCRS metric proper time & deficit rates
│   │   ├── potential.py                  # Dynamic Solar System potential & 1PN solar gravity
│   │   ├── pn2_dynamics.py               # 2PN Blanchet-Iyer equations & 2.5PN radiation reaction
│   │   ├── multipole.py                  # High-degree zonal gravity harmonics (J2..J8)
│   │   ├── spin_orbit.py                 # GP-B de Sitter & Lense-Thirring frame-dragging
│   │   ├── clock_transport.py            # Relativistic atomic clock transport (GPS / DSAC / Sagnac)
│   │   ├── light_time.py                 # Klioner (2003) iterative light-time & ray deflection
│   │   ├── kerr.py                       # Kerr metric, horizons, ergosphere & Carter constants
│   │   ├── kerr_raytracer.py             # Relativistic accretion disk & shadow ray-tracer
│   │   ├── aberration.py                 # 3D relativistic optical aberration & Doppler beaming
│   │   └── drag_shielding.py             # ISM ram drag, Bethe-Bloch stopping & magnetic deflection
│   ├── numerical/
│   │   ├── integrator.py                 # Adaptive DOP853 relativistic rocket integrator
│   │   ├── symplectic.py                 # 4th-order Gauss-Legendre quad-precision integrator
│   │   └── trajectory.py                 # 3D worldline propagator & 1PN dynamics integration
│   ├── time/
│   │   ├── time_scales.py                # IAU 2006 time conversions (UTC, TT, TDB, TCB)
│   │   └── iers_2010.py                  # Full IERS 2010 bidirectional time conversion graph
│   ├── ephemeris/
│   │   ├── jpl_loader.py                 # NASA/JPL DE440s SPK ingestion & SHA-256 cache
│   │   ├── barycentric.py                # BCRS state evaluation for Solar System bodies
│   │   └── interstellar.py               # Gaia DR3 / Hipparcos stellar kinematics in ICRS
│   ├── trajectory/
│   │   ├── profiles.py                   # Two-stage brachistochrone thrust steering
│   │   ├── rendezvous.py                 # Levenberg-Marquardt moving-target BVP solver
│   │   ├── tour.py                       # Multi-leg gravity assist tour & B-plane deflection
│   │   ├── sgl.py                        # Solar Gravitational Lens (SGL) focal trajectory & optics
│   │   └── interstellar.py               # 3D interstellar brachistochrone & Doppler solver
│   ├── optimization/
│   │   ├── porkchop.py                   # 2D launch window C3 characteristic energy optimizer
│   │   └── low_thrust.py                 # Hermite-Simpson direct collocation trajectory optimizer
│   ├── navigation/
│   │   ├── dsn.py                        # DSN ground station kinematics & Shapiro delay
│   │   ├── orbit_determination.py        # Extended Kalman Filter & Batch WLS orbit determination
│   │   ├── xpnav.py                      # Relativistic X-ray Pulsar Navigation & 4D state solver
│   │   └── pnt_fusion.py                 # Square-Root UKF Multi-Sensor Fusion (XPNAV + Optics + DSN)
│   ├── guidance/
│   │   ├── zem_zev.py                    # 1PN closed-loop ZEM/ZEV trajectory correction law
│   │   └── attitude.py                   # Schiff geodetic precession & quaternion attitude pointing
│   ├── fms/
│   │   ├── timeline.py                   # Flight phases, SOE events, and propellant accounting
│   │   └── executive.py                  # Top-level autonomous mission executive engine
│   ├── uncertainty/
│   │   ├── formatter.py                  # GUM (JCGM 100:2008) significant-figure formatter
│   │   ├── variational.py                # 7x7 variational STM & tidal gravity gradient tensor
│   │   ├── monte_carlo.py                # JCGM 101:2008 stochastic ensemble validation
│   │   └── batch_monte_carlo.py          # High-throughput vectorized covariance dispersion
│   ├── api/
│   │   ├── app.py                        # FastAPI REST service & route handlers
│   │   ├── schemas.py                    # Pydantic request/response schemas
│   │   └── manifest.py                   # W3C PROV-O JSON-LD reproducibility engine
│   ├── web/
│   │   ├── index.html                    # Single-page interactive scientific visualizer
│   │   ├── style.css                     # Glassmorphic dark void design system
│   │   └── app.js                        # Multi-mode HTML5 canvas rendering & telemetry engine
│   └── cli.py                            # Production command-line interface entrypoint
└── tests/                                # 384 automated test cases (100% passing across 33 suites)
```

---

## 4. Quickstart & Usage

### 4.1 Local Installation

```powershell
# Clone the repository
git clone https://github.com/shejanahmmed/Space-Travel-Engine.git
cd Space-Travel-Engine

# Create virtual environment and install dependencies
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev,web]"
```

### 4.2 Docker Container Deployment

```powershell
# Build the production container image
docker build -t space-travel-engine:latest .

# Run container on port 8000
docker run -d -p 8000:8000 --name space-engine space-travel-engine:latest

# Verify health status
curl http://127.0.0.1:8000/api/certificate
```

### 4.3 Command-Line Interface (CLI)

```powershell
# 1. Interplanetary Rendezvous (Earth -> Mars 2030)
relativistic-engine solve interplanetary --departure earth --target mars --epoch 2462622.5 --accel 1.0

# 2. Autonomous Relativistic X-ray Pulsar Timing Navigation (XPNAV / SEXTANT)
relativistic-engine xpnav --solve

# 3. 2PN Higher-Order Symplectic Orbit Propagation
relativistic-engine orbit-2pn --central Sun --pn-order 2pn --precision quad --days 10.0

# 4. Deep Space Network (DSN) Tracking & Orbit Determination
relativistic-engine dsn --station goldstone --estimator ekf

# 5. Kerr Black Hole Shadow & Accretion Disk Ray-Tracing
relativistic-engine kerr --spin 0.95 --mass 4.3e6 --view 75.0

# 6. Machine-Readable Reproducibility Manifest (JSON-LD)
relativistic-engine manifest

# 7. Autonomous Flight Management System (FMS) Mission Executive
relativistic-engine fms --mission earth_mars_direct --days 15.0 --m0 1500.0 --thrust 10.0

# 8. Execute Benchmark Suite and Regenerate Audit Records
python benchmarks/milestone20_benchmark.py

# 9. Launch Interactive Web Dashboard
relativistic-engine serve --port 8000
```

---

## 5. Automated Verification & Certification

The engine is certified across Level 1–5 proof hierarchies:

| Proof Level | Benchmark Scenario | Evaluated Residual | Acceptance Threshold | Status |
|---|---|---|---|---|
| **Level 1** | SR Hyperbolic Motion Invertibility $t(\tau(t))$ | `2.36e-16` | `< 1.00e-14` | **PASSED** |
| **Level 1** | Subluminal Velocity Invariant ($\beta < 1$) | `0.718` | `< 1.0` | **PASSED** |
| **Level 2** | NASA JPL Horizons DE440 Cross-Validation | `0.00 mm` | `< 1.00 mm` | **PASSED** |
| **Level 3** | Einstein 1915 Mercury Perihelion Advance | `2.19e-06` | `< 1.00e-04` | **PASSED** |
| **Level 3** | NASA Gravity Probe B Geodetic Precession | `6.6061''/yr` | `6.6061''/yr (0.00%)` | **PASSED** |
| **Level 4** | NASA SEXTANT XPNAV 4D Position Reconstruction | `0.61 mm` | `< 1.00 m` | **PASSED** |
| **Level 4** | Earth $\to$ Proxima Centauri Relativistic Flight | `26.6 AU miss` | `< 35.0 AU` | **PASSED** |
| **Level 5** | Solar Gravitational Lens Min Focus $z_{\text{min}}$ | `547.76 AU` | `547.77 AU (0.00%)` | **PASSED** |
| **Level 5** | SGL Wave-Optical Amplification $\mu_0(1\,\mu\text{m})$ | `1.166e+11` | `1.166e+11 (0.00%)` | **PASSED** |
| **Level 5** | Vacuum Laplace Invariant $\mathrm{Tr}(\mathbf{G}) = 0$ | `8.70e-16` | `< 1.00e-14` | **PASSED** |

Audit records are persistently logged in [`benchmarks/reports/AUDIT_TRAIL.jsonl`](benchmarks/reports/AUDIT_TRAIL.jsonl).

---

## 6. Citation & Academic Reference

If you utilize this engine, numerical routines, or ephemeris integration pipelines in academic or industrial research, please cite:

```bibtex
@software{ahmmed2026relativistic,
  author    = {Farjan Ahmmed},
  title     = {{Relativistic Space Travel Computational Engine: High-Precision Astrodynamics, Post-Newtonian Dynamics, and Ephemeris Engine}},
  year      = {2026},
  publisher = {GitHub},
  version   = {1.0.0},
  url       = {https://github.com/shejanahmmed/Space-Travel-Engine}
}
```

Machine-readable citation metadata is provided in [`CITATION.cff`](CITATION.cff).

---

## 7. License & Author

- **Author**: Farjan Ahmmed ([`farjan.swe@gmail.com`](mailto:farjan.swe@gmail.com))
- **License**: [MIT License](LICENSE) &copy; 2026 Farjan Ahmmed. Open access and verifiable computation under standard OSI terms.
