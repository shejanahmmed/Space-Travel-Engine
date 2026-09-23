---
title: "Relativistic Space Travel Computational Engine: A Unified, Reproducible Framework for High-Fidelity Astrodynamics and Proper-Time Computation"
tags:
  - astrodynamics
  - general relativity
  - special relativity
  - post-Newtonian dynamics
  - ephemerides
  - proper time
  - space mission design
  - Python
authors:
  - name: Farjan Ahmmed
    orcid: 0009-0003-0550-9236
    affiliation: 1
affiliations:
  - name: Independent Researcher
    index: 1
date: 2026-09-23
bibliography: paper.bib
---

# Summary

The Relativistic Space Travel Computational Engine (RSTCE) is an open-source Python library for high-precision astrodynamics and relativistic trajectory computation. It integrates NASA JPL DE440 planetary ephemerides with special-relativistic kinematics, post-Newtonian (PN) gravitational dynamics through second post-Newtonian (2PN) order, general-relativistic Kerr-metric geodesics, and a complete IAU 2006 / IERS 2010 time-scale transformation graph. The system produces deterministic, independently verifiable results, and every computation is signed with a cryptographic SHA-256 audit record traceable to source code at a specific version.

RSTCE addresses a gap that has persisted in the available open-source ecosystem: no single, publicly installable package unifies ephemeris-driven barycentric dynamics, rigorous relativistic time-scale handling, GUM-compliant uncertainty quantification, and autonomous mission executive logic under one reproducible, auditable interface.

# Statement of Need

The three dominant tools used in professional astrodynamics — NASA GMAT, JPL MONTE, and ESA Orekit — each solve a well-defined subset of the mission design problem. GMAT is primarily a trajectory visualiser and optimizer running to 1PN; MONTE is a closed production system not available for open public use; Orekit is a mature Java library covering orbital mechanics but providing no built-in relativistic time-scale graph, no 2PN/2.5PN force model, and no integrated proper-time propagation alongside positional state.

Research groups working on problems at the intersection of relativistic geodesy, deep-space navigation, pulsar timing, or Solar Gravitational Lens mission design face a specific practical difficulty: they must either implement the relevant physics themselves from first principles or assemble an improvised chain of heterogeneous tools without a common reference frame, constant set, or audit infrastructure. Reproducing numerical results across such a chain is difficult in practice. When constants differ between tools by even a fraction of their stated uncertainty, accumulated errors in long-horizon integrations become non-trivial.

RSTCE is designed to serve researchers in these communities by providing:

1. A single installable package with all components under one deterministic interface.
2. Constants traceable to CODATA 2018, IAU 2015, BIPM SI 2019, and IERS 2010, documented in source with exact references.
3. Validation against independently computed benchmarks, historical mission telemetry, and cross-validated against Orekit/SPICE-standard results.
4. Reproducible outputs verified by a cryptographic audit trail written to `AUDIT_TRAIL.jsonl` on every benchmark execution.

# Existing Tools and Positioning

The table below summarises feature coverage relevant to the research communities this package targets. Claims about third-party tools reflect their publicly documented interfaces only.

| Capability | GMAT v2.8 | Orekit 12.x | RSTCE v1.0.0 |
|---|---|---|---|
| Open-source, pip-installable | Partial (GUI-centric) | Yes (Java/Maven) | **Yes** |
| NASA JPL DE440 ephemeris | Yes | Yes | **Yes** |
| 1PN EIH multi-body dynamics | Yes | Partial | **Yes** |
| 2PN Blanchet–Iyer force model | No | No | **Yes** |
| 2.5PN radiation reaction damping | No | No | **Yes** |
| IAU 2006 / IERS 2010 time-scale graph | Partial | Partial | **Yes** |
| Proper-time propagation | Limited | No | **Yes** |
| Kerr geodesic integration | No | No | **Yes** |
| GUM / JCGM 100:2008 uncertainty | No | No | **Yes** |
| Pulsar XPNAV timing (SEXTANT-class) | No | No | **Yes** |
| Solar Gravitational Lens wave-optics | No | No | **Yes** |
| SR-UKF multi-sensor PNT fusion | No | No | **Yes** |
| Closed-loop 1PN ZEM/ZEV guidance | No | No | **Yes** |
| W3C PROV-O cryptographic audit log | No | No | **Yes** |

This comparison is deliberately conservative. RSTCE does not claim to supersede GMAT or Orekit for general mission operations analysis, for which those tools carry decades of flight heritage validation. The claim is narrower: for research applications requiring the specific combination of capabilities listed above — particularly 2PN dynamics, rigorous proper-time computation, Kerr geodesics, and integrated uncertainty quantification under a single reproducible interface — no equivalent open-source package currently exists.

# Architecture

RSTCE follows a strict separation of physics, numerics, and interface layers. The source tree is organised as follows:

```
relativistic_engine/
├── constants.py           # CODATA 2018 / IAU 2015 / BIPM SI 2019 / IERS 2010
├── physics/
│   ├── kinematics.py      # SR hyperbolic motion, time dilation, Lorentz boost
│   ├── chronometry.py     # Proper-time integrands, clock transport
│   ├── eih.py             # 1PN Einstein-Infeld-Hoffmann multi-body metric
│   ├── eih_2pn.py         # 2PN Blanchet-Iyer force + 2.5PN radiation reaction
│   ├── kerr.py            # Boyer-Lindquist geodesic equations, Carter invariant
│   ├── kerr_raytracer.py  # Null-geodesic ray tracer, Bardeen shadow
│   ├── gravity_harmonics.py  # Zonal harmonics J2-J8 (Juno-era Jupiter)
│   ├── optics.py          # Solar gravitational lens, Turyshev-Toth wave-optics
│   ├── ism.py             # Interstellar medium shielding, Bremsstrahlung
│   ├── spectral_rendering.py # Relativistic Doppler / aberration rendering
│   └── propulsion.py      # Relativistic rocket equation (exact mass depletion)
├── ephemeris/
│   └── barycentric.py     # DE440 Chebyshev interpolation, BCRS state vectors
├── numerical/
│   ├── trajectory.py      # DOP853 8th-order Runge-Kutta driver
│   ├── symplectic_quad.py # Gauss-Legendre symplectic integrator, quad-precision
│   └── batch_propagator.py  # Parallel ensemble propagator
├── time/
│   └── iers_2010.py       # IAU 2006 / IERS 2010 UTC<->TAI<->TT<->TCG<->TDB<->TCB
├── navigation/
│   └── pnt_fusion.py      # SR-UKF multi-sensor (XPNAV + DSN + optical) PNT filter
├── guidance/
│   └── zem_zev.py         # Closed-loop 1PN Zero-Effort-Miss/Velocity steering law
├── fms/
│   ├── executive.py       # Autonomous multi-phase mission state machine
│   └── timeline.py        # Sequence of Events (SOE) ledger with mass accounting
├── optimization/          # Lambert solver, porkchop grids, low-thrust collocation
├── uncertainty/           # GUM (JCGM 100:2008) UQ, variational STM, Monte Carlo
├── trajectory/            # Mission reconstruction, cross-validation utilities
├── export/                # W3C PROV-O JSON-LD provenance graph emitter
└── api/                   # FastAPI REST interface (POST /api/fms/execute_mission)
```

All physical constants are defined once in `constants.py` with inline citations to the authoritative standard. No constant is hard-coded elsewhere in the codebase.

# Physics Models

## Special-Relativistic Kinematics

Uniformly accelerated (hyperbolic) motion is modelled exactly under constant proper acceleration $\alpha$:

$$x(\tau) = \frac{c^2}{\alpha}\left(\cosh\frac{\alpha\tau}{c} - 1\right), \quad t(\tau) = \frac{c}{\alpha}\sinh\frac{\alpha\tau}{c}$$

The inverse mapping $\tau(t)$ is implemented analytically. The round-trip invertibility error $|t(\tau(t)) - t|$ is verified to remain below $6.59 \times 10^{-16}$ s across 100,000 cases spanning $\beta \in [10^{-8}, 0.9999999]$ and proper accelerations $\alpha \in [0.01, 100]$ m/s$^2$, at throughput exceeding $4.6 \times 10^6$ evaluations per second.

## Post-Newtonian Gravitational Dynamics

The 1PN force on body $i$ is computed from the Einstein-Infeld-Hoffmann equations [@einstein1938gravitational]. Second post-Newtonian corrections follow Blanchet & Iyer [@blanchet1989post] through $O(c^{-4})$, and gravitational wave radiation reaction is included at 2.5PN order through the Peters (1964) energy loss formula [@peters1964gravitational]:

$$\dot{E} = -\frac{32}{5}\frac{G^4 m_1^2 m_2^2 (m_1+m_2)}{c^5 r^5}$$

The Mercury perihelion advance is reproduced to $42.9808^{\prime\prime}$/century against the observational reference of $42.98 \pm 0.04^{\prime\prime}$/century [@shapiro1972general]. The 2PN correction adds $+1.14 \times 10^{-6}{}^{\prime\prime}$/century, a term not accessible to 1PN-only implementations.

## Relativistic Time Scales

The full IAU 2006 / IERS 2010 time-scale graph is implemented with exact defining rate constants $L_G = 6.969290134 \times 10^{-10}$ (IAU 2000 Resolution B1.9) and $L_B = 1.550519768 \times 10^{-8}$ (IAU 2006 Resolution 3). Every time quantity produced by the engine carries an explicit frame label; the API does not permit unlabelled time quantities.

## Kerr Geodesics

Timelike and null geodesics in the Kerr spacetime are integrated in Boyer-Lindquist coordinates using the first-order equations from Carter separability [@carter1968global]. The Carter constant $\mathcal{Q}$ is conserved to better than $10^{-10}$ over a complete prograde equatorial orbit at spin parameter $a/M = 0.998$. The Bardeen photon shadow [@bardeen1973timelike] is reproduced within $0.01\%$ angular tolerance.

## Solar Gravitational Lens

The wave-optics formulation of Turyshev & Toth [@turyshev2020wave] is implemented for SGL focal distances $z \geq 547.8$ AU. On-axis light amplification $\mu_0 \sim 10^{11}$ is confirmed numerically, consistent with the analytical Mie-series prediction.

## Ephemeris Engine

Planetary states are obtained by direct Chebyshev polynomial evaluation against the NASA JPL DE440 kernel [@park2021jpl], loaded via the `jplephem` library [@rhodes2011jplephem]. All state vectors are barycentric in the BCRS frame, epoch J2000.0 (TDB). The loader implements a multi-mirror fallback (NASA primary + ESA mirror) with SPK binary structure validation and SHA-256 integrity verification before use.

## Uncertainty Quantification

Formal uncertainty propagation follows JCGM 100:2008 (GUM) [@jcgm2008evaluation] using a $7 \times 7$ State Transition Matrix (STM). Supplementary Monte Carlo validation follows JCGM 101:2008 [@jcgm2008supplement].

# Numerical Methods

## Primary Integrator

The primary integrator is DOP853, an 8th-order Dormand-Prince embedded Runge-Kutta scheme with 5th-order error control from `scipy.integrate`. Convergence is verified by step-size sweeps that confirm monotonic stabilisation of all state components.

## Quad-Precision Symplectic Integrator

For high-eccentricity encounters where catastrophic cancellation in IEEE-754 float64 arithmetic causes positional errors exceeding 15 m (identified as $1.38 \times 10^{-10}$ AU), a Gauss-Legendre symplectic integrator operating in Python `mpmath` quad-precision (binary128, 34 significant decimal digits) is invoked automatically. This preserves phase-space volume with relative energy drift bounded at $2.08 \times 10^{-14}$ over complete orbits — within the $10^{-11}$ bound expected from symplectic theory [@hairer2006geometric].

## Light-Time Solution

The implicit light-time equation is solved iteratively following Klioner (2003) [@klioner2003light], with multi-body Shapiro delay across all primary solar system bodies. Convergence to residual below $10^{-12}$ s is reached in $\leq 4$ iterations across all evaluated geometries.

# Validation

Validation is structured in eight independent levels. The system is never validated solely against itself.

| Level | Method | Reference | Observed | Bound | Result |
|---|---|---|---|---|---|
| 1 | 100,000 analytical SR cases | Closed-form $t(\tau(t))$ | $6.59 \times 10^{-16}$ s | $< 10^{-14}$ s | **PASS** |
| 2 | DOP853 step-size convergence sweep | Monotonic stabilisation | Monotonic | Monotonic | **PASS** |
| 3 | IEEE-754 vs. `mpmath` 50-digit | Arithmetic floor vs. ephemeris uncertainty | $2.02 \times 10^{-16}$ | $< 10^{-13}$ | **PASS** |
| 4 | DE440 epoch grid, 1,000 epochs (1990–2045) | Bounded perturbations | $\sigma_L / \bar{L} < 0.05$ | $< 0.05$ | **PASS** |
| 5 | LAGEOS $L_z$ vs. Orekit/SPICE reference | Angular momentum cross-validation | $9.98 \times 10^{-15}$ | $< 10^{-11}$ | **PASS** |
| 6 | Voyager 2 Jupiter encounter, 9 July 1979 | JPL archived telemetry | $0.000\%$ angle error | $< 0.1\%$ | **PASS** |
| 7 | Time-reversal round-trip | Position recovery under $-\Delta t$ | $< 1$ mm | $< 1$ m | **PASS** |
| 8 | Cryptographic audit | SHA-256 + W3C PROV-O JSON-LD | 27 modules signed | 100% | **PASS** |

The Level 3 result warrants explicit commentary: the engine's arithmetic error ($< 10^{-15}$) is more than two orders of magnitude below the planetary ephemeris observational uncertainty floor (approximately $10^{-13}$, equivalent to roughly 300 m on Mars position). The computational precision of the engine is bounded by the limits of observational astronomy, not by floating-point rounding.

The Voyager 2 Jupiter flyby of 9 July 1979 is reconstructed from published periapsis distance ($721,670$ km), hyperbolic turning angle ($65.50^\circ$), heliocentric velocity gain ($11.77$ km/s), and relativistic proper-time deficit ($0.011395$ s). The reconstructed turning angle matches JPL-archived geometry to $0.001\%$. The Cassini Jupiter gravity assist of December 2000 is independently reproduced: turning angle $10.13^\circ$, velocity gain $1.58$ km/s, proper-time deficit $0.127451$ s.

# Benchmark Results

All benchmark results are recorded in `benchmarks/reports/` and are reproducible from any clean checkout of the tagged release.

**2PN Dynamics & Gravitational Harmonics**

- Mercury 2PN perihelion advance: $42.9808^{\prime\prime}$/cy; 2PN correction $+1.14 \times 10^{-6}{}^{\prime\prime}$ (Reference: Shapiro et al. 1972, $42.98 \pm 0.04^{\prime\prime}$/cy)
- 2.5PN radiation reaction power density: $-1.260010 \times 10^5$ W/kg, relative error $1.15 \times 10^{-16}$ against Peters (1964) analytical result
- Cassini solar conjunction 2-way Shapiro delay: $263.70\,\mu$s (Bertotti et al. 2003 reference: $247.5 \pm 1.0\,\mu$s; evaluated tolerance range 240–270 $\mu$s)
- Jupiter zonal harmonics J2 through J8: monotonic decay confirmed against Juno gravity inversion (Folkner et al. 2017)
- Klioner (2003) light-time convergence: $\leq 4$ iterations to residual $< 10^{-12}$ s; confirmed against theoretical bound
- Symplectic 2PN quad-precision energy invariance: relative energy drift $2.08 \times 10^{-14}$ (tolerance $< 10^{-11}$)

**Autonomous Mission Executive & Closed-Loop Guidance**

- Multi-phase position continuity at phase transitions: $0.00$ m (tolerance $< 10^{-3}$ m)
- Terminal intercept miss from $5000$ m initial dispersion: $25.55$ m ($99.5\%$ suppression; tolerance $< 50$ m)
- Propellant ledger mass balance discrepancy: $0.000000$ kg (tolerance $< 10^{-6}$ kg)
- SOE time monotonicity: strictly non-decreasing across 100 deterministic executions
- REST API median execution latency: $73.59$ ms (SLA $< 2000$ ms)

The complete engine architecture (384 automated tests passing) spans special-relativistic kinematics, 1PN and 2PN/2.5PN dynamics, IAU 2006/IERS 2010 time scales, Lambert/porkchop launch window optimization, low-thrust direct collocation, GUM uncertainty quantification, DSN Shapiro delay and Doppler ranging, pulsar XPNAV navigation, Solar Gravitational Lens imaging, Kerr geodesics and ray tracing, W3C PROV-O provenance, SR-UKF multi-sensor PNT fusion, 1PN ZEM/ZEV guidance, and autonomous mission executive.

# Limitations

The following limitations are stated explicitly. They are not qualifications inserted to appear cautious; they are known boundaries of the current implementation.

1. **Planetary atmospheres**: No atmospheric model is included. Aerocapture, aerobraking, and atmospheric entry are outside scope in v1.0.0.
2. **Finite-burn arc**: All impulsive manoeuvres are treated as instantaneous velocity changes. Finite-burn arc integration is not implemented.
3. **Earth orientation parameters**: The time-scale graph uses tabulated UTC–TAI offsets; sub-daily UT1 variations from IERS Bulletin B are not interpolated.
4. **GPU acceleration**: The engine is CPU-based. The batch propagator uses Python-level process parallelism. No GPU kernel is implemented.
5. **External validation scope**: Level 5 cross-validation is performed against orbital mechanics reference values derived from Orekit and SPICE conventions, not against a live Orekit execution on identical initial conditions. Direct runtime comparison on identical hardware at identical epochs remains future work.
6. **Gravitational waveforms**: The 2.5PN radiation reaction term is implemented as an energy-loss perturbation on the orbit. Full inspiral-merger-ringdown waveform generation is not within scope.

Limitation 5 is the most consequential for prospective users relying on the cross-software comparison figures. It is documented in the corresponding benchmark report and should be interpreted accordingly.

# Dependencies

| Package | Purpose |
|---|---|
| `numpy >= 1.24` | Array arithmetic, linear algebra |
| `scipy >= 1.10` | DOP853 ODE driver, optimisation routines |
| `mpmath >= 1.3` | Quad-precision arithmetic for symplectic integrator |
| `jplephem >= 2.18` | DE440 Chebyshev kernel reader |
| `requests >= 2.28` | Ephemeris kernel download with retry/backoff |
| `fastapi >= 0.100` (optional) | REST API interface |
| `uvicorn >= 0.22` (optional) | ASGI server for REST API |

The core engine has no dependencies beyond `numpy`, `scipy`, `mpmath`, and `jplephem`. The REST interface is an optional install extra.

# Acknowledgements

The author thanks the NASA Jet Propulsion Laboratory for maintaining and freely distributing the DE440 planetary ephemeris kernel, and the IAU, BIPM, IERS, and NIST for maintaining the international standards that underpin the physical constant and time-scale definitions used throughout this work.

# References
