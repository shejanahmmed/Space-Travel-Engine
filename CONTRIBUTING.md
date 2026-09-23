# Contributing to the Relativistic Space Travel Computational Engine

Thank you for your interest in contributing to the **Relativistic Space Travel Computational Engine**.

This project maintains high standards of scientific rigor, reproducibility, and precision. We welcome contributions from researchers, computational physicists, astrodynamicists, and software engineers.

---

## 1. Guiding Scientific Principles

Every contribution must adhere to the core principles of the engine:
- **Evidence > Claims**: Never assert accuracy without quantitative proof. All changes to physical models must include analytical or benchmark validation.
- **Deterministic Physics**: Core scientific algorithms must remain deterministic and testable.
- **Explicit Units**: Use explicit SI units ($m$, $s$, $kg$, $rad$) and document the coordinate system, reference frame, and epoch.
- **Zero AI-Style Comments**: Code comments must focus on non-obvious mathematical rationale, physical invariants, and standards citations (IAU, IERS, BIPM, NIST). Avoid superficial or conversational comments.

---

## 2. Development Setup

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/shejanahmmed/Space-Travel-Engine.git
   cd Space-Travel-Engine
   ```

2. **Create a Virtual Environment**:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install in Editable Mode with Development Dependencies**:
   ```bash
   pip install --upgrade pip build wheel
   pip install -e ".[dev,web]"
   ```

---

## 3. Running Tests and Benchmarks

Before submitting any code, verify that all tests and benchmarks pass:

- **Execute the Full Test Suite**:
  ```bash
  pytest -v
  ```
  All tests must pass with zero regressions.

- **Execute Verification Benchmarks**:
  ```bash
  python benchmarks/milestone16_benchmark.py
  python benchmarks/milestone17_benchmark.py
  python benchmarks/milestone20_benchmark.py
  ```

---

## 4. Submitting Pull Requests

1. **Create a Topic Branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```
2. **Follow Conventional Commits**:
   - `feat(...)`: New feature or physics capability.
   - `fix(...)`: Bug fix or numerical correction.
   - `test(...)`: Additional test cases or validation benchmarks.
   - `docs(...)`: Documentation updates.
3. **Submit a Pull Request**:
   - Provide a clear technical description of the mathematical model, assumptions, and validation methodology.
