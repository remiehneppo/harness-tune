# AGENTS.md

## Project: Harness Tuning & Agent Optimization Framework

### Commands
- Run verification: `python3 -m unittest discover -s tests -p "*_test.py"` (khi có test suite)
- Format check: `python3 -m py_compile **/*.py`

### Guidelines
- Follow `rule://*` standards.
- Every harness modification must follow `HarnessManifest` bounded editable surfaces.
- Never edit protected boundaries: `secrets/**`, `auth/**`, `billing/**`, `sandbox/**`.
- Optimization gate must enforce non-regression on held-out split: $\Delta p_{ho} \ge 0$.
