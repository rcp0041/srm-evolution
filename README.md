# srm-evolution
Evolutionary design optimization for solid rocket motors.

## Getting Started

1. Clone the repository and navigate to the root directory.
   1. `git clone git@github.com:rcp0041/srm-evolution.git`
2. (Recommended) Create a virtual environment for the project and activate it.
   1. `python3 -m venv .venv`
   2. `source .venv/bin/activate` (Linux) or
   `.venv\Scripts\activate` (Windows)
3. Install the project requirements.
   1. `pip install -e .`
4. Run a design optimizer, e.g. ./full-cp-sounding-rocket.py.

## Project Goals

Optimization of a conceptual design for a solid rocket motor, requiring only a high-level description of the performance parameter(s) to be optimized.

## Prerequisites

- Python 3
- NumPy >= 1.0
- SciPy >= 1.0
- srm >= 1.0
- LEAP

## Author

- Ray Patrick
