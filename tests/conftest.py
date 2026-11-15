"""Gemeinsame Fixtures: das Handbeispiel (L = 2, β = 0, α = 9/10), an dem jede Einheit von Hand geprüft wird (DEMO-PLAYBOOK §1b)."""

import sys
from fractions import Fraction
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture
def hand():
    """Lager L = 2, β = 0, α = 9/10 (α_krit = 1, kritische Periode 4L − 2 = 6). Handrechnung: P(z) = z² − z + α; bei z = e^{iπ/3} ist z² − z + 1 = 0, also P = α − 1 = −1/10
    und H_u = α z / (α − 1) = −9 z, |H_u| = 9, |H_i| = 10."""
    return {"L": 2, "beta": 0.0, "alpha": 0.9, "alpha_exact": Fraction(9, 10), "T": 6}
