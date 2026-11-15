"""Eigenschaftstests (ORACLE-PLAYBOOK §5.2): Aussagen, die für das Modell gelten müssen, ohne dass man das richtige Ergebnis kennt.

Jede Eigenschaft ist eine belegte Aussage über das lineare, zeitinvariante Modell (nicht nur das heutige Verhalten des Codes); sie gilt im stabilen Bereich.
Zufallsfälle kommen aus SplitMix64 mit festem Zufalls-Seed (nie Python-`hash()`); β und α sind keine Zweierpotenzen."""

import math

import numpy as np
import pytest

import fgl_core as K
import fgl_sim as S
from fgl_rng import SplitMix64

N_CASES = 30


def _cases(seed, n=N_CASES):
    """(L, β, α, T) mit stabilem Lager (Spektralradius unter 0,99)."""
    rng = SplitMix64(seed)
    out = []
    while len(out) < n:
        L = 1 + int(rng.random() * 8)
        beta = [0.0, 0.3, 0.55, 0.8, 1.0][int(rng.random() * 5)]
        alpha = 0.05 + 1.9 * rng.random()
        T = 2 + int(rng.random() * 40)
        if K.spectral_radius(L, alpha, beta) < 0.99:
            out.append((L, beta, alpha, T))
    return out


def test_dc_gain_of_the_order_is_exactly_one_and_of_the_stock_follows_the_formula():
    for (L, beta, alpha, T) in _cases(1):
        z1 = np.array([1.0 + 0j])
        assert abs(K.Hu(z1, L, alpha, beta)[0] - 1.0) < 1e-12
        assert K.Hi(z1, L, alpha, beta)[0].real == pytest.approx(-(1.0 + alpha * beta * (L - 1)) / alpha, rel=1e-11)


def test_gain_is_symmetric_under_complex_conjugation():
    for (L, beta, alpha, T) in _cases(2):
        z = np.exp(1j * 2 * math.pi / T)
        assert K.Hu(np.conj(z), L, alpha, beta) == pytest.approx(np.conj(K.Hu(z, L, alpha, beta)), rel=1e-12)
        assert K.Hi(np.conj(z), L, alpha, beta) == pytest.approx(np.conj(K.Hi(z, L, alpha, beta)), rel=1e-12)


def test_the_simulation_is_linear_double_demand_gives_double_response_and_sums_add():
    for (L, beta, alpha, T) in _cases(3, 12):
        n = 300
        t = np.arange(n)
        e1 = np.cos(2 * math.pi / T * t)
        e2 = np.sin(2 * math.pi / 13 * t)
        i1, u1 = S.simulate(L, alpha, beta, e1)
        i2, u2 = S.simulate(L, alpha, beta, e2)
        i3, u3 = S.simulate(L, alpha, beta, 2.0 * e1 + 3.0 * e2)
        assert u3 == pytest.approx(2.0 * u1 + 3.0 * u2, abs=1e-9 * max(1.0, np.abs(u3).max()))
        assert i3 == pytest.approx(2.0 * i1 + 3.0 * i2, abs=1e-9 * max(1.0, np.abs(i3).max()))


def test_time_shift_of_the_demand_shifts_the_steady_state_phase_only():
    """Bedarf um d Perioden verschoben: gleiche Amplitude, Phase um 2π d/T gedreht (eingeschwungen, ganzzahlige Verschiebung)."""
    for (L, beta, alpha, T) in _cases(4, 10):
        if T == 2:
            continue
        rho = float(K.spectral_radius(L, alpha, beta))
        w = 2 * math.pi / T
        tr = int(math.ceil(math.log(1e-12) / math.log(rho))) + 10
        M = 20 * T
        d = 3
        e0 = np.cos(w * np.arange(tr + M))
        e1 = np.cos(w * (np.arange(tr + M) - d))
        _, u0 = S.simulate(L, alpha, beta, e0)
        _, u1 = S.simulate(L, alpha, beta, e1)
        c0 = S.demodulate(u0, T, tr, M)
        c1 = S.demodulate(u1, T, tr, M)
        assert abs(c1) == pytest.approx(abs(c0), rel=1e-8)
        assert abs(c1 / c0 - np.exp(-1j * w * d)) < 1e-7                      # Drehung um −w d (als komplexe Zahl verglichen, damit ±π nicht stört)


def test_at_beta_1_the_order_gain_does_not_depend_on_L():
    for alpha in (0.37, 0.93, 1.45, 1.8):
        for T in (2, 3, 7, 19, 60):
            z = np.exp(2j * math.pi / T)
            ref = abs(K.Hu(z, 1, alpha, 1.0))
            for L in (2, 3, 6, 12):
                assert abs(K.Hu(z, L, alpha, 1.0)) == pytest.approx(ref, rel=1e-11)


def test_resonance_gain_grows_with_alpha_at_beta_0_for_every_L():
    for L in range(1, 13):
        ak = K.alpha_crit_beta0(L)
        gains = [K.hu_hi_at_period(L, q * ak, 0.0, 4 * L - 2)[0] for q in (0.21, 0.37, 0.55, 0.73, 0.91, 0.97)]
        assert all(b > a for a, b in zip(gains, gains[1:])), (L, gains)


def test_very_long_periods_give_gain_one():
    for (L, beta, alpha, T) in _cases(5, 10):
        hu = abs(K.Hu(np.exp(2j * math.pi / 1e6), L, alpha, beta))
        assert hu == pytest.approx(1.0, abs=1e-6)


def test_same_seed_same_orbit_and_larger_noise_larger_variance():
    a = S.noise_run(3, 0.4, 0.55, 7, 0.5, 0.3, 77, 30_000)
    b = S.noise_run(3, 0.4, 0.55, 7, 0.5, 0.3, 77, 30_000)
    assert a == b
    v1 = S.noise_run(3, 0.4, 0.55, 7, 0.0, 0.2, 77, 30_000)["var_u"]
    v2 = S.noise_run(3, 0.4, 0.55, 7, 0.0, 0.4, 77, 30_000)["var_u"]
    assert v2 == pytest.approx(4.0 * v1, rel=1e-9)             # gleiche Standardnormalwerte, doppeltes σ: vierfache Varianz


def test_rounding_noise_in_the_parameters_changes_nothing_visible():
    """Der Rundungstest: α und β um 1e-9 verschoben (auch an den Randwerten β = 1, T = 2 des Presets „Leitung voll“): die Kennzahlen ändern sich um weniger als 1e-6."""
    for (L, beta, alpha, T) in ((3, 1.0, 1.5, 2), (3, 0.55, 1.5, 7), (3, 0.0, 0.56, 10), (2, 0.0, 0.9, 6)):
        base = K.hu_hi_at_period(L, alpha, beta, T)
        for da, db in ((1e-9, 0.0), (-1e-9, 0.0), (0.0, 1e-9 if beta < 1 else -1e-9)):
            moved = K.hu_hi_at_period(L, alpha + da, beta + db, T)
            assert moved == pytest.approx(base, rel=1e-6, abs=1e-6)
        assert K.peak(L, alpha + 1e-9, beta)[0] == pytest.approx(K.peak(L, alpha, beta)[0], rel=1e-6)
