"""Orakel-Tests frequenzgang-lager-demo (ORACLE-PLAYBOOK §2 und §5, DEMO-PLAYBOOK §12): jede Kernroutine gegen einen anderen Rechenweg.

Die Übertragungsfunktionen H_u, H_i (`fgl_core`, z-Transformation) werden geprüft gegen
 (a) die Zustandsmatrix c (zI − A)^(−1) B, die hier **aus der Modellgleichung selbst** durch Einheitsvektoren gewonnen wird (nicht aus `fgl_core.state_space`),
 (b) die direkte Simulation mit Sinusbedarf (Betrag und Phase),
 (c) die Lyapunov-Gleichung (`scipy.linalg.solve_discrete_lyapunov`) und die geschlossenen Formeln für das weiße Rauschen,
 (d) die Winkelform von |P|² gegen die komplexe Auswertung,
 (e) die Zerlegung Saison plus Rauschen gegen die Simulation (SplitMix64, fester Zufalls-Seed),
 (f) die Einschwingzeit ln 0,01/ln ρ gegen die gemessene (Band),
 (g) die Formel mit Meldeverzug P_s gegen die Zustandsmatrix mit s > 0,
 (h) den Spektralradius gegen die Wurzeln des Polynoms und die Stabilitätsgrenze gegen die Simulation, den Verstärkungsbeginn gegen eine volle Rasterung.
Gegenproben für Absolutaussagen mit gestreuten Zufallsfällen (ORACLE-PLAYBOOK §6) stehen in `test_claims.py`.
Plattformabhängiges (BLAS, Gleitkomma-Summen) wird nur mit Toleranz oder als Band geprüft. β und α sind nirgends Zweierpotenzen außer in den benannten Randfällen."""

import math

import numpy as np
import pytest
from scipy.linalg import solve_discrete_lyapunov

import fgl_core as K
import fgl_sim as S

# ---------------------------------------------------------------------------------------------------------------------------------------
# Orakel (a): Zustandsmatrix aus der Modellgleichung, ohne fgl_core.state_space
# ---------------------------------------------------------------------------------------------------------------------------------------


def _step(x, e, L, s, alpha, beta):
    """Ein Schritt der Modellgleichungen. Zustand x = (i_t, i_{t-1}, ..., i_{t-s}, u_{t-1}, ..., u_{t-L+1}); Rückgabe (x', u_t)."""
    ni = s + 1
    i_hist, u_hist = list(x[:ni]), list(x[ni:])
    u_t = -alpha * (i_hist[s] + beta * sum(u_hist))
    arriving = u_t if L == 1 else u_hist[L - 2]          # u_{t+1-L}
    i_next = i_hist[0] + arriving - e
    new_i = [i_next] + i_hist[:-1] if s > 0 else [i_next]
    new_u = ([u_t] + u_hist[:-1]) if L > 1 else []
    return np.array(new_i + new_u), u_t


def _matrices(L, s, alpha, beta):
    n = (s + 1) + (L - 1)
    A = np.zeros((n, n))
    c = np.zeros(n)
    zero = np.zeros(n)
    for k in range(n):
        unit = np.zeros(n)
        unit[k] = 1.0
        nxt, ut = _step(unit, 0.0, L, s, alpha, beta)
        A[:, k] = nxt
        c[k] = ut
    B = _step(zero, 1.0, L, s, alpha, beta)[0]
    e0 = np.zeros(n)
    e0[0] = 1.0
    return A, B, c, e0


def _transfer(L, s, alpha, beta, z):
    A, B, c, e0 = _matrices(L, s, alpha, beta)
    x = np.linalg.solve(z * np.eye(len(B)) - A, B)
    return complex(c @ x), complex(e0 @ x)


def test_hu_hi_equal_the_state_space_transfer_on_a_grid():
    """(a) 12 L-Werte nicht nötig: 9 L, vier β (auch β > 1 und instabile α), fünf α, sieben Winkel: größte relative Abweichung unter 1e-11."""
    worst = 0.0
    count = 0
    for L in (1, 2, 3, 4, 5, 6, 8, 10, 12):
        for beta in (0.0, 0.3, 0.55, 1.3):
            for alpha in (0.13, 0.41, 0.9, 1.7, 2.3):
                for w in np.linspace(0.07, math.pi, 7):
                    z = np.exp(1j * w)
                    hu_s, hi_s = _transfer(L, 0, alpha, beta, z)
                    hu, hi = complex(K.Hu(z, L, alpha, beta)), complex(K.Hi(z, L, alpha, beta))
                    worst = max(worst, abs(hu - hu_s) / abs(hu_s), abs(hi - hi_s) / abs(hi_s))
                    count += 1
    assert count == 9 * 4 * 5 * 7
    assert worst < 1e-11, worst


def test_state_space_of_the_module_equals_the_one_from_the_model_equation():
    for (L, beta, alpha) in ((1, 0.2, 0.7), (2, 0.55, 0.9), (4, 1.0, 1.4), (6, 0.3, 0.2)):
        A, B, c, e0 = _matrices(L, 0, alpha, beta)
        A2, B2, c2, e02 = K.state_space(L, alpha, beta)
        assert np.allclose(A, A2) and np.allclose(B, B2) and np.allclose(c, c2) and np.allclose(e0, e02)


def test_meldeverzug_formula_matches_the_state_space_with_s_greater_than_zero():
    """(g) P_s, H_u, H_i mit Meldeverzug s = 1 bis 3 gegen die Zustandsmatrix mit s > 0; keine App-Funktion, nur Formelprüfung (gehört zu `meldeverzug-demo`)."""
    worst = 0.0
    count = 0
    for L in (1, 2, 3, 5):
        for s in (1, 2, 3):
            for beta in (0.0, 0.3, 0.55, 1.0):
                for alpha in (0.2, 0.6, 1.1):
                    for w in np.linspace(0.11, math.pi, 5):
                        z = np.exp(1j * w)
                        hu_s, hi_s = _transfer(L, s, alpha, beta, z)
                        hu, hi = complex(K.Hu(z, L, alpha, beta, s)), complex(K.Hi(z, L, alpha, beta, s))
                        worst = max(worst, abs(hu - hu_s) / abs(hu_s), abs(hi - hi_s) / abs(hi_s))
                        count += 1
    assert count == 4 * 3 * 4 * 3 * 5
    assert worst < 1e-11, worst


def test_at_beta_0_the_gain_depends_on_L_and_s_only_through_their_sum():
    z = np.exp(1j * 0.9)
    for total in (3, 5, 6):
        vals = [abs(K.Hu(z, total - s, 0.19, 0.0, s)) for s in range(0, total)]
        assert max(vals) - min(vals) < 1e-12


# ---------------------------------------------------------------------------------------------------------------------------------------
# Orakel (b): direkte Simulation mit Sinusbedarf
# ---------------------------------------------------------------------------------------------------------------------------------------


def _steady(L, alpha, beta, T, rho):
    w = 2.0 * math.pi / T
    tr = int(math.ceil(math.log(1e-13) / math.log(rho))) + 10
    M = 40 * T
    N = tr + M
    e = np.cos(w * np.arange(N))
    i_s, u_s = S.simulate(L, alpha, beta, e)
    t = np.arange(tr, N)
    fac = 1.0 if T == 2 else 2.0
    cu = fac * np.mean(u_s[tr:N] * np.exp(-1j * w * t))
    ci = fac * np.mean(i_s[tr:N] * np.exp(-1j * w * t))
    return cu, ci


def test_amplitude_and_phase_equal_the_simulation_in_60_cases():
    """(b) Simulation aus dem Ruhezustand, Einschwingen auf 1e-13, Fenster 40 Perioden; Betrag und Phase von Bestellung und Bestand."""
    from fgl_rng import SplitMix64

    rng = SplitMix64(19)
    periods = [2, 3, 4, 5, 6, 7, 10, 12, 22, 52]
    Ls = [1, 2, 3, 4, 6, 8, 12]
    betas = [0.0, 0.3, 0.55, 0.7, 1.0, 1.25]
    cases = []
    while len(cases) < 60:
        L = Ls[int(rng.random() * len(Ls))]
        beta = betas[int(rng.random() * len(betas))]
        alpha = 0.05 + 2.15 * rng.random()
        T = periods[int(rng.random() * len(periods))]
        rho = float(K.spectral_radius(L, alpha, beta))
        if rho < 0.99:
            cases.append((L, beta, alpha, T, rho))
    worst_amp = worst_phase = 0.0
    for (L, beta, alpha, T, rho) in cases:
        cu, ci = _steady(L, alpha, beta, T, rho)
        z = np.exp(2j * math.pi / T)
        hu, hi = complex(K.Hu(z, L, alpha, beta)), complex(K.Hi(z, L, alpha, beta))
        worst_amp = max(worst_amp, abs(cu - hu) / abs(hu), abs(ci - hi) / abs(hi))
        if T != 2:                                  # bei T = 2 ist die Phase der Simulation nicht auswertbar (cos(π t) = ±1)
            worst_phase = max(worst_phase, abs(np.angle(cu / hu)), abs(np.angle(ci / hi)))
    assert worst_amp < 1e-9 and worst_phase < 1e-9, (worst_amp, worst_phase)


# ---------------------------------------------------------------------------------------------------------------------------------------
# Orakel (c): weißes Rauschen
# ---------------------------------------------------------------------------------------------------------------------------------------


def test_mean_gain2_equals_the_lyapunov_equation_in_13_cases():
    worst = 0.0
    count = 0
    for (L, beta, alphas) in [(3, 0.0, (0.1, 0.3, 0.5)), (3, 0.55, (0.2, 0.9, 1.5)), (3, 1.0, (0.1, 0.7, 1.5)), (1, 1.0, (0.3,)), (8, 0.3, (0.3,)), (2, 0.0, (0.3, 0.9))]:
        for alpha in alphas:
            A, B, c, e0 = _matrices(L, 0, alpha, beta)
            P = solve_discrete_lyapunov(A, np.outer(B, B))
            lu, li = float(c @ P @ c), float(e0 @ P @ e0)
            mu, mi = K.mean_gain2(L, alpha, beta)
            worst = max(worst, abs(mu - lu) / lu, abs(mi - li) / li)
            count += 1
    assert count == 13 and worst < 1e-11, (count, worst)


def test_mean_gain2_equals_the_closed_forms():
    for L in (1, 3, 8):
        for alpha in (0.3, 0.7, 1.2, 1.5):
            mu, mi = K.mean_gain2(L, alpha, 1.0)
            assert mu == pytest.approx(alpha / (2 - alpha), rel=1e-12)                              # β = 1: var(u)/var(e) = α/(2−α)
            assert mi == pytest.approx((L - 1) + 1.0 / (alpha * (2 - alpha)), rel=1e-12)          # Bestandsvarianz
    for alpha in (0.3, 0.7, 0.9):
        mu, _ = K.mean_gain2(2, alpha, 0.0)
        assert mu == pytest.approx(alpha * (1 + alpha) / ((1 - alpha) * (2 + alpha)), rel=1e-12)  # L = 2, β = 0: AR(2)


# ---------------------------------------------------------------------------------------------------------------------------------------
# Orakel (e): Saison plus Rauschen
# ---------------------------------------------------------------------------------------------------------------------------------------


def test_season_plus_noise_variance_decomposition_in_a_band_of_1_percent():
    """400 000 Perioden, Zufalls-Seed 19, L = 3, β = 0,55, α = 0,4, T = 7, A = 1, σ = 0,5: var gegen A²|H|²/2 + σ² Mittel|H|²; über drei Zufalls-Seeds im Band."""
    for seed in (19, 20, 21):
        r = S.noise_run(3, 0.4, 0.55, 7, 1.0, 0.5, seed, 400_000)
        assert r["var_u"] == pytest.approx(r["pred_season_u"] + r["pred_noise_u"], rel=0.01)
        assert r["var_i"] == pytest.approx(r["pred_season_i"] + r["pred_noise_i"], rel=0.01)


def test_noise_run_is_reproducible_for_the_same_seed_and_differs_between_seeds():
    a = S.noise_run(3, 0.4, 0.55, 7, 1.0, 0.5, 35, 20_000)
    b = S.noise_run(3, 0.4, 0.55, 7, 1.0, 0.5, 35, 20_000)
    c = S.noise_run(3, 0.4, 0.55, 7, 1.0, 0.5, 36, 20_000)
    assert a == b and a["var_u"] != c["var_u"]


# ---------------------------------------------------------------------------------------------------------------------------------------
# Orakel (f): Einschwingzeit
# ---------------------------------------------------------------------------------------------------------------------------------------


def test_settling_time_of_the_resonance_matches_ln001_over_ln_rho_in_a_band():
    """Nahe der Grenze (q ≥ 0,7 der Grenze, Periode 4L − 2, L = 3): gemessen gegen ln 0,01/ln ρ im Band von 5 %. Für q = 0,5 gilt das nicht (20 gegen 23, siehe README)."""
    ak = K.alpha_crit_beta0(3)
    for q in (0.7, 0.9):
        a = q * ak
        r = S.season_run(3, a, 0.0, 10)
        assert r["settle_measured"] == pytest.approx(r["settle_estimate"], rel=0.05), q


# ---------------------------------------------------------------------------------------------------------------------------------------
# Orakel (h): Stabilität
# ---------------------------------------------------------------------------------------------------------------------------------------


def test_stable_limit_against_simulation_growth_and_decay():
    """Dicht unter der Grenze klingt ein Stoß ab, dicht darüber wächst er (Simulation des Stoßes e_0 = 1, 3000 Perioden), für β = 0 und β = 0,55."""
    for (L, beta) in ((3, 0.0), (6, 0.0), (4, 0.55), (3, 0.55)):
        lim = K.stable_limit(L, beta)
        for factor, grows in ((0.98, False), (1.02, True)):
            e = np.zeros(3000)
            e[0] = 1.0
            _, u = S.simulate(L, factor * lim, beta, e)
            late = np.abs(u[2500:]).max()
            early = np.abs(u[:500]).max()
            assert (late > early) == grows, (L, beta, factor, early, late)


def test_amplification_onset_equals_a_full_grid_scan_and_the_approximation():
    """Die Abkürzung `amplification_onset_measured` (Klammer um die Näherung) gegen eine volle Rasterung, die die Näherung nicht benutzt; Näherung und Messung auf 1e-3."""
    wscan = np.linspace(1e-5, math.pi, 20001)
    for (L, beta) in ((3, 0.0), (12, 0.0), (3, 0.55), (6, 0.7), (4, 0.3)):
        lim = K.stable_limit(L, beta)
        grid = np.linspace(1e-3, 0.999 * lim, 400)
        first = next(a for a in grid if float(np.max(K.gain_u(wscan, L, a, beta))) > 1 + 1e-9)
        measured = K.amplification_onset_measured(L, beta, lim)
        assert grid[np.searchsorted(grid, measured) - 1] < measured <= first + 1e-12, (L, beta)
        assert measured == pytest.approx(K.amplification_onset_formula(L, beta), abs=1e-3)


def test_peak_equals_a_dense_grid_maximum():
    wd = np.linspace(1e-6, math.pi, 400001)
    for (L, beta, alpha) in ((3, 0.0, 0.56), (2, 0.0, 0.9), (3, 0.55, 1.5), (3, 1.0, 1.5), (3, 0.0, 0.12)):
        pk, pT = K.peak(L, alpha, beta)
        gmax = float(np.max(K.gain_u(wd, L, alpha, beta)))
        assert pk == pytest.approx(max(gmax, 1.0), rel=1e-8)
    assert math.isinf(K.peak(3, 0.12, 0.0)[1]) and K.peak(3, 1.5, 1.0)[1] == 2.0
