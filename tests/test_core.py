"""Kern gegen Handrechnung in Brüchen (DEMO-PLAYBOOK §1b): die Handbeispiele des Plans, mit `fractions` gerechnet, gegen `fgl_core` und `fgl_sim`.

Handbeispiel 1 bis 5: L = 2, β = 0, α = 9/10 (Periode 6 und 4, weißes Rauschen); β = 1, α = 3/2, L = 3 (Periode 2); L = 3, β = 0 (Resonanz bei Periode 10).
Der Ring Q[z]/(z² − z + 1) rechnet mit z = e^{iπ/3} exakt (z² = z − 1)."""

import cmath
import math
from fractions import Fraction as F

import numpy as np
import pytest

import fgl_core as K
import fgl_sim as S

# ---------- exakter Ring: Elemente a + b z mit z² = z − 1, |z| = 1 ----------


def mul(p, q):
    a, b = p
    c, d = q
    return (a * c - b * d, a * d + b * c + b * d)


def zpow(n):
    r = (F(1), F(0))
    for _ in range(n):
        r = mul(r, (F(0), F(1)))
    return r


def add(p, q):
    return (p[0] + q[0], p[1] + q[1])


def scal(k, p):
    return (k * p[0], k * p[1])


def abs2(p):
    """|a + b z|² = a² + b² + a b (weil z + conj z = 1)."""
    a, b = p
    return a * a + b * b + a * b


ALPHA = F(9, 10)


def test_P_at_period_6_is_alpha_minus_one_in_fractions():
    P = add(add(zpow(2), scal(F(-1), zpow(1))), (ALPHA, F(0)))
    assert P == (F(-1, 10), F(0))
    assert ALPHA / (ALPHA - 1) == -9 and abs2((F(0), ALPHA / (ALPHA - 1))) == 81     # H_u = -9 z, Betrag 9


def test_hand_example_gain_9_and_10_against_the_code(hand):
    hu, hi = K.hu_hi_at_period(hand["L"], hand["alpha"], hand["beta"], hand["T"])
    assert hu == pytest.approx(9.0, abs=1e-12) and hi == pytest.approx(10.0, abs=1e-12)
    assert K.alpha_crit_beta0(2) == pytest.approx(1.0, abs=1e-15)
    assert hu == pytest.approx(0.9 / (K.alpha_crit_beta0(2) - 0.9), rel=1e-12)        # α/(α_krit − α)


def test_periodic_solution_in_fractions_satisfies_both_equations():
    """u_t = −9 cos(π(t+1)/3), i_t = 10 cos(π(t+1)/3) bei Bedarf e_t = cos(π t/3), 24 Perioden exakt."""
    cos6 = [F(1), F(1, 2), F(-1, 2), F(-1), F(-1, 2), F(1, 2)]
    c = lambda k: cos6[k % 6]
    for t in range(24):
        assert 10 * c(t + 2) == 10 * c(t + 1) + (-9) * c(t) - c(t)              # i_{t+1} = i_t + u_{t-1} - e_t (L = 2)
        assert -9 * c(t + 1) == -ALPHA * (10 * c(t + 1))                          # u_t = -α i_t
    # Gegenprobe der Simulation (Gleitkomma) aus dem Ruhezustand: nähert sich der periodischen Lösung
    e = np.cos(math.pi * np.arange(2001) / 3)
    i_s, u_s = S.simulate(2, 0.9, 0.0, e)
    ref_i = np.array([10 * math.cos(math.pi * (t + 1) / 3) for t in range(2001)])
    assert np.max(np.abs(i_s[1500:] - ref_i[1500:])) < 1e-9
    assert K.settling_estimate(math.sqrt(0.9)) == pytest.approx(87.4, abs=0.05)      # ρ = √0,9, Einschwingzeit auf 1 %


def test_period_4_is_damped_81_over_101(hand):
    P2 = (ALPHA - 1) ** 2 + 1                       # |P(i)|² = (α−1)² + 1
    assert ALPHA ** 2 / P2 == F(81, 101)
    hu, _ = K.hu_hi_at_period(2, 0.9, 0.0, 4)
    assert hu == pytest.approx(math.sqrt(81 / 101), rel=1e-12) and hu < 1


def test_amplification_range_edge_is_cos_1_over_18():
    """Für L = 2 gilt |P|² = α² + 2(1−c)(1 − α(2c+1)), c = cos ω: verstärkt genau für cos ω > (1−α)/(2α) = 1/18, also T > 4,147."""
    c0 = F(1, 18)
    assert (1 - ALPHA) / (2 * ALPHA) == c0
    w_edge = math.acos(1 / 18)
    assert 2 * math.pi / w_edge == pytest.approx(4.1467, abs=1e-4)
    assert float(K.gain_u(w_edge, 2, 0.9, 0.0)) == pytest.approx(1.0, abs=1e-12)
    iv = K.amplified_intervals(2, 0.9, 0.0)
    assert len(iv) == 1 and math.isinf(iv[0][1]) and iv[0][0] == pytest.approx(2 * math.pi / w_edge, abs=1e-9)
    assert K.amplification_onset_formula(2, 0.0) == pytest.approx(1 / 3, abs=1e-15)    # Verstärkungsbeginn 1/(2L−1)


# ---------- weißes Rauschen exakt (Lyapunov-Gleichung in Brüchen) ----------


def _gauss(M, b):
    n = len(M)
    M = [row[:] + [b[i]] for i, row in enumerate(M)]
    for col in range(n):
        piv = next(r for r in range(col, n) if M[r][col] != 0)
        M[col], M[piv] = M[piv], M[col]
        pv = M[col][col]
        M[col] = [x / pv for x in M[col]]
        for r in range(n):
            if r != col and M[r][col] != 0:
                f = M[r][col]
                M[r] = [x - f * y for x, y in zip(M[r], M[col])]
    return [M[i][n] for i in range(n)]


def _lyap_exact(A, B):
    """P = A P A^T + B B^T in Brüchen (n² Unbekannte)."""
    n = len(A)
    M = [[F(0)] * (n * n) for _ in range(n * n)]
    rhs = [F(0)] * (n * n)
    for i in range(n):
        for j in range(n):
            row = i * n + j
            M[row][row] += 1
            for k in range(n):
                for m in range(n):
                    M[row][k * n + m] -= A[i][k] * A[j][m]
            rhs[row] = B[i] * B[j]
    sol = _gauss(M, rhs)
    return [[sol[i * n + j] for j in range(n)] for i in range(n)]


def _hand_state(L, a, beta):
    """Zustand (i_t, u_{t-1}, ..., u_{t-L+1}) von Hand aufgeschrieben (L >= 2): x' = A x + B e, u = c x."""
    cvec = [F(0)] * L
    cvec[0] = -a
    for j in range(1, L):
        cvec[j] = -a * beta
    A = [[F(0)] * L for _ in range(L)]
    A[0][0] = F(1)
    A[0][L - 1] += 1
    A[1] = cvec[:]
    for j in range(2, L):
        A[j][j - 1] = F(1)
    B = [F(0)] * L
    B[0] = F(-1)
    return A, B, cvec


def test_white_noise_variance_ratios_171_over_29_and_1900_over_261(hand):
    A, B, c = _hand_state(2, ALPHA, F(0))
    P = _lyap_exact(A, B)
    var_u = sum(c[i] * P[i][j] * c[j] for i in range(2) for j in range(2))
    assert var_u == ALPHA * (1 + ALPHA) / ((1 - ALPHA) * (2 + ALPHA)) == F(171, 29)
    assert P[0][0] == (1 + ALPHA) / (ALPHA * (1 - ALPHA) * (2 + ALPHA)) == F(1900, 261)
    mu, mi = K.mean_gain2(2, 0.9, 0.0)
    assert mu == pytest.approx(171 / 29, abs=1e-9) and mi == pytest.approx(1900 / 261, abs=1e-9)
    lu, li = S.lyapunov_variances(2, 0.9, 0.0)
    assert lu == pytest.approx(171 / 29, abs=1e-9) and li == pytest.approx(1900 / 261, abs=1e-9)
    assert 9.0 ** 2 > 171 / 29                      # das Quadrat der Resonanzverstärkung liegt weit über dem Mittel
    pk, pT = K.peak(2, 0.9, 0.0)
    assert pk ** 2 == pytest.approx(112.2, abs=0.05) and pT == pytest.approx(6.19, abs=0.005)


# ---------- β = 1, α = 3/2, L = 3, Periode 2 ----------


def test_beta_1_period_2_in_fractions():
    a4 = F(3, 2)
    P = F(1) * (F(-1) - 1 + a4)                                      # P = z^(L-1) (z − 1 + α) bei z = −1: −1/2
    assert P == F(-1, 2)
    assert a4 / P == -3 and -(F(1) + a4 * (F(-1) + 1)) / P == 2       # H_u(−1) = −3, H_i(−1) = 2
    s = lambda t: (-1) ** t
    for t in range(20):
        assert 2 * s(t + 1) == 2 * s(t) + (-3) * s(t - 2) - s(t)     # i_{t+1} = i_t + u_{t-2} − e_t (L = 3)
        assert -3 * s(t) == -a4 * (2 * s(t) + (-3 * s(t - 1)) + (-3 * s(t - 2)))
    A, B, c = _hand_state(3, a4, F(1))
    Pm = _lyap_exact(A, B)
    assert sum(c[i] * Pm[i][j] * c[j] for i in range(3) for j in range(3)) == 3 and Pm[0][0] == F(10, 3)
    assert K.hu_hi_at_period(3, 1.5, 1.0, 2) == pytest.approx((3.0, 2.0), abs=1e-12)
    pk, pT = K.peak(3, 1.5, 1.0)
    assert pk == pytest.approx(3.0, abs=1e-9) and pT == pytest.approx(2.0)                 # α/(2−α) bei Periode 2
    pk_lo, pT_lo = K.peak(3, 0.6, 1.0)
    assert pk_lo == pytest.approx(1.0) and math.isinf(pT_lo)                                # α = 3/5: kein Winkel über 1
    mu, mi = K.mean_gain2(3, 1.5, 1.0)
    assert mu == pytest.approx(3.0, abs=1e-9) and mi == pytest.approx(10 / 3, abs=1e-9)


# ---------- L = 3, β = 0: Resonanz bei Periode 10 ----------


def test_critical_period_10_and_gain_alpha_over_gap():
    ak = K.alpha_crit_beta0(3)
    assert ak == pytest.approx((math.sqrt(5) - 1) / 2, abs=1e-15)
    z = cmath.exp(1j * math.pi / 5)
    for a, gain in ((0.52, 5.30), (0.6, 33.27), (0.9 * ak, 9.00)):
        assert abs(z ** 3 - z ** 2 + a) == pytest.approx(ak - a, abs=1e-14)              # |P| = α_krit − α
        assert K.hu_hi_at_period(3, a, 0.0, 10)[0] == pytest.approx(gain, abs=0.005)
    rho = K.spectral_radius(3, 0.6, 0.0)
    assert rho == pytest.approx(0.99145, abs=1e-5) and K.settling_estimate(rho) == pytest.approx(536, abs=1)


# ---------- Einheiten ----------


def test_charpoly_equals_the_formula_and_the_companion_matrix_radius():
    for L in (1, 2, 3, 5):
        for beta in (0.0, 0.3, 0.55, 1.0):
            for alpha in (0.2, 0.9, 1.7):
                coeffs = np.zeros(L + 1)
                coeffs[0] = 1.0
                coeffs[1] -= 1 - alpha * beta
                coeffs[L] += alpha * (1 - beta)
                r_roots = float(np.max(np.abs(np.roots(coeffs)))) if L > 0 else 0.0
                assert K.spectral_radius(L, alpha, beta) == pytest.approx(r_roots, abs=1e-9)
                z = 0.7 + 0.4j
                assert K.charpoly(z, L, alpha, beta) == pytest.approx(np.polyval(coeffs, z), abs=1e-12)


def test_L1_has_radius_abs_one_minus_alpha_and_limit_2():
    for alpha in (0.3, 1.4, 1.9, 2.3):
        assert K.spectral_radius(1, alpha, 0.0) == pytest.approx(abs(1 - alpha))
        assert K.spectral_radius(1, alpha, 0.7) == pytest.approx(abs(1 - alpha))     # β ändert bei L = 1 nichts
    assert K.stable_limit(1, 0.0) == pytest.approx(2.0) and K.stable_limit(1, 0.7) == pytest.approx(2.0, abs=1e-9)
    assert K.stable_limit(3, 0.3) == pytest.approx(1.0158, abs=1e-4)


def test_stable_limit_closed_form_matches_the_eigenvalue_scan_at_beta_0():
    """Die geschlossene Formel 2 sin(π/(4L−2)) gegen den Spektralradius dicht davor und dahinter."""
    for L in (1, 2, 3, 4, 6, 8, 12):
        ak = K.alpha_crit_beta0(L)
        assert K.spectral_radius(L, 0.999 * ak, 0.0) < 1.0 < K.spectral_radius(L, 1.001 * ak, 0.0)
        assert K.stable_limit(L, 0.0) == ak


def test_dc_gain_and_the_angle_form_agree_with_the_complex_form():
    for (L, beta, alpha) in ((3, 0.0, 0.3), (3, 0.55, 0.4), (4, 1.0, 0.9), (1, 0.0, 0.7)):
        z1 = np.array([1.0 + 0j])
        assert abs(K.Hu(z1, L, alpha, beta)[0] - 1.0) < 1e-12
        assert K.Hi(z1, L, alpha, beta)[0].real == pytest.approx(K.dc_gain_i(L, alpha, beta), rel=1e-12)
        w = np.linspace(0.05, math.pi, 17)
        z = np.exp(1j * w)
        assert K.gain_u(w, L, alpha, beta) == pytest.approx(np.abs(K.Hu(z, L, alpha, beta)), rel=1e-10)
        assert K.gain_i(w, L, alpha, beta) == pytest.approx(np.abs(K.Hi(z, L, alpha, beta)), rel=1e-10)
        assert K.abs2_P_angle(w, L, alpha, beta) == pytest.approx(np.abs(K.charpoly(z, L, alpha, beta)) ** 2, rel=1e-9)


def test_state_space_matrices_reproduce_the_transfer_functions_on_the_mini_instance():
    A, B, c, e0 = K.state_space(2, 0.9, 0.0)
    z = cmath.exp(1j * math.pi / 3)
    x = np.linalg.solve(z * np.eye(2) - A, B)
    assert abs(c @ x) == pytest.approx(9.0, abs=1e-12) and abs(e0 @ x) == pytest.approx(10.0, abs=1e-12)
