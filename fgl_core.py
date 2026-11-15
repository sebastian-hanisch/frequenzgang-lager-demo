"""Kern: Frequenzgang des Lagers mit linearer Bestellregel (nur numpy und scipy).

Modell (Abweichungen vom Soll und vom Mittelbedarf, Zählung in Perioden):
    i_{t+1} = i_t + u_{t+1-L} - e_t,        u_t = -alpha (i_{t-s} + beta sum_{j=1..L-1} u_{t-j}).
i Bestand, u Bestellung, e Bedarfsabweichung, L Lieferzeit, alpha Anpassung, beta Leitungsgewicht, s Meldeverzug (hier s = 0; s > 0 nur für die Formelprüfung).

Eigene Herleitung in diesem Modell (z-Transformation, x_t <-> X(z), x_{t+1} <-> z X(z)); keine Quelle:
    P(z)   = z^L - (1 - alpha beta) z^(L-1) + alpha (1 - beta)                       (s = 0; allgemein siehe `charpoly`)
    H_u(z) = U/E =  alpha z^(L-1) / P(z)                                               (Bedarfsabweichung -> Bestellung)
    H_i(z) = I/E = -(z^(L-1) + alpha beta sum_{k=0..L-2} z^k) / P(z)                   (Bedarfsabweichung -> Bestand)
Der Frequenzgang ist z = exp(i w), w = 2 pi / T für die Periode T. Er gilt für den eingeschwungenen Zustand und nur, solange alle Wurzeln von P
im Einheitskreis liegen (Spektralradius rho < 1); die Bestellung D + u ist nicht geklemmt.

Alles hier ist eigener Rechenweg dieser Demo. Die Tests (`tests/test_oracle_frequenzgang_lager.py`) prüfen gegen die Zustandsmatrix,
die direkte Simulation, die Lyapunov-Gleichung und exakte Brüche, die dieses Modul nicht benutzen."""

import math
from functools import lru_cache

import numpy as np
from scipy.optimize import brentq, minimize_scalar

import fgl_constants as C

# ---------------------------------------------------------------------------------------------------------------------------------------
# Charakteristisches Polynom und Stabilität
# ---------------------------------------------------------------------------------------------------------------------------------------


def alpha_crit_beta0(L: int) -> float:
    """Stabilitätsgrenze ohne Leitungsgewicht: 2 sin(pi / (4L - 2)) (die kritische Schwingung hat die Periode 4L - 2)."""
    return 2.0 * math.sin(math.pi / (4 * L - 2))


def charpoly(z, L: int, alpha: float, beta: float, s: int = 0):
    """P_s(z) = z^(L+s) - (1 - alpha beta) z^(L+s-1) - alpha beta z^s + alpha; für s = 0: z^L - (1 - alpha beta) z^(L-1) + alpha (1 - beta)."""
    z = np.asarray(z, dtype=complex)
    return z ** (L + s) - (1 - alpha * beta) * z ** (L + s - 1) - alpha * beta * z ** s + alpha


def spectral_radius(L: int, alphas, beta: float) -> np.ndarray:
    """Größter Betrag der Wurzeln von P (s = 0) für jedes alpha; vektorisiert über Begleitmatrizen. Gibt ein Feld in der Form von `alphas` zurück."""
    a = np.atleast_1d(np.asarray(alphas, dtype=float))
    n = a.size
    if L == 1:
        r = np.abs(a - 1.0)                     # P(z) = z - 1 + alpha, unabhängig von beta
    else:
        M = np.zeros((n, L, L))
        M[:, 0, 0] = 1.0 - a * beta             # Begleitmatrix: erste Zeile = Koeffizienten von z^(L-1) und z^0 mit umgekehrtem Vorzeichen
        M[:, 0, L - 1] = -a * (1.0 - beta)
        for k in range(1, L):
            M[:, k, k - 1] = 1.0
        r = np.abs(np.linalg.eigvals(M)).max(axis=1)
    return r if np.ndim(alphas) else float(r[0])


def is_stable(L: int, alpha: float, beta: float) -> bool:
    return bool(spectral_radius(L, float(alpha), beta) < 1.0 - C.STABLE_EPS)


@lru_cache(maxsize=512)
def stable_limit(L: int, beta: float, hi: float = 10.0, n: int = 2000) -> float:
    """Kleinstes alpha, ab dem der Spektralradius 1 überschreitet (Rasterung bis `hi`, danach Bisektion); ohne Leitungsgewicht die geschlossene Formel."""
    if beta == 0.0:
        return alpha_crit_beta0(L)
    grid = np.linspace(1e-4, hi, n)
    rho = spectral_radius(L, grid, beta)
    bad = np.nonzero(rho > 1.0 + 1e-9)[0]
    if bad.size == 0:
        return float("inf")
    k = int(bad[0])
    lo, up = (grid[k - 1] if k > 0 else 1e-4), grid[k]
    for _ in range(45):
        mid = 0.5 * (lo + up)
        if spectral_radius(L, mid, beta) > 1.0 + 1e-12:
            up = mid
        else:
            lo = mid
    return float(up)


def settling_estimate(rho: float, fraction: float = C.SETTLE_FRACTION) -> float:
    """Einschwingzeit auf den Bruchteil `fraction`: ln(fraction) / ln(rho) (Schätzung aus dem Spektralradius, keine Regel; Messung siehe `fgl_sim`)."""
    if rho <= 0.0:
        return 0.0
    return math.log(fraction) / math.log(rho)


# ---------------------------------------------------------------------------------------------------------------------------------------
# Übertragungsfunktionen und Verstärkung
# ---------------------------------------------------------------------------------------------------------------------------------------


def _geo(z, L: int):
    """sum_{k=0..L-2} z^k (leer für L = 1)."""
    out = np.zeros_like(np.asarray(z, dtype=complex))
    for k in range(L - 1):
        out = out + z ** k
    return out


def Hu(z, L: int, alpha: float, beta: float, s: int = 0):
    """Bedarfsabweichung -> Bestellung: alpha z^(L-1) / P_s(z)."""
    z = np.asarray(z, dtype=complex)
    return alpha * z ** (L - 1) / charpoly(z, L, alpha, beta, s)


def Hi(z, L: int, alpha: float, beta: float, s: int = 0):
    """Bedarfsabweichung -> Bestand: -z^s (z^(L-1) + alpha beta sum_{k=0..L-2} z^k) / P_s(z)."""
    z = np.asarray(z, dtype=complex)
    return -z ** s * (z ** (L - 1) + alpha * beta * _geo(z, L)) / charpoly(z, L, alpha, beta, s)


def omega_of_period(T):
    return 2.0 * np.pi / np.asarray(T, dtype=float)


def abs2_P_angle(w, L: int, alpha: float, beta: float):
    """|P(e^{iw})|² in reellen Winkelfunktionen (s = 0); stimmt mit der komplexen Auswertung überein (Test)."""
    w = np.asarray(w, dtype=float)
    a = 1.0 - alpha * beta
    return (1.0 + a * a - 2.0 * a * np.cos(w) + alpha ** 2 * (1.0 - beta) ** 2
            + 2.0 * alpha * (1.0 - beta) * (np.cos(L * w) - a * np.cos((L - 1) * w)))


def gain_u(w, L: int, alpha: float, beta: float):
    """|H_u(e^{iw})| über die Winkelform (schnell, vektorisiert)."""
    return alpha / np.sqrt(abs2_P_angle(w, L, alpha, beta))


def gain_i(w, L: int, alpha: float, beta: float):
    """|H_i(e^{iw})| über die Winkelform: |z^(L-1) + alpha beta sum z^k|² / |P|², der Zähler gliedweise als Betragsquadrat einer Teilsumme."""
    w = np.asarray(w, dtype=float)
    k = np.arange(L - 1)
    num = np.exp(1j * (L - 1) * w) + alpha * beta * np.exp(1j * np.multiply.outer(w, k)).sum(axis=-1) if L > 1 else np.exp(1j * (L - 1) * w)
    return np.abs(num) / np.sqrt(abs2_P_angle(w, L, alpha, beta))


def hu_hi_at_period(L: int, alpha: float, beta: float, T: float) -> tuple[float, float]:
    """(|H_u|, |H_i|) bei der Periode T über die komplexe Auswertung."""
    z = np.exp(1j * 2.0 * math.pi / T)
    return float(abs(Hu(z, L, alpha, beta))), float(abs(Hi(z, L, alpha, beta)))


def dc_gain_i(L: int, alpha: float, beta: float) -> float:
    """H_i(1) = -(1 + alpha beta (L-1)) / alpha: ein dauerhaft höherer Bedarf verschiebt den Bestand dauerhaft (H_u(1) = 1)."""
    return -(1.0 + alpha * beta * (L - 1)) / alpha


def peak(L: int, alpha: float, beta: float, n: int = C.N_ANGLES_PEAK) -> tuple[float, float]:
    """(Spitze von |H_u| über alle Winkel 0..pi, Periode der Spitze); Periode `inf`, wenn das Maximum bei w -> 0 liegt (dort ist |H_u| = 1).

    Grobes Gitter einschließlich w = 0 und w = pi, danach Verfeinerung um den besten Gitterpunkt."""
    w = np.concatenate([[0.0], np.linspace(1e-6, math.pi, n)])
    g = gain_u(w, L, alpha, beta)
    k = int(np.argmax(g))
    if k <= 1:                                   # Maximum bei w -> 0 (Gleichstrom, |H_u| = 1) oder direkt daneben
        if g[k] <= 1.0 + 1e-12:
            return 1.0, float("inf")
    if k == len(w) - 1:                          # Rand bei w = pi (Periode 2): kein inneres Maximum
        return float(g[k]), 2.0
    lo, up = w[max(k - 1, 0)], w[k + 1]
    res = minimize_scalar(lambda x: -float(gain_u(x, L, alpha, beta)), bounds=(lo, up), method="bounded", options={"xatol": 1e-13})
    wp, gp = float(res.x), float(-res.fun)
    if gp < float(g[k]):
        wp, gp = float(w[k]), float(g[k])
    if wp < 1e-3 and gp <= 1.0 + 1e-12:
        return 1.0, float("inf")
    return gp, 2.0 * math.pi / wp


_W_MEAN = 2.0 * np.pi * (np.arange(C.N_ANGLES_MEAN) + 0.5) / C.N_ANGLES_MEAN     # Mittelpunktsregel über den ganzen Kreis


def mean_gain2(L: int, alpha: float, beta: float) -> tuple[float, float]:
    """(Mittel von |H_u|², Mittel von |H_i|²) über den Winkelkreis (Mittelpunktsregel auf 2^17 Punkten): das Verhältnis var(u)/var(e) bzw. var(i)/var(e)
    bei weißem Rauschen im Bedarf."""
    return float(np.mean(gain_u(_W_MEAN, L, alpha, beta) ** 2)), float(np.mean(gain_i(_W_MEAN, L, alpha, beta) ** 2))


_W_SCAN = np.linspace(1e-5, math.pi, C.N_ANGLES_SCAN)


def amplified_share(L: int, alpha: float, beta: float) -> float:
    """Anteil der Winkelachse (0, pi], auf dem |H_u| > 1 gilt (gleichmäßiges Gitter in w, nicht in der Periode)."""
    return float(np.mean(gain_u(_W_SCAN, L, alpha, beta) > 1.0))


def amplifies(L: int, alpha: float, beta: float) -> bool:
    """Wird irgendeine Periode verstärkt (max |H_u| auf dem Gitter über 1 + AMP_EPS)?"""
    return float(np.max(gain_u(_W_SCAN, L, alpha, beta))) > 1.0 + C.AMP_EPS


def amplified_intervals(L: int, alpha: float, beta: float) -> list[tuple[float, float]]:
    """Perioden (von, bis) mit |H_u| > 1, aufsteigend; `bis = inf`, wenn der Bereich bis w -> 0 reicht. Ränder per Nullstellensuche von |H_u| - 1."""
    w = _W_SCAN
    flag = gain_u(w, L, alpha, beta) > 1.0 + 1e-12
    out: list[tuple[float, float]] = []
    n = len(w)
    k = 0
    while k < n:
        if not flag[k]:
            k += 1
            continue
        j = k
        while j + 1 < n and flag[j + 1]:
            j += 1
        # Lauf k..j; Ränder in w
        def edge(a, b):
            f = lambda x: float(gain_u(x, L, alpha, beta)) - 1.0
            return brentq(f, a, b, xtol=1e-14)
        w_lo = w[0] if k == 0 else edge(w[k - 1], w[k])
        w_hi = w[-1] if j == n - 1 else edge(w[j], w[j + 1])
        t_from = 2.0 * math.pi / w_hi
        t_to = float("inf") if k == 0 else 2.0 * math.pi / w_lo
        out.append((t_from, t_to))
        k = j + 1
    out.sort()
    return out


def amplification_onset_formula(L: int, beta: float) -> float:
    """Näherung für den Beginn der Verstärkung langer Perioden (kleines w): positive Nullstelle von a x² + b x - 1,
    a = beta (1-beta) (L-1)², b = beta + (1-beta)(2L-1); für beta = 0: 1/(2L-1), für beta = 1: 1.
    Eigene Herleitung nur der Näherung bei kleinem w; die Aussage „keine andere Periode verstärkt früher“ ist gemessen, nicht bewiesen."""
    a = beta * (1.0 - beta) * (L - 1) ** 2
    b = beta + (1.0 - beta) * (2 * L - 1)
    if a == 0.0:
        return 1.0 / b
    return (-b + math.sqrt(b * b + 4.0 * a)) / (2.0 * a)


def amplification_onset_measured(L: int, beta: float, limit: float | None = None, n_grid: int = 700) -> float:
    """Kleinstes alpha unterhalb der Stabilitätsgrenze, bei dem irgendeine Periode verstärkt wird: grobe Rasterung, dann Bisektion; `inf`, wenn nie."""
    lim = stable_limit(L, beta) if limit is None else limit
    # Abkürzung: die Näherung nur als Klammer benutzen und prüfen, dass die Klammer stimmt (unten eine volle Rasterung als Rückfall)
    f = amplification_onset_formula(L, beta)
    lo, up = 0.5 * f, min(1.5 * f, 0.999 * lim)
    if lo < up and not amplifies(L, lo, beta) and amplifies(L, up, beta):
        for _ in range(45):
            mid = 0.5 * (lo + up)
            if amplifies(L, mid, beta):
                up = mid
            else:
                lo = mid
        return up
    grid = np.linspace(1e-3, 0.999 * lim, n_grid)
    prev = None
    for a in grid:
        if amplifies(L, float(a), beta):
            if prev is None:
                return float(a)
            lo, up = float(prev), float(a)
            for _ in range(40):
                mid = 0.5 * (lo + up)
                if amplifies(L, mid, beta):
                    up = mid
                else:
                    lo = mid
            return up
        prev = a
    return float("inf")


# ---------------------------------------------------------------------------------------------------------------------------------------
# Zustandsmatrix (für die Lyapunov-Gleichung der Rauschanzeige)
# ---------------------------------------------------------------------------------------------------------------------------------------


def state_space(L: int, alpha: float, beta: float):
    """Zustand x_t = (i_t, u_{t-1}, ..., u_{t-L+1}), x_{t+1} = A x_t + B e_t, u_t = c x_t, i_t = e0 x_t (s = 0)."""
    n = L
    c = np.zeros(n)
    c[0] = -alpha
    c[1:] = -alpha * beta
    A = np.zeros((n, n))
    A[0, 0] = 1.0
    if L == 1:
        A[0, :] += c
    else:
        A[0, L - 1] += 1.0
        A[1, :] = c
        for j in range(2, L):
            A[j, j - 1] = 1.0
    B = np.zeros(n)
    B[0] = -1.0
    e0 = np.zeros(n)
    e0[0] = 1.0
    return A, B, c, e0
