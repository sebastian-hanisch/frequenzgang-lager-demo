"""Direkte Simulation des Lagers mit Sinus- und Rauschbedarf (nur numpy und scipy).

Die Simulation rechnet die Modellgleichungen von Periode zu Periode nach (`simulate`) und benutzt **keine** Übertragungsfunktion: sie ist der Gegenwert zum
Frequenzgang in `fgl_core`. Amplitude und Phase der eingeschwungenen Schwingung werden aus dem Ende der Bahn entnommen (Demodulation über ganze Perioden)."""

import math

import numpy as np
from scipy.linalg import solve_discrete_lyapunov

import fgl_constants as C
import fgl_core as K
from fgl_rng import normal_array


def simulate(L: int, alpha: float, beta: float, e) -> tuple[np.ndarray, np.ndarray]:
    """Direkte Rekursion i_{t+1} = i_t + u_{t+1-L} - e_t, u_t = -alpha (i_t + beta sum_{j=1..L-1} u_{t-j}) aus dem Ruhezustand (leere Leitung, i_0 = 0).
    Gibt (i_0..i_{n-1}, u_0..u_{n-1}) zurück; Bestellungen vor Periode 0 sind 0."""
    e = [float(x) for x in e]
    n = len(e)
    u: list[float] = []
    i = [0.0]
    for t in range(n):
        pipe = sum(u[max(0, t - L + 1):t])
        ut = -alpha * (i[t] + beta * pipe)
        u.append(ut)
        arriving = u[t + 1 - L] if t + 1 - L >= 0 else 0.0
        i.append(i[t] + arriving - e[t])
    return np.array(i[:n]), np.array(u)


def demodulate(x: np.ndarray, T: int, start: int, count: int) -> complex:
    """Komplexe Amplitude der Komponente mit Periode T in x[start:start+count] (count ein Vielfaches von T): 2 mean(x e^{-i w t}); bei T = 2 ohne den Faktor 2."""
    w = 2.0 * math.pi / T
    t = np.arange(start, start + count)
    fac = 1.0 if T == 2 else 2.0
    return fac * np.mean(x[start:start + count] * np.exp(-1j * w * t))


def season_run(L: int, alpha: float, beta: float, T: int, cap: int = C.SIM_CAP) -> dict:
    """Sinusbedarf e_t = cos(2 pi t / T) (Amplitude 1) aus dem Ruhezustand, Periode T ganzzahlig.

    Rückgabe: Zeitreihen e, u, i (ganze Länge N), die gemessenen Amplituden `amp_u`, `amp_i` (Fenster 40 Perioden am Ende), `settled` (Wurzelrest nach dem Einschwingen
    vor dem Messfenster unter 1e-6), die gemessene Einschwingzeit `settle_measured` (Perioden bis die Abweichung der Bestellung von der eingeschwungenen Schwingung dauerhaft unter 1 %
    ihrer Amplitude liegt; `None`, wenn das innerhalb der Länge nicht eintritt) und die Schätzung `settle_estimate`. Instabile Fälle (rho >= 1) werden nicht gerechnet."""
    rho = float(K.spectral_radius(L, float(alpha), beta))
    if rho >= 1.0 - C.STABLE_EPS:
        raise ValueError("instabil: kein Frequenzgang")
    M = 40 * T
    tr = int(math.ceil(math.log(1e-9) / math.log(rho))) + 10 if rho > 0 else 10
    N = min(tr + M, max(cap, M + 20))
    w = 2.0 * math.pi / T
    e = np.cos(w * np.arange(N))
    i_s, u_s = simulate(L, alpha, beta, e)
    start = N - M
    cu = demodulate(u_s, T, start, M)
    ci = demodulate(i_s, T, start, M)
    z = np.exp(1j * w)
    hu = complex(K.Hu(z, L, alpha, beta))
    uss = np.real(hu * np.exp(1j * w * np.arange(N)))
    err = np.abs(u_s - uss)
    over = np.nonzero(err > C.SETTLE_FRACTION * abs(hu))[0]
    t1 = int(over.max()) + 1 if len(over) else 0
    residual = rho ** start if rho > 0 else 0.0
    return {
        "e": e, "u": u_s, "i": i_s, "n": N, "amp_u": abs(cu), "amp_i": abs(ci), "phase_u": float(np.angle(cu / hu)),
        "settled": residual < 1e-6, "settle_measured": (t1 if t1 < N - 1 else None), "settle_estimate": K.settling_estimate(rho), "rho": rho,
        "u_max": float(np.max(np.abs(u_s))),
    }


def noise_run(L: int, alpha: float, beta: float, T: int, A: float, sigma: float, seed: int, n: int = C.N_NOISE) -> dict:
    """Bedarf e_t = A cos(2 pi t / T) + sigma * N(0,1) (Normalwerte aus SplitMix64, Zufalls-Seed `seed`), direkte Simulation, Varianzen nach dem Einschwingen.

    Vorhersage der Zerlegung: var(u) = A² |H_u(T)|² / 2 + sigma² Mittel|H_u|² (bei T = 2 fällt der Faktor 1/2 weg, weil cos(pi t) = ±1 Varianz A² hat); ebenso var(i)."""
    rho = float(K.spectral_radius(L, float(alpha), beta))
    if rho >= 1.0 - C.STABLE_EPS:
        raise ValueError("instabil: kein Frequenzgang")
    w = 2.0 * math.pi / T
    e = A * np.cos(w * np.arange(n)) + sigma * normal_array(seed, n)
    i_s, u_s = simulate(L, alpha, beta, e)
    burn = min(int(math.log(1e-6) / math.log(rho)) + 1 if rho > 0 else 1, n // 4)
    gu, gi = K.hu_hi_at_period(L, alpha, beta, T)
    mu, mi = K.mean_gain2(L, alpha, beta)
    season = 1.0 if T == 2 else 0.5
    return {
        "var_u": float(np.var(u_s[burn:])), "var_i": float(np.var(i_s[burn:])), "burn": burn, "n": n,
        "pred_season_u": A * A * gu * gu * season, "pred_noise_u": sigma * sigma * mu,
        "pred_season_i": A * A * gi * gi * season, "pred_noise_i": sigma * sigma * mi,
    }


def lyapunov_variances(L: int, alpha: float, beta: float) -> tuple[float, float]:
    """(var(u)/var(e), var(i)/var(e)) für weißes Rauschen aus der Lyapunov-Gleichung P = A P A^T + B B^T der Zustandsmatrix."""
    A, B, c, e0 = K.state_space(L, alpha, beta)
    P = solve_discrete_lyapunov(A, np.outer(B, B))
    return float(c @ P @ c), float(e0 @ P @ e0)
