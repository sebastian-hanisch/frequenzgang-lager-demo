"""Kennzahlen, Analysen der Experimente und Verdikt-Logik (reine Funktionen, die Tests prüfen jede Zahl im Text an handgesetzten Werten).

Alle Analysen sind Wörterbücher aus Zahlen und Feldern, damit `st.cache_data` sie halten kann und die Verdikte ohne Streamlit testbar sind."""

import math
from dataclasses import dataclass

import numpy as np

import fgl_constants as C
import fgl_core as K
import fgl_sim as S


@dataclass(frozen=True)
class Settings:
    L: int = C.DEFAULT_L
    beta: float = C.DEFAULT_BETA
    alpha: float = C.DEFAULT_ALPHA
    T: int = C.DEFAULT_T
    A: float = C.DEFAULT_A
    seed: int = C.DEFAULT_SEED


# ---------------------------------------------------------------------------------------------------------------------------------------
# Zahlen als deutscher Text
# ---------------------------------------------------------------------------------------------------------------------------------------


def de(x: float, nd: int = 2, sign: bool = False) -> str:
    """Zahl mit Dezimalkomma (Anzeige)."""
    s = f"{x:+.{nd}f}" if sign else f"{x:.{nd}f}"
    return s.replace(".", ",")


def para(x: float) -> str:
    """Reglerwert ohne überflüssige Nullen: 0,9 / 0,56 / 1,5 / 2."""
    return f"{x:g}".replace(".", ",")


def int_text(n: int) -> str:
    """Ganze Zahl mit geschütztem Leerzeichen als Tausendertrenner."""
    return f"{n:,}".replace(",", " ")


def period_text(t: float, nd: int = 2) -> str:
    return "∞" if math.isinf(t) else de(t, nd)


def intervals_text(intervals: list[tuple[float, float]]) -> str:
    """Perioden mit Verstärkung als Satzteil: „alle Perioden über 4,15“ oder „die Perioden von 2,00 bis 2,53 und über 3,71“."""
    if len(intervals) == 1 and math.isinf(intervals[0][1]):
        lo = intervals[0][0]
        return "alle Perioden" if lo <= 2.0 + 1e-9 else f"alle Perioden über {de(lo)}"
    parts = []
    for lo, hi in intervals:
        parts.append(f"über {de(lo)}" if math.isinf(hi) else f"von {de(lo)} bis {de(hi)}")
    return "die Perioden " + " und ".join(parts)


# ---------------------------------------------------------------------------------------------------------------------------------------
# Ergebnis: Bode-Bild
# ---------------------------------------------------------------------------------------------------------------------------------------


def bode_analysis(s: Settings) -> dict:
    """Frequenzgang von Bedarf zu Bestellung und Bestand über die Periode (2 bis 200, logarithmisch) und seine Kennzahlen."""
    rho = float(K.spectral_radius(s.L, float(s.alpha), s.beta))
    limit = K.stable_limit(s.L, s.beta)
    stable = rho < 1.0 - C.STABLE_EPS
    out = {"L": s.L, "beta": s.beta, "alpha": s.alpha, "T": s.T, "rho": rho, "limit": limit, "stable": stable}
    if not stable:
        return out
    periods = np.geomspace(C.T_BODE_MIN, C.T_BODE_MAX, C.N_BODE)
    w = 2.0 * np.pi / periods
    pk, pkT = K.peak(s.L, s.alpha, s.beta)
    hu_T, hi_T = K.hu_hi_at_period(s.L, s.alpha, s.beta, s.T)
    mu, mi = K.mean_gain2(s.L, s.alpha, s.beta)
    out.update({
        "periods": periods, "gain_u": K.gain_u(w, s.L, s.alpha, s.beta), "gain_i": K.gain_i(w, s.L, s.alpha, s.beta),
        "peak": float(pk), "peak_T": float(pkT), "hu_T": hu_T, "hi_T": hi_T, "mean_u": mu, "mean_i": mi,
        "share": K.amplified_share(s.L, s.alpha, s.beta),
        "intervals": [(float(a), float(b)) for a, b in K.amplified_intervals(s.L, s.alpha, s.beta)],
        "settle_estimate": K.settling_estimate(rho), "resonance_T": 4 * s.L - 2,
    })
    return out


def verdict_bode(a: dict) -> tuple[str, str]:
    """Kernfrage 1: Welche Perioden des Bedarfs verstärkt die Regel?"""
    head = f"L = {a['L']}, β = {para(a['beta'])}, α = {para(a['alpha'])}: "
    if not a["stable"]:
        return "warning", (head + f"**instabil** (Spektralradius ρ = {de(a['rho'], 3)} ≥ 1, Grenze α = {de(a['limit'], 3)}): die Bestellung wächst ohne Obergrenze, "
                           "ein Frequenzgang existiert dort nicht.")
    ints, pk = a["intervals"], a["peak"]
    exact = ""
    if a["beta"] == 0.0 and a["T"] == a["resonance_T"]:
        exact = " (genau α/(α_krit − α))"
    at_T = f"bei Periode {a['T']}: {de(a['hu_T'])}{exact}"
    if not ints and pk <= 1.0 + C.AMP_EPS:
        return "success", head + f"**keine Periode wird verstärkt**: die Spitze von |H_u| ist {de(pk)} (nur für sehr lange Perioden, T → ∞), {at_T}."
    kind = "warning" if pk >= 5.0 else "info"
    return kind, (head + f"verstärkt werden **{intervals_text(ints)}**; Spitze **{de(pk)}** bei Periode {period_text(a['peak_T'])}, {at_T}; "
                  f"{de(100.0 * a['share'], 0)} % der Winkelachse.")


def verdict_stable_but_amplifies(a: dict) -> tuple[str, str]:
    """Kernfrage 2: Stabil heißt „ein Stoß klingt ab“, nicht „verstärkt nicht“."""
    if not a["stable"]:
        return "error", f"Hier ist das Lager nicht stabil (ρ = {de(a['rho'], 3)}); die Frage nach der Verstärkung stellt sich erst darunter."
    q = a["alpha"] / a["limit"]
    base = (f"Das Lager ist stabil (ρ = {de(a['rho'], 3)} < 1, α ist {de(100.0 * q, 0)} % der Grenze {de(a['limit'], 3)}): ein einzelner Stoß klingt auf 1 % in rund "
            f"{de(a['settle_estimate'], 0)} Perioden ab (Schätzung aus ρ).")
    if a["peak"] > 1.0 + C.AMP_EPS:
        return "warning", base + f" Trotzdem verstärkt es eine Saison bis zum **{de(a['peak'])}fachen** (bei Periode {period_text(a['peak_T'])}): „stabil“ heißt nicht „verstärkt nicht“."
    return "success", base + " Es verstärkt auch keine Saison: die Spitze von |H_u| ist 1."


# ---------------------------------------------------------------------------------------------------------------------------------------
# Experiment 1: Saisonbedarf (direkte Simulation)
# ---------------------------------------------------------------------------------------------------------------------------------------


def season_analysis(s: Settings) -> dict:
    out = {"L": s.L, "beta": s.beta, "alpha": s.alpha, "T": s.T, "A": s.A}
    rho = float(K.spectral_radius(s.L, float(s.alpha), s.beta))
    out["rho"] = rho
    out["stable"] = rho < 1.0 - C.STABLE_EPS
    out["limit"] = K.stable_limit(s.L, s.beta)
    if not out["stable"]:
        return out
    r = S.season_run(s.L, s.alpha, s.beta, s.T)
    hu, hi = K.hu_hi_at_period(s.L, s.alpha, s.beta, s.T)
    n_plot = int(min(r["n"], max(120, round(2.0 * r["settle_estimate"]) + 3 * s.T, 6 * s.T), 2000))
    out.update({
        "hu": hu, "hi": hi, "amp_u": r["amp_u"], "amp_i": r["amp_i"], "settle_measured": r["settle_measured"], "settle_estimate": r["settle_estimate"],
        "settled": r["settled"], "n_sim": r["n"], "n_plot": n_plot, "t": np.arange(n_plot), "e": r["e"][:n_plot], "u": r["u"][:n_plot], "i": r["i"][:n_plot],
        "u_max_transient": r["u_max"], "phase_u": r["phase_u"],
    })
    return out


def season_warnings(a: dict) -> list[tuple[str, str]]:
    """Hinweise der Saison-Simulation als (Art, Text); Mittelbedarf d = 1, Bedarfsamplitude A in Vielfachen von d."""
    if not a["stable"]:
        return [("error", f"α = {para(a['alpha'])} liegt über der Stabilitätsgrenze {de(a['limit'], 3)}: die Bestellung wächst ohne Obergrenze, es gibt keinen Frequenzgang.")]
    out = []
    steady = a["A"] * a["hu"]
    if steady > 1.0:
        out.append(("warning", f"Klemmung: A·|H_u| = {de(steady)} liegt über dem Mittelbedarf (1). Die Bestellung d + u würde unter 0 fallen; das Modell ist linear und ohne "
                               "Klemmung, die Zahlen gelten dort nicht mehr (Klemmung und Chaos: `lager-chaos-demo`)."))
    elif a["A"] * a["u_max_transient"] > 1.0:
        out.append(("info", f"Im Einschwingen überschreitet die Bestellung bei A = {para(a['A'])} die Grenze 0 (A·größte Bestellung = {de(a['A'] * a['u_max_transient'])} > 1); "
                            "im eingeschwungenen Zustand nicht."))
    if not a["settled"]:
        out.append(("info", "Die Simulation ist auf 30 000 Perioden begrenzt und noch nicht eingeschwungen; die gemessene Amplitude weicht vom Frequenzgang ab."))
    return out


def verdict_season(a: dict) -> tuple[str, str]:
    if not a["stable"]:
        return "warning", f"α = {para(a['alpha'])} liegt über der Grenze {de(a['limit'], 3)}: die Bestellung wächst ohne Obergrenze, ein Frequenzgang existiert dort nicht."
    meas = "nicht gemessen (Simulation begrenzt)" if a["settle_measured"] is None else f"gemessen {a['settle_measured']}"
    return "info", (f"Eingeschwungene Bestellamplitude **{de(a['amp_u'])}·A** (Frequenzgang {de(a['hu'])}), Bestandsamplitude **{de(a['amp_i'])}·A** (Frequenzgang {de(a['hi'])}); "
                    f"Einschwingzeit auf 1 % rund **{de(a['settle_estimate'], 0)} Perioden** (Schätzung aus ρ), {meas}; vorher gilt der Frequenzgang noch nicht.")


# ---------------------------------------------------------------------------------------------------------------------------------------
# Experiment 2: Zielkonflikt α bei einer Saison; Experiment 5: Leitung im Vergleich
# ---------------------------------------------------------------------------------------------------------------------------------------

Q_LO, Q_HI = 0.2, 0.85           # Bezugspunkte des Zielkonflikt-Verdikts: 20 % und 85 % der Stabilitätsgrenze ohne Leitungsgewicht


def _alpha_grid() -> np.ndarray:
    return np.round(np.arange(1, int(round(C.ALPHA_MAX / 0.01)) + 1) * 0.01, 2)


def tradeoff_analysis(L: int, T: int) -> dict:
    """α ↦ |H_u|, |H_i| bei der Periode T für β = 0, 0,55 und 1 (nur im stabilen Bereich, sonst NaN)."""
    alphas = _alpha_grid()
    z = np.exp(1j * 2.0 * math.pi / T)
    curves = {}
    for beta in C.BETA_COMPARE:
        rho = K.spectral_radius(L, alphas, beta)
        ok = rho < 1.0 - C.STABLE_EPS
        hu = np.where(ok, np.abs(K.Hu(z, L, alphas, beta)), np.nan)
        hi = np.where(ok, np.abs(K.Hi(z, L, alphas, beta)), np.nan)
        curves[beta] = {"hu": hu, "hi": hi, "limit": K.stable_limit(L, beta)}
    lim0 = curves[0.0]["limit"]
    ref = {"alpha_lo": Q_LO * lim0, "alpha_hi": Q_HI * lim0}
    for beta in C.BETA_COMPARE:
        ref[beta] = {name: (K.hu_hi_at_period(L, ref["alpha_" + name], beta, T) if K.is_stable(L, ref["alpha_" + name], beta) else (float("nan"), float("nan")))
                     for name in ("lo", "hi")}   # (|H_u|, |H_i|) bei gleichem α für alle β; NaN, wenn dort instabil
    return {"L": L, "T": T, "alphas": alphas, "curves": curves, "ref": ref, "limit0": lim0}


def verdict_tradeoff(a: dict) -> tuple[str, str]:
    """Zielkonflikt bei einer Saison: bei zwei gleichen α (20 % und 85 % der Grenze ohne Leitungsgewicht) die Bestellamplitude |H_u| und die Bestandsamplitude |H_i| je β."""
    ref = a["ref"]
    parts = []
    for beta in C.BETA_COMPARE:
        (u_lo, i_lo), (u_hi, i_hi) = ref[beta]["lo"], ref[beta]["hi"]
        if math.isnan(u_lo) or math.isnan(u_hi):
            parts.append(f"β = {para(beta)}: bei einem der beiden α instabil")
            continue
        parts.append(f"β = {para(beta)}: Bestellung {de(u_lo)} → {de(u_hi)}, Bestand {de(i_lo)} → {de(i_hi)} ({'steigt' if i_hi > i_lo else 'fällt'} mit α)")
    return "info", (f"L = {a['L']}, Periode {a['T']}: von α = {de(ref['alpha_lo'], 3)} ({de(100 * Q_LO, 0)} % der Grenze {de(a['limit0'], 3)} ohne Leitungsgewicht) "
                    f"auf α = {de(ref['alpha_hi'], 3)} ({de(100 * Q_HI, 0)} %) ändern sich die Amplituden von Bedarf zu Bestellung und zu Bestand so: " + "; ".join(parts) + ".")


def compare_analysis(L: int, alpha: float) -> dict:
    """Bode-Kurven bei gleichem L und α für β = 0, 0,55, 1; instabile Kurven fehlen."""
    periods = np.geomspace(C.T_BODE_MIN, C.T_BODE_MAX, C.N_BODE)
    w = 2.0 * np.pi / periods
    out = {"L": L, "alpha": alpha, "periods": periods, "curves": {}}
    for beta in C.BETA_COMPARE:
        rho = float(K.spectral_radius(L, float(alpha), beta))
        lim = K.stable_limit(L, beta)
        if rho >= 1.0 - C.STABLE_EPS:
            out["curves"][beta] = {"stable": False, "limit": lim, "rho": rho}
            continue
        pk, pkT = K.peak(L, alpha, beta)
        mu, _ = K.mean_gain2(L, alpha, beta)
        out["curves"][beta] = {"stable": True, "limit": lim, "rho": rho, "gain_u": K.gain_u(w, L, alpha, beta), "peak": float(pk), "peak_T": float(pkT), "mean_u": mu}
    return out


def verdict_compare(a: dict) -> tuple[str, str]:
    parts = []
    for beta in C.BETA_COMPARE:
        c = a["curves"][beta]
        if not c["stable"]:
            parts.append(f"β = {para(beta)}: instabil (Grenze {de(c['limit'], 3)})")
        elif c["peak"] <= 1.0 + C.AMP_EPS:
            parts.append(f"β = {para(beta)}: stabil, keine Verstärkung (Mittel von |H_u|² {de(c['mean_u'])})")
        else:
            parts.append(f"β = {para(beta)}: stabil, Spitze {de(c['peak'])} bei Periode {period_text(c['peak_T'])} (Mittel von |H_u|² {de(c['mean_u'])})")
    return "info", f"L = {a['L']}, α = {para(a['alpha'])}: " + "; ".join(parts) + "."


# ---------------------------------------------------------------------------------------------------------------------------------------
# Experiment 3: Rauschen als Mittelwert
# ---------------------------------------------------------------------------------------------------------------------------------------


def noise_analysis(s: Settings, sigma: float) -> dict:
    out = {"L": s.L, "beta": s.beta, "alpha": s.alpha, "T": s.T, "A": s.A, "sigma": sigma, "seed": s.seed}
    rho = float(K.spectral_radius(s.L, float(s.alpha), s.beta))
    out["stable"] = rho < 1.0 - C.STABLE_EPS
    out["limit"] = K.stable_limit(s.L, s.beta)
    out["rho"] = rho
    if not out["stable"]:
        return out
    w = np.linspace(1e-3, math.pi, 600)
    pk, pkT = K.peak(s.L, s.alpha, s.beta)
    mu, mi = K.mean_gain2(s.L, s.alpha, s.beta)
    vu, vi = S.lyapunov_variances(s.L, s.alpha, s.beta)
    sim = S.noise_run(s.L, s.alpha, s.beta, s.T, s.A, sigma, s.seed)
    out.update({"w": w, "g2_u": K.gain_u(w, s.L, s.alpha, s.beta) ** 2, "peak": float(pk), "peak_T": float(pkT), "mean_u": mu, "mean_i": mi, "lyap_u": vu, "lyap_i": vi, "sim": sim})
    return out


def verdict_noise(a: dict) -> tuple[str, str]:
    """Das Mittel von |H_u|² ist das Verhältnis var(u)/var(e) bei weißem Rauschen; die Spitze² zeigt, was das Mittel verbirgt."""
    if not a["stable"]:
        return "warning", "Instabil: weder Frequenzgang noch Varianzverhältnis existieren."
    pk2, mu = a["peak"] ** 2, a["mean_u"]
    if a["peak"] <= 1.0 + C.AMP_EPS:
        return "success", (f"Keine Periode wird verstärkt, das Mittel von |H_u|² ist {de(mu)} (Lyapunov-Gleichung: {de(a['lyap_u'])}): weißes Rauschen im Bedarf wird in der Bestellung "
                           f"{'gedämpft' if mu < 1 else 'nicht gedämpft'}.")
    return "info", (f"Spitze² = {de(pk2, 1)} (Spitze {de(a['peak'])} bei Periode {period_text(a['peak_T'])}) gegen das Mittel von |H_u|² = **{de(mu)}** (Lyapunov-Gleichung: {de(a['lyap_u'])}): "
                    f"Spitze² ist das {de(pk2 / mu, 1)}fache des Mittels; das Rauschmaß allein sagt nicht, welche Periode die Regel am stärksten verstärkt.")


def variance_rows(a: dict) -> list[dict]:
    """Zerlegung var = Saison + Rauschen gegen die Simulation, je für Bestellung und Bestand (Prozent Abweichung der Simulation von der Summe)."""
    sim = a["sim"]
    rows = []
    for name, ps, pn, v in (("Bestellung u", sim["pred_season_u"], sim["pred_noise_u"], sim["var_u"]), ("Bestand i", sim["pred_season_i"], sim["pred_noise_i"], sim["var_i"])):
        tot = ps + pn
        rows.append({"Größe": name, "Saison": ps, "Rauschen": pn, "Summe": tot, "Simulation": v, "Abweichung (%)": 100.0 * (v - tot) / tot if tot > 0 else float("nan")})
    return rows


# ---------------------------------------------------------------------------------------------------------------------------------------
# Experiment 4: Wo beginnt das Verstärken?
# ---------------------------------------------------------------------------------------------------------------------------------------

_W_MAP = np.geomspace(1e-4, math.pi, 1500)


def onset_map(beta: float) -> dict:
    """Karte (L = 1 bis 30, q = α/α_krit): max |H_u| über die Winkel; dazu die Näherungslinie α*(L)/α_krit(L)."""
    Ls = list(range(1, C.L_MAP_MAX + 1))
    qs = np.linspace(0.02, 0.98, C.N_MAP_Q)
    z = np.zeros((len(qs), len(Ls)))
    limits, formula_ratio = [], []
    for k, L in enumerate(Ls):
        lim = K.stable_limit(L, beta, 4.0, 500)
        limits.append(lim)
        formula_ratio.append(K.amplification_onset_formula(L, beta) / lim)
        for j, q in enumerate(qs):
            z[j, k] = float(np.max(K.gain_u(_W_MAP, L, q * lim, beta)))
    return {"beta": beta, "Ls": Ls, "qs": qs, "max_gain": z, "limits": limits, "formula_ratio": formula_ratio}


def onset_analysis(L: int, beta: float) -> dict:
    """Gemessener Verstärkungsbeginn bei L und bei L = 30 (Bisektion auf dem Gitter von 20 001 Winkeln) neben der Näherung."""
    out = {"L": L, "beta": beta}
    for key, LL in (("", L), ("30", C.L_MAP_MAX)):
        lim = K.stable_limit(LL, beta)
        onset = K.amplification_onset_measured(LL, beta, lim)
        out["limit" + key] = lim
        out["onset" + key] = onset
        out["formula" + key] = K.amplification_onset_formula(LL, beta)
    return out


def verdict_onset(a: dict) -> tuple[str, str]:
    r, r30 = 100.0 * a["onset"] / a["limit"], 100.0 * a["onset30"] / a["limit30"]
    if a["beta"] == 1.0:
        return "info", (f"β = 1: Verstärkung erst ab α = {de(a['onset'], 3)} (gemessen), für jedes L; die Stabilitätsgrenze liegt bei {de(a['limit'], 3)}. "
                        f"L = {a['L']}: {de(r, 1)} % der Grenze, L = {C.L_MAP_MAX}: {de(r30, 1)} %.")
    tail = f", 1/π = {de(100.0 / math.pi, 1)} %" if a["beta"] == 0.0 else ""
    return "info", (f"β = {para(a['beta'])}, L = {a['L']}: Verstärkung ab α = {de(a['onset'], 3)} (gemessen; Näherung {de(a['formula'], 3)}), das sind {de(r, 1)} % der Stabilitätsgrenze {de(a['limit'], 3)}; "
                    f"L = {C.L_MAP_MAX}: {de(r30, 1)} % der Grenze{tail}.")
