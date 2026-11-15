"""Claims-Tests: jede Zahl der README steht hier als Band (Spalte „Test“ der README-Tabelle nennt die Funktionen dieser Datei).

Messbedingungen überall: Modell ohne Klemmung, im stabilen Bereich, Frequenzgang aus der z-Transformation (`fgl_core`), Simulationen direkt aus den Modellgleichungen (`fgl_sim`),
Zufallsfolgen aus SplitMix64 mit festen Zufalls-Seeds. α und β nie als Zweierpotenz (Ausnahmen: der benannte Randfall β = 1, α = 1 in test_edge_cases.py und die Randwerte β = 1,
T = 2 des Presets „Leitung voll“). Zahlen, die auf einer anderen Plattform leicht kippen können (Zählungen an Schwellen, Simulationsvarianzen), stehen als Band."""

import math

import numpy as np
import pytest

import fgl_constants as C
import fgl_core as K
import fgl_evaluation as E
import fgl_sim as S
from fgl_rng import SplitMix64

AK3 = K.alpha_crit_beta0(3)


def _near(value, expected, tol):
    assert abs(value - expected) <= tol, (value, expected, tol)


# --- M1: Handbeispiel und Schnellstart-Tabelle -----------------------------------------------------------------------------------------------

PRESET_TABLE = {  # Name: (|H_u|, |H_i|, Spitze, Periode der Spitze (inf = T → ∞), Mittel|H_u|², ρ, Anteil, Einschwingzeit Schätzung)
    "Von Hand": (9.0000, 10.0000, 10.5903, 6.19, 5.8966, 0.94868, 0.482, 87),
    "Resonanz": (9.6495, 17.2313, 11.5463, 10.30, 3.8347, 0.97184, 0.287, 161),
    "Saison nahe der Lieferzeit": (3.8388, 7.3823, 6.4481, 10.57, 2.0947, 0.95126, 0.279, 92),
    "Saison träge": (0.2978, 2.4818, 1.0000, math.inf, 0.0815, 0.82271, 0.000, 24),
    "Saison lang": (1.0411, 3.4705, 1.3863, 14.87, 0.3500, 0.81271, 0.194, 22),
    "Leitung halb": (4.6448, 6.0928, 6.2293, 6.37, 4.6628, 0.90586, 0.750, 47),
    "Leitung voll": (3.0000, 2.0000, 3.0000, 2.00, 3.0000, 0.50000, 1.000, 7),
}


@pytest.mark.parametrize("name", list(PRESET_TABLE))
def test_preset_table_values(name):
    hu, hi, pk, pT, mu, rho, share, settle = PRESET_TABLE[name]
    p = C.PRESETS[name]
    L, beta, alpha, T = p["L"], p["beta"], p["alpha"], p["T"]
    _near(K.hu_hi_at_period(L, alpha, beta, T)[0], hu, 5e-5)
    _near(K.hu_hi_at_period(L, alpha, beta, T)[1], hi, 5e-5)
    gpk, gT = K.peak(L, alpha, beta)
    _near(gpk, pk, 5e-5)
    if math.isinf(pT):
        assert math.isinf(gT)
    else:
        _near(gT, pT, 0.006)
    _near(K.mean_gain2(L, alpha, beta)[0], mu, 5e-5)
    _near(K.spectral_radius(L, alpha, beta), rho, 5e-6)
    _near(K.amplified_share(L, alpha, beta), share, 0.0006)
    _near(K.settling_estimate(K.spectral_radius(L, alpha, beta)), settle, 0.6)


def test_simulation_gives_the_same_amplitudes_as_the_table_for_every_preset():
    for name, row in PRESET_TABLE.items():
        p = C.PRESETS[name]
        r = S.season_run(p["L"], p["alpha"], p["beta"], p["T"])
        _near(r["amp_u"], row[0], 5e-5)
        _near(r["amp_i"], row[1], 5e-5)


def test_limits_quoted_in_the_readme():
    _near(K.stable_limit(2, 0.55), 2.2222, 5e-5)         # L = 2: min(1/(1−β), 2/(2β−1))
    _near(K.stable_limit(12, 0.55), 2.0038, 5e-5)


def test_limits_of_the_presets():
    _near(K.stable_limit(2, 0.0), 1.0, 1e-12)
    _near(K.stable_limit(3, 0.0), 0.6180, 5e-5)
    _near(K.stable_limit(3, 0.55), 2.0, 1e-9)
    _near(K.stable_limit(3, 1.0), 2.0, 1e-9)


def test_hand_example_edge_period_and_share():
    iv = K.amplified_intervals(2, 0.9, 0.0)
    assert len(iv) == 1 and abs(iv[0][0] - 4.1467) < 1e-4 and math.isinf(iv[0][1])


def test_preset_help_texts_numbers():
    """Die Zahlen der Hilfetexte (gerundet) stimmen mit den Messungen der Tabelle überein."""
    def hu_hi(name):
        p = C.PRESETS[name]
        return K.hu_hi_at_period(p["L"], p["alpha"], p["beta"], p["T"])
    assert round(100 * 0.56 / AK3) == 91 and "91 %" in C.PRESET_HELP["Resonanz"] and 9.0 < hu_hi("Resonanz")[0] < 10.0 and "fast zehnfach" in C.PRESET_HELP["Resonanz"]
    hu, hi = hu_hi("Saison nahe der Lieferzeit")
    assert 3.5 < hu < 4.0 and "fast vierfach" in C.PRESET_HELP["Saison nahe der Lieferzeit"] and round(hi, 1) == 7.4 and "7,4fache" in C.PRESET_HELP["Saison nahe der Lieferzeit"]
    hu, hi = hu_hi("Saison träge")
    assert 0.29 < hu < 0.3 and "knapp 0,3" in C.PRESET_HELP["Saison träge"] and round(hi, 1) == 2.5 and "2,5fache" in C.PRESET_HELP["Saison träge"]
    hu, hi = hu_hi("Saison lang")
    assert round(hu, 2) == 1.04 and "Verstärkung 1,04" in C.PRESET_HELP["Saison lang"] and round(hi, 1) == 3.5 and "3,5fache" in C.PRESET_HELP["Saison lang"]
    hu, _ = hu_hi("Leitung halb")
    assert round(hu, 1) == 4.6 and "4,6fach" in C.PRESET_HELP["Leitung halb"] and round(K.peak(3, 1.5, 0.55)[1], 1) == 6.4 and "Periode 6,4" in C.PRESET_HELP["Leitung halb"]
    assert "(Grenze 2)" in C.PRESET_HELP["Leitung halb"] and K.stable_limit(3, 0.55) == pytest.approx(2.0)
    assert K.hu_hi_at_period(3, 1.5, 1.0, 2)[0] == pytest.approx(1.5 / (2 - 1.5)) and "α/(2 − α) = 3" in C.PRESET_HELP["Leitung voll"]


def test_each_preset_shows_the_expected_verdict_lines():
    expect = {
        "Von Hand": ("warning", "verstärkt werden **alle Perioden über 4,15**; Spitze **10,59** bei Periode 6,19, bei Periode 6: 9,00 (genau α/(α_krit − α)); 48 % der Winkelachse."),
        "Resonanz": ("warning", "verstärkt werden **alle Perioden über 6,98**; Spitze **11,55** bei Periode 10,30, bei Periode 10: 9,65 (genau α/(α_krit − α)); 29 % der Winkelachse."),
        "Saison nahe der Lieferzeit": ("warning", "verstärkt werden **alle Perioden über 7,18**; Spitze **6,45** bei Periode 10,57, bei Periode 12: 3,84; 28 % der Winkelachse."),
        "Saison träge": ("success", "**keine Periode wird verstärkt**: die Spitze von |H_u| ist 1,00 (nur für sehr lange Perioden, T → ∞), bei Periode 12: 0,30."),
        "Saison lang": ("info", "verstärkt werden **alle Perioden über 10,32**; Spitze **1,39** bei Periode 14,87, bei Periode 52: 1,04; 19 % der Winkelachse."),
        "Leitung halb": ("warning", "verstärkt werden **die Perioden von 2,00 bis 2,53 und über 3,71**; Spitze **6,23** bei Periode 6,37, bei Periode 7: 4,64; 75 % der Winkelachse."),
        "Leitung voll": ("info", "verstärkt werden **alle Perioden**; Spitze **3,00** bei Periode 2,00, bei Periode 2: 3,00; 100 % der Winkelachse."),
    }
    for name, (kind, tail) in expect.items():
        p = C.PRESETS[name]
        a = E.bode_analysis(E.Settings(**p))
        k, text = E.verdict_bode(a)
        assert k == kind, name
        assert text.endswith(tail), (name, text)


def test_season_verdicts_of_the_presets_name_measured_and_estimated_settling():
    for name, est, meas in (("Von Hand", 87, 84), ("Resonanz", 161, 159), ("Saison lang", 22, 22), ("Leitung voll", 7, 7)):
        p = C.PRESETS[name]
        se = E.season_analysis(E.Settings(**p))
        text = E.verdict_season(se)[1]
        assert f"rund **{est} Perioden**" in text, name
        assert abs(se["settle_measured"] - meas) <= 2, (name, se["settle_measured"])
        assert E.season_warnings(se) == [], name                     # mit A = 0,1 greift in keinem Preset eine Warnung


# --- M2: Resonanz bei β = 0 ------------------------------------------------------------------------------------------------------------------

def test_resonance_gain_alpha_over_gap_in_34_cases():
    """L = 1, 2, 3, 4, 6, 8, 12 und q = α/α_krit = 0,3 / 0,5 / 0,7 / 0,9 / 0,99 (ohne L = 1, q = 0,5: dort wäre α = 1 eine Zweierpotenz): |H_u| bei Periode 4L − 2 gleich α/(α_krit − α)."""
    worst, n = 0.0, 0
    for L in (1, 2, 3, 4, 6, 8, 12):
        ak = K.alpha_crit_beta0(L)
        for q in (0.3, 0.5, 0.7, 0.9, 0.99):
            if L == 1 and q == 0.5:
                continue
            a = q * ak
            g = float(K.gain_u(math.pi / (2 * L - 1), L, a, 0.0))
            worst = max(worst, abs(g - a / (ak - a)) / (a / (ak - a)))
            n += 1
    assert n == 34 and worst < 1e-9, (n, worst)


def test_resonance_gain_in_200_random_cases():
    """Gegenprobe (ORACLE-PLAYBOOK §6): L = 1 bis 40, α zufällig in 0,01 bis 0,995 der Grenze (SplitMix64, Zufalls-Seed 1919), 200 Fälle; die Vormessung hatte 1 000 Fälle."""
    rng = SplitMix64(1919)
    worst = 0.0
    for _ in range(200):
        L = 1 + int(rng.random() * 40)
        ak = K.alpha_crit_beta0(L)
        a = (0.01 + 0.985 * rng.random()) * ak
        g = float(K.gain_u(math.pi / (2 * L - 1), L, a, 0.0))
        worst = max(worst, abs(g - a / (ak - a)) / (a / (ak - a)))
    assert worst < 1e-9, worst


RESULT_TABLE = {  # q: (α, ρ, |H_u(10)|, Spitze, Periode der Spitze, Schätzung, gemessene Einschwingzeit)
    0.3: (0.1854, 0.7097, 0.4286, 1.0000, math.inf, 13, 12),
    0.5: (0.3090, 0.8196, 1.0000, 1.4476, 14.42, 23, 20),
    0.7: (0.4326, 0.9022, 2.3333, 2.9725, 11.42, 45, 44),
    0.9: (0.5562, 0.9699, 9.0000, 10.7839, 10.33, 151, 149),
    0.99: (0.6119, 0.9971, 99.0000, 116.4625, 10.03, 1580, 1563),
}


@pytest.mark.parametrize("q", list(RESULT_TABLE))
def test_result_table_L3_beta0(q):
    a_, rho_, hu_, pk_, pT_, est_, meas_ = RESULT_TABLE[q]
    a = q * AK3
    _near(a, a_, 1e-4)
    rho = K.spectral_radius(3, a, 0.0)
    _near(rho, rho_, 1e-4)
    _near(K.hu_hi_at_period(3, a, 0.0, 10)[0], hu_, 5e-5)
    pk, pT = K.peak(3, a, 0.0)
    _near(pk, pk_, 5e-4)
    if math.isinf(pT_):
        assert math.isinf(pT)
    else:
        _near(pT, pT_, 0.006)
    _near(K.settling_estimate(rho), est_, 0.6)
    r = S.season_run(3, a, 0.0, 10)
    assert abs(r["settle_measured"] - meas_) <= 2, (q, r["settle_measured"])


def test_peak_period_lies_behind_4L_minus_2_and_comes_close_only_near_the_limit():
    """L = 2 bis 12: die Spitzenperiode liegt bei q = 0,5 um 43,8 bis 44,9 % hinter 4L − 2, bei q = 0,9 um 3,2 bis 3,3 %, bei q = 0,99 um 0,28 bis 0,29 %."""
    for q, lo, hi in ((0.5, 43.7, 45.0), (0.7, 13.9, 14.4), (0.9, 3.1, 3.4), (0.99, 0.27, 0.30)):
        ratios = []
        for L in range(2, 13):
            _, pT = K.peak(L, q * K.alpha_crit_beta0(L), 0.0)
            ratios.append(100.0 * (pT / (4 * L - 2) - 1.0))
        assert lo <= min(ratios) and max(ratios) <= hi, (q, min(ratios), max(ratios))


# --- M3: Beginn der Verstärkung ----------------------------------------------------------------------------------------------------------------

ONSET_RATIO = {1: 0.5000, 2: 0.3333, 3: 0.3236, 4: 0.3210, 6: 0.3194, 8: 0.3189, 12: 0.3186, 20: 0.3184, 30: 0.3184}


def test_amplification_onset_beta0_equals_one_over_2L_minus_1_and_the_ratio_to_the_limit():
    for L, ratio in ONSET_RATIO.items():
        ak = K.alpha_crit_beta0(L)
        am = K.amplification_onset_measured(L, 0.0, ak)
        assert abs(am * (2 * L - 1) - 1.0) < 5e-5, L                       # gemessen gleich 1/(2L − 1) auf 5·10⁻⁵ relativ (gemessen 2·10⁻⁵)
        _near(am / ak, ratio, 5e-5)
    _near(1.0 / math.pi, 0.3183, 5e-5)
    assert K.amplification_onset_formula(3, 0.0) == pytest.approx(0.2) and K.amplification_onset_formula(30, 0.0) == pytest.approx(1 / 59)


def test_amplification_onset_with_pipeline_weight_formula_against_measurement_in_25_cases():
    worst, n = 0.0, 0
    for beta in (0.3, 0.55, 0.7, 0.9, 1.0):
        for L in (2, 3, 4, 6, 12):
            lim = K.stable_limit(L, beta)
            am = K.amplification_onset_measured(L, beta, lim)
            worst = max(worst, abs(am - K.amplification_onset_formula(L, beta)))
            n += 1
    assert n == 25 and worst < 1e-4, worst                                   # gemessen 1,8·10⁻⁵
    _near(K.amplification_onset_measured(2, 0.55, K.stable_limit(2, 0.55)), 0.4945, 5e-4)
    _near(K.amplification_onset_measured(3, 0.55, 2.0), 0.3208, 5e-4)
    _near(K.amplification_onset_measured(12, 0.55, K.stable_limit(12, 0.55)), 0.0759, 5e-4)
    for L in (2, 3, 4, 6, 12):
        _near(K.amplification_onset_measured(L, 1.0, 2.0), 1.0, 1e-4)         # β = 1: ab α = 1 für jedes L


def test_no_amplification_below_and_always_above_the_onset_in_200_random_cases():
    """β = 0: unter α = 1/(2L−1) verstärkt keine Periode, darüber (bis 0,97 der Spanne zur Grenze) immer; β = 1 und α < 1: nie. 200 Fälle je Aussage, L = 1 bis 40, 60 001 Winkel."""
    rng = SplitMix64(1919)
    wq = np.linspace(1e-6, math.pi, 60001)
    wrong_below = wrong_above = wrong_one = 0
    for _ in range(200):
        L = 1 + int(rng.random() * 40)
        ak = K.alpha_crit_beta0(L)
        thr = 1.0 / (2 * L - 1)
        a_below = thr * (0.2 + 0.799 * rng.random())
        a_above = thr * (1.02 + (0.97 * (ak / thr - 1.0) - 0.02) * rng.random())
        wrong_below += float(np.max(K.gain_u(wq, L, a_below, 0.0))) > 1 + 1e-9
        wrong_above += not float(np.max(K.gain_u(wq, L, a_above, 0.0))) > 1
    for _ in range(200):
        L = 1 + int(rng.random() * 40)
        a = 0.01 + 0.989 * rng.random()
        wrong_one += float(np.max(K.gain_u(wq, L, a, 1.0))) > 1 + 1e-9
    assert (wrong_below, wrong_above, wrong_one) == (0, 0, 0)


# --- M4: Leitungsgewicht 0,55 und 1 -----------------------------------------------------------------------------------------------------------

def test_beta_1_peak_is_alpha_over_2_minus_alpha_at_period_2_for_every_L():
    for L in (3, 6, 12):
        for a, pk in ((1.2, 1.5), (1.5, 3.0), (1.9, 19.0)):
            g, T = K.peak(L, a, 1.0)
            _near(g, pk, 1e-6)
            assert T == 2.0
            _near(K.mean_gain2(L, a, 1.0)[0], a / (2 - a), 1e-9)
        assert K.peak(L, 0.5 + 0.01, 1.0)[0] == pytest.approx(1.0, abs=1e-9)        # α ≤ 1: keine Verstärkung (Spitze 1 bei T → ∞)
    for a in (0.6, 0.97):
        assert math.isinf(K.peak(3, a, 1.0)[1])


def test_beta_055_peaks_for_L_3_6_12():
    """β = 0,55 (statt ½): stabil für alle gerechneten L und α; die Zahlen von β = ½ der Vormessung (11,3 / 38,0 / 140,3 bei α = 1,5) sind nicht übertragbar."""
    expect = {(3, 1.2): (3.9892, 6.865, 2.5495), (3, 1.5): (6.2293, 6.366, 4.6628), (3, 1.9): (19.0000, 2.0, 14.2446),
              (6, 1.2): (6.7805, 12.985, 3.0079), (6, 1.5): (8.5121, 12.414, 4.9212), (6, 1.9): (17.8906, 2.396, 14.1220),
              (12, 1.2): (8.8252, 25.028, 3.0924), (12, 1.5): (9.5570, 24.426, 4.9320), (12, 1.9): (18.7023, 2.181, 14.1226)}
    for (L, a), (pk, pT, mu) in expect.items():
        assert K.is_stable(L, a, 0.55)
        g, T = K.peak(L, a, 0.55)
        _near(g, pk, 5e-4)
        _near(T, pT, 0.01)
        _near(K.mean_gain2(L, a, 0.55)[0], mu, 5e-4)


# --- M5: Zielkonflikt bei einer Saison ---------------------------------------------------------------------------------------------------------

def test_tradeoff_numbers_L3():
    _near(K.hu_hi_at_period(3, 0.12, 0.0, 12)[0], 0.2978, 5e-5)
    _near(K.hu_hi_at_period(3, 0.12, 0.0, 12)[1], 2.4818, 5e-5)
    _near(K.hu_hi_at_period(3, 0.52, 0.0, 12)[0], 3.8388, 5e-5)
    _near(K.hu_hi_at_period(3, 0.52, 0.0, 12)[1], 7.3823, 5e-5)
    # T = 52, β = 0: die Bestandsamplitude fällt mit α, die Bestellung wird kaum verändert
    _near(K.hu_hi_at_period(3, 0.1, 0.0, 52)[1], 7.5816, 5e-5)
    _near(K.hu_hi_at_period(3, 0.6, 0.0, 52)[1], 1.7369, 5e-5)
    _near(K.hu_hi_at_period(3, 0.3, 0.0, 52)[0], 1.0411, 5e-5)
    _near(K.hu_hi_at_period(3, 0.3, 0.0, 52)[1], 3.4705, 5e-5)
    # β = 1, T = 7: die Bestandsamplitude steigt mit α (anders als bei weißem Rauschen)
    _near(K.hu_hi_at_period(3, 0.1, 1.0, 7)[1], 1.2720, 5e-5)
    _near(K.hu_hi_at_period(3, 1.1, 1.0, 7)[1], 2.2630, 5e-5)


def test_tradeoff_verdict_for_L3_T12_and_T52_and_T7():
    t12 = E.tradeoff_analysis(3, 12)["ref"]
    _near(t12["alpha_lo"], 0.1236, 5e-5)
    _near(t12["alpha_hi"], 0.5253, 5e-5)
    assert t12[0.0]["lo"][1] < t12[0.0]["hi"][1] and t12[1.0]["lo"][1] < t12[1.0]["hi"][1]          # T = 12: steigt mit α
    t52 = E.tradeoff_analysis(3, 52)["ref"]
    assert t52[0.0]["lo"][1] > t52[0.0]["hi"][1] and t52[0.55]["lo"][1] > t52[0.55]["hi"][1]         # T = 52: fällt mit α
    t7 = E.tradeoff_analysis(3, 7)
    assert "β = 1: Bestellung 0,15 → 0,66, Bestand 1,30 → 1,91 (steigt mit α)" in E.verdict_tradeoff(t7)[1]


# --- M6: Rauschen ----------------------------------------------------------------------------------------------------------------------------

def test_default_preset_noise_numbers():
    """Von Hand (L = 2, β = 0, α = 0,9): Spitze² 112,2 gegen Mittel 5,90 = 171/29 (Verhältnis 19,0)."""
    a = E.noise_analysis(E.Settings(**C.PRESETS["Von Hand"]), 0.25)
    _near(a["peak"] ** 2, 112.2, 0.05)
    _near(a["mean_u"], 171 / 29, 1e-9)
    _near(a["lyap_u"], 171 / 29, 1e-9)
    _near(a["peak"] ** 2 / a["mean_u"], 19.0, 0.05)
    assert E.verdict_noise(a)[1].startswith("Spitze² = 112,2 (Spitze 10,59 bei Periode 6,19) gegen das Mittel von |H_u|² = **5,90** (Lyapunov-Gleichung: 5,90): Spitze² ist das 19,0fache")


def test_leitung_voll_noise_peak_squared_is_9_against_mean_3():
    """Plan-Korrektur: Bei β = 1, α = 1,5 ist die Spitze 3 (Verstärkung), ihr Quadrat 9; das Mittel von |H_u|² ist 3. „Spitze = Mittel“ vergliche Verstärkung mit Quadrat."""
    a = E.noise_analysis(E.Settings(**C.PRESETS["Leitung voll"]), 0.25)
    _near(a["peak"], 3.0, 1e-9)
    _near(a["peak"] ** 2, 9.0, 1e-8)
    _near(a["mean_u"], 3.0, 1e-9)


def test_season_plus_noise_numbers_for_the_readme():
    """L = 3, β = 0,55, α = 0,4, T = 7, A = 1, σ = 0,5, 400 000 Perioden, Zufalls-Seed 19: Simulation 0,2809 / 2,5124 gegen Vorhersage 0,2806 / 2,5100 (Band 1 %)."""
    r = S.noise_run(3, 0.4, 0.55, 7, 1.0, 0.5, 19, 400_000)
    _near(r["pred_season_u"] + r["pred_noise_u"], 0.28062, 5e-5)
    _near(r["pred_season_i"] + r["pred_noise_i"], 2.51000, 5e-4)
    assert abs(r["var_u"] / (r["pred_season_u"] + r["pred_noise_u"]) - 1.0) < 0.01
    assert abs(r["var_i"] / (r["pred_season_i"] + r["pred_noise_i"]) - 1.0) < 0.01
    _near(r["var_u"], 0.2809, 0.0015)
    _near(r["var_i"], 2.5124, 0.0125)
    _near(r["pred_season_u"], 0.1969, 5e-4)
    _near(r["pred_noise_u"], 0.0837, 5e-4)


def test_app_noise_run_with_100000_periods_stays_within_3_percent_in_every_preset():
    for name, p in C.PRESETS.items():
        s = E.Settings(**p)
        a = E.noise_analysis(s, C.DEFAULT_SIGMA)
        rows = E.variance_rows(a)
        assert all(abs(r["Abweichung (%)"]) < 3.0 for r in rows), (name, rows)


# --- M7: Einschwingen und Instabilität ----------------------------------------------------------------------------------------------------------

def test_settling_estimate_holds_within_10_percent_in_36_of_48_cases_and_always_near_the_limit():
    """L = 2, 3, 6; β = 0; q = 0,3 / 0,5 / 0,7 / 0,9; Perioden 4L − 2, 7, 12, 52: Schätzung ln 0,01/ln ρ gegen die gemessene Einschwingzeit auf 1 %."""
    total = good = 0
    per_q = {0.3: 0, 0.5: 0, 0.7: 0, 0.9: 0}
    ratios = []
    for L in (2, 3, 6):
        for q in per_q:
            a = q * K.alpha_crit_beta0(L)
            for T in (4 * L - 2, 7, 12, 52):
                r = S.season_run(L, a, 0.0, T)
                ok = abs(r["settle_measured"] - r["settle_estimate"]) <= 0.1 * r["settle_estimate"]
                good += ok
                per_q[q] += ok
                total += 1
                ratios.append(r["settle_measured"] / r["settle_estimate"])
    assert total == 48
    assert abs(good - 36) <= 3, good
    assert per_q[0.9] == 12 and abs(per_q[0.3] - 6) <= 2, per_q
    assert min(ratios) > 0.7 and max(ratios) < 1.2, (min(ratios), max(ratios))


def test_unstable_demand_response_grows_without_bound():
    """L = 3, β = 0, α = 1,05 α_krit, Bedarf cos(2πt/10) aus dem Ruhezustand: größte Bestellung in den Perioden 0–99, 100–199, 300–399: 69,7 / 339 / 5 790 (Band 1 %)."""
    a = 1.05 * AK3
    e = np.cos(2 * math.pi / 10 * np.arange(400))
    _, u = S.simulate(3, a, 0.0, e)
    for lo, hi, expect in ((0, 100, 69.7), (100, 200, 339.0), (300, 400, 5793.0)):
        assert abs(np.abs(u[lo:hi]).max() / expect - 1.0) < 0.01
    _near(K.spectral_radius(3, a, 0.0), 1.0143, 5e-5)


# --- M8: Verwandtes (nur Rechnung, anderes Modell) ------------------------------------------------------------------------------------------------

def test_mean_of_the_phi_squared_formula_of_another_model_equals_1_625_in_three_cases():
    """arXiv 2208.04459, Gl. 11 (wörtlich gelesen, anderes Modell: Order-up-to mit gleitendem Mittel, Länge P): Mittel von φ² über den Winkel ist 1 + 2(L/P + L²/P²). Reine Rechnung, kein Nachbau."""
    n = 1 << 14
    w = 2 * np.pi * (np.arange(n) + 0.5) / n
    for (L, P) in ((1, 4), (2, 8), (4, 16)):
        phi2 = 1 + 2 * (L / P + L ** 2 / P ** 2) * (1 - np.cos(P * w))
        _near(float(phi2.mean()), 1.625, 1e-9)
        _near(1 + 2 * (L / P + L ** 2 / P ** 2), 1.625, 1e-12)


# --- M9: Gleichstrom --------------------------------------------------------------------------------------------------------------------------

def test_dc_gain_numbers():
    for beta, alpha, hi in ((0.0, 0.3, -3.333333), (0.0, 0.4, -2.5), (0.55, 0.3, -4.433333), (1.0, 0.3, -5.333333)):
        _near(K.dc_gain_i(3, alpha, beta), hi, 5e-4)
        z1 = np.array([1.0 + 0j])
        _near(K.Hu(z1, 3, alpha, beta)[0].real, 1.0, 1e-12)
        _near(K.Hi(z1, 3, alpha, beta)[0].real, hi, 5e-4)


# --- M10: Verdikte der Experimente an den echten Analysen ----------------------------------------------------------------------------------------

def test_introduction_numbers_L3_beta0_alpha_03():
    """Warum-Absatz der README: α = 0,3: Periode 52: 1,04, Periode 12: 1,25, Periode 4: 0,18; α = 0,56, Periode 10: 9,65."""
    _near(K.hu_hi_at_period(3, 0.3, 0.0, 52)[0], 1.0411, 5e-5)
    _near(K.hu_hi_at_period(3, 0.3, 0.0, 12)[0], 1.2462, 5e-5)
    _near(K.hu_hi_at_period(3, 0.3, 0.0, 4)[0], 0.1829, 5e-5)
    _near(K.hu_hi_at_period(3, 0.56, 0.0, 10)[0], 9.6495, 5e-5)


def test_white_noise_trade_off_at_beta_1_variance_ratio_alpha_over_2_minus_alpha_and_stock_minimum_at_alpha_1():
    """β = 1, L = 3: var(u)/var(e) = α/(2−α) < 1 für α < 1; die Bestandsvarianz (L−1) + 1/(α(2−α)) ist bei α = 1 am kleinsten (dort 3,0 = L; nur dieser benannte Fall rechnet mit α = 1)."""
    mus = {a: K.mean_gain2(3, a, 1.0) for a in (0.1, 0.5, 0.9, 0.99, 1.01, 1.1, 1.5)}
    assert all(mus[a][0] < 1.0 for a in (0.1, 0.5, 0.9, 0.99))
    assert min(mus, key=lambda a: mus[a][1]) in (0.99, 1.01)
    _near(K.mean_gain2(3, 1.0, 1.0)[1], 3.0, 1e-9)
    _near(mus[0.1][1], 7.2632, 5e-5)                      # α = 0,1: 7,26 gegen 3,0 bei α = 1 (Vormessung V5: 7,26)


def test_tradeoff_verdict_texts_of_the_readme_for_L3_T12_and_T52():
    t12 = E.verdict_tradeoff(E.tradeoff_analysis(3, 12))[1]
    assert ("β = 0: Bestellung 0,31 → 3,85, Bestand 2,50 → 7,33 (steigt mit α); β = 0,55: Bestellung 0,27 → 1,31, Bestand 2,42 → 3,60 (steigt mit α); "
            "β = 1: Bestellung 0,25 → 0,83, Bestand 2,36 → 2,93 (steigt mit α).") in t12
    t52 = E.verdict_tradeoff(E.tradeoff_analysis(3, 52))[1]
    assert ("β = 0: Bestellung 0,85 → 1,04, Bestand 6,90 → 1,99 (fällt mit α); β = 0,55: Bestellung 0,79 → 1,02, Bestand 7,22 → 3,05 (fällt mit α); "
            "β = 1: Bestellung 0,74 → 0,99, Bestand 7,42 → 3,84 (fällt mit α).") in t52


def test_onset_verdict_for_L3_beta_0_and_beta_1_and_beta_055():
    assert E.verdict_onset(E.onset_analysis(3, 0.0))[1] == ("β = 0, L = 3: Verstärkung ab α = 0,200 (gemessen; Näherung 0,200), das sind 32,4 % der Stabilitätsgrenze 0,618; "
                                                           "L = 30: 31,8 % der Grenze, 1/π = 31,8 %.")
    assert E.verdict_onset(E.onset_analysis(3, 1.0))[1] == "β = 1: Verstärkung erst ab α = 1,000 (gemessen), für jedes L; die Stabilitätsgrenze liegt bei 2,000. L = 3: 50,0 % der Grenze, L = 30: 50,0 %."
    t = E.verdict_onset(E.onset_analysis(3, 0.55))[1]
    assert "β = 0,55, L = 3: Verstärkung ab α = 0,321 (gemessen; Näherung 0,321), das sind 16,0 % der Stabilitätsgrenze 2,000; L = 30: 1,5 % der Grenze." == t


def test_compare_verdict_of_the_presets_leitung_halb():
    p = C.PRESETS["Leitung halb"]
    text = E.verdict_compare(E.compare_analysis(p["L"], p["alpha"]))[1]
    assert text == ("L = 3, α = 1,5: β = 0: instabil (Grenze 0,618); β = 0,55: stabil, Spitze 6,23 bei Periode 6,37 (Mittel von |H_u|² 4,66); "
                    "β = 1: stabil, Spitze 3,00 bei Periode 2,00 (Mittel von |H_u|² 3,00).")


def test_noise_table_deviation_is_small_for_the_readme_case():
    """Von Hand mit dem Zufalls-Seed der App (35), σ = 0,25 und A = 0,1: Simulation gegen Zerlegung (100 000 Perioden)."""
    a = E.noise_analysis(E.Settings(**C.PRESETS["Von Hand"]), 0.25)
    rows = E.variance_rows(a)
    _near(rows[0]["Summe"], 0.7735, 5e-4)
    _near(rows[1]["Summe"], 0.9550, 5e-4)
    assert all(abs(r["Abweichung (%)"]) < 3.0 for r in rows)
