"""Randfälle (DEMO-PLAYBOOK §12): L = 1, β = 1 mit α = 1, α an und über der Grenze, T = 2, sehr lange Perioden, Spektralradius nahe 1, β > 1.

Die benannten Randfälle β = 1 und α = 1 (|H_u| = 1 für jede Periode) und T = 2 sind Randwerte des Modells; das Modell ist linear und ohne Klemmung, ein Rundungsartefakt
wie bei Zweierpotenzen im geklemmten Lager kann nicht auftreten (die Parameter-Rundungsprobe steht in test_eigenschaften_frequenzgang_lager.py)."""

import math

import numpy as np
import pytest

import fgl_constants as C
import fgl_core as K
import fgl_evaluation as E
import fgl_sim as S


def test_L_equals_1_one_period_of_dead_time():
    """L = 1: P(z) = z − 1 + α, H_u = α/(z − 1 + α); stabil bis α = 2, Spitze α/(2−α) bei Periode 2 für α > 1 (hier β egal)."""
    for beta in (0.0, 0.55, 1.0):
        assert K.stable_limit(1, beta) == pytest.approx(2.0, abs=1e-9)
        pk, pT = K.peak(1, 1.4, beta)
        assert pk == pytest.approx(1.4 / 0.6, rel=1e-9) and pT == pytest.approx(2.0)
    assert K.hu_hi_at_period(1, 0.7, 0.0, 6)[1] == pytest.approx(abs(1 / (np.exp(1j * math.pi / 3) - 1 + 0.7)), rel=1e-12)


def test_beta_1_alpha_1_named_case_has_gain_one_at_every_period():
    """β = 1, α = 1: die Leitungsrechnung macht aus dem Lager eine reine Verzögerung ohne Verstärkung; |H_u| = 1 bei jeder Periode (Spektralradius 0)."""
    for L in (1, 3, 8):
        assert K.spectral_radius(L, 1.0, 1.0) == pytest.approx(0.0, abs=1e-9)
        for T in (2, 3, 7, 52):
            assert K.hu_hi_at_period(L, 1.0, 1.0, T)[0] == pytest.approx(1.0, abs=1e-12)
        pk, pT = K.peak(L, 1.0, 1.0)
        assert pk == pytest.approx(1.0, abs=1e-9)


def test_alpha_at_the_limit_gives_no_frequency_response_only_a_band():
    """α = α_krit: Wurzel auf dem Einheitskreis, die App meldet „instabil“ (Spektralradius ≥ 1 − 1e-9); knapp darunter gibt es Zahlen."""
    ak = K.alpha_crit_beta0(3)
    s = E.Settings(L=3, beta=0.0, alpha=ak, T=10)
    a = E.bode_analysis(s)
    assert not a["stable"]
    assert E.verdict_bode(a)[0] == "warning" and "instabil" in E.verdict_bode(a)[1]
    assert E.bode_analysis(E.Settings(L=3, beta=0.0, alpha=0.999 * ak, T=10))["stable"]
    with pytest.raises(ValueError):
        S.season_run(3, ak, 0.0, 10)


def test_above_the_limit_the_app_shows_no_numbers_and_the_season_warning_names_the_limit():
    s = E.Settings(L=3, beta=0.0, alpha=1.5, T=10)
    a = E.bode_analysis(s)
    se = E.season_analysis(s)
    assert not a["stable"] and "periods" not in a and not se["stable"]
    kind, text = E.season_warnings(se)[0]
    assert kind == "error" and "0,618" in text and "wächst ohne Obergrenze" in text
    assert E.verdict_stable_but_amplifies(a)[0] == "error"


def test_period_2_nyquist_amplitude_and_phase_convention():
    """T = 2: e_t = cos(π t) = ±1; die Amplitude der Simulation ist ohne den Faktor 2 gerechnet, der Betrag stimmt mit |H| überein."""
    r = S.season_run(3, 0.3, 0.55, 2)
    hu, hi = K.hu_hi_at_period(3, 0.3, 0.55, 2)
    assert r["amp_u"] == pytest.approx(hu, rel=1e-9) and r["amp_i"] == pytest.approx(hi, rel=1e-9)


def test_very_long_period_has_gain_close_to_one():
    assert K.hu_hi_at_period(3, 0.3, 0.0, 6000)[0] == pytest.approx(1.0, abs=1e-5)
    assert K.hu_hi_at_period(3, 0.3, 0.0, 60)[0] > 1.0


def test_spectral_radius_near_one_the_simulation_is_capped_and_flagged():
    """L = 12, β = 0, α = 99,9 % der Grenze: Einschwingzeit (Schätzung) über 70 000 Perioden; die Simulation läuft höchstens SIM_CAP Perioden und meldet „nicht eingeschwungen“.
    Bei 99 % der Grenze (Schätzung rund 7 400 Perioden) reichen die 30 000 Perioden aus."""
    assert S.season_run(12, 0.99 * K.alpha_crit_beta0(12), 0.0, 46)["settled"]
    a = 0.999 * K.alpha_crit_beta0(12)
    r = S.season_run(12, a, 0.0, 46)
    assert r["n"] == C.SIM_CAP
    assert r["settle_estimate"] > 70000 and not r["settled"] and r["settle_measured"] is None
    se = E.season_analysis(E.Settings(L=12, beta=0.0, alpha=a, T=46))
    assert any("noch nicht eingeschwungen" in t for _, t in E.season_warnings(se))
    assert se["n_plot"] <= 2000
    # kurze Einschwingzeit: eingeschwungen
    assert S.season_run(3, 0.3, 0.0, 12)["settled"]


def test_beta_above_one_and_beta_0_below_the_onset():
    """β = 1,3 (nur Formelprüfung, nicht im Regler): H stimmt mit der Simulation überein; β = 0 und α unter 1/(2L−1): keine Periode wird verstärkt."""
    z = np.exp(2j * math.pi / 7)
    r = S.season_run(3, 0.31, 1.3, 7) if K.spectral_radius(3, 0.31, 1.3) < 1 else None
    assert r is not None
    assert r["amp_u"] == pytest.approx(abs(K.Hu(z, 3, 0.31, 1.3)), rel=1e-9)
    for L in (2, 3, 6):
        a = 0.97 / (2 * L - 1)
        assert not K.amplifies(L, a, 0.0)
        assert K.amplifies(L, 1.03 / (2 * L - 1), 0.0)


def test_amplified_intervals_can_be_disjoint():
    """L = 3, β = 0,55, α = 1,5: zwei Bereiche (Perioden 2 bis 2,53 und über 3,71), 75 % der Winkelachse."""
    iv = K.amplified_intervals(3, 1.5, 0.55)
    assert len(iv) == 2
    assert iv[0][0] == pytest.approx(2.0) and iv[0][1] == pytest.approx(2.534, abs=1e-3) and iv[1][0] == pytest.approx(3.707, abs=1e-3) and math.isinf(iv[1][1])
    assert K.amplified_share(3, 1.5, 0.55) == pytest.approx(0.75, abs=0.001)
    assert K.amplified_intervals(3, 0.12, 0.0) == []
