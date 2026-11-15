"""Verdikte an von Hand gesetzten Werten (DEMO-PLAYBOOK §12): jede Zahl im Text und jede Schwelle der Art steht hier.

Ein Verdikt ist eine reine Funktion (Art, Text) und bekommt Wörterbücher, nicht Seeds. Damit prüft der Test die Grenzen (Spitze ab 5 als Warnung, Klemmung ab A·|H_u| > 1)
und jede genannte Zahl direkt; die Verdikte der Presets an den echten Analysen stehen in `test_claims.py`."""

import math

import fgl_evaluation as ev


def _bode(**over):
    base = {"L": 2, "beta": 0.0, "alpha": 0.9, "T": 6, "rho": 0.94868, "limit": 1.0, "stable": True, "peak": 10.5903, "peak_T": 6.19, "hu_T": 9.0, "hi_T": 10.0,
            "share": 0.482, "intervals": [(4.1467, math.inf)], "settle_estimate": 87.4, "resonance_T": 6, "mean_u": 5.8966}
    base.update(over)
    return base


# ---------- Formatierung ----------


def test_number_formatting_uses_decimal_commas_and_trims_parameters():
    assert ev.de(4.1467) == "4,15" and ev.de(0.9, 3) == "0,900" and ev.de(1.5, 1, True) == "+1,5"
    assert ev.para(0.9) == "0,9" and ev.para(2.0) == "2" and ev.para(0.55) == "0,55" and ev.para(0.56) == "0,56"
    assert ev.period_text(math.inf) == "∞" and ev.period_text(6.19) == "6,19"
    assert ev.int_text(100000) == "100 000"


def test_intervals_text_for_one_open_and_two_disjoint_ranges():
    assert ev.intervals_text([(4.1467, math.inf)]) == "alle Perioden über 4,15"
    assert ev.intervals_text([(2.0, math.inf)]) == "alle Perioden"
    assert ev.intervals_text([(2.0, 2.534), (3.707, math.inf)]) == "die Perioden von 2,00 bis 2,53 und über 3,71"


# ---------- Kernfrage 1 ----------


def test_bode_verdict_hand_example_names_edge_peak_resonance_and_share():
    kind, text = ev.verdict_bode(_bode())
    assert kind == "warning"                                       # Spitze 10,59 ≥ 5
    assert text == ("L = 2, β = 0, α = 0,9: verstärkt werden **alle Perioden über 4,15**; Spitze **10,59** bei Periode 6,19, bei Periode 6: 9,00 (genau α/(α_krit − α)); 48 % der Winkelachse.")


def test_bode_verdict_threshold_for_warning_is_5_inclusive():
    assert ev.verdict_bode(_bode(peak=5.0))[0] == "warning"
    assert ev.verdict_bode(_bode(peak=4.99))[0] == "info"


def test_bode_verdict_no_exact_remark_off_resonance_or_with_beta():
    assert "genau" not in ev.verdict_bode(_bode(T=7))[1]
    assert "genau" not in ev.verdict_bode(_bode(beta=0.55))[1]


def test_bode_verdict_without_amplification_is_a_success():
    kind, text = ev.verdict_bode(_bode(L=3, alpha=0.12, T=12, peak=1.0, peak_T=math.inf, hu_T=0.2978, intervals=[], share=0.0, resonance_T=10))
    assert kind == "success"
    assert text == "L = 3, β = 0, α = 0,12: **keine Periode wird verstärkt**: die Spitze von |H_u| ist 1,00 (nur für sehr lange Perioden, T → ∞), bei Periode 12: 0,30."


def test_bode_verdict_unstable_names_rho_and_limit():
    kind, text = ev.verdict_bode({"L": 3, "beta": 0.0, "alpha": 1.5, "stable": False, "rho": 1.4321, "limit": 0.618})
    assert kind == "warning" and "**instabil** (Spektralradius ρ = 1,432 ≥ 1, Grenze α = 0,618)" in text and "kein Frequenzgang" not in text and "ein Frequenzgang existiert dort nicht" in text


def test_bode_verdict_two_ranges_and_infinite_peak_period():
    kind, text = ev.verdict_bode(_bode(L=3, beta=0.55, alpha=1.5, T=7, peak=6.2293, peak_T=6.37, hu_T=4.6448, share=0.75, intervals=[(2.0, 2.534), (3.707, math.inf)], resonance_T=10))
    assert kind == "warning"
    assert "die Perioden von 2,00 bis 2,53 und über 3,71" in text and "Spitze **6,23** bei Periode 6,37" in text and "bei Periode 7: 4,64;" in text and "75 % der Winkelachse" in text


# ---------- Kernfrage 2 ----------


def test_stable_but_amplifies_verdict_both_branches_and_the_unstable_one():
    kind, text = ev.verdict_stable_but_amplifies(_bode())
    assert kind == "warning"
    assert text == ("Das Lager ist stabil (ρ = 0,949 < 1, α ist 90 % der Grenze 1,000): ein einzelner Stoß klingt auf 1 % in rund 87 Perioden ab (Schätzung aus ρ). "
                    "Trotzdem verstärkt es eine Saison bis zum **10,59fachen** (bei Periode 6,19): „stabil“ heißt nicht „verstärkt nicht“.")
    kind, text = ev.verdict_stable_but_amplifies(_bode(peak=1.0, peak_T=math.inf, alpha=0.12, limit=0.618, rho=0.823, settle_estimate=23.6))
    assert kind == "success" and "α ist 19 % der Grenze 0,618" in text and text.endswith("Es verstärkt auch keine Saison: die Spitze von |H_u| ist 1.")
    kind, text = ev.verdict_stable_but_amplifies({"stable": False, "rho": 1.2})
    assert kind == "error" and "ρ = 1,200" in text


# ---------- Experiment 1 ----------


def _season(**over):
    base = {"L": 3, "beta": 0.0, "alpha": 0.56, "T": 10, "A": 0.1, "rho": 0.97184, "stable": True, "limit": 0.618, "hu": 9.6495, "hi": 17.2313, "amp_u": 9.6495, "amp_i": 17.2313,
            "settle_measured": 159, "settle_estimate": 161.0, "settled": True, "n_sim": 1136, "u_max_transient": 9.826}
    base.update(over)
    return base


def test_season_verdict_names_amplitudes_settling_estimate_and_measurement():
    kind, text = ev.verdict_season(_season())
    assert kind == "info"
    assert text == ("Eingeschwungene Bestellamplitude **9,65·A** (Frequenzgang 9,65), Bestandsamplitude **17,23·A** (Frequenzgang 17,23); "
                    "Einschwingzeit auf 1 % rund **161 Perioden** (Schätzung aus ρ), gemessen 159; vorher gilt der Frequenzgang noch nicht.")
    assert "nicht gemessen (Simulation begrenzt)" in ev.verdict_season(_season(settle_measured=None))[1]
    assert ev.verdict_season({"stable": False, "alpha": 1.5, "limit": 0.618})[0] == "warning"


def test_season_warning_threshold_is_steady_amplitude_above_the_mean_demand():
    """Klemmung ab A·|H_u| > 1 (Mittelbedarf 1): genau 1 warnt nicht."""
    assert ev.season_warnings(_season(A=0.1, hu=9.6495)) == []                               # 0,965
    assert ev.season_warnings(_season(A=0.1037, hu=9.6495))[0][0] == "warning"             # 1,0007; die Grenze liegt bei A = 0,10363
    assert ev.season_warnings(_season(A=0.5, hu=2.0, u_max_transient=0.1)) == []             # genau 1: keine Warnung
    kind, text = ev.season_warnings(_season(A=0.5, hu=2.0001, u_max_transient=0.1))[0]
    assert kind == "warning" and "A·|H_u| = 1,00 liegt über dem Mittelbedarf (1)" in text and "lager-chaos-demo" in text


def test_season_warning_for_the_transient_and_the_cap_and_the_unstable_case():
    kind, text = ev.season_warnings(_season(A=0.1, hu=9.6495, u_max_transient=10.5))[0]
    assert kind == "info" and "Im Einschwingen" in text and "1,05" in text
    assert any("auf 30 000 Perioden begrenzt" in t for _, t in ev.season_warnings(_season(settled=False)))
    kind, text = ev.season_warnings({"stable": False, "alpha": 1.5, "limit": 0.618})[0]
    assert kind == "error" and "α = 1,5" in text and "0,618" in text


# ---------- Experiment 2 und 5 ----------


def test_tradeoff_verdict_direction_words_follow_the_numbers():
    nan = math.nan
    ref = {"alpha_lo": 0.1236, "alpha_hi": 0.5253,
           0.0: {"lo": (0.31, 2.50), "hi": (3.85, 7.33)}, 0.55: {"lo": (0.27, 2.42), "hi": (1.31, 3.60)}, 1.0: {"lo": (0.74, 7.42), "hi": (0.99, 3.84)}}
    kind, text = ev.verdict_tradeoff({"L": 3, "T": 12, "limit0": 0.618, "ref": ref})
    assert kind == "info"
    assert text.startswith("L = 3, Periode 12: von α = 0,124 (20 % der Grenze 0,618 ohne Leitungsgewicht) auf α = 0,525 (85 %)")
    assert "β = 0: Bestellung 0,31 → 3,85, Bestand 2,50 → 7,33 (steigt mit α)" in text
    assert "β = 0,55: Bestellung 0,27 → 1,31, Bestand 2,42 → 3,60 (steigt mit α)" in text
    assert "β = 1: Bestellung 0,74 → 0,99, Bestand 7,42 → 3,84 (fällt mit α)" in text
    ref[0.55] = {"lo": (nan, nan), "hi": (nan, nan)}
    assert "β = 0,55: bei einem der beiden α instabil" in ev.verdict_tradeoff({"L": 3, "T": 12, "limit0": 0.618, "ref": ref})[1]


def test_compare_verdict_names_unstable_stable_with_peak_and_stable_without_amplification():
    curves = {0.0: {"stable": False, "limit": 0.618}, 0.55: {"stable": True, "peak": 6.2293, "peak_T": 6.37, "mean_u": 4.6628, "limit": 2.0},
              1.0: {"stable": True, "peak": 1.0, "peak_T": math.inf, "mean_u": 0.39, "limit": 2.0}}
    kind, text = ev.verdict_compare({"L": 3, "alpha": 1.5, "curves": curves})
    assert kind == "info"
    assert text == ("L = 3, α = 1,5: β = 0: instabil (Grenze 0,618); β = 0,55: stabil, Spitze 6,23 bei Periode 6,37 (Mittel von |H_u|² 4,66); "
                    "β = 1: stabil, keine Verstärkung (Mittel von |H_u|² 0,39).")


# ---------- Experiment 3 ----------


def test_noise_verdict_peak_squared_against_the_mean():
    a = {"stable": True, "peak": 10.5903, "peak_T": 6.19, "mean_u": 5.8966, "lyap_u": 5.8966}
    kind, text = ev.verdict_noise(a)
    assert kind == "info"
    assert text == ("Spitze² = 112,2 (Spitze 10,59 bei Periode 6,19) gegen das Mittel von |H_u|² = **5,90** (Lyapunov-Gleichung: 5,90): "
                    "Spitze² ist das 19,0fache des Mittels; das Rauschmaß allein sagt nicht, welche Periode die Regel am stärksten verstärkt.")
    kind, text = ev.verdict_noise({"stable": True, "peak": 1.0, "peak_T": math.inf, "mean_u": 0.0815, "lyap_u": 0.0815})
    assert kind == "success" and "gedämpft" in text and "nicht gedämpft" not in text
    assert "nicht gedämpft" in ev.verdict_noise({"stable": True, "peak": 1.0, "peak_T": math.inf, "mean_u": 1.0, "lyap_u": 1.0})[1]
    assert ev.verdict_noise({"stable": False})[0] == "warning"


def test_variance_rows_sum_and_deviation():
    sim = {"pred_season_u": 0.4, "pred_noise_u": 0.1, "var_u": 0.51, "pred_season_i": 2.0, "pred_noise_i": 1.0, "var_i": 2.97}
    rows = ev.variance_rows({"sim": sim})
    assert rows[0]["Summe"] == 0.5 and abs(rows[0]["Abweichung (%)"] - 2.0) < 1e-9
    assert rows[1]["Größe"] == "Bestand i" and rows[1]["Summe"] == 3.0 and abs(rows[1]["Abweichung (%)"] + 1.0) < 1e-9


# ---------- Experiment 4 ----------


def test_onset_verdict_for_beta_0_adds_one_over_pi_and_for_beta_1_the_every_L_sentence():
    a = {"L": 3, "beta": 0.0, "onset": 0.2, "limit": 0.618, "formula": 0.2, "onset30": 0.01695, "limit30": 0.05324}
    kind, text = ev.verdict_onset(a)
    assert kind == "info"
    assert text == ("β = 0, L = 3: Verstärkung ab α = 0,200 (gemessen; Näherung 0,200), das sind 32,4 % der Stabilitätsgrenze 0,618; L = 30: 31,8 % der Grenze, 1/π = 31,8 %.")
    kind, text = ev.verdict_onset({"L": 3, "beta": 1.0, "onset": 1.0, "limit": 2.0, "formula": 1.0, "onset30": 1.0, "limit30": 2.0})
    assert text == "β = 1: Verstärkung erst ab α = 1,000 (gemessen), für jedes L; die Stabilitätsgrenze liegt bei 2,000. L = 3: 50,0 % der Grenze, L = 30: 50,0 %."
    text = ev.verdict_onset({"L": 3, "beta": 0.55, "onset": 0.3208, "limit": 2.0, "formula": 0.3208, "onset30": 0.0, "limit30": 2.0})[1]
    assert "1/π" not in text and "β = 0,55, L = 3: Verstärkung ab α = 0,321" in text and "16,0 % der Stabilitätsgrenze 2,000" in text
