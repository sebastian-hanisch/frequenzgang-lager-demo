# Frequenzgang des Lagers – Streamlit-Demo

**[→ Demo live ausprobieren](https://sebastianhanisch-frequenzgang-lager-demo.streamlit.app/)**

Stück 5 der Linie **Rückkopplung und Verzögerung** in der „Konzepte“-Reihe für die Website „Sebastian Hanisch – Operations Research und Machine Learning“. Die Linie fragt, wann ein Kreis aus Bestand, Fluss und Verzögerung schwingt, wächst oder kippt; ihr Vehikel ist ein Lager mit linearer Bestellregel (Lieferzeit L, Anpassung α, Leitungsgewicht β).
Dieses Stück hängt als Kante `fix` an den Stücken 1 und 2 ([`lager-lieferzeit-demo`](https://github.com/sebastian-hanisch/lager-lieferzeit-demo): Grenze α_krit, Periode 4L − 2; [`leitung-mitrechnen-demo`](https://github.com/sebastian-hanisch/leitung-mitrechnen-demo): das Leitungsgewicht β): Grenze und Rauschverstärkung (bei β = 1: α/(2 − α)) sagen nicht, **welche** Schwankungen des Bedarfs die Regel verstärkt und wie stark.
„Stabil“ heißt dort „ein Stoß klingt ab“, nicht „verstärkt nicht“. Die Nachbarn der Linie: [`meldeverzug-demo`](https://github.com/sebastian-hanisch/meldeverzug-demo) (späte Meldung des Bestands), [`streuende-lieferzeit-demo`](https://github.com/sebastian-hanisch/streuende-lieferzeit-demo) (Lieferzeit mit Streuung). Die Schwesterlinie Dynamische Systeme hat dasselbe Lager mit Klemmung
([`lager-chaos-demo`](https://github.com/sebastian-hanisch/lager-chaos-demo)) und in stetiger Zeit ([`verzoegerung-hopf-demo`](https://github.com/sebastian-hanisch/verzoegerung-hopf-demo)); die Ketten aus mehreren Stufen mit Prognose rechnet [`bullwhip-demo`](https://github.com/sebastian-hanisch/bullwhip-demo).
Hier steht die **Einzelschleife** ohne Prognose, ohne Kette, ohne Kosten, dafür das Frequenzbild.

## Warum dieses Problem

Eine Lagerregel verstärkt nicht jede Schwankung gleich. Beispiel L = 3, β = 0, α = 0,3: eine Saison der Periode 52 wird kaum verändert (Verstärkung 1,04), die Periode 12 auf das 1,25fache verstärkt und die Periode 4 auf das 0,18fache gedämpft; mit α = 0,56 wird die Periode 10 auf das 9,65fache verstärkt. Wer nur die Stabilitätsgrenze oder das Rauschverhältnis kennt, sieht das nicht:
Das weiße Rauschen enthält alle Perioden gleich stark, seine Wirkung auf die Varianz ist nur der **Mittelwert** von |H|² über alle Perioden; eine schmale Spitze geht darin weitgehend unter. Die Demo fragt: **Welche Perioden des Bedarfs verstärkt die Regel, wie stark, ab welchem α, und was verbirgt das Rauschmaß?**
Der Frequenzgang als Werkzeug ist in der regelungstheoretischen Lieferketten-Literatur verbreitet (nach Suchtreffern, **nicht gelesen**); alle Herleitungen hier sind **eigene Herleitungen in diesem Modell**, keine Neuheitsaussage.

## Modell

Abweichungen vom Soll-Bestand (i), vom Mittelbedarf (u, e), Perioden als Zähleinheit, **ohne Klemmung**, feste Lieferzeit, kein Meldeverzug:

- Lieferung: **i_{t+1} = i_t + u_{t+1−L} − e_t**, e_t die Bedarfsabweichung.
- Regel: **u_t = −α (i_t + β Σ_{j=1..L−1} u_{t−j})**; β ist das Gewicht der unterwegs befindlichen Ware.
- **Übertragungsfunktionen** (z-Transformation; eigene Herleitung): P(z) = z^L − (1 − αβ) z^(L−1) + α(1 − β), **H_u = α z^(L−1) / P(z)** (Bedarf → Bestellung), **H_i = −(z^(L−1) + αβ Σ_{k=0..L−2} z^k) / P(z)** (Bedarf → Bestand), Frequenzgang bei z = e^{iω}, ω = 2π/T für die Periode T.
- Der Frequenzgang gilt im eingeschwungenen Zustand und im stabilen Bereich (alle Wurzeln von P im Einheitskreis). **Stabil** heißt hier Spektralradius ρ < 1 − 10⁻⁹.
- **Verstärkt** heißt |H_u| > 1 (auf dem Gitter über 1 + 10⁻⁹). **Anteil der Winkelachse** ist der Anteil der Winkel in (0, π] mit |H_u| > 1 auf einem gleichmäßigen Gitter von 20 001 Punkten in ω (nicht in der Periode).
- Gleichstrom: H_u(1) = 1 (ein dauerhaft höherer Bedarf geht voll in die Bestellung), H_i(1) = −(1 + αβ(L − 1))/α.
- Die Formeln stehen im Expander 📐 der App; L, β, α, die Periode T und die Amplitude A der Saison (Vielfache des Mittelbedarfs 1) sind Regler.

## Methodik

| Verfahren | Wie es arbeitet | Grenze |
|---|---|---|
| **Übertragungsfunktion** (`fgl_core.Hu`, `Hi`, Winkelform `gain_u`, `gain_i`) | geschlossene Formel an z = e^{iω}; die Winkelform rechnet \|P\|² mit reellen Kosinus | gilt nur im stabilen Bereich ohne Klemmung |
| **Direkte Simulation** (`fgl_sim.simulate`) | die Modellgleichungen Periode für Periode aus dem Ruhezustand (leere Leitung, i₀ = 0), nie über H selbst (nur die Einschwingzeit vergleicht die Bahn mit dem Verlauf aus H); Amplitude und Phase aus dem Ende der Bahn (Fenster 40 Perioden) | höchstens 30 000 Perioden; darüber „nicht eingeschwungen“ |
| **Mittel von \|H\|²** (`fgl_core.mean_gain2`) | Mittelpunktsregel auf 2¹⁷ Punkten über den Winkelkreis | Quadratur, kein Beweis |
| **Lyapunov-Gleichung** (`fgl_sim.lyapunov_variances`) | `scipy.linalg.solve_discrete_lyapunov` der Zustandsmatrix: Varianzverhältnis bei weißem Rauschen | eine zweite Rechnung zum selben Mittel |
| **Spitze** (`fgl_core.peak`) | Gitter von 100 001 Winkeln in (0, π] plus ω = 0, dann Verfeinerung um den besten Punkt | keine geschlossene Form für die Periode der Spitze |
| **Beginn der Verstärkung** (`amplification_onset_measured`) | Bisektion auf dem Gitter von 20 001 Winkeln in einer Klammer um die Näherung α* (die Klammer wird geprüft, sonst volle Rasterung); die Näherung α* ist die Nullstelle von a x² + b x − 1 | Messung, kein Beweis, dass keine andere Periode früher verstärkt |
| **Einschwingzeit** (`settling_estimate`, `season_run`) | Schätzung ln 0,01/ln ρ; gemessen als Zeit, nach der die Bestellung dauerhaft unter 1 % der eingeschwungenen Amplitude vom eingeschwungenen Verlauf abweicht | die Schätzung ist keine Regel (siehe Befunde) |

Der Zufall (Rauschen in Experiment 3) kommt aus einem eigenen **SplitMix64** (`fgl_rng.py`, Kopie von `lgc_rng.py` aus `lager-chaos-demo`, vektorisiert gegen die Klasse geprüft); nie Python-`hash()`. Alle Rechnungen der App sind **live**; es gibt keine vorgerechneten Reihen.
Die Live-Rechnung braucht je Eingabe höchstens 0,7 Sekunden (Messung auf dem Entwicklungsrechner, nicht festgeschrieben; die Karte in Experiment 4 rechnet für β > 0 bis 0,6 Sekunden und wird je β zwischengespeichert, die Simulation mit 100 000 Perioden 0,2 Sekunden).

## Befunde (gemessen, keine Behauptungen)

Alle Zahlen: in diesem Modell, ohne Klemmung, im stabilen Bereich; α und β nie Zweierpotenzen (die Randwerte β = 1 und T = 2 des Presets „Leitung voll“ sind benannt). Die Tests prüfen **Bänder**; die Zahlen sind die des Entwicklungsrechners.
Die Spalte „Test“ nennt die Funktionen in `tests/test_claims.py`, sofern keine andere Datei genannt ist.

| Frage | Befund | Test |
|---|---|---|
| Stimmt die Formel für H? | Die Übertragungsfunktionen stimmen gegen die Zustandsmatrix c (zI − A)⁻¹ B, die aus der Modellgleichung durch Einheitsvektoren gewonnen wird, in 1 260 Punkten (L = 1 bis 12 in 9 Werten, β = 0; 0,3; 0,55; 1,3, fünf α auch über der Grenze, sieben Winkel) auf unter 10⁻¹¹ relativ; Betrag **und** Phase gegen die direkte Simulation in 60 Zufallsfällen (Zufalls-Seed 19, ρ < 0,99, Perioden 2 bis 52) auf unter 10⁻⁹. Die Vormessung `out_rk_v19_frequenzgang.txt` (nicht im Repo) hatte 9 000 Punkte mit 4,5·10⁻¹⁴ und 150 Fälle mit 1,2·10⁻¹³ | `test_oracle_frequenzgang_lager.py::test_hu_hi_equal_the_state_space_transfer_on_a_grid`, `::test_amplitude_and_phase_equal_the_simulation_in_60_cases` |
| Handbeispiel von Hand (L = 2, β = 0, α = 9/10) | Bei Periode 6 = 4L − 2 genau **9** (α/(α_krit − α) mit α_krit = 1; in Brüchen: P(e^{iπ/3}) = −1/10), Bestand 10; Periode 4 wird gedämpft (\|H_u\|² = 81/101); verstärkt werden alle Perioden über **4,15** (cos ω > 1/18); Spitze **10,59** bei Periode 6,19; **48 %** der Winkelachse; Mittel von \|H_u\|² = **171/29 = 5,90**, von \|H_i\|² 1 900/261 = 7,28; das Quadrat der Spitze (112,2) ist das **19,0fache** des Mittels | `test_core.py` (Brüche), `test_preset_table_values[Von Hand]`, `test_default_preset_noise_numbers` |
| Wie stark ist die Resonanz ohne Leitung? | Bei β = 0 und der Periode 4L − 2 ist \|H_u\| **genau α/(α_krit − α)**: 34 Fälle (L = 1, 2, 3, 4, 6, 8, 12; q = α/α_krit = 0,3 / 0,5 / 0,7 / 0,9 / 0,99, ohne L = 1 und q = 0,5) auf unter 10⁻⁹ relativ, dazu 200 Zufallsfälle (L = 1 bis 40, α zufällig in 0,01 bis 0,995 der Grenze, Zufalls-Seed 1919) auf unter 10⁻⁹ (gemessen 4,9·10⁻¹¹). Die Vormessung hatte 1 000 Fälle (andere Folge, 1,7·10⁻¹⁰). Beispiel L = 3: das **Neunfache** bei 90 % der Grenze, das 99fache bei 99 % | `test_resonance_gain_alpha_over_gap_in_34_cases`, `test_resonance_gain_in_200_random_cases` |
| Wo liegt die Spitze? | Die Spitze über alle Perioden liegt höher als der Wert bei 4L − 2 und **hinter** dieser Periode. L = 2 bis 12, Spitzenperiode über 4L − 2: bei q = 0,5 um 43,8 bis 44,9 %, bei q = 0,7 um 14,0 bis 14,3 %, bei q = 0,9 um 3,2 bis 3,3 %, bei q = 0,99 um 0,28 bis 0,29 %. „Resonanz nahe 4L − 2“ gilt also erst nahe der Grenze | `test_peak_period_lies_behind_4L_minus_2_and_comes_close_only_near_the_limit` |
| Ergebnistabelle L = 3, β = 0, Periode 10 (unten) | q = 0,3: keine Verstärkung; q = 0,5: Spitze 1,45 bei Periode 14,42; q = 0,7: 2,97 bei 11,42; q = 0,9: 10,78 bei 10,33; q = 0,99: 116,46 bei 10,03 | `test_result_table_L3_beta0` |
| Ab welchem α beginnt die Verstärkung (β = 0)? | Ab **α = 1/(2L − 1)** (eigene Herleitung der Näherung bei kleinem ω; Messung auf 20 001 Winkeln, Bisektion): gemessen gleich auf höchstens 2,1·10⁻⁵ relativ (Test: unter 5·10⁻⁵) für L = 1, 2, 3, 4, 6, 8, 12, 20, 30. Anteil an der Stabilitätsgrenze: 0,5000 / 0,3333 / 0,3236 / 0,3210 / 0,3194 / 0,3189 / 0,3186 / 0,3184 / 0,3184, gegen 1/π = 0,3183 für große L; also für L ≥ 3 rund 32 % der Grenze | `test_amplification_onset_beta0_equals_one_over_2L_minus_1_and_the_ratio_to_the_limit` |
| Gilt das für jede Periode? | Gegenprobe mit 200 Zufallsfällen (L = 1 bis 40, 60 001 Winkel, Zufalls-Seed 1919): unter α = 1/(2L − 1) wird in **0 von 200** Fällen eine Periode verstärkt, darüber (bis 0,97 der Spanne zur Grenze) in 200 von 200 immer; bei β = 1 und α < 1 in 0 von 200 Fällen. Das ist Messung, kein Beweis (Vormessung: 1 000 Fälle, ebenfalls 0 Verstöße) | `test_no_amplification_below_and_always_above_the_onset_in_200_random_cases` |
| Und mit Leitungsgewicht? | Die positive Nullstelle α* von a x² + b x − 1 (a = β(1−β)(L−1)², b = β + (1−β)(2L−1)) stimmt mit dem gemessenen Beginn in 25 Fällen (β = 0,3; 0,55; 0,7; 0,9; 1; L = 2, 3, 4, 6, 12) auf höchstens 1,8·10⁻⁵ absolut (4,5·10⁻⁵ relativ) überein. β = 0,55: L = 2: 0,4945 (22,3 % der Grenze 2,222), L = 3: 0,3208 (16,0 % von 2,000), L = 12: 0,0759 (3,8 % von 2,004); β = 1: genau **1 für jedes L** (die Grenze ist 2) | `test_amplification_onset_with_pipeline_weight_formula_against_measurement_in_25_cases` |
| β = 1 und β = 0,55 bei großem α | β = 1: für α ≤ 1 keine Verstärkung (Spitze 1 bei T → ∞), für α > 1 Spitze **α/(2 − α) bei Periode 2**, unabhängig von L (1,5 / 3,0 / 19,0 bei α = 1,2 / 1,5 / 1,9, L = 3, 6, 12). β = 0,55 (statt ½, siehe Korrekturen): stabil für L = 3, 6, 12 bei α = 1,2 / 1,5 / 1,9; Spitzen L = 3: 3,99 / 6,23 / 19,00, L = 6: 6,78 / 8,51 / 17,89, L = 12: 8,83 / 9,56 / 18,70; Mittel von \|H_u\|² bei α = 1,5: 4,66 / 4,92 / 4,93 | `test_beta_1_peak_is_alpha_over_2_minus_alpha_at_period_2_for_every_L`, `test_beta_055_peaks_for_L_3_6_12` |
| Zielkonflikt bei einer Saison (L = 3) | β = 0, Periode 12: α = 0,12: Bestellung 0,298, Bestand 2,48; α = 0,52: 3,84 und 7,38. Periode 52, β = 0: α = 0,3 gibt 1,04 und 3,47; die Bestandsamplitude **fällt** mit α von 7,58 (α = 0,1) auf 1,74 (α = 0,6). β = 1, Periode 7: die Bestandsamplitude **steigt** mit α von 1,27 (α = 0,1) auf 2,26 (α = 1,1), anders als der Zielkonflikt bei weißem Rauschen. Verdikt der App (20 % und 85 % der Grenze ohne Leitung, β = 0 / 0,55 / 1): Periode 12 steigt für alle drei, Periode 52 fällt für alle drei | `test_tradeoff_numbers_L3`, `test_tradeoff_verdict_texts_of_the_readme_for_L3_T12_and_T52` |
| Weißes Rauschen gegen Frequenzgang | Mittel von \|H\|² gegen die Lyapunov-Gleichung in 13 Fällen auf unter 10⁻¹¹ relativ (die Vormessung: 21 Fälle, 1,7·10⁻¹⁵) und gegen α/(2 − α) bzw. (L − 1) + 1/(α(2 − α)) bei β = 1 und gegen α(1 + α)/((1 − α)(2 + α)) bei L = 2, β = 0. Bei β = 1, α = 1,5: Spitze 3,0, **Spitze² 9,0**, Mittel 3,0 | `test_oracle_frequenzgang_lager.py::test_mean_gain2_equals_the_lyapunov_equation_in_13_cases`, `::test_mean_gain2_equals_the_closed_forms`, `test_leitung_voll_noise_peak_squared_is_9_against_mean_3` |
| Saison plus Rauschen | var = A²\|H(T)\|²/2 + σ² · Mittel\|H\|²: L = 3, β = 0,55, α = 0,4, T = 7, A = 1, σ = 0,5, **400 000 Perioden**, Zufalls-Seed 19: Simulation var(u) = 0,2809 gegen Vorhersage 0,2806 (Saison 0,1969 + Rauschen 0,0837), var(i) = 2,5124 gegen 2,5100; über die Zufalls-Seeds 19, 20, 21 im Band von 1 %. Die App rechnet 100 000 Perioden (Band 3 % in jedem Preset) | `test_season_plus_noise_numbers_for_the_readme`, `test_app_noise_run_with_100000_periods_stays_within_3_percent_in_every_preset`, `test_oracle_frequenzgang_lager.py::test_season_plus_noise_variance_decomposition_in_a_band_of_1_percent` |
| Wie lange dauert das Einschwingen? | L = 3, β = 0, Periode 10, Start im Ruhezustand: Einschwingzeit auf 1 % der eingeschwungenen Amplitude gemessen 12 / 20 / 44 / 149 / 1 563 Perioden bei q = 0,3 / 0,5 / 0,7 / 0,9 / 0,99 (Schätzung ln 0,01/ln ρ: 13 / 23 / 45 / 151 / 1 580): **die stärkste Resonanz braucht die längste Zeit**. Die Schätzung gilt nicht allgemein: in 36 von 48 Fällen (L = 2, 3, 6; q = 0,3 bis 0,9; Perioden 4L − 2, 7, 12, 52) auf 10 %, bei q = 0,9 in 12 von 12, bei q = 0,3 nur in 6 von 12; das Verhältnis gemessen/geschätzt liegt zwischen 0,7 und 1,2. Beispiele: Von Hand 84 (Schätzung 87), Resonanz 159 (161), Saison träge 13 (24) | `test_result_table_L3_beta0`, `test_settling_estimate_holds_within_10_percent_in_36_of_48_cases_and_always_near_the_limit`, `test_season_verdicts_of_the_presets_name_measured_and_estimated_settling` |
| Was gilt hinter der Grenze? | L = 3, β = 0, α = 1,05 α_krit, Bedarf cos(2πt/10) aus dem Ruhezustand: ρ = 1,014; größte Bestellung 69,7 / 339 / 5 790 in den Perioden 0–99 / 100–199 / 300–399. Ein Frequenzgang existiert dort nicht, die App zeigt keine Zahlen | `test_unstable_demand_response_grows_without_bound` |
| Meldeverzug (nur Formelprüfung) | P_s(z) = z^(L+s) − (1 − αβ) z^(L+s−1) − αβ z^s + α, H_u = α z^(L−1)/P_s, H_i = −z^s (z^(L−1) + αβ Σ z^k)/P_s stimmen gegen die Zustandsmatrix mit s = 1 bis 3 in 720 Punkten auf unter 10⁻¹¹ (die Vormessung: 1 296 Punkte, 6·10⁻¹⁵); bei β = 0 zählt nur L + s. Keine Rechnung in der App | `test_oracle_frequenzgang_lager.py::test_meldeverzug_formula_matches_the_state_space_with_s_greater_than_zero`, `::test_at_beta_0_the_gain_depends_on_L_and_s_only_through_their_sum` |
| Verwandtes (nur Rechnung, anderes Modell) | arXiv 2208.04459, Gl. 11 (Order-up-to mit gleitendem Mittel der Länge P, Lieferzeit L, im Wortlaut gelesen): φ²(ω) = 1 + 2(L/P + L²/P²)(1 − cos Pω); das Mittel über den Winkel ist 1 + 2(L/P + L²/P²), also 1,625 für (L, P) = (1, 4), (2, 8), (4, 16). `bullwhip-demo` nennt für Stufe 1 die Form 1 + 2m/p + 2m²/p²; ob die Modelle übereinstimmen, ist nicht geprüft. Nicht nachgebaut | `test_mean_of_the_phi_squared_formula_of_another_model_equals_1_625_in_three_cases` |

### Schnellstart-Beispiele (Messung, `test_preset_table_values`)

Spalten: |H_u| und |H_i| bei der Periode T des Beispiels, Spitze von |H_u| (Periode), Mittel von |H_u|², Spektralradius ρ, Anteil der Winkelachse mit Verstärkung, Einschwingzeit auf 1 % (Schätzung). Die Simulation mit Sinusbedarf der Periode T gibt dieselben |H_u|, |H_i| auf 4 Stellen (`test_simulation_gives_the_same_amplitudes_as_the_table_for_every_preset`).
Alle sieben Beispiele sind stabil; die Grenze α_krit ist 1,000 (L = 2, β = 0), 0,618 (L = 3, β = 0) und 2,000 (L = 3, β = 0,55 und β = 1). Mit der Voreinstellung A = 0,1 greift in keinem Beispiel die Klemmwarnung.

| Beispiel | L | β | α | T | \|H_u\| | \|H_i\| | Spitze (Periode) | Mittel \|H_u\|² | ρ | Anteil | Einschwingzeit |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Von Hand | 2 | 0 | 0,9 | 6 | 9,00 | 10,00 | 10,59 (6,19) | 5,90 | 0,949 | 48 % | 87 |
| Resonanz | 3 | 0 | 0,56 | 10 | 9,65 | 17,23 | 11,55 (10,30) | 3,83 | 0,972 | 29 % | 161 |
| Saison nahe der Lieferzeit | 3 | 0 | 0,52 | 12 | 3,84 | 7,38 | 6,45 (10,57) | 2,09 | 0,951 | 28 % | 92 |
| Saison träge | 3 | 0 | 0,12 | 12 | 0,30 | 2,48 | 1,00 (T → ∞) | 0,08 | 0,823 | 0 % | 24 |
| Saison lang | 3 | 0 | 0,3 | 52 | 1,04 | 3,47 | 1,39 (14,87) | 0,35 | 0,813 | 19 % | 22 |
| Leitung halb | 3 | 0,55 | 1,5 | 7 | 4,64 | 6,09 | 6,23 (6,37) | 4,66 | 0,906 | 75 % | 47 |
| Leitung voll | 3 | 1 | 1,5 | 2 | 3,00 | 2,00 | 3,00 (2,00) | 3,00 | 0,500 | 100 % | 7 |

Verdikt der Kernfrage 1 je Beispiel (Test: `test_each_preset_shows_the_expected_verdict_lines`): Von Hand „verstärkt werden alle Perioden über 4,15“; Resonanz „über 6,98“; Saison nahe der Lieferzeit „über 7,18“; Saison träge „keine Periode wird verstärkt“; Saison lang „über 10,32“;
Leitung halb „die Perioden von 2,00 bis 2,53 und über 3,71“ (zwei getrennte Bereiche); Leitung voll „alle Perioden“. Im Beispiel „Leitung halb“ gilt bei gleichem α = 1,5 und L = 3: β = 0 instabil (Grenze 0,618), β = 0,55 stabil mit Spitze 6,23 bei Periode 6,37 (Mittel 4,66), β = 1 stabil mit Spitze 3,00 bei Periode 2,00 (Mittel 3,00) (`test_compare_verdict_of_the_presets_leitung_halb`).

### Ergebnistabelle (L = 3, β = 0, α_krit = 0,618, Periode 10 = 4L − 2; `test_result_table_L3_beta0`)

| q = α/α_krit | α | ρ | Verstärkung bei Periode 10 | Spitze (bei Periode) | Einschwingzeit: Schätzung | gemessen |
|---|---|---|---|---|---|---|
| 0,3 | 0,185 | 0,7097 | 0,4286 | 1,0000 (T → ∞, keine Verstärkung) | 13 | 12 |
| 0,5 | 0,309 | 0,8196 | 1,0000 | 1,4476 (14,42) | 23 | 20 |
| 0,7 | 0,433 | 0,9022 | 2,3333 | 2,9725 (11,42) | 45 | 44 |
| 0,9 | 0,556 | 0,9699 | 9,0000 | 10,7839 (10,33) | 151 | 149 |
| 0,99 | 0,612 | 0,9971 | 99,0000 | 116,4625 (10,03) | 1 580 | 1 563 |

### Befunde und Korrekturen gegenüber dem Plan

- **β = 0,55 statt ½ (Zweierpotenz-Regel):** Die Zahlen der Vormessung für β = ½ (Spitze bei α = 1,5: 11,3 / 38,0 / 140,3 für L = 3 / 6 / 12) sind **nicht** auf β = 0,55 übertragbar (dort 6,23 / 8,51 / 9,56): schon der kleine Schritt von ½ auf 0,55 ändert die Spitzen stark (Spektralradius für L = 3 und α = 1,5: 0,949 bei β = ½ in der Vormessung, 0,906 bei β = 0,55 hier). Die Ursache ist nicht untersucht. Die Gruppe β ∈ {0,25; 0,5; 0,75} des Verstärkungsbeginns wurde durch β ∈ {0,3; 0,55; 0,7; 0,9} ersetzt; die neue Messung gilt.
- **Falscher Vergleich im Plan:** Der Plan schrieb für β = 1, α = 1,5, „Spitze 3,0 = Mittel 3,0“. Das vergleicht eine Verstärkung (3) mit dem Mittel von |H|² (3); Spitze² ist 9,0, das Mittel 3,0, also das **3fache**. Die App nennt immer Spitze² gegen das Mittel; kein Satz „Spitze gleich Mittel“.
- **Einschwingzeit:** Der Plan sah die Schätzung ln 0,01/ln ρ als Band von 5 % vor. Sie gilt so nur nahe der Grenze (q = 0,9: 12 von 12 auf 10 %); insgesamt 36 von 48 Fällen auf 10 %, bei q = 0,3 nur 6 von 12. Die App nennt die Schätzung als „Schätzung aus ρ“ neben der gemessenen Zeit. Die Zahl des Plan-Verdikts (Resonanz: 161) stimmt als Schätzung; gemessen sind es 159.
- **Spitzenperiode:** „Ab q ≥ 0,9 liegt die Spitzenperiode höchstens 3,4 % über 4L − 2“ ist mit L = 2 bis 12 gemessen 3,2 bis 3,3 %; bei q = 0,5 sind es 43,8 bis 44,9 % (Plan: 44 %).
- **Genauigkeit des Verstärkungsbeginns:** Der Plan nannte 2·10⁻³ für β > 0; die Messung stimmt bis 1,8·10⁻⁵ absolut überein (25 Fälle).
- **Rauschen:** Die Vormessung nutzte den numpy-Generator und β = ½; die App benutzt SplitMix64 (Zufalls-Seed 19 im Test) und β = 0,55. Neue Zahlen: 0,2809 gegen 0,2806 (0,11 %) und 2,5124 gegen 2,5100 (0,09 %).
- **Voreinstellung A = 0,1:** Der Plan warnt bei A·|H_u| > Mittelbedarf. Mit A = 0,2 hätte schon das Beispiel „Von Hand“ (9 · 0,2 = 1,8) beim Start eine Klemmwarnung; die Voreinstellung ist deshalb A = 0,1 (alle Beispiele ohne Warnung).
- **Zuschnitt der Experimente:** Das Bode-Bild steht als „Ergebnis“ mit den Kernfragen darunter; es folgen fünf Experimente (Saison, Zielkonflikt, Rauschen, Beginn der Verstärkung, Leitung im Vergleich). Die Verdikte des Zielkonflikts nennen statt fester α-Werte 20 % und 85 % der Grenze ohne Leitungsgewicht; die Werte des Plans (α = 0,12 und 0,52, Periode 52 mit α = 0,3) stehen in der Tabelle oben.
- **Beispiel „Leitung voll“:** β = 1 und T = 2 sind Zweierpotenzen. Sie sind Randwerte des Modells (volle Leitungsrechnung, kürzeste Periode); das Modell ist linear und ohne Klemmung, ein Rundungsartefakt wie in der Schwesterlinie kann nicht auftreten. `test_rounding_noise_in_the_parameters_changes_nothing_visible` prüft die Zahlen unter einer Verschiebung von α und β um 10⁻⁹.

## Quellen

| Quelle | Stufe | Was hier verwendet wird |
|---|---|---|
| arXiv 2208.04459 (Yü, Cai, Gao: Lieferketten als Netz, Order-up-to mit gleitendem Mittel) | Anfang des Aufsatzes und Gl. 11 im Wortlaut gelesen (2026-11-15) | nur die Formel φ²(ω) als Verwandtes eines **anderen Modells**; nicht nachgebaut |
| Dejonckheere, Disney, Lambrecht, Towill (2003); Disney und Towill (APIOBPCS-Familie); Wang und Disney (2016) | nur Verzeichnis- und Suchtreffer, **nicht gelesen** | nichts; hier steht kein „nach“ und keine Neuheitsaussage, alle Befunde sind „eigene Herleitung in diesem Modell“ |

## Ehrliche Grenzen

- **Linear, ohne Klemmung.** Die Bestellung darf negativ werden. Der Frequenzgang gilt nur, solange A·|H_u| den Mittelbedarf nicht übersteigt; die App warnt, wenn die Saison allein die Bestellung unter 0 drücken würde; Rauschen (Experiment 3) berücksichtigt die Warnung nicht. Das Verhalten darüber (Grenzzyklus, Chaos) ist hier **nicht gemessen**; es steht in [`lager-chaos-demo`](https://github.com/sebastian-hanisch/lager-chaos-demo).
- **Eingeschwungener Zustand.** Der Frequenzgang beschreibt nur ihn; die Einschwingzeit wächst mit der Nähe zur Grenze zugleich mit der Resonanz (Beispiel: 1 563 Perioden bei 99 % der Grenze, L = 3, β = 0, Periode 10; näher an der Grenze mehr; die Simulation der App ist auf 30 000 Perioden begrenzt und meldet „noch nicht eingeschwungen“, wenn das nicht reicht).
- **Verstärkungsbeginn.** Die Näherung α* gilt für kleines ω; dass bei α < α* **keine andere Periode** verstärkt, ist gemessen (200 Zufallsfälle bis L = 40 hier, 1 000 in der Vormessung), **nicht bewiesen**. Die Karte (Experiment 4) rechnet auf 1 500 logarithmisch verteilten Winkeln; knapp über dem Beginn kann die Verstärkung auf einem sehr kleinen Winkelbereich liegen.
- **Feste Lieferzeit, kein Meldeverzug, weißes Rauschen und Sinus.** Streuende Lieferzeit (Erlang-Verteilung) und späte Meldung ändern den Frequenzgang; für sie gibt es hier keine Rechnung (die Formel für den Meldeverzug ist nur geprüft). Farbiger Bedarf (verzögert korreliert, Trend) und die Prognose-Rückkopplung der Kette sind nicht gemessen.
- **β < 0 und β > 1** nur in der Formelprüfung (Test mit β = 1,3), nicht im Verstärkungsbeginn.
- **Phase.** Die Phase von Bestellung und Bestand stimmt gegen die Simulation (60 Fälle), ist aber nicht als Kennzahl (Verzug in Perioden) ausgewertet.
- **Spitzenperiode.** Für q < 1 hat die Periode der Spitze keine geschlossene Form; sie ist numerisch bestimmt.
- **Praxis.** Welche Perioden und α in der Praxis vorkommen, ist nicht belegt; die Schnellstart-Beispiele sind Lehrbeispiele. Das Modell ist das eines einzelnen Lagers, nicht der Kette.
- **Plattform.** Zählungen an Schwellen (Einschwingzeit auf 10 %, Anteile) und Simulationsvarianzen stehen in den Tests als Band; Stellen hinter den hier genannten können auf anderen Rechnern abweichen.
- **Literatur.** Die regelungstheoretische Bullwhip-Literatur wurde nicht gelesen. Keine Aussage dieser Demo beruft sich auf sie.

## Verifikation

- **Handwerte:** die Handbeispiele des Plans in Brüchen (`fractions`): P(e^{iπ/3}) = −1/10, |H_u| = 9, |H_i| = 10, 81/101, cos ω_Rand = 1/18, 171/29, 1 900/261, H_u(−1) = −3 bei β = 1, α_krit(L = 3) = (√5 − 1)/2 mit |P| = α_krit − α (`tests/test_core.py`).
- **Orakel** (`tests/test_oracle_frequenzgang_lager.py`, andere Rechenwege): H_u, H_i gegen die Zustandsmatrix, die aus der Modellgleichung durch Einheitsvektoren gewonnen wird; Betrag und Phase gegen die direkte Simulation; Mittel von |H|² gegen `scipy.linalg.solve_discrete_lyapunov` und geschlossene Formeln; Zerlegung Saison plus Rauschen gegen die Simulation; Einschwingzeit gegen ln 0,01/ln ρ; die Formel mit Meldeverzug gegen die Zustandsmatrix mit s > 0; Stabilitätsgrenze gegen das Wachsen und Abklingen der Simulation; Verstärkungsbeginn gegen eine volle Rasterung; Spitze gegen ein Gitter von 400 001 Winkeln.
- **Eigenschaften** (`tests/test_eigenschaften_frequenzgang_lager.py`): H_u(1) = 1 und H_i(1) nach Formel; Spiegelung; Linearität und Superposition der Simulation; Zeitverschiebung ändert nur die Phase; bei β = 1 hängt H_u nicht von L ab; der Betrag bei 4L − 2 wächst mit α (L = 1 bis 12); sehr lange Perioden gleich 1; gleicher Zufalls-Seed gleiche Bahn; Rundungsprobe der Parameter.
- **Randfälle** (`tests/test_edge_cases.py`): L = 1; β = 1 mit α = 1 (benannter Fall, |H_u| = 1 für jede Periode); α an der Grenze (nur „instabil“); über der Grenze (keine Zahlen); T = 2; T sehr lang; ρ nahe 1 (Simulation begrenzt und gekennzeichnet); β > 1; zwei getrennte Verstärkungsbereiche.
- **Claims** (`tests/test_claims.py`): jede Zahl dieser README, an den festen Beispielen und Zufalls-Seeds als Band; Hilfetexte und Verdikte je Beispiel; Gegenproben mit gestreuten Zufallsfällen. **Verdikte** als reine Funktionen (`tests/test_verdicts.py`): Schwellen (Spitze ab 5 als Warnung, Klemmung ab A·|H_u| > 1) und jede Zahl im Text.
- **Oberfläche:** AppTest-Rauchtest (jeder Button, jeder Regler am Minimum und Maximum), jedes Beispiel, Permalink, instabile Einstellung; Layout (Schnellstart oben in der Seitenleiste, fester Kopf, aktives Beispiel hervorgehoben, jede Regleränderung hebt die Markierung auf); Footer; Laufzeit-Importe.
- **Zahl der Tests:** 144 Tests (`pytest --collect-only -q tests`), alle grün.

## Dateistruktur

| Datei | Zweck |
|---|---|
| `.github/workflows/tests.yml` | CI: Push, Pull Request, wöchentlicher Canary |
| `.gitignore` | venv, Caches, Streamlit-Secrets |
| `.streamlit/config.toml` | Streamlit-Einstellungen (keine Nutzungsstatistik) |
| `app.py` | Streamlit-App (Gerüst nach DEMO-PLAYBOOK §2) |
| `fgl_constants.py` | Regler-Grenzen, feste Annahmen, Farben, Schnellstart-Beispiele |
| `fgl_core.py` | Kern: charakteristisches Polynom, Spektralradius, Stabilitätsgrenze, H_u, H_i, Spitze, Mittel, Verstärkungsbereiche, Verstärkungsbeginn |
| `fgl_sim.py` | Direkte Simulation mit Sinus- und Rauschbedarf, Lyapunov-Varianzen |
| `fgl_rng.py` | SplitMix64 (Zufalls-Seed) |
| `fgl_evaluation.py` | Analysen der Experimente, Verdikte als reine Funktionen, Zahlenformat |
| `fgl_layout.py` | Fester Kopf (Stylesheet) |
| `fgl_presets.py` | Permalink (Begrenzen), Schnellstart in der Seitenleiste, aktives Beispiel hervorgehoben, Zufalls-Seed-Knopf |
| `fgl_visualization.py` | Plotly-Abbildungen (Achsen gesperrt) |
| `requirements-dev.txt` | Laufzeit plus pytest |
| `requirements.txt` | Laufzeit-Bibliotheken (ungepinnt): streamlit, numpy, scipy, plotly |
| `tests/` | Handbeispiele, Orakel, Eigenschaften, Randfälle, Claims, Verdikte, Presets, Layout, AppTest-Smoke-Test, Footer, Laufzeit-Importe |

## Bewusst nicht umgesetzt

- **Meldeverzug als Regler:** gehört zu [`meldeverzug-demo`](https://github.com/sebastian-hanisch/meldeverzug-demo); hier nur die geprüfte Formel im Expander.
- **Klemmung mit Sinusbedarf (Beschreibungsfunktion, Chaos):** nicht gemessen; Warnung und Verweis auf [`lager-chaos-demo`](https://github.com/sebastian-hanisch/lager-chaos-demo).
- **Streuende Lieferzeit (Frequenzgang für Erlang-Verteilung):** nicht gemessen; [`streuende-lieferzeit-demo`](https://github.com/sebastian-hanisch/streuende-lieferzeit-demo).
- **Farbiger Bedarf (verzögert korrelierter Zufall, Trend) und Prognose-Rückkopplung (Kette):** nicht gemessen; die Kette rechnet [`bullwhip-demo`](https://github.com/sebastian-hanisch/bullwhip-demo).
- **Reglerentwurf** (Bandbreitenwahl, Phasenreserve) und **Parameterschätzung aus Daten:** nicht Teil der Linie.
- **Phasenverzug als Kennzahl** und ein **PDF-Export:** nicht gebaut.

## Lokal ausführen

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

## Tests ausführen

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

Gebaut mit Streamlit, Plotly, NumPy und SciPy.

---

Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). Mehr zur Reihe: [Rückkopplung und Verzögerung: das Lager](https://sebastianhanisch.net/konzepte-rueckkopplung.html).
