"""Konstanten: Regler-Grenzen, feste Annahmen, Farben und Presets.

Nie α, β oder T als Zweierpotenz voreinstellen (Rechner-Artefakt der Schwesterlinie, siehe README „Ehrliche Grenzen“). Zwei Werte der Presets sind
Randwerte des Modells und keine Reglerwahl: β = 1 (volle Leitungsrechnung) und T = 2 (kürzeste Periode, Nyquist) im Preset „Leitung voll“; das
Modell ist linear und ohne Klemmung, ein Rundungsartefakt wie dort kann nicht auftreten (`tests/test_claims.py` prüft die Zahlen unter
Rundungsrauschen der Parameter mit)."""

L_MIN, L_MAX, DEFAULT_L = 1, 12, 2
BETA_MIN, BETA_MAX, DEFAULT_BETA, BETA_STEP = 0.0, 1.0, 0.0, 0.05
ALPHA_MIN, ALPHA_MAX, DEFAULT_ALPHA, ALPHA_STEP = 0.02, 2.5, 0.9, 0.01
T_MIN, T_MAX, DEFAULT_T = 2, 60, 6
A_MIN, A_MAX, DEFAULT_A, A_STEP = 0.05, 1.0, 0.1, 0.05     # Amplitude des Bedarfs in Vielfachen des Mittelbedarfs
SEED_MAX, DEFAULT_SEED = 999999, 35
SIGMA_MIN, SIGMA_MAX, DEFAULT_SIGMA, SIGMA_STEP = 0.0, 1.0, 0.25, 0.05   # Rauschen (Standardabweichung) in Vielfachen des Mittelbedarfs

# Messkonventionen (feste Annahmen der Demo, alle in der README genannt)
N_ANGLES_MEAN = 1 << 17          # Mittelpunktsregel für das Mittel von |H|² über den Winkelkreis
N_ANGLES_SCAN = 20001            # Gitter von ω in (0, π] für „Anteil der Winkelachse“ und den Verstärkungsbeginn
N_ANGLES_PEAK = 100001           # grobes Gitter für die Spitze, danach Verfeinerung
AMP_EPS = 1e-9                   # „verstärkt“ heißt max |H_u| > 1 + AMP_EPS
STABLE_EPS = 1e-9                # „stabil“ heißt Spektralradius < 1 - STABLE_EPS
SETTLE_FRACTION = 0.01           # Einschwingzeit: Abweichung unter 1 % der eingeschwungenen Amplitude
SIM_CAP = 30000                  # längste Simulation (Perioden) in der App; darüber steht „nicht eingeschwungen“
N_NOISE = 100_000                # Perioden der Rausch-Simulation in der App (die README-Zahl mit 400 000 steht in tests/test_claims.py)
T_BODE_MIN, T_BODE_MAX, N_BODE = 2.0, 200.0, 400
BETA_COMPARE = (0.0, 0.55, 1.0)  # Leitungsgewichte der Vergleiche (Experiment 4 und 5); 0,55 statt ½ (Zweierpotenz)
L_MAP_MAX = 30
N_MAP_Q = 60

COLOR_U = "#c4782b"
COLOR_I = "#2F6B65"
COLOR_E = "rgba(120,120,120,0.9)"
COLOR_LINE = "#4c78a8"
COLOR_AMP = "rgba(192,57,43,0.12)"
COLOR_RED = "#c0392b"
COLOR_BETA = {0.0: "#4c78a8", 0.55: "#c4782b", 1.0: "#2F6B65"}

_BASE = {"L": DEFAULT_L, "beta": DEFAULT_BETA, "alpha": DEFAULT_ALPHA, "T": DEFAULT_T, "A": DEFAULT_A, "seed": DEFAULT_SEED}
PRESETS = {
    "Von Hand": dict(_BASE),
    "Resonanz": {**_BASE, "L": 3, "alpha": 0.56, "T": 10},
    "Saison nahe der Lieferzeit": {**_BASE, "L": 3, "alpha": 0.52, "T": 12},
    "Saison träge": {**_BASE, "L": 3, "alpha": 0.12, "T": 12},
    "Saison lang": {**_BASE, "L": 3, "alpha": 0.3, "T": 52},
    "Leitung halb": {**_BASE, "L": 3, "beta": 0.55, "alpha": 1.5, "T": 7},
    "Leitung voll": {**_BASE, "L": 3, "beta": 1.0, "alpha": 1.5, "T": 2},
}
PRESET_HELP = {
    "Von Hand": "Das Handbeispiel: Lieferzeit 2, ohne Leitungsgewicht, α = 0,9 (α_krit = 1), Periode 6. Bei Periode 6 genau neunfache Verstärkung, in Brüchen nachrechenbar.",
    "Resonanz": "Lieferzeit 3, α = 0,56 (91 % der Grenze 0,618), Periode 10 = 4L − 2: die Periode der kritischen Schwingung wird fast zehnfach verstärkt.",
    "Saison nahe der Lieferzeit": "Lieferzeit 3, α = 0,52, Periode 12 (nahe 4L − 2 = 10): die Bestellung wird fast vierfach verstärkt, der Bestand schwankt mit der 7,4fachen Amplitude.",
    "Saison träge": "Lieferzeit 3, α = 0,12, Periode 12: die Regel dämpft die Bestellung auf knapp 0,3, der Bestand schwankt dafür mit der 2,5fachen Amplitude.",
    "Saison lang": "Lieferzeit 3, α = 0,3, Periode 52 (ein Jahr in Wochen): die Bestellung folgt dem Bedarf (Verstärkung 1,04), der Bestand schwankt mit der 3,5fachen Amplitude.",
    "Leitung halb": "Lieferzeit 3, Leitungsgewicht 0,55, α = 1,5, Periode 7: stabil (Grenze 2), aber die Bestellung wird 4,6fach verstärkt, die Spitze liegt bei Periode 6,4.",
    "Leitung voll": "Lieferzeit 3, Leitungsgewicht 1, α = 1,5, Periode 2: Spitze α/(2 − α) = 3 liegt bei der kürzesten Periode.",
}
