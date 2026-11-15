"""Presets: Vollständigkeit, gültige Werte, Standard-Preset = Standardeinstellungen, keine Zweierpotenz in α, β, T außer den benannten Randwerten, Stabilität."""

import fgl_constants as C
import fgl_core as K
import fgl_evaluation as ev
import fgl_presets as P


def _is_power_of_two(x):
    """Zweierpotenz im Sinn der Rechner-Artefakt-Regel: 2^k mit ganzzahligem k (auch 1/2, 1/4, 1)."""
    import math

    if x <= 0:
        return False
    k = math.log2(x)
    return abs(k - round(k)) < 1e-12


def test_every_preset_has_help_and_all_keys():
    assert set(C.PRESETS) == set(C.PRESET_HELP)
    for name, preset in C.PRESETS.items():
        assert set(preset) == set(P.PRESET_KEYS), name
        assert C.PRESET_HELP[name]


def test_preset_values_are_inside_the_slider_bounds():
    for preset in C.PRESETS.values():
        assert C.L_MIN <= preset["L"] <= C.L_MAX and C.BETA_MIN <= preset["beta"] <= C.BETA_MAX and C.ALPHA_MIN <= preset["alpha"] <= C.ALPHA_MAX
        assert C.T_MIN <= preset["T"] <= C.T_MAX and C.A_MIN <= preset["A"] <= C.A_MAX and 0 <= preset["seed"] <= C.SEED_MAX


def test_default_preset_equals_the_default_settings():
    assert ev.Settings(**C.PRESETS["Von Hand"]) == ev.Settings()


def test_setting_specs_cover_every_preset_key():
    assert set(P.PRESET_KEYS.values()) == set(P.SETTING_SPECS)


def test_no_power_of_two_in_alpha_beta_t_except_the_two_named_edge_values():
    """β = 1 und T = 2 des Presets „Leitung voll“ sind Randwerte des Modells; sonst ist keiner der drei Werte eine Zweierpotenz."""
    allowed = {("Leitung voll", "beta"), ("Leitung voll", "T")}
    for name, preset in C.PRESETS.items():
        for key in ("alpha", "beta", "T"):
            if key == "beta" and preset[key] == 0.0:
                continue                                      # β = 0 (keine Leitungsrechnung) ist keine Zweierpotenz im Sinn der Regel (0 ist kein 2^k)
            if _is_power_of_two(float(preset[key])):
                assert (name, key) in allowed, (name, key, preset[key])
    assert not _is_power_of_two(C.DEFAULT_ALPHA) and not _is_power_of_two(C.DEFAULT_T)


def test_every_preset_is_stable_and_has_the_expected_limit():
    limits = {"Von Hand": 1.0, "Resonanz": 0.6180, "Saison nahe der Lieferzeit": 0.6180, "Saison träge": 0.6180, "Saison lang": 0.6180, "Leitung halb": 2.0, "Leitung voll": 2.0}
    for name, preset in C.PRESETS.items():
        assert K.is_stable(preset["L"], preset["alpha"], preset["beta"]), name
        assert abs(K.stable_limit(preset["L"], preset["beta"]) - limits[name]) < 5e-4, name


def test_preset_help_texts_name_the_values_of_their_preset():
    """Die Zahlen der Hilfetexte stimmen mit den Presets überein (Lieferzeit, α, Periode)."""
    for name, preset in C.PRESETS.items():
        text = C.PRESET_HELP[name]
        assert f"Lieferzeit {preset['L']}" in text
        assert ("α = " + ev.para(preset["alpha"])) in text, name
        assert f"Periode {preset['T']}" in text, name
