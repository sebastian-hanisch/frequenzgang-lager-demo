"""End-to-end Smoke-Test via Streamlits offiziellem AppTest-Framework: laedt app.py mit den
Standardeinstellungen, klickt jeden Preset-Button und faehrt jeden Slider an seine Grenzen und prueft, dass
kein Python-Fehler auftritt - insbesondere `streamlit.errors.StreamlitDuplicateElementId` (mehrere
st.plotly_chart-Aufrufe ohne eindeutiges key= koennen zufaellig identischen Inhalt rendern und kollidieren)
und `StreamlitAPIException` bei Slidern mit berechneten Grenzen (min == max)."""

import os

from streamlit.testing.v1 import AppTest

APP_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.py")
TIMEOUT = int(os.environ.get("APPTEST_TIMEOUT", "300"))  # unter Last (parallele Läufe) reichen 180 s nicht immer; Umgebungsvariable zum Anpassen
AUTOPLAY_MARKER = "\u25b6"  # Auto-Play-Buttons (Animation mit sleep) im Smoke-Test auslassen


def _fresh_app() -> AppTest:
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    return at


def test_app_loads_without_exception():
    _fresh_app()


def test_every_button_click_does_not_raise():
    labels = [b.label for b in _fresh_app().button]
    assert labels, "keine Buttons gefunden"
    for index, label in enumerate(labels):
        if AUTOPLAY_MARKER in label:
            continue
        at = _fresh_app()
        at.button[index].click().run(timeout=TIMEOUT)
        assert not at.exception, (label, [str(e) for e in at.exception])


def test_every_slider_at_its_min_and_max_does_not_raise():
    labels = [s.label for s in _fresh_app().slider]
    assert labels, "keine Slider gefunden"
    for index, label in enumerate(labels):
        for edge in ("min", "max"):
            at = _fresh_app()
            slider = at.slider[index]
            value = slider.min if edge == "min" else slider.max
            if isinstance(slider.value, int):
                value = int(value)
            slider.set_value(value).run(timeout=TIMEOUT)
            assert not at.exception, (label, edge, [str(e) for e in at.exception])


# --- Zusätze für diese Demo (unterhalb der Vorlage) ------------------------------------------------------------------------------------------

import pytest  # noqa: E402

import fgl_constants as C  # noqa: E402


def _texts(at):
    return [e.value for e in list(at.success) + list(at.info) + list(at.warning) + list(at.error)]


@pytest.mark.parametrize("name", list(C.PRESETS))
def test_each_preset_runs_and_shows_the_bode_verdict(name):
    at = _fresh_app()
    at.sidebar.button(key=f"preset_{name}").click().run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    p = C.PRESETS[name]
    head = f"L = {p['L']}, β = {str(p['beta']).rstrip('0').rstrip('.').replace('.', ',') or '0'}, α = "
    assert any(t.startswith(head) for t in _texts(at)), (name, head)
    assert any("heißt nicht" in t or "Es verstärkt auch keine Saison" in t for t in _texts(at))      # Kernfrage 2
    assert not [e for e in at.error]                                                                    # kein Hinweis „instabil“ als Fehler bei einem Preset


def test_unstable_alpha_shows_the_limit_and_no_bode_chart():
    at = _fresh_app()
    at.sidebar.slider(key="alpha_slider").set_value(2.5).run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    assert any("liegt über der Stabilitätsgrenze 1,000" in w.value for w in at.warning)
    assert any("Hier ist das Lager nicht stabil" in e.value for e in at.error)
    assert not [c for c in at.main.metric if c.label == "α / α_krit"]            # keine Kennzahlen ohne Frequenzgang


def test_permalink_sets_the_regulators_and_clamps_to_the_bounds():
    at = AppTest.from_file(APP_PATH)
    at.query_params["L"] = "3"
    at.query_params["a"] = "0.56"
    at.query_params["T"] = "999"          # über der Grenze: auf T_MAX begrenzt
    at.run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    assert at.sidebar.slider(key="L_slider").value == 3
    assert at.sidebar.slider(key="alpha_slider").value == pytest.approx(0.56)
    assert at.sidebar.slider(key="T_slider").value == C.T_MAX


def test_noise_slider_in_the_main_area_changes_the_table():
    at = _fresh_app()
    assert at.slider(key="sigma_slider").value == C.DEFAULT_SIGMA
    before = [m.value for m in at.markdown if m.value.startswith("| Größe")]
    at.slider(key="sigma_slider").set_value(0.5).run(timeout=TIMEOUT)
    after = [m.value for m in at.markdown if m.value.startswith("| Größe")]
    assert not at.exception and before and after and before != after
