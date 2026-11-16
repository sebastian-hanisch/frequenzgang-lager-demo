"""Frequenzgang des Lagers - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Welche Schwankungen des Bedarfs verstärkt eine lineare Bestellregel mit Lieferzeit L, Anpassung α und Leitungsgewicht β? Frequenzgang von Bedarf zu Bestellung und Bestand,
Saison in der direkten Simulation, Zielkonflikt, Rauschen als Mittelwert, Beginn der Verstärkung. Eigene Herleitung in diesem Modell, ohne Klemmung, im stabilen Bereich.
Aufbau und Regeln: DEMO-PLAYBOOK.md (§2).

Lauffähig mit: streamlit run app.py
"""

import streamlit as st

import fgl_constants as C
import fgl_core as K
import fgl_evaluation as E
from fgl_evaluation import de, para
from fgl_layout import dunkles_design, kopf_css
from fgl_presets import (
    bounds,
    init_session_state_defaults,
    load_permalink_settings,
    randomize_seed,
    sidebar_presets,
    sync_query_params,
)
from fgl_visualization import build_bode, build_compare, build_map, build_noise, build_season, build_tradeoff

st.set_page_config(page_title="Frequenzgang des Lagers – Sebastian Hanisch", layout="wide")


@st.cache_data(show_spinner=False)
def _bode(L, beta, alpha, T):
    return E.bode_analysis(E.Settings(L, beta, alpha, T))


@st.cache_data(show_spinner=False)
def _season(L, beta, alpha, T, A):
    return E.season_analysis(E.Settings(L, beta, alpha, T, A))


@st.cache_data(show_spinner=False)
def _tradeoff(L, T):
    return E.tradeoff_analysis(L, T)


@st.cache_data(show_spinner="Simuliere 100 000 Perioden (unter einer Sekunde) …")
def _noise(L, beta, alpha, T, A, seed, sigma):
    return E.noise_analysis(E.Settings(L, beta, alpha, T, A, seed), sigma)


@st.cache_data(show_spinner="Rechne die Karte über L und α (rund eine Sekunde) …")
def _map(beta):
    return E.onset_map(beta)


@st.cache_data(show_spinner=False)
def _onset(L, beta):
    return E.onset_analysis(L, beta)


@st.cache_data(show_spinner=False)
def _compare(L, alpha):
    return E.compare_analysis(L, alpha)


def _show(verdict):
    kind, text = verdict
    getattr(st, kind)(text)


# Fester Kopf: nur der Titel bleibt beim Scrollen oben (Stylesheet in fgl_layout.py).
st.html(kopf_css(dunkles_design()))
with st.container(key="kopf"):
    st.title("📈 Frequenzgang des Lagers")
st.markdown(
    """
Ein **Lager** bestellt nach einer linearen Regel nach: Bestellung = Mittelbedarf minus **α** mal (Bestandsabweichung plus **β** mal die unterwegs befindliche Ware); die Ware kommt nach **L Perioden**.
Der **Frequenzgang** sagt, wie stark das Lager eine Schwankung des Bedarfs mit der Periode T in der Bestellung und im Bestand verstärkt. **Stabil sein heißt nicht, nicht zu verstärken:** in diesem Modell
(linear, ohne Klemmung, im stabilen Bereich, eigene Herleitung) beginnt die Verstärkung langer Perioden ohne Leitungsgewicht schon bei α = 1/(2L − 1), und das weiße Rauschen ist nur der Mittelwert über alle Perioden.

Weiter unten: **❓ Kernfragen**, fünf Experimente (**🔬**), **🚧 Wo die Annahmen enden** und der Expander **📐 Mathematische Formulierung**.
"""
)

with st.expander("So funktioniert der Frequenzgang", expanded=True):
    st.markdown(
        """
1. **Modell:** Das Lager hat in Periode t die Abweichung **i** vom Soll-Bestand; der Bedarf weicht um **e** vom Mittelbedarf ab, die Bestellung um **u**. Die Regel ist linear: u = −α · (i + β · Summe der unterwegs befindlichen Bestellungen).
2. **Frequenzgang:** Ein Bedarf, der wie ein Sinus mit der Periode T schwankt, erzeugt nach dem Einschwingen eine Bestellung und einen Bestand, die ebenfalls mit der Periode T schwanken, mit anderer Amplitude und verschoben. Das Amplitudenverhältnis **|H_u|** (Bestellung) und **|H_i|** (Bestand) hängt bei festem Lager nur von der Periode ab.
3. **Rauschen:** Weißes Rauschen enthält alle Perioden gleich stark; die Bestellvarianz ist das Mittel von |H_u|² über alle Perioden. Eine schmale Spitze bei einer Periode geht in diesem Mittel weitgehend unter.
        """
    )

st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    sidebar_presets()  # Schnellstart oben, das aktive Preset ist hervorgehoben
    st.header("⚙️ Einstellungen")
    st.markdown("**Lager**")
    L = st.slider("Lieferzeit L (Perioden)", *bounds("L_slider"), key="L_slider", help="Wie viele Perioden die Ware unterwegs ist.")
    beta = st.slider("Leitungsgewicht β", *bounds("beta_slider"), step=C.BETA_STEP, key="beta_slider",
                     help="0: die Regel sieht nur den Bestand; 1: die Regel zieht die unterwegs befindliche Ware voll vom Bedarf ab.")
    alpha = st.slider("Anpassung α", *bounds("alpha_slider"), step=C.ALPHA_STEP, key="alpha_slider", help="Anteil der Abweichung, den die Bestellung je Periode ausgleicht.")
    limit = K.stable_limit(int(L), float(beta))
    st.caption(f"Stabilitätsgrenze α_krit = {de(limit, 3)}; α ist {de(100.0 * alpha / limit, 0)} % davon." + (" Darüber ist das Lager instabil." if alpha >= limit else ""))
    st.markdown("**Bedarf**")
    T = st.slider("Periode T des Bedarfs", *bounds("T_slider"), key="T_slider", help="Der Bedarf schwankt wie ein Sinus mit dieser Periode (in Perioden).")
    A = st.slider("Amplitude A (Vielfache des Mittelbedarfs)", *bounds("A_slider"), step=C.A_STEP, key="A_slider",
                  help="Mittelbedarf = 1; ab A·|H_u| > 1 fiele die Bestellung unter 0 (Klemmung, nicht modelliert).")
    st.markdown("**Zufall**")
    seed = st.number_input("Zufalls-Seed", *bounds("seed_input"), key="seed_input", step=1, help="Gilt für das Rauschen in Experiment 3.")
    st.button("🎲 Neuer Zufalls-Seed", width="stretch", on_click=randomize_seed)

L, beta, alpha, T, A, seed = int(L), float(beta), float(alpha), int(T), float(A), int(seed)
sync_query_params({"L_slider": L, "beta_slider": beta, "alpha_slider": alpha, "T_slider": T, "A_slider": A, "seed_input": seed})

# --- Ergebnis ------------------------------------------------------------------------------------------------------------------------------

a = _bode(L, beta, alpha, T)
st.markdown("## 🎯 Ergebnis: Bode-Bild")
if a["stable"]:
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("α / α_krit", de(alpha / a["limit"], 2), help="Unter 1 ist die Regel stabil (Spektralradius unter 1).")
    m2.metric(f"|H_u| bei Periode {T}", de(a["hu_T"]), help="Bestellamplitude je Bedarfsamplitude.")
    m3.metric(f"|H_i| bei Periode {T}", de(a["hi_T"]), help="Bestandsamplitude je Bedarfsamplitude.")
    m4.metric("Spitze von |H_u|", de(a["peak"]), help=f"Bei Periode {E.period_text(a['peak_T'])}; Werte über 1 sind Verstärkung.")
    m5.metric("Mittel von |H_u|²", de(a["mean_u"]), help="Varianzverhältnis var(u)/var(e) bei weißem Rauschen im Bedarf.")
    st.plotly_chart(build_bode(a), width="stretch", key="bode_chart")
    st.caption(
        "Oben: um wie viel die Regel eine Schwankung des Bedarfs in der Bestellung verstärkt (über der gestrichelten 1: Verstärkung, rot hinterlegt); unten: wie stark der Bestand schwankt. "
        "Raute: die gewählte Periode, Stern: die Spitze, gepunktet: die Periode 4L − 2 der kritischen Schwingung ohne Leitungsgewicht. Beide Achsen logarithmisch."
    )
else:
    st.warning(
        f"α = {para(alpha)} liegt über der Stabilitätsgrenze {de(limit, 3)} (Spektralradius ρ = {de(a['rho'], 3)} ≥ 1): die Bestellung wächst ohne Obergrenze, "
        "ein Frequenzgang existiert dort nicht. Kleineres α, größeres β oder kleineres L wählen."
    )

# --- Kernfragen ----------------------------------------------------------------------------------------------------------------------------

st.markdown("## ❓ Kernfragen")
st.markdown("**1. Welche Perioden des Bedarfs verstärkt die Regel, und wie stark?**")
_show(E.verdict_bode(a))
st.markdown("**2. Heißt „stabil“ auch „verstärkt nicht“?**")
_show(E.verdict_stable_but_amplifies(a))

st.markdown("---")

# --- Experimente ---------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 1: Saisonbedarf in der Simulation")
st.caption("Bedarf = Mittelbedarf + A · cos(2πt/T), direkte Simulation der Modellgleichungen aus dem Ruhezustand (nicht über den Frequenzgang gerechnet). Die gemessene Amplitude am Ende der Bahn wird mit dem Frequenzgang verglichen.")
se = _season(L, beta, alpha, T, A)
for kind, text in E.season_warnings(se):
    getattr(st, kind)(text)
if se["stable"]:
    st.plotly_chart(build_season(se), width="stretch", key="season_chart")
    st.caption(f"Gezeigt sind die ersten {se['n_plot']} Perioden (von {se['n_sim']} simulierten); gestrichelt die Hülle des eingeschwungenen Zustands.")
    _show(E.verdict_season(se))              # über der Grenze steht der Hinweis schon oben (season_warnings)

st.markdown("---")

st.subheader("🔬 Experiment 2: Zielkonflikt bei einer Saison")
st.caption(f"Bestellamplitude und Bestandsamplitude über α bei Periode {T} und Lieferzeit {L}, für drei Leitungsgewichte nebeneinander (nur im stabilen Bereich). "
           "Bei weißem Rauschen und β = 1 glättet α < 1 die Bestellung (Varianzverhältnis α/(2 − α)) und kostet Bestandsvarianz (kleinste bei α = 1); bei einer Saison kommt es auf die Periode an.")
tr = _tradeoff(L, T)
st.plotly_chart(build_tradeoff(tr), width="stretch", key="tradeoff_chart")
_show(E.verdict_tradeoff(tr))

st.markdown("---")

st.subheader("🔬 Experiment 3: Rauschen als Mittelwert")
st.caption("Weißes Rauschen im Bedarf enthält alle Perioden gleich stark; die Bestellvarianz ist das Mittel von |H_u|² über den Winkelkreis. Die Simulation (SplitMix64, Zufalls-Seed der Seitenleiste) liefert Saison plus Rauschen zugleich.")
sigma = st.slider("Rauschen σ (Standardabweichung, Vielfache des Mittelbedarfs)", C.SIGMA_MIN, C.SIGMA_MAX, C.DEFAULT_SIGMA, step=C.SIGMA_STEP, key="sigma_slider",
                  help="Gilt nur für dieses Experiment; die Amplitude A und die Periode T der Saison stehen in der Seitenleiste.")
na = _noise(L, beta, alpha, T, A, seed, float(sigma))
if na["stable"]:
    st.plotly_chart(build_noise(na), width="stretch", key="noise_chart")
    rows = E.variance_rows(na)
    table = "| Größe | Saison | Rauschen | Summe (Vorhersage) | Simulation | Abweichung |\n|---|---|---|---|---|---|\n"
    for r in rows:
        table += f"| {r['Größe']} | {de(r['Saison'], 4)} | {de(r['Rauschen'], 4)} | {de(r['Summe'], 4)} | {de(r['Simulation'], 4)} | {de(r['Abweichung (%)'], 1, True)} % |\n"
    st.markdown(table)
    st.caption(f"Varianzen nach dem Einschwingen ({E.int_text(na['sim']['n'])} Perioden, ohne die ersten {na['sim']['burn']}). Vorhersage: A²·|H(T)|²/2 + σ²·Mittel|H|²; Lyapunov-Gleichung für weißes Rauschen: "
               f"var(u)/var(e) = {de(na['lyap_u'], 4)}, var(i)/var(e) = {de(na['lyap_i'], 4)}.")
_show(E.verdict_noise(na))

st.markdown("---")

st.subheader("🔬 Experiment 4: Wo beginnt das Verstärken?")
st.caption("Karte der größten Verstärkung über alle Perioden, Lieferzeit L horizontal, α als Anteil der Stabilitätsgrenze vertikal, für das Leitungsgewicht der Seitenleiste. Weiß: keine Periode wird verstärkt.")
mp = _map(beta)
st.plotly_chart(build_map(mp, L, alpha / limit if limit and limit != float("inf") else None), width="stretch", key="map_chart")
on = _onset(L, beta)
_show(E.verdict_onset(on))

st.markdown("---")

st.subheader("🔬 Experiment 5: Leitung im Vergleich")
st.caption(f"Drei Bode-Kurven bei gleichem α = {para(alpha)} und L = {L}: ohne, mit teilweiser (β = 0,55) und mit voller Leitungsrechnung. β = 0,55 ist ein Lehrwert; die Schwelle β = ½ der Stabilitätsgrenze gehört zu `leitung-mitrechnen-demo`.")
cp = _compare(L, alpha)
st.plotly_chart(build_compare(cp), width="stretch", key="compare_chart")
_show(E.verdict_compare(cp))

st.markdown("---")

# --- Grenzen -------------------------------------------------------------------------------------------------------------------------------

st.subheader("🚧 Wo die Annahmen enden")
st.markdown(
    """
| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Linear, ohne Klemmung** (die Bestellung darf negativ werden) | Der Frequenzgang gilt nur, solange die Bestellamplitude (A mal Verstärkung) höchstens den Mittelbedarf erreicht; die App warnt, wenn die Saison allein die Bestellung unter 0 drücken würde, das Rauschen (Experiment 3) berücksichtigt die Warnung nicht. Das Verhalten darüber (Grenzzyklus, Chaos) ist hier nicht gemessen. | `lager-chaos-demo` |
| **Stabil** (alle Wurzeln im Einheitskreis) | Über der Grenze wächst die Bestellung ohne Obergrenze; es gibt keinen Frequenzgang. Direkt unter der Grenze dauert das Einschwingen extrem lange (L = 3, β = 0, Periode 10, 99 % der Grenze: gemessen 1 563 Perioden bis auf 1 %). | `lager-lieferzeit-demo` |
| **Eingeschwungener Zustand** | Der Frequenzgang beschreibt nur die Schwingung nach dem Einschwingen; die Einschwingzeit wächst mit der Nähe zur Grenze zugleich mit der Resonanz (Experiment 1; näher an der Grenze mehr als die 1 563 Perioden oben). | – |
| **Feste Lieferzeit, kein Meldeverzug** | Streuende Lieferzeit und späte Meldung ändern den Frequenzgang; hier nicht gemessen (die Formel für den Meldeverzug steht im Expander und ist gegen die Zustandsmatrix geprüft). | `streuende-lieferzeit-demo`, `meldeverzug-demo` |
| **Sinus und weißes Rauschen als Bedarf** | Farbiger Bedarf (verzögert korreliert, mit Trend) und Prognose-Rückkopplung sind nicht gemessen. | `bullwhip-demo` |
| **Lehrbeispiele, keine Praxiswerte** | Welche Perioden und α in der Praxis vorkommen, ist nicht belegt; die Schnellstart-Beispiele sind gewählt, um die Effekte zu zeigen. | – |
"""
)

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Lesehilfe.** Gegeben sind die Lieferzeit $L$, die Anpassung $\alpha$ und das Leitungsgewicht $\beta$; gesucht ist, wie stark das Lager eine Bedarfsschwankung der Periode $T$ in der Bestellung und im Bestand verstärkt.

| Symbol | Bedeutung |
|---|---|
| $i_t$ | Bestand in Periode $t$ minus Soll-Bestand |
| $u_t$ | Bestellung in Periode $t$ minus Mittelbedarf |
| $e_t$ | Bedarf in Periode $t$ minus Mittelbedarf |
| $L$, $\alpha$, $\beta$ | Lieferzeit (Perioden), Anpassung, Leitungsgewicht |
| $z$, $\omega$ | Variable der z-Transformation, $z = e^{i\omega}$ mit $\omega = 2\pi/T$ |
| $P(z)$ | charakteristisches Polynom; seine Wurzeln bestimmen die Stabilität |
| $H_u$, $H_i$ | Übertragungsfunktion vom Bedarf zur Bestellung bzw. zum Bestand |

**Modell** (Abweichungen vom Soll und vom Mittelbedarf, in Perioden gezählt):

$$i_{t+1} = i_t + u_{t+1-L} - e_t$$

$$u_t = -\alpha \Big( i_t + \beta \sum_{j=1}^{L-1} u_{t-j} \Big)$$

**Übertragungsfunktionen** (eigene Herleitung in diesem Modell durch z-Transformation, keine Quelle):

$$P(z) = z^L - (1-\alpha\beta)\, z^{L-1} + \alpha(1-\beta)$$

$$H_u(z) = \frac{\alpha\, z^{L-1}}{P(z)}, \qquad H_i(z) = -\,\frac{z^{L-1} + \alpha\beta \sum_{k=0}^{L-2} z^k}{P(z)}$$

Der Frequenzgang ist $H(e^{i\omega})$ mit $\omega = 2\pi/T$; er gilt im eingeschwungenen Zustand und nur, solange alle Wurzeln von $P$ im Einheitskreis liegen. Gleichstrom: $H_u(1) = 1$ und $H_i(1) = -\big(1+\alpha\beta(L-1)\big)/\alpha$.

**Resonanz ohne Leitungsgewicht** ($\beta = 0$, $\alpha_{\mathrm{krit}} = 2\sin\frac{\pi}{4L-2}$): bei der Periode $4L-2$ ist $|P(e^{i\pi/(2L-1)})| = \alpha_{\mathrm{krit}} - \alpha$, also

$$|H_u| = \frac{\alpha}{\alpha_{\mathrm{krit}} - \alpha}.$$

**Beginn der Verstärkung** (Näherung für kleines $\omega$, eigene Herleitung nur dieser Näherung): $|H_u|^2 \approx \alpha^2/(\alpha^2 + K\omega^2)$; lange Perioden werden verstärkt, sobald $K<0$, das ist die positive Nullstelle $\alpha^*$ von

$$a x^2 + b x - 1 = 0, \qquad a = \beta(1-\beta)(L-1)^2, \quad b = \beta + (1-\beta)(2L-1).$$

Für $\beta = 0$ ist $\alpha^* = 1/(2L-1)$, für $\beta = 1$ ist $\alpha^* = 1$. Dass bei $\alpha < \alpha^*$ keine andere Periode verstärkt wird, ist gemessen (Gitter, bis $L = 40$), nicht bewiesen.

**Weißes Rauschen:** Für unkorrelierten Bedarf mit Varianz $\sigma^2$ ist die Varianz der Bestellung

$$\operatorname{var}(u) = \sigma^2 \cdot \frac{1}{2\pi}\int_{-\pi}^{\pi} |H_u(e^{i\omega})|^2\, d\omega,$$

das Mittel von $|H_u|^2$ über den Winkelkreis (ebenso für $H_i$); die App rechnet es mit der Mittelpunktsregel auf $2^{17}$ Punkten und prüft es mit der Lyapunov-Gleichung der Zustandsmatrix. Eine Saison mit Amplitude $A$ und Rauschen addieren ihre Varianzen (bei $T = 2$ ohne den Faktor $\tfrac{1}{2}$):

$$\operatorname{var}(u) = \tfrac{1}{2} A^2 |H_u(e^{2\pi i/T})|^2 + \sigma^2 \,\overline{|H_u|^2}.$$

**Meldeverzug $s$** (nur Formelprüfung, keine Rechnung in der App; gehört zu `meldeverzug-demo`):

$$P_s(z) = z^{L+s} - (1-\alpha\beta)\, z^{L+s-1} - \alpha\beta\, z^{s} + \alpha, \qquad H_u = \frac{\alpha\, z^{L-1}}{P_s(z)}.$$

**Literatur.** Für das Verhältnis zu anderen Modellen: arXiv 2208.04459 (Yü, Cai, Gao) gibt für einen *anderen* Fall, die Order-up-to-Regel mit gleitendem Mittel der Länge $P$ und Lieferzeit $L$, im Wortlaut gelesen die Verstärkung
$\varphi^2(\omega) = 1 + 2(L/P + L^2/P^2)(1-\cos P\omega)$; das ist hier nicht nachgebaut. Der Frequenzgang als Werkzeug der regelungstheoretischen Bullwhip-Literatur wurde nicht gelesen; die Herleitungen oben sind eigene Herleitungen in diesem Modell.
        """
    )

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). "
    "Mehr zur Reihe: [Rückkopplung und Verzögerung: das Lager](https://sebastianhanisch.net/konzepte-rueckkopplung.html)."
)
