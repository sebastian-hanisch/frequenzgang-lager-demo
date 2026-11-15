"""Plotly-Abbildungen. Alle Achsen sind gesperrt (fixedrange), damit Touch-Geräte beim Scrollen nicht am Diagramm hängen bleiben."""

import math

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import fgl_constants as C


def lock_axes(fig):
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def _base(fig, height=360):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=30, b=10), legend=dict(orientation="h", y=-0.2), plot_bgcolor="rgba(0,0,0,0)")
    return lock_axes(fig)


def _log_range(values, lo_floor=1e-3, hi_cap=1e6, pad=1.4):
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v) & (v > 0)]
    lo = max(float(v.min()) / pad, lo_floor)
    hi = min(float(v.max()) * pad, hi_cap)
    lo = min(lo, 0.7)                      # die Marke 1 bleibt im Bild
    hi = max(hi, 1.5)
    return [math.log10(lo), math.log10(hi)]


def build_bode(a: dict):
    """Oben |H_u|, unten |H_i| über der Periode (logarithmisch); Linie bei 1, Marke bei 4L − 2, Verstärkungsbereiche schraffiert (rot hinterlegt), gewählte Periode und Spitze markiert."""
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08, subplot_titles=("Bestellung: |H_u| (Verstärkung der Bedarfsamplitude)", "Bestand: |H_i| (Bestandsamplitude je Bedarfsamplitude)"))
    p = a["periods"]
    for row, key, color, name in ((1, "gain_u", C.COLOR_U, "|H_u|"), (2, "gain_i", C.COLOR_I, "|H_i|")):
        fig.add_trace(go.Scatter(x=p, y=a[key], mode="lines", line=dict(color=color, width=3), name=name, hovertemplate="Periode %{x:.1f}: %{y:.2f}<extra></extra>"), row=row, col=1)
    fig.add_hline(y=1.0, line=dict(color="rgba(120,120,120,0.9)", dash="dash"), row=1, col=1)
    for lo, hi in a["intervals"]:
        fig.add_vrect(x0=max(lo, C.T_BODE_MIN), x1=min(hi, C.T_BODE_MAX), fillcolor=C.COLOR_AMP, line_width=0, row=1, col=1)
    for row in (1, 2):
        fig.add_vline(x=a["resonance_T"], line=dict(color="rgba(120,120,120,0.8)", dash="dot"), row=row, col=1)
    fig.add_annotation(x=math.log10(a["resonance_T"]), y=1.0, xref="x", yref="y domain", text="4L − 2", showarrow=False, yanchor="top", xanchor="left", font=dict(size=11))
    fig.add_trace(go.Scatter(x=[a["T"]], y=[a["hu_T"]], mode="markers", marker=dict(size=11, color="#222", symbol="diamond"), name=f"Periode {a['T']}",
                             hovertemplate="Periode %{x}: %{y:.2f}<extra></extra>"), row=1, col=1)
    fig.add_trace(go.Scatter(x=[a["T"]], y=[a["hi_T"]], mode="markers", marker=dict(size=11, color="#222", symbol="diamond"), showlegend=False,
                             hovertemplate="Periode %{x}: %{y:.2f}<extra></extra>"), row=2, col=1)
    if math.isfinite(a["peak_T"]) and C.T_BODE_MIN <= a["peak_T"] <= C.T_BODE_MAX and a["peak"] > 1.0 + C.AMP_EPS:
        fig.add_trace(go.Scatter(x=[a["peak_T"]], y=[a["peak"]], mode="markers", marker=dict(size=11, color=C.COLOR_RED, symbol="star"), name="Spitze",
                                 hovertemplate="Spitze bei Periode %{x:.2f}: %{y:.2f}<extra></extra>"), row=1, col=1)
    fig.update_xaxes(type="log", range=[math.log10(C.T_BODE_MIN), math.log10(C.T_BODE_MAX)], row=1, col=1)
    fig.update_xaxes(type="log", range=[math.log10(C.T_BODE_MIN), math.log10(C.T_BODE_MAX)], title_text="Periode des Bedarfs (Perioden, logarithmisch)", row=2, col=1)
    fig.update_yaxes(type="log", range=_log_range(a["gain_u"]), row=1, col=1)
    fig.update_yaxes(type="log", range=_log_range(a["gain_i"]), row=2, col=1)
    return _base(fig, 520)


def build_season(a: dict):
    """Bedarf e, Bestellung u und Bestand i als Zeitreihe (direkte Simulation, mit der Amplitude A skaliert); gestrichelt die Hülle ±A·|H|."""
    t, A = a["t"], a["A"]
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1, subplot_titles=("Bedarfsabweichung e und Bestellung u (Vielfache des Mittelbedarfs)", "Bestand i (Abweichung vom Soll)"))
    fig.add_trace(go.Scatter(x=t, y=A * a["e"], mode="lines", line=dict(color=C.COLOR_E, width=1.5), name="Bedarf e"), row=1, col=1)
    fig.add_trace(go.Scatter(x=t, y=A * a["u"], mode="lines", line=dict(color=C.COLOR_U, width=2.5), name="Bestellung u"), row=1, col=1)
    fig.add_trace(go.Scatter(x=t, y=A * a["i"], mode="lines", line=dict(color=C.COLOR_I, width=2.5), name="Bestand i"), row=2, col=1)
    for row, h in ((1, a["hu"]), (2, a["hi"])):
        for sign in (1, -1):
            fig.add_hline(y=sign * A * h, line=dict(color="rgba(120,120,120,0.9)", dash="dash", width=1), row=row, col=1)
    fig.add_annotation(x=0.99, y=1.0, xref="x domain", yref="y domain", text="Hülle ±A·|H_u|", showarrow=False, xanchor="right", yanchor="bottom", font=dict(size=11))
    fig.add_annotation(x=0.99, y=1.0, xref="x2 domain", yref="y2 domain", text="Hülle ±A·|H_i|", showarrow=False, xanchor="right", yanchor="bottom", font=dict(size=11))
    for row in (1, 2):
        lim = A * (a["hu"] if row == 1 else a["hi"])
        fig.update_yaxes(range=[-1.25 * max(lim, A * 1.0), 1.25 * max(lim, A * 1.0)], row=row, col=1)
    fig.update_xaxes(title_text="Periode t", row=2, col=1)
    return _base(fig, 520)


def build_tradeoff(a: dict):
    """α ↦ |H_u| (oben) und |H_i| (unten) bei der gewählten Periode für β = 0, 0,55 und 1; nur im stabilen Bereich."""
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08, subplot_titles=("Bestellamplitude |H_u|", "Bestandsamplitude |H_i|"))
    allu, alli = [], []
    for beta in C.BETA_COMPARE:
        c = a["curves"][beta]
        col = C.COLOR_BETA[beta]
        allu.append(c["hu"]); alli.append(c["hi"])
        fig.add_trace(go.Scatter(x=a["alphas"], y=c["hu"], mode="lines", line=dict(color=col, width=3), name=f"β = {beta:g}".replace(".", ","), legendgroup=str(beta),
                                 hovertemplate="α = %{x:.2f}: %{y:.2f}<extra></extra>"), row=1, col=1)
        fig.add_trace(go.Scatter(x=a["alphas"], y=c["hi"], mode="lines", line=dict(color=col, width=3), showlegend=False, legendgroup=str(beta),
                                 hovertemplate="α = %{x:.2f}: %{y:.2f}<extra></extra>"), row=2, col=1)
    fig.add_hline(y=1.0, line=dict(color="rgba(120,120,120,0.9)", dash="dash"), row=1, col=1)
    fig.update_yaxes(type="log", range=_log_range(np.concatenate(allu)), row=1, col=1)
    fig.update_yaxes(type="log", range=_log_range(np.concatenate(alli)), row=2, col=1)
    fig.update_xaxes(range=[0, C.ALPHA_MAX], title_text="Anpassung α", row=2, col=1)
    fig.update_xaxes(range=[0, C.ALPHA_MAX], row=1, col=1)
    return _base(fig, 520)


def build_noise(a: dict):
    """|H_u|² über dem Winkel (Einheit π) mit der Mittellinie: das Mittel über den Kreis ist das Varianzverhältnis bei weißem Rauschen."""
    w = a["w"] / math.pi
    periods = 2.0 / w
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=w, y=a["g2_u"], mode="lines", line=dict(color=C.COLOR_U, width=3), name="|H_u|²", customdata=periods,
                             hovertemplate="ω/π = %{x:.3f} (Periode %{customdata:.1f}): %{y:.2f}<extra></extra>"))
    fig.add_hline(y=a["mean_u"], line=dict(color=C.COLOR_LINE, dash="dash", width=2), annotation_text=f"Mittel {a['mean_u']:.2f}".replace(".", ","), annotation_position="top left")
    fig.add_hline(y=1.0, line=dict(color="rgba(120,120,120,0.7)", dash="dot"))
    fig.update_xaxes(range=[0, 1], title_text="Winkel ω/π (Periode T = 2π/ω; rechts kurze, links lange Perioden)")
    fig.update_yaxes(type="log", range=_log_range(a["g2_u"], pad=2.0), title_text="|H_u|²")
    return _base(fig, 340)


def build_map(m: dict, L: int, q: float):
    """Karte: größte Verstärkung (max |H_u|) über alle Perioden, L horizontal, α/α_krit vertikal; weiß heißt keine Verstärkung. Linie: Näherung α*(L)/α_krit(L)."""
    z = np.log10(np.maximum(m["max_gain"], 1.0))
    fig = go.Figure()
    fig.add_trace(go.Heatmap(x=m["Ls"], y=m["qs"], z=z, zmin=0.0, zmax=1.5, colorscale=[[0.0, "#ffffff"], [0.02, "#fde0dd"], [1.0, "#a50f15"]],
                             colorbar=dict(title=dict(text="max |H_u|", side="top"), tickvals=[0, 0.5, 1, 1.5], ticktext=["1", "3,2", "10", "32+"], len=0.8),
                             hovertemplate="L = %{x}, α/α_krit = %{y:.2f}: max |H_u| = %{customdata:.2f}<extra></extra>", customdata=m["max_gain"]))
    fig.add_trace(go.Scatter(x=m["Ls"], y=np.minimum(m["formula_ratio"], 1.0), mode="lines+markers", line=dict(color="#222", width=2), marker=dict(size=4), name="Näherung α*/α_krit"))
    if q is not None and 0.0 < q <= 1.0:
        fig.add_trace(go.Scatter(x=[L], y=[q], mode="markers", marker=dict(size=12, color=C.COLOR_LINE, symbol="diamond", line=dict(color="#fff", width=1)), name="gewählt"))
    fig.update_xaxes(title_text="Lieferzeit L", range=[0.5, m["Ls"][-1] + 0.5])
    fig.update_yaxes(title_text="α / α_krit", range=[0, 1])
    return _base(fig, 420)


def build_compare(a: dict):
    """|H_u| über der Periode bei gleichem α und L für β = 0, 0,55 und 1 (instabile Kurven fehlen)."""
    fig = go.Figure()
    allg = []
    for beta in C.BETA_COMPARE:
        c = a["curves"][beta]
        if not c["stable"]:
            continue
        allg.append(c["gain_u"])
        fig.add_trace(go.Scatter(x=a["periods"], y=c["gain_u"], mode="lines", line=dict(color=C.COLOR_BETA[beta], width=3), name=f"β = {beta:g}".replace(".", ","),
                                 hovertemplate="Periode %{x:.1f}: %{y:.2f}<extra></extra>"))
    fig.add_hline(y=1.0, line=dict(color="rgba(120,120,120,0.9)", dash="dash"))
    fig.update_xaxes(type="log", range=[math.log10(C.T_BODE_MIN), math.log10(C.T_BODE_MAX)], title_text="Periode des Bedarfs (Perioden, logarithmisch)")
    fig.update_yaxes(type="log", range=_log_range(np.concatenate(allg)) if allg else [-1, 1], title_text="|H_u|")
    return _base(fig, 380)
