"""Chart helpers: Plotly when available (interactive), matplotlib otherwise.

Every function returns nothing -- it renders directly via st.plotly_chart/st.pyplot so
call sites don't need to branch on which backend is active.
"""
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

try:
    import plotly.graph_objects as go
    HAVE_PLOTLY = True
except ImportError:
    HAVE_PLOTLY = False

from dashboard.theme import CARD_BG, FAMILY_COLORS, NAVY_TEXT

_PLOTLY_LAYOUT = dict(paper_bgcolor=CARD_BG, plot_bgcolor=CARD_BG,
                     font=dict(color=NAVY_TEXT), margin=dict(l=40, r=20, t=40, b=40),
                     xaxis=dict(gridcolor="#e7ecf5"), yaxis=dict(gridcolor="#e7ecf5"))


def bar_chart(categories, series, title="", y_title=""):
    """series: {name -> list[float|None] aligned with categories}."""
    if HAVE_PLOTLY:
        fig = go.Figure()
        for name, values in series.items():
            fig.add_trace(go.Bar(x=list(categories), y=values, name=name,
                                 marker_color=FAMILY_COLORS.get(name)))
        fig.update_layout(title=title, yaxis_title=y_title, barmode="group", **_PLOTLY_LAYOUT)
        st.plotly_chart(fig, width="stretch")
        return
    fig, ax = plt.subplots(figsize=(7, 3.5))
    x = np.arange(len(categories))
    width = 0.8 / max(len(series), 1)
    for i, (name, values) in enumerate(series.items()):
        vals = [v if v is not None else np.nan for v in values]
        ax.bar(x + (i - (len(series) - 1) / 2) * width, vals, width, label=name,
              color=FAMILY_COLORS.get(name))
    ax.set_xticks(x, categories, rotation=35, ha="right", fontsize=8)
    ax.set_title(title)
    ax.set_ylabel(y_title)
    ax.legend(fontsize=7)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def line_chart(x, series, title="", y_title=""):
    if HAVE_PLOTLY:
        fig = go.Figure()
        for name, y in series.items():
            fig.add_trace(go.Scatter(x=list(x), y=y, mode="lines+markers", name=name))
        fig.update_layout(title=title, yaxis_title=y_title, **_PLOTLY_LAYOUT)
        st.plotly_chart(fig, width="stretch")
        return
    fig, ax = plt.subplots(figsize=(7, 3.5))
    for name, y in series.items():
        ax.plot(x, y, marker="o", label=name, markersize=3)
    ax.set_title(title)
    ax.set_ylabel(y_title)
    ax.legend(fontsize=7)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def confusion_matrix_heatmap(cm, labels, title="Confusion matrix"):
    cm = np.asarray(cm)
    if HAVE_PLOTLY:
        fig = go.Figure(data=go.Heatmap(
            z=cm, x=[f"Pred {l}" for l in labels], y=[f"True {l}" for l in labels],
            colorscale="Blues", text=cm, texttemplate="%{text}", showscale=False))
        fig.update_layout(title=title, **_PLOTLY_LAYOUT)
        st.plotly_chart(fig, width="stretch")
        return
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1], [f"Pred {l}" for l in labels])
    ax.set_yticks([0, 1], [f"True {l}" for l in labels])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, int(cm[i][j]), ha="center", va="center",
                   color="white" if cm[i][j] > cm.max() / 2 else "black")
    ax.set_title(title)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def pie_chart(labels, values, title=""):
    if HAVE_PLOTLY:
        fig = go.Figure(data=go.Pie(labels=labels, values=values, hole=0.45))
        fig.update_layout(title=title, **_PLOTLY_LAYOUT)
        st.plotly_chart(fig, width="stretch")
        return
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.pie(values, labels=labels, autopct="%1.0f%%")
    ax.set_title(title)
    st.pyplot(fig)
    plt.close(fig)
