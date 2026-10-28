"""Light theme: component styling only. The page/sidebar background and base text colors
come from .streamlit/config.toml (Streamlit's own theme engine), not from CSS overrides
here -- a hand-forced `.stApp { background-color: ... }` rule is exactly the kind of thing
that can fight with Streamlit's own styling across versions, so this module intentionally
does not set one.
"""
import streamlit as st

NAVY_TEXT = "#0f1b2d"
MUTED_TEXT = "#5b6b84"
ACCENT = "#2563eb"
ACCENT_SOFT = "#60a5fa"
CARD_BG = "#ffffff"
CARD_BORDER = "#dbe3f0"
WARN_BG = "#fff7e6"
WARN_BORDER = "#f0c36d"
WARN_TEXT = "#7a5b00"

FAMILY_COLORS = {"orig": "#2563eb", "aug": "#d97706", "edgeaug": "#16a34a",
                 "baseline_pad": "#7c3aed", "baseline_resize": "#db2777"}


def inject_css():
    """Light, restrained component styling: cards, badges, the disclaimer banner. Headings,
    body text, and page/sidebar background are left to Streamlit's theme (config.toml)."""
    st.markdown(f"""
    <style>
      .cxr-card {{
          background-color: {CARD_BG}; border: 1px solid {CARD_BORDER}; border-radius: 12px;
          padding: 16px 20px; margin-bottom: 12px;
          box-shadow: 0 1px 3px rgba(15, 27, 45, 0.06);
      }}
      .cxr-metric-label {{ color: {MUTED_TEXT}; font-size: 0.8rem; text-transform: uppercase;
                            letter-spacing: 0.04em; margin-bottom: 2px; }}
      .cxr-metric-value {{ color: {NAVY_TEXT}; font-size: 1.8rem; font-weight: 700; }}
      .cxr-badge {{
          display: inline-block; padding: 2px 10px; border-radius: 999px; font-size: 0.78rem;
          font-weight: 600; background-color: {ACCENT}; color: white; margin-right: 6px;
      }}
      .cxr-disclaimer {{
          background-color: {WARN_BG}; border: 1px solid {WARN_BORDER}; border-radius: 8px;
          padding: 10px 14px; color: {WARN_TEXT}; font-size: 0.86rem;
      }}
    </style>
    """, unsafe_allow_html=True)


def disclaimer(text):
    st.markdown(f'<div class="cxr-disclaimer">{text}</div>', unsafe_allow_html=True)


def metric_card(label, value, help_text=None):
    st.markdown(
        f'<div class="cxr-card"><div class="cxr-metric-label">{label}</div>'
        f'<div class="cxr-metric-value">{value}</div></div>', unsafe_allow_html=True)
    if help_text:
        st.caption(help_text)


def badge(text):
    st.markdown(f'<span class="cxr-badge">{text}</span>', unsafe_allow_html=True)


def fmt_pct(v, digits=2):
    return "N/A" if v is None else f"{v * 100:.{digits}f}%"


def fmt_num(v, digits=4):
    return "N/A" if v is None else f"{v:.{digits}f}"
