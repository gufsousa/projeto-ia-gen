"""Estilos visuais leves (proximo ao padrao nativo do Streamlit)."""

from __future__ import annotations

import streamlit as st


BASE_UI_CSS = """
<style>
/* Mantem visual nativo; apenas pequenos ajustes de espacamento */
.block-container {
    padding-top: 1rem !important;
}

[data-testid="stSidebar"] {
    border-right: 1px solid rgba(0, 0, 0, 0.08);
}
</style>
"""


CANVAS_UI_CSS = """
<style>
/* Sem tema dark custom; visual clean no estilo Streamlit padrao */
.canvas-topbar {
    border: 1px solid rgba(255, 255, 255, 0.16);
    border-radius: 10px;
    padding: 10px 12px;
    margin-bottom: 10px;
    background: transparent;
}

.canvas-eyebrow {
    text-transform: uppercase;
    letter-spacing: 0.06em;
    font-size: 0.72rem;
    color: rgba(255, 255, 255, 0.75);
    font-weight: 700;
}

.canvas-title {
    margin: 0.2rem 0 0;
    font-size: 1.4rem;
    font-weight: 700;
    color: #ffffff;
}

.canvas-subtitle {
    margin: 0.2rem 0 0;
    color: rgba(255, 255, 255, 0.82);
    font-size: 0.9rem;
}

.section-title {
    margin: 0;
    font-size: 1rem;
    font-weight: 700;
}

.section-subtitle {
    margin: 0.2rem 0 0;
    color: #6b7280;
    font-size: 0.84rem;
}

.mode-chip {
    display: inline-block;
    margin-left: 6px;
    padding: 1px 8px;
    border-radius: 999px;
    border: 1px solid rgba(0, 0, 0, 0.12);
    font-size: 0.74rem;
}

.canvas-empty {
    border: 1px dashed rgba(0, 0, 0, 0.2);
    border-radius: 10px;
    padding: 18px 12px;
    color: #6b7280;
    text-align: center;
}
</style>
"""


def inject_professional_css() -> None:
    st.markdown(BASE_UI_CSS, unsafe_allow_html=True)


def inject_canvas_css() -> None:
    st.markdown(CANVAS_UI_CSS, unsafe_allow_html=True)
