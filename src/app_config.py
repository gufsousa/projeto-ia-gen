"""Configuracao e ciclo de vida do UI connector."""
from __future__ import annotations

from copy import deepcopy

import streamlit as st

from src.ui.petri_ui_connector import PetriUIConnector


def _theme_color_defaults() -> tuple[str, str]:
    """Return default DOT colors based on current Streamlit theme."""
    base = (st.get_option("theme.base") or "dark").strip().lower()
    if base == "light":
        return "#FFFFFF", "#000000"
    return "#0e1117", "#FFFFFF"


def build_sidebar_config(app_runtime_version: str) -> dict:
    """Monta configuracao da aplicacao a partir da sidebar."""
    default_bg, default_fg = _theme_color_defaults()
    orientation_options = {"Horizontal": "LR", "Vertical": "TB"}
    current_rankdir = st.session_state.get("cfg_rankdir", "LR")
    current_label = "Vertical" if current_rankdir == "TB" else "Horizontal"

    with st.sidebar:
        with st.expander("Configuracao DOT", expanded=False):
            c_bg, c_fg = st.columns(2)
            with c_bg:
                bg_color = st.color_picker(
                    "Fundo DOT",
                    value=st.session_state.get("cfg_bg_color", default_bg),
                    key="cfg_bg_color",
                )
            with c_fg:
                fg_color = st.color_picker(
                    "Desenho DOT",
                    value=st.session_state.get("cfg_fg_color", default_fg),
                    key="cfg_fg_color",
                )
            orientation = st.selectbox(
                "Orientacao",
                options=list(orientation_options.keys()),
                index=list(orientation_options.keys()).index(current_label),
                key="cfg_orientation_label",
            )
            rankdir = orientation_options.get(orientation, "LR")
            st.session_state["cfg_rankdir"] = rankdir

    return {
        "graph_bg_color": bg_color,
        "graph_fg_color": fg_color,
        "graph_rankdir": rankdir,
        "enable_flow_preview": False,
        "enable_float_panel": True,
        "app_runtime_version": app_runtime_version,
    }


def should_rebuild_connector(llm_config: dict) -> bool:
    """Indica se o connector precisa ser reconstruido."""
    last = st.session_state.get("last_llm_config")
    if last != llm_config:
        st.session_state["last_llm_config"] = deepcopy(llm_config)
        return True
    return "ui_connector" not in st.session_state


def build_ui_connector(llm_config: dict) -> PetriUIConnector:
    """Cria o connector da UI."""
    return PetriUIConnector(llm_config)


def reset_stream_state() -> None:
    """Reseta estados temporarios de streaming."""
    st.session_state.pop("stream_buffer", None)
