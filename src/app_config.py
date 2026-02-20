"""Configuracao e ciclo de vida do UI connector."""
from __future__ import annotations

from copy import deepcopy

import streamlit as st

from src.ui.petri_ui_connector import PetriUIConnector


def build_sidebar_config(app_runtime_version: str) -> dict:
    """Monta configuracao da aplicacao a partir da sidebar."""
    with st.sidebar:
        with st.expander("Color Config", expanded=False):
            c_bg, c_fg = st.columns(2)
            with c_bg:
                bg_color = st.color_picker(
                    "Fundo DOT",
                    value=st.session_state.get("cfg_bg_color", "#000000"),
                    key="cfg_bg_color",
                )
            with c_fg:
                fg_color = st.color_picker(
                    "Desenho DOT",
                    value=st.session_state.get("cfg_fg_color", "#ffffff"),
                    key="cfg_fg_color",
                )

    return {
        "graph_bg_color": bg_color,
        "graph_fg_color": fg_color,
        "enable_flow_preview": True,
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
