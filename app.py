"""Entry point da aplicacao Streamlit."""
from __future__ import annotations

import streamlit as st
from PIL import Image

# 1. SETUP DA PÁGINA (Deve ser o primeiro comando)
try:
    # Use o logo1.jpg (quadrado) para o ícone da aba do navegador
    img_icone = Image.open("assets/icone.png") 
    st.set_page_config(
        page_title="SED Workspace", 
        page_icon=img_icone, 
        layout="wide"
    )
except Exception:
    st.set_page_config(page_title="SED Workspace", page_icon="⚙️", layout="wide")

# 2. LOGO RETANGULAR (7x4) NA SIDEBAR
# O parâmetro 'icon_image' aparece quando a barra lateral está recolhida
# Logo Menu
try:
    st.logo("assets/logo_horizontal_1.png")
except:
    pass

from src.app_config import (
    build_sidebar_config,
    build_ui_connector,
    reset_stream_state,
    should_rebuild_connector,
)

APP_RUNTIME_VERSION = "2026-02-20-sed-petri-ui-connector-v1"


def main() -> None:
    st.set_page_config(
        page_title="Modelagem - SED",
        page_icon="Chat",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    llm_config = build_sidebar_config(APP_RUNTIME_VERSION)
    if should_rebuild_connector(llm_config):
        st.session_state.ui_connector = build_ui_connector(llm_config)
        reset_stream_state()
    st.session_state.ui_connector.ensure_theme()

    st.session_state.ui_connector.display_chat()


if __name__ == "__main__":
    main()
