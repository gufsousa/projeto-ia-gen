"""Unified secrets resolver for local (.env) and Streamlit Cloud (st.secrets)."""
from __future__ import annotations

import os


def _normalize(value: str | None) -> str | None:
    if value is None:
        return None
    v = str(value).strip()
    if not v:
        return None
    if (v.startswith('"') and v.endswith('"')) or (v.startswith("'") and v.endswith("'")):
        v = v[1:-1].strip()
    return v or None


def get_secret(*names: str, default: str | None = None) -> str | None:
    """Read first available secret name from st.secrets then environment."""
    try:
        import streamlit as st  # type: ignore

        for name in names:
            if name in st.secrets:
                value = _normalize(st.secrets.get(name))
                if value:
                    return value
    except Exception:
        pass

    for name in names:
        value = _normalize(os.getenv(name))
        if value:
            return value

    return _normalize(default)

