"""Petri net graph builders for IA and Manual modes."""
from __future__ import annotations

import graphviz
import streamlit as st


def _apply_theme(dot: graphviz.Digraph, bg_color: str, fg_color: str) -> None:
    """Apply graph-wide theme colors."""
    dot.attr(bgcolor=bg_color, fontcolor=fg_color)
    dot.attr("edge", color=fg_color, penwidth="1.2")


def _token_label(tokens: int) -> str:
    """Render a compact token marker inside place nodes."""
    if tokens <= 0:
        return ""
    return "\u25CF" * min(tokens, 3)


def _add_place(dot: graphviz.Digraph, node_id: str, place_name: str, tokens: int = 0) -> None:
    """Create place with outside label and optional tokens."""
    dot.node(node_id, _token_label(tokens), xlabel=place_name)


def _tem_palavra(descricao: str, *palavras: str) -> bool:
    if not descricao:
        return False
    d = descricao.lower()
    return any(p in d for p in palavras)


def build_petri_net(descricao: str, bg_color: str = "#000000", fg_color: str = "#ffffff") -> graphviz.Digraph:
    """Build Petri net in IA mode."""
    dot = graphviz.Digraph()
    _apply_theme(dot, bg_color=bg_color, fg_color=fg_color)
    dot.attr(rankdir="LR")

    dot.attr(
        "node",
        shape="circle",
        fixedsize="true",
        width="0.42",
        height="0.42",
        color=fg_color,
        fontcolor=fg_color,
        fontsize="10",
    )
    _add_place(dot, "alpha_01", "P1", tokens=1)
    _add_place(dot, "alpha_02", "P2", tokens=0)
    _add_place(dot, "alpha_03", "P3", tokens=0)

    tem_transicao_q = _tem_palavra(descricao, "motor", "maquina", "m?quina")
    tem_lugar_alpha = _tem_palavra(descricao, "estoque", "buffer")

    if tem_lugar_alpha:
        _add_place(dot, "alpha_11", "P11(k)", tokens=0)

    dot.attr(
        "node",
        shape="box",
        style="filled",
        fillcolor=fg_color,
        color=fg_color,
        fontcolor=fg_color,
        width="0.06",
        height="0.6",
        fixedsize="true",
    )
    dot.node("t1", "", xlabel="t1", labelangle="45", labeldistance="2")
    dot.node("t2", "", xlabel="t2", labelangle="45", labeldistance="2")
    if tem_lugar_alpha:
        dot.node("t0", "", xlabel="t0", labelangle="45", labeldistance="2")
    if tem_transicao_q:
        dot.node("q1", "", xlabel="q1", labelangle="45", labeldistance="2")

    dot.attr(
        "node",
        shape="circle",
        fixedsize="true",
        width="0.42",
        height="0.42",
        style="",
        fillcolor="",
        color=fg_color,
        fontcolor=fg_color,
        fontsize="10",
    )

    if tem_lugar_alpha and tem_transicao_q:
        dot.edge("alpha_01", "t0")
        dot.edge("t0", "alpha_11")
        dot.edge("alpha_11", "t1")
        dot.edge("t1", "alpha_02")
        dot.edge("alpha_02", "q1")
        dot.edge("q1", "alpha_03")
    elif tem_lugar_alpha:
        dot.edge("alpha_01", "t0")
        dot.edge("t0", "alpha_11")
        dot.edge("alpha_11", "t1")
        dot.edge("t1", "alpha_02")
        dot.edge("alpha_02", "t2")
        dot.edge("t2", "alpha_03")
    elif tem_transicao_q:
        dot.edge("alpha_01", "t1")
        dot.edge("t1", "alpha_02")
        dot.edge("alpha_02", "q1")
        dot.edge("q1", "alpha_03")
    else:
        dot.edge("alpha_01", "t1")
        dot.edge("t1", "alpha_02")
        dot.edge("alpha_02", "t2")
        dot.edge("t2", "alpha_03")

    return dot


def build_petri_net_manual(
    n_lugares: int,
    n_transicoes: int,
    bg_color: str = "#000000",
    fg_color: str = "#ffffff",
) -> graphviz.Digraph:
    """Build Petri net in Manual mode as linear chain."""
    conexoes = []
    for i in range(n_transicoes):
        pre = [f"alpha_{i + 1:02d}"] if i + 1 <= n_lugares else []
        post = [f"alpha_{i + 2:02d}"] if i + 2 <= n_lugares else []
        conexoes.append({"pre": pre, "post": post})
    return build_petri_net_manual_conexoes(
        n_lugares,
        n_transicoes,
        conexoes,
        tokens_por_lugar=None,
        bg_color=bg_color,
        fg_color=fg_color,
    )


def build_petri_net_manual_conexoes(
    n_lugares: int,
    n_transicoes: int,
    conexoes: list[dict],
    tokens_por_lugar: dict[str, int] | None = None,
    bg_color: str = "#000000",
    fg_color: str = "#ffffff",
) -> graphviz.Digraph:
    """Build Petri net in Manual mode with custom Pre/Post mapping."""
    dot = graphviz.Digraph()
    _apply_theme(dot, bg_color=bg_color, fg_color=fg_color)
    dot.attr(rankdir="LR")

    dot.attr(
        "node",
        shape="circle",
        fixedsize="true",
        width="0.42",
        height="0.42",
        color=fg_color,
        fontcolor=fg_color,
        fontsize="10",
    )
    for i in range(1, n_lugares + 1):
        node_id = f"alpha_{i:02d}"
        if tokens_por_lugar is not None:
            tokens = int(tokens_por_lugar.get(node_id, 0))
        else:
            tokens = 1 if i == 1 else 0
        _add_place(dot, node_id, f"P{i}", tokens=tokens)

    dot.attr(
        "node",
        shape="box",
        style="filled",
        fillcolor=fg_color,
        color=fg_color,
        fontcolor=fg_color,
        width="0.06",
        height="0.6",
        fixedsize="true",
    )
    for i in range(1, n_transicoes + 1):
        dot.node(f"t{i}", "", xlabel=f"t{i}", labelangle="45", labeldistance="2")

    dot.attr(
        "node",
        shape="circle",
        fixedsize="true",
        width="0.42",
        height="0.42",
        style="",
        fillcolor="",
        color=fg_color,
        fontcolor=fg_color,
        fontsize="10",
    )

    for i in range(n_transicoes):
        t_id = f"t{i + 1}"
        pre = conexoes[i].get("pre", [])
        post = conexoes[i].get("post", [])
        for p in pre:
            dot.edge(p, t_id)
        for p in post:
            dot.edge(t_id, p)

    return dot


def render_grafo_com_fallback(dot: graphviz.Digraph) -> None:
    """Render graph or show DOT code if Graphviz engine is unavailable."""
    try:
        st.graphviz_chart(dot.source)
    except Exception:
        st.error("Graphviz nao esta disponivel. Use o codigo DOT abaixo.")
        st.code(dot.source, language="dot")
