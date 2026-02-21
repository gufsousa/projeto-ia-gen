"""LangGraph pipeline for friendly chat and neuro-symbolic Petri generation."""
from __future__ import annotations

from typing import Literal, TypedDict

from langgraph.graph import END, StateGraph

from sed.grafo import build_petri_net_from_spec
from sed.llm_factory import UnifiedLLMClient
from src.neuro_symbolic import (
    fallback_spec_from_text,
    parse_and_validate_petri_spec,
    petri_json_prompt,
    summarize_spec,
)


class ChatState(TypedDict, total=False):
    user_text: str
    model_name: str
    llm_client: UnifiedLLMClient
    bg_color: str
    fg_color: str
    rankdir: str
    intent: Literal["joke", "petri", "chat"]
    assistant_text: str
    raw_model_output: str
    parse_error: str
    petri_spec: dict
    dot_source: str
    generation_label: str


def _detect_intent(state: ChatState) -> ChatState:
    text = (state.get("user_text") or "").lower()
    normalized = text.strip()
    is_joke = any(k in text for k in ["piada", "joke", "engracad"])
    is_greeting = normalized in {
        "oi",
        "ola",
        "olá",
        "bom dia",
        "boa tarde",
        "boa noite",
        "e ai",
        "e aí",
        "opa",
    }
    has_modeling_signal = any(
        k in text
        for k in [
            "rede de petri",
            "petri",
            "modele",
            "modelar",
            "modelagem",
            "maquina",
            "maquinas",
            "máquina",
            "máquinas",
            "buffer",
            "estoque",
            "fila",
            "lugar",
            "transicao",
            "transição",
            "token",
            "pre",
            "post",
            "grafo",
            "diagrama",
        ]
    )
    if is_joke:
        state["intent"] = "joke"
    elif is_greeting or not has_modeling_signal:
        state["intent"] = "chat"
    else:
        state["intent"] = "petri"
    return state


def _generate_graph_from_json(state: ChatState) -> ChatState:
    if state.get("intent") in {"joke", "chat"}:
        state["generation_label"] = "IA (chat)"
        return state

    user_text = state.get("user_text", "")
    bg_color = state.get("bg_color", "#000000")
    fg_color = state.get("fg_color", "#ffffff")
    rankdir = state.get("rankdir", "LR")
    client = state.get("llm_client")

    prompt = petri_json_prompt(user_text)
    raw = client.complete_text(prompt).strip() if client else ""
    state["raw_model_output"] = raw

    spec, parse_error = parse_and_validate_petri_spec(raw)
    if spec is None:
        spec = fallback_spec_from_text(user_text)
        state["parse_error"] = parse_error or "unknown parse error"
    else:
        state["parse_error"] = ""

    state["petri_spec"] = spec
    dot = build_petri_net_from_spec(spec, bg_color=bg_color, fg_color=fg_color, rankdir=rankdir)
    state["dot_source"] = dot.source
    state["generation_label"] = "IA (langgraph-neuro-symbolic)"
    return state


def _build_assistant_reply(state: ChatState) -> ChatState:
    if state.get("intent") == "joke":
        state["assistant_text"] = (
            "Por que o programador foi ao psicologo?\n\n"
            "Porque tinha muitos problemas nao resolvidos! :)\n\n"
            "Posso ajudar voce com mais alguma funcionalidade para a Rede de Petri "
            "ou criar uma nova aplicacao web?"
        )
        return state
    if state.get("intent") == "chat":
        user_text = state.get("user_text", "")
        client = state.get("llm_client")
        chat_prompt = (
            "Responda em pt-BR de forma amigavel e curta (maximo 2 frases) a pergunta abaixo. "
            "Se for apenas saudacao, cumprimente e diga que pode ajudar. Pergunta: "
            + user_text
        )
        partial = client.complete_text(chat_prompt).strip() if client else ""
        if not partial:
            partial = "Entendi! Posso te ajudar com isso."
        state["assistant_text"] = (
            partial
            + "\n\nPosso ajudar voce com mais alguma funcionalidade para a Rede de Petri?"
        )
        return state

    spec = state.get("petri_spec") or {}
    reply = summarize_spec(spec)
    if state.get("parse_error"):
        reply += " JSON invalido no modelo, aplicado fallback simbolico."
    state["assistant_text"] = reply
    return state


def _build_graph():
    graph = StateGraph(ChatState)
    graph.add_node("detect_intent", _detect_intent)
    graph.add_node("generate_graph", _generate_graph_from_json)
    graph.add_node("assistant_reply", _build_assistant_reply)

    graph.set_entry_point("detect_intent")
    graph.add_edge("detect_intent", "generate_graph")
    graph.add_edge("generate_graph", "assistant_reply")
    graph.add_edge("assistant_reply", END)
    return graph.compile()


_PIPELINE = _build_graph()


def run_langgraph_chat(
    *,
    user_text: str,
    model_name: str,
    llm_client: UnifiedLLMClient,
    bg_color: str,
    fg_color: str,
    rankdir: str,
) -> ChatState:
    """Run the friendly LangGraph pipeline and return final state."""
    initial: ChatState = {
        "user_text": user_text,
        "model_name": model_name,
        "llm_client": llm_client,
        "bg_color": bg_color,
        "fg_color": fg_color,
        "rankdir": rankdir,
    }
    return _PIPELINE.invoke(initial)
