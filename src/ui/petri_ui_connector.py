"""Connector de UI para modelagem de Rede de Petri."""
from __future__ import annotations

import html
import re
import time
import urllib.parse
import urllib.request
import base64  # Adicionado para conversão de imagem

import graphviz
from pydantic import BaseModel
import streamlit as st

from sed.grafo import build_petri_net_manual, build_petri_net_manual_conexoes
from sed.llm_factory import UnifiedLLMClient
from sed.ui_css import inject_canvas_css, inject_professional_css
from src.langgraph_chat import run_langgraph_chat

try:
    from streamlit_flow import (
        StreamlitFlowEdge,
        StreamlitFlowNode,
        StreamlitFlowState,
        streamlit_flow,
    )
except Exception:
    StreamlitFlowEdge = None
    StreamlitFlowNode = None
    StreamlitFlowState = None
    streamlit_flow = None

try:
    from streamlit_float import float_box, float_init
except Exception:
    float_box = None
    float_init = None


class FlowNodeModel(BaseModel):
    id: str
    label: str
    kind: str


class FlowEdgeModel(BaseModel):
    id: str
    source: str
    target: str


class PetriUIConnector:
    """Encapsula renderizacao da UI, estado e interacoes."""

    def __init__(self, config: dict) -> None:
        self.config = config
        self._init_session_state()

    def ensure_theme(self) -> None:
        """Aplica tema global da aplicacao."""
        inject_professional_css()
        inject_canvas_css()
        if float_init:
            float_init(theme=False)

    def _img_to_base64(self, image_path: str) -> str | None:
        """Converte imagem local para string base64 para uso em HTML."""
        try:
            with open(image_path, "rb") as img_file:
                return f"data:image/png;base64,{base64.b64encode(img_file.read()).decode()}"
        except Exception:
            return None

    def _init_session_state(self) -> None:
        if "workspace_id" not in st.session_state:
            st.session_state["workspace_id"] = "default"
        if "petri_store" not in st.session_state:
            st.session_state["petri_store"] = {}

        workspace_id = st.session_state["workspace_id"]
        if workspace_id not in st.session_state["petri_store"]:
            st.session_state["petri_store"][workspace_id] = {
                "modo_ia": True,
                "dot_source": "",
                "ultima_geracao": "Nenhuma",
                "graph_bg_color": "#0e1117",
                "graph_fg_color": "#ffffff",
                "graph_rankdir": "LR",
                "diagram_scale": 3.0,  # Padrão inicial de 300%
                "diagram_box_height": 640,
                "llm_model": "",
                "llm_last_output": "",
                "last_raw_model_output": "",
                "last_petri_spec": {},
                "chat_history": [],
                "manual": {
                    "n_lugares": 3,
                    "n_transicoes": 2,
                    "conexoes": {},
                    "tokens": {"alpha_01": 1},
                },
            }
        else:
            ws = st.session_state["petri_store"][workspace_id]
            ws.setdefault("modo_ia", True)
            ws.setdefault("dot_source", "")
            ws.setdefault("ultima_geracao", "Nenhuma")
            ws.setdefault("graph_bg_color", "#0e1117")
            ws.setdefault("graph_fg_color", "#ffffff")
            ws.setdefault("graph_rankdir", "LR")
            ws.setdefault("diagram_scale", 3.0)  # Padrão de 300%
            ws.setdefault("diagram_box_height", 640)
            ws.setdefault("llm_model", "")
            ws.setdefault("llm_last_output", "")
            ws.setdefault("last_raw_model_output", "")
            ws.setdefault("last_petri_spec", {})
            ws.setdefault("chat_history", [])
            ws.setdefault("manual", {})
            ws["manual"].setdefault("n_lugares", 3)
            ws["manual"].setdefault("n_transicoes", 2)
            ws["manual"].setdefault("conexoes", {})
            ws["manual"].setdefault("tokens", {"alpha_01": 1})

        if "unified_llm_client" not in st.session_state:
            st.session_state["unified_llm_client"] = UnifiedLLMClient(provider="mock")

    def _workspace_state(self) -> dict:
        workspace_id = st.session_state["workspace_id"]
        return st.session_state["petri_store"][workspace_id]

    def _stream_text(self, text: str, delay_s: float = 0.015):
        for token in (text or "").split(" "):
            yield token + " "
            time.sleep(delay_s)

    def _provider_defaults(self, provider: str) -> str:
        if provider == "openai":
            return "gpt-4o-mini"
        if provider == "gemini":
            return "gemini-2.5-flash"
        if provider == "groq":
            return "meta-llama/llama-4-maverick-17b-128e-instruct"
        return "meta-llama/llama-4-maverick-17b-128e-instruct"

    def _provider_from_model(self, model: str) -> str:
        model_l = (model or "").strip().lower()
        if "gemini" in model_l:
            return "gemini"
        if (
            "llama" in model_l
            or "mixtral" in model_l
            or "qwen" in model_l
            or "kimi" in model_l
            or "groq" in model_l
            or "openai/gpt-oss" in model_l
        ):
            return "groq"
        if model_l.startswith("gpt") or model_l.startswith("o") or "openai" in model_l:
            return "openai"
        return "mock"

    def _sync_llm_client(self, model: str) -> tuple[UnifiedLLMClient, str]:
        provider = self._provider_from_model(model)
        client: UnifiedLLMClient = st.session_state["unified_llm_client"]
        if client.provider != provider or client.model != model:
            client.set_provider(provider=provider, model=model)
        return client, provider

    def _manual_defaults(self, ws: dict) -> tuple[int, int]:
        manual = ws["manual"]
        return int(manual.get("n_lugares", 3)), int(manual.get("n_transicoes", 2))

    def _ensure_connection_defaults(self, ws: dict, n_transicoes: int) -> None:
        conexoes = ws["manual"]["conexoes"]
        for i in range(1, n_transicoes + 1):
            key = f"t{i}"
            if key not in conexoes:
                conexoes[key] = {"pre": [], "post": []}

    def _ensure_token_defaults(self, ws: dict, opcoes_lugares: list[str]) -> None:
        tokens = ws["manual"].setdefault("tokens", {})
        for place_id in opcoes_lugares:
            tokens.setdefault(place_id, 1 if place_id == "alpha_01" else 0)

    def _transition_dimensions(self, rankdir: str) -> tuple[str, str]:
        r = (rankdir or "LR").upper()
        if r in {"TB", "BT"}:
            return "0.6", "0.06"
        return "0.06", "0.6"

    def _retheme_dot_source(self, dot_source: str, bg_color: str, fg_color: str, rankdir: str) -> str:
        if not dot_source:
            return dot_source

        updated = dot_source
        updated = re.sub(r'\bbgcolor="[^"]*"', f'bgcolor="{bg_color}"', updated)
        updated = re.sub(r'\bfontcolor="[^"]*"', f'fontcolor="{fg_color}"', updated)
        updated = re.sub(r'\bfillcolor="[^"]*"', f'fillcolor="{fg_color}"', updated)
        updated = re.sub(r'(?<![A-Za-z])color="[^"]*"', f'color="{fg_color}"', updated)
        if re.search(r"\brankdir=(LR|TB|RL|BT)\b", updated):
            updated = re.sub(r"\brankdir=(LR|TB|RL|BT)\b", f"rankdir={rankdir}", updated)
        else:
            updated = updated.replace("{", "{\n\trankdir=" + rankdir, 1)

        trans_w, trans_h = self._transition_dimensions(rankdir)
        lines: list[str] = []
        for line in updated.splitlines():
            if "shape=box" in line:
                if re.search(r"\bwidth=[0-9.]+\b", line):
                    line = re.sub(r"\bwidth=[0-9.]+\b", f"width={trans_w}", line)
                else:
                    line = line.replace("]", f" width={trans_w}]", 1)
                if re.search(r"\bheight=[0-9.]+\b", line):
                    line = re.sub(r"\bheight=[0-9.]+\b", f"height={trans_h}", line)
                else:
                    line = line.replace("]", f" height={trans_h}]", 1)
            lines.append(line)
        updated = "\n".join(lines)
        return updated

    def _scaled_dot_source(self, dot_source: str, scale: float) -> str:
        if not dot_source or abs(scale - 1.0) < 1e-6:
            return dot_source
        s = max(0.5, min(scale, 5.0))

        def _scale_attr(match: re.Match[str]) -> str:
            key = match.group(1)
            num = float(match.group(2))
            quote = match.group(3) or ""
            scaled = max(0.01, num * s)
            if key in {"fontsize"}:
                val = str(int(round(scaled)))
            else:
                val = f"{scaled:.2f}".rstrip("0").rstrip(".")
            return f"{key}={quote}{val}{quote}"

        pattern = r"\b(width|height|fontsize|penwidth|labeldistance)=(\"?)([0-9]+(?:\.[0-9]+)?)(\"?)"

        def _replace(m: re.Match[str]) -> str:
            key = m.group(1)
            q1 = m.group(2)
            num = float(m.group(3))
            q2 = m.group(4)
            quote = q1 if q1 else q2
            scaled = max(0.01, num * s)
            if key == "fontsize":
                val = str(int(round(scaled)))
            else:
                val = f"{scaled:.2f}".rstrip("0").rstrip(".")
            return f"{key}={quote}{val}{quote}"

        return re.sub(pattern, _replace, dot_source)

    def new_thread(self) -> None:
        ws = self._workspace_state()
        ws["graph_bg_color"] = self.config.get("graph_bg_color", ws.get("graph_bg_color", "#0e1117"))
        ws["graph_fg_color"] = self.config.get("graph_fg_color", ws.get("graph_fg_color", "#ffffff"))
        ws["chat_history"] = []
        ws["llm_last_output"] = ""

    def clear_graph(self) -> None:
        ws = self._workspace_state()
        ws["dot_source"] = ""
        ws["ultima_geracao"] = "Nenhuma"

    def _parse_dot_to_flow(self, dot_source: str) -> tuple[list[FlowNodeModel], list[FlowEdgeModel]]:
        if not dot_source:
            return [], []

        node_labels: dict[str, str] = {}
        edges: list[FlowEdgeModel] = []
        seen_edges: set[tuple[str, str]] = set()

        for raw in dot_source.splitlines():
            line = raw.strip()
            if not line or line.startswith("digraph") or line in ("{", "}"):
                continue

            edge_match = re.match(r"([A-Za-z0-9_]+)\s*->\s*([A-Za-z0-9_]+)", line)
            if edge_match:
                source = edge_match.group(1)
                target = edge_match.group(2)
                key = (source, target)
                if key not in seen_edges:
                    seen_edges.add(key)
                    edges.append(FlowEdgeModel(id=f"{source}-{target}", source=source, target=target))
                continue

            node_match = re.match(r"([A-Za-z0-9_]+)\s*\[(.+)\]", line)
            if node_match:
                node_id = node_match.group(1)
                attrs = node_match.group(2)
                xlabel_match = re.search(r"xlabel=([^\s\]]+)", attrs)
                label_match = re.search(r"label=([^\s\]]+)", attrs)
                label = node_id
                if xlabel_match:
                    label = xlabel_match.group(1).strip('"')
                elif label_match:
                    val = label_match.group(1).strip('"')
                    if val:
                        label = val
                node_labels[node_id] = label

        node_ids = set(node_labels.keys())
        for edge in edges:
            node_ids.add(edge.source)
            node_ids.add(edge.target)

        nodes: list[FlowNodeModel] = []
        for node_id in sorted(node_ids):
            label = node_labels.get(node_id, node_id)
            kind = "place" if node_id.startswith("alpha_") else "transition"
            nodes.append(FlowNodeModel(id=node_id, label=label, kind=kind))
        return nodes, edges

    def _render_flow_preview(self, dot_source: str) -> None:
        if not self.config.get("enable_flow_preview", True):
            return
        if not streamlit_flow or not StreamlitFlowState or not StreamlitFlowNode or not StreamlitFlowEdge:
            return

        nodes_cfg, edges_cfg = self._parse_dot_to_flow(dot_source)
        if not nodes_cfg:
            return

        flow_nodes = []
        for idx, node in enumerate(nodes_cfg):
            x_pos = 70 + (idx * 160)
            y_pos = 120 if node.kind == "place" else 210
            style = {
                "backgroundColor": "#0f172a" if node.kind == "place" else "#111827",
                "color": "#e5e7eb",
                "border": "1px solid #475569",
                "borderRadius": "10px",
                "padding": "6px 10px",
            }
            flow_nodes.append(
                StreamlitFlowNode(
                    id=node.id,
                    pos=(x_pos, y_pos),
                    data={"content": node.label, "label": node.label},
                    style=style,
                )
            )

        flow_edges = [
            StreamlitFlowEdge(
                id=edge.id,
                source=edge.source,
                target=edge.target,
                edge_type="smoothstep",
                style={"stroke": "#93c5fd", "strokeWidth": 1.5},
                animated=False,
            )
            for edge in edges_cfg
        ]

        flow_state = StreamlitFlowState(nodes=flow_nodes, edges=flow_edges)
        streamlit_flow(
            key="petri_flow_preview",
            state=flow_state,
            height=360,
            fit_view=True,
            show_controls=True,
            show_minimap=False,
            allow_new_edges=False,
            pan_on_drag=True,
            allow_zoom=True,
            hide_watermark=True,
        )

    def _dot_to_png_bytes(self, dot_source: str) -> bytes:
        try:
            return graphviz.Source(dot_source).pipe(format="png")
        except Exception:
            url = "https://quickchart.io/graphviz?format=png&graph=" + urllib.parse.quote(dot_source)
            with urllib.request.urlopen(url, timeout=20) as resp:
                return resp.read()

    def _render_float_panel(self, ws: dict) -> None:
        if not self.config.get("enable_float_panel", True):
            return
        if not float_box:
            return
        float_box(
            markdown=(
                f"**Modo:** {'IA' if ws['modo_ia'] else 'Manual'}  \n"
                f"**Modelo:** `{ws.get('llm_model') or 'meta-llama/llama-4-maverick-17b-128e-instruct'}`  \n"
                f"**Geracao:** {ws.get('ultima_geracao', 'Nenhuma')}"
            ),
            width="280px",
            height="130px",
            right="1rem",
            bottom="1rem",
            background="#0f1115",
            border="1px solid #334155",
            shadow="0 8px 24px rgba(0,0,0,0.45)",
            z_index="999",
        )

    def display_chat(self) -> None:
        ws = self._workspace_state()
        
        ws.setdefault("graph_bg_color", "#0e1117")
        ws.setdefault("graph_fg_color", "#ffffff")
        ws.setdefault("graph_rankdir", "LR")
        
        # Garante fallback em 300% (3.0)
        if float(ws.get("diagram_scale", 3.0)) < 0.5:
            ws["diagram_scale"] = 3.0
            
        if int(ws.get("diagram_box_height", 520)) < 640:
            ws["diagram_box_height"] = 640
        ws.setdefault("dot_theme_applied", (ws["graph_bg_color"], ws["graph_fg_color"], ws["graph_rankdir"]))
        
        if ws.get("dot_source") and ws["dot_theme_applied"] != (
            ws["graph_bg_color"],
            ws["graph_fg_color"],
            ws["graph_rankdir"],
        ):
            ws["dot_source"] = self._retheme_dot_source(
                ws["dot_source"],
                bg_color=ws["graph_bg_color"],
                fg_color=ws["graph_fg_color"],
                rankdir=ws["graph_rankdir"],
            )
            ws["dot_theme_applied"] = (ws["graph_bg_color"], ws["graph_fg_color"], ws["graph_rankdir"])

        # Carregamento do Logo e Definição do Header Estilizado
        logo_base64 = self._img_to_base64("assets/logo.png")
        img_tag = f'<img src="{logo_base64}" width="40" style="margin-right: 15px; vertical-align: middle;">' if logo_base64 else ""

        st.markdown(
            f"""
            <style>
            .header-estilizado {{
                display: flex;
                align-items: center;
                padding: 10px 12px;
                margin-bottom: 10px;
                border: 1px solid rgba(255, 255, 255, 0.16);
                border-radius: 10px;
                background: transparent;
                font-size: 1.4rem;
                font-weight: 700;
                color: #ffffff;
            }}
            .canvas-subtitle {{
                margin: -5px 0 10px 12px;
                color: rgba(255, 255, 255, 0.82);
                font-size: 0.9rem;
            }}
            
            .chat-scroll-box {{
                max-height: 38vh;
                overflow-y: auto;
                padding-right: 6px;
                margin-bottom: 8px;
            }}
            .chat-scroll-box::-webkit-scrollbar {{
                width: 6px;
            }}
            .chat-scroll-box::-webkit-scrollbar-track {{
                background: rgba(255,255,255,0.05);
                border-radius: 4px;
            }}
            .chat-scroll-box::-webkit-scrollbar-thumb {{
                background: rgba(255,255,255,0.2);
                border-radius: 4px;
            }}
            .chat-scroll-box::-webkit-scrollbar-thumb:hover {{
                background: rgba(255,255,255,0.3);
            }}
            </style>
            
            <div class="header-estilizado">
                {img_tag} SED Workspace
            </div>
            <p class="canvas-subtitle">
                Arquitetura pronta para evoluir para Canvas Infinito com estado persistente por workspace.
            </p>
            """,
            unsafe_allow_html=True,
        )

        with st.expander("⚙️ Configurações Visuais do Diagrama (DOT)", expanded=False):
            c_bg, c_fg, c_ori = st.columns(3)
            with c_bg:
                new_bg = st.color_picker("Fundo DOT", value=ws["graph_bg_color"], key="ui_cfg_bg")
            with c_fg:
                new_fg = st.color_picker("Desenho DOT", value=ws["graph_fg_color"], key="ui_cfg_fg")
            with c_ori:
                ori_opts = {"Horizontal": "LR", "Vertical": "TB"}
                curr_lbl = "Vertical" if ws["graph_rankdir"] == "TB" else "Horizontal"
                new_lbl = st.selectbox(
                    "Orientação",
                    options=list(ori_opts.keys()),
                    index=list(ori_opts.keys()).index(curr_lbl),
                    key="ui_cfg_ori"
                )
                new_rank = ori_opts[new_lbl]
            
            if new_bg != ws["graph_bg_color"] or new_fg != ws["graph_fg_color"] or new_rank != ws["graph_rankdir"]:
                ws["graph_bg_color"] = new_bg
                ws["graph_fg_color"] = new_fg
                ws["graph_rankdir"] = new_rank
                if ws.get("dot_source"):
                    ws["dot_source"] = self._retheme_dot_source(
                        ws["dot_source"], bg_color=new_bg, fg_color=new_fg, rankdir=new_rank
                    )
                    ws["dot_theme_applied"] = (new_bg, new_fg, new_rank)
                st.rerun()

        with st.sidebar:
            st.markdown("### Painel de Modelagem")
            modo_ia = st.toggle(
                "Gerador automático (IA)",
                value=ws["modo_ia"],
                key="modo_toggle",
                help="Desative para usar o modo Manual.",
            )
            ws["modo_ia"] = modo_ia
            st.caption("Modo ativo: Automático (IA)" if modo_ia else "Modo ativo: Manual (automático desligado)")
            
            if modo_ia:
                col_btn1, col_btn2 = st.columns(2)
                with col_btn1:
                    if st.button("New Chat", width="stretch"):
                        self.new_thread()
                with col_btn2:
                    if st.button("Clear Graph", width="stretch"):
                        self.clear_graph()
            else:
                if st.button("Clear Graph", width="stretch"):
                    self.clear_graph()

            if modo_ia:
                if st.session_state.get("clear_side_chat_prompt", False):
                    st.session_state["side_chat_prompt"] = ""
                    st.session_state["clear_side_chat_prompt"] = False

                st.markdown("**Chat**")
                if ws["chat_history"]:
                    messages_html = []
                    for msg in ws["chat_history"][-40:]:
                        is_user = msg["role"] == "user"
                        align = "flex-end" if is_user else "flex-start"
                        safe = html.escape(str(msg["content"])).replace("\n", "<br>")
                        if is_user:
                            messages_html.append(
                                (
                                    f'<div style="display:flex; justify-content:{align}; margin:6px 0;">'
                                    '<div style="max-width:85%; background:#1a1c24; color:#f9fafb; '
                                    "padding:8px 10px; border-radius:12px; border:1px solid rgba(255,255,255,0.12); "
                                    f'font-size:0.9rem; line-height:1.25;">{safe}</div></div>'
                                )
                            )
                        else:
                            messages_html.append(
                                (
                                    f'<div style="display:flex; justify-content:{align}; margin:6px 0;">'
                                    f'<div style="max-width:90%; color:#ffffff; font-size:0.9rem; line-height:1.3;">{safe}</div>'
                                    "</div>"
                                )
                            )
                    
                    chat_html = (
                        '<div class="chat-scroll-box">'
                        + "".join(messages_html)
                        + "</div>"
                    )
                    st.markdown(chat_html, unsafe_allow_html=True)
                else:
                    st.caption("Sem mensagens ainda.")

                live_reply_slot = st.empty()
                model_options = [
                    "meta-llama/llama-4-maverick-17b-128e-instruct",
                    "gpt-4o-mini",
                    "gpt-4.1-mini",
                    "gemini-3-pro-preview",
                    "gemini-3-flash-preview",
                    "gemini-2.5-flash",
                    "gemini-2.5-flash-lite",
                    "gemini-2.5-pro",
                    "llama-3.1-8b-instant",
                    "meta-llama/llama-4-scout-17b-16e-instruct",
                    "qwen/qwen3-32b",
                    "moonshotai/kimi-k2-instruct-0905",
                    "openai/gpt-oss-20b",
                    "openai/gpt-oss-120b",
                ]
                selected_model = ws["llm_model"] if ws["llm_model"] in model_options else model_options[0]
                chat_prompt = st.text_area(
                    "Mensagem",
                    value=st.session_state.get("side_chat_prompt", ""),
                    placeholder="Digite sua mensagem...",
                    height=90,
                    key="side_chat_prompt",
                )
                row_model, row_btn = st.columns([0.72, 0.28], gap="small")
                with row_model:
                    model = st.selectbox(
                        "Modelo",
                        options=model_options,
                        index=model_options.index(selected_model),
                        key="llm_model_select",
                        label_visibility="collapsed"
                    )
                with row_btn:
                    send_chat = st.button("↑", width="stretch", key="btn_side_send")
                ws["llm_model"] = model

                if send_chat and chat_prompt.strip():
                    client, auto_provider = self._sync_llm_client(model=model)
                    msg = chat_prompt.strip()

                    ws["chat_history"].append({"role": "user", "content": msg})
                    result = run_langgraph_chat(
                        user_text=msg,
                        model_name=model,
                        llm_client=client,
                        bg_color=ws["graph_bg_color"],
                        fg_color=ws["graph_fg_color"],
                        rankdir=ws["graph_rankdir"],
                    )
                    assistant_text = result.get("assistant_text", "")
                    streamed = live_reply_slot.write_stream(self._stream_text(assistant_text))
                    live_reply_slot.empty()
                    if isinstance(streamed, str) and streamed.strip():
                        assistant_text = streamed
                    ws["chat_history"].append({"role": "assistant", "content": assistant_text})
                    ws["llm_last_output"] = assistant_text
                    ws["last_raw_model_output"] = result.get("raw_model_output", "")
                    ws["last_petri_spec"] = result.get("petri_spec", {})

                    dot_source = result.get("dot_source", "")
                    if dot_source:
                        ws["dot_source"] = dot_source
                        ws["dot_theme_applied"] = (ws["graph_bg_color"], ws["graph_fg_color"], ws["graph_rankdir"])
                    ws["ultima_geracao"] = result.get("generation_label", f"IA ({auto_provider})")
                    st.session_state["clear_side_chat_prompt"] = True
                    st.rerun()
            else:
                modo_manual = st.radio(
                    "Modo Manual",
                    options=["Padrão Fixo", "Personalizado"],
                    horizontal=True,
                    key="manual_mode_kind",
                )

                if modo_manual == "Padrão Fixo":
                    st.caption("Topologia padrão fixa: 3 lugares e 2 transições em cadeia.")
                    if st.button("Gerar Manual", width="stretch", key="btn_manual_fixed"):
                        dot = build_petri_net_manual(
                            3,
                            2,
                            bg_color=ws["graph_bg_color"],
                            fg_color=ws["graph_fg_color"],
                            rankdir=ws["graph_rankdir"],
                        )
                        ws["dot_source"] = dot.source
                        ws["ultima_geracao"] = "Manual (Padrão Fixo)"
                        st.success("Grafo padrão gerado.")
                else:
                    with st.expander("Configuração Manual Avançada", expanded=False):
                        n_lugares_default, n_transicoes_default = self._manual_defaults(ws)
                        n_lugares = int(
                            st.number_input(
                                "Lugares (alpha)",
                                min_value=1,
                                max_value=20,
                                value=n_lugares_default,
                                step=1,
                                key="n_lugares",
                            )
                        )
                        n_transicoes = int(
                            st.number_input(
                                "Transicoes (t)",
                                min_value=1,
                                max_value=20,
                                value=n_transicoes_default,
                                step=1,
                                key="n_transicoes",
                            )
                        )

                        ws["manual"]["n_lugares"] = n_lugares
                        ws["manual"]["n_transicoes"] = n_transicoes
                        self._ensure_connection_defaults(ws, n_transicoes)

                        st.caption("Conexoes Pre/Post")
                        opcoes_lugares = [f"alpha_{i:02d}" for i in range(1, n_lugares + 1)]
                        self._ensure_token_defaults(ws, opcoes_lugares)

                        conexoes_list: list[dict] = []
                        for i in range(1, n_transicoes + 1):
                            t_key = f"t{i}"
                            atual = ws["manual"]["conexoes"][t_key]
                            pre_atual = [p for p in atual.get("pre", []) if p in opcoes_lugares]
                            post_atual = [p for p in atual.get("post", []) if p in opcoes_lugares]

                            pre = st.multiselect(
                                f"Pre {t_key}",
                                options=opcoes_lugares,
                                default=pre_atual,
                                key=f"pre_{t_key}",
                            )
                            post = st.multiselect(
                                f"Post {t_key}",
                                options=opcoes_lugares,
                                default=post_atual,
                                key=f"post_{t_key}",
                            )
                            ws["manual"]["conexoes"][t_key] = {"pre": pre, "post": post}
                            conexoes_list.append({"pre": pre, "post": post})

                        st.caption("Tokens por lugar")
                        for place_id in opcoes_lugares:
                            current = int(ws["manual"]["tokens"].get(place_id, 0))
                            tokens = int(
                                st.number_input(
                                    f"{place_id} token(s)",
                                    min_value=0,
                                    max_value=9,
                                    value=current,
                                    step=1,
                                    key=f"tok_{place_id}",
                                )
                            )
                            ws["manual"]["tokens"][place_id] = tokens

                if st.button("Gerar Manual", width="stretch", key="btn_manual_custom"):
                    dot = build_petri_net_manual_conexoes(
                        n_lugares,
                        n_transicoes,
                        conexoes_list,
                        tokens_por_lugar=ws["manual"]["tokens"],
                        bg_color=ws["graph_bg_color"],
                        fg_color=ws["graph_fg_color"],
                        rankdir=ws["graph_rankdir"],
                    )
                    ws["dot_source"] = dot.source
                    ws["dot_theme_applied"] = (ws["graph_bg_color"], ws["graph_fg_color"], ws["graph_rankdir"])
                    ws["ultima_geracao"] = "Manual (Personalizado)"
                    st.success("Grafo manual personalizado gerado.")

        with st.container():
            mode_label = "IA" if ws["modo_ia"] else "Manual"
            with st.container(border=True):
                
                # NOVO SLIDER DE ZOOM
                zoom_val = int(float(ws.get('diagram_scale', 3.0)) * 100)
                zoom_pct = st.slider(
                    "🔍 Zoom do Diagrama (%)", 
                    min_value=50, 
                    max_value=800, 
                    value=zoom_val, 
                    step=25,
                    help="Ajuste o nível de zoom do grafo gerado."
                )
                ws["diagram_scale"] = zoom_pct / 100.0

                st.markdown(
                    f"""
                    <p class="section-title">Canvas de Grafo <span class="mode-chip">{mode_label}</span></p>
                    <p class="section-subtitle">Ultima geracao: {ws["ultima_geracao"]}</p>
                    """,
                    unsafe_allow_html=True,
                )
                box_h = int(ws.get("diagram_box_height", 520))
                
                # CSS PARA CENTRALIZAÇÃO PERFEITA
                st.markdown(
                    f"""
                    <style>
                    div[data-testid='stGraphvizChart'] {{
                        min-height: {box_h}px;
                        display: flex;
                        justify-content: center;
                        align-items: center;
                        overflow: auto;
                        width: 100%;
                    }}
                    div[data-testid='stGraphvizChart'] svg {{
                        margin: auto;
                    }}
                    </style>
                    """,
                    unsafe_allow_html=True,
                )

                if ws["dot_source"]:
                    try:
                        dot_to_render = self._scaled_dot_source(
                            ws["dot_source"], float(ws.get("diagram_scale", 3.0))
                        )
                        st.graphviz_chart(dot_to_render, use_container_width=False) # graphviz chart doesn't support width yet
                    except Exception:
                        st.error("Graphviz nao disponivel no ambiente. Exibindo DOT.")
                        st.code(ws["dot_source"], language="dot")

                    try:
                        png_bytes = self._dot_to_png_bytes(ws["dot_source"])
                        st.download_button(
                            "Baixar PNG",
                            data=png_bytes,
                            file_name="rede_petri.png",
                            mime="image/png",
                            key="download_petri_png",
                        )
                    except Exception:
                        st.caption("Download PNG indisponivel neste ambiente.")

                    if self.config.get("enable_flow_preview", True):
                        st.markdown("**Preview Canvas (streamlit-flow)**")
                        self._render_flow_preview(ws["dot_source"])

                    with st.expander("Codigo DOT", expanded=False):
                        st.code(ws["dot_source"], language="dot")
                else:
                    st.markdown(
                        """
                        <div class="canvas-empty">
                          Nenhum grafo gerado ainda.<br>
                          Configure o painel e clique em gerar.
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

        if ws.get("llm_last_output"):
            with st.expander("Ultima saida do LLM", expanded=False):
                st.write(ws["llm_last_output"])
        if ws.get("last_raw_model_output"):
            with st.expander("Raw retorno do modelo", expanded=False):
                st.code(ws["last_raw_model_output"], language="json")
        if ws.get("last_petri_spec"):
            with st.expander("JSON validado (neuro-simbolico)", expanded=False):
                st.json(ws["last_petri_spec"])

        self._render_float_panel(ws)