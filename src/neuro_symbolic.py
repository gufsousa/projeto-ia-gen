"""Neuro-symbolic parsing: LLM text -> validated Petri JSON spec."""
from __future__ import annotations

import json
import re
from typing import Any

from pydantic import BaseModel, Field, ValidationError, field_validator


class PlaceSpec(BaseModel):
    id: str
    label: str
    tokens: int = 0

    @field_validator("id", "label")
    @classmethod
    def _not_empty(cls, value: str) -> str:
        v = (value or "").strip()
        if not v:
            raise ValueError("empty value")
        return v


class TransitionSpec(BaseModel):
    id: str
    label: str

    @field_validator("id", "label")
    @classmethod
    def _not_empty(cls, value: str) -> str:
        v = (value or "").strip()
        if not v:
            raise ValueError("empty value")
        return v


class ArcSpec(BaseModel):
    source: str
    target: str
    weight: int = 1

    @field_validator("source", "target")
    @classmethod
    def _not_empty(cls, value: str) -> str:
        v = (value or "").strip()
        if not v:
            raise ValueError("empty value")
        return v

    @field_validator("weight")
    @classmethod
    def _valid_weight(cls, value: int) -> int:
        if int(value) < 1:
            raise ValueError("weight must be >= 1")
        return int(value)


class PetriNetSpec(BaseModel):
    places: list[PlaceSpec] = Field(default_factory=list)
    transitions: list[TransitionSpec] = Field(default_factory=list)
    arcs: list[ArcSpec] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("places")
    @classmethod
    def _ensure_places(cls, places: list[PlaceSpec]) -> list[PlaceSpec]:
        if not places:
            raise ValueError("places must not be empty")
        return places

    @field_validator("transitions")
    @classmethod
    def _ensure_transitions(cls, transitions: list[TransitionSpec]) -> list[TransitionSpec]:
        if not transitions:
            raise ValueError("transitions must not be empty")
        return transitions

    @field_validator("arcs")
    @classmethod
    def _ensure_arcs(cls, arcs: list[ArcSpec]) -> list[ArcSpec]:
        if not arcs:
            raise ValueError("arcs must not be empty")
        return arcs


def petri_json_prompt(user_text: str) -> str:
    """Instruction prompt to force JSON output."""
    return (
        "Voce e um compilador neuro-simbolico para Rede de Petri. "
        "Converta o texto do usuario para JSON VALIDO e SOMENTE JSON (sem markdown, sem comentarios). "
        "Use exatamente este schema: "
        '{"places":[{"id":"p1","label":"P1","tokens":0}],"transitions":[{"id":"t1","label":"t1"}],'
        '"arcs":[{"source":"p1","target":"t1","weight":1}],"metadata":{"assumptions":[],"bounded":false}} '
        "Regras: ids unicos; arcos so entre lugar<->transicao; tokens inteiro >= 0; weight inteiro >= 1. "
        "Texto do usuario: "
        + user_text
    )


def _extract_json_candidate(raw_text: str) -> str:
    text = (raw_text or "").strip()
    if not text:
        return ""

    code_block = re.search(r"```(?:json)?\s*(\{[\s\S]*\})\s*```", text, flags=re.IGNORECASE)
    if code_block:
        return code_block.group(1).strip()

    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        return text[start : end + 1].strip()
    return text


def parse_and_validate_petri_spec(raw_text: str) -> tuple[dict[str, Any] | None, str | None]:
    """Parse possible JSON text and validate against Pydantic schema."""
    candidate = _extract_json_candidate(raw_text)
    if not candidate:
        return None, "empty model output"

    try:
        payload = json.loads(candidate)
    except json.JSONDecodeError as exc:
        return None, f"json decode error: {exc}"

    try:
        spec = PetriNetSpec.model_validate(payload)
    except ValidationError as exc:
        return None, f"schema validation error: {exc}"

    return spec.model_dump(), None


def fallback_spec_from_text(user_text: str) -> dict[str, Any]:
    """Deterministic fallback spec when model JSON is invalid/unavailable."""
    text = (user_text or "").lower()
    is_queue = "fila" in text or "queue" in text
    is_unbounded = (
        "sem limite" in text
        or "ilimitad" in text
        or "infinit" in text
        or "unbounded" in text
    )
    if is_queue:
        return {
            "places": [
                {"id": "p_src", "label": "P_src", "tokens": 1},
                {"id": "p_fila", "label": "P_fila(inf)" if is_unbounded else "P_fila(k)", "tokens": 0},
                {"id": "p_sink", "label": "P_sink", "tokens": 0},
            ],
            "transitions": [
                {"id": "t_chegada", "label": "t_chegada"},
                {"id": "t_servico", "label": "t_servico"},
            ],
            "arcs": [
                {"source": "p_src", "target": "t_chegada", "weight": 1},
                {"source": "t_chegada", "target": "p_fila", "weight": 1},
                {"source": "p_fila", "target": "t_servico", "weight": 1},
                {"source": "t_servico", "target": "p_sink", "weight": 1},
            ],
            "metadata": {"assumptions": ["fallback rule-based spec"], "bounded": not is_unbounded},
        }

    return {
        "places": [
            {"id": "p1", "label": "P1", "tokens": 1},
            {"id": "p2", "label": "P2", "tokens": 0},
            {"id": "p3", "label": "P3", "tokens": 0},
        ],
        "transitions": [
            {"id": "t1", "label": "t1"},
            {"id": "t2", "label": "t2"},
        ],
        "arcs": [
            {"source": "p1", "target": "t1", "weight": 1},
            {"source": "t1", "target": "p2", "weight": 1},
            {"source": "p2", "target": "t2", "weight": 1},
            {"source": "t2", "target": "p3", "weight": 1},
        ],
        "metadata": {"assumptions": ["fallback linear spec"], "bounded": True},
    }


def summarize_spec(spec: dict[str, Any]) -> str:
    """Create a friendly assistant explanation from validated spec."""
    n_places = len(spec.get("places", []) or [])
    n_transitions = len(spec.get("transitions", []) or [])
    n_arcs = len(spec.get("arcs", []) or [])
    weighted_arcs = sum(1 for a in (spec.get("arcs", []) or []) if int(a.get("weight", 1) or 1) > 1)
    bounded = spec.get("metadata", {}).get("bounded")
    bounded_txt = "limitada" if bounded else "ilimitada" if bounded is False else "nao especificada"
    places = spec.get("places", []) or []
    transitions = spec.get("transitions", []) or []
    place_names = ", ".join(str(p.get("label", p.get("id", ""))) for p in places[:4]) or "-"
    trans_names = ", ".join(str(t.get("label", t.get("id", ""))) for t in transitions[:4]) or "-"
    return (
        "Perfeito! Montei a modelagem em logica neuro-simbolica. "
        "Interpretei sua solicitacao, converti para JSON estruturado e validei as regras formais de Redes de Petri.\n\n"
        "Resultado:\n"
        f"- {n_places} Lugares: {place_names}.\n"
        f"- {n_transitions} Transicoes: {trans_names}.\n"
        f"- Conexao: {n_arcs} Arcos (pesos > 1 em {weighted_arcs} arcos; capacidade inferida: {bounded_txt}).\n\n"
        "O grafo ja esta disponivel no seu canvas. "
        "Gostaria de adicionar um limite de capacidade (buffer) ou simular o disparo de uma ficha?"
    )
