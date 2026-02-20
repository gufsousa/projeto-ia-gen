"""Extração simulada de tags industriais (sensores/atuadores) a partir de texto."""
import pandas as pd

TAG_RULES = [
    ("sensor de presença", "S01", "Digital Input"),
    ("sensor de temperatura", "S02", "Analog Input"),
    ("sensor de nível", "S03", "Analog Input"),
    ("sensor de pressão", "S04", "Analog Input"),
    ("sensor fim de curso", "S05", "Digital Input"),
    ("fim de curso", "S05", "Digital Input"),
    ("atuador", "A01", "Digital Output"),
    ("válvula", "A02", "Digital Output"),
    ("bomba", "A03", "Digital Output"),
    ("motor", "A04", "Digital Output"),
]


def extrair_tags(texto: str) -> pd.DataFrame:
    """Simula extração de tags industriais (sensores/atuadores) a partir do texto."""
    if not texto or not texto.strip():
        return pd.DataFrame(columns=["Tag", "Tipo", "Descrição"])
    texto_lower = texto.lower().strip()
    linhas = []
    for frase, tag, tipo in TAG_RULES:
        if frase in texto_lower:
            linhas.append({"Tag": tag, "Tipo": tipo, "Descrição": frase})
    return pd.DataFrame(linhas) if linhas else pd.DataFrame(columns=["Tag", "Tipo", "Descrição"])
