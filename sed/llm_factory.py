"""Factory e strategies para hot-swapping de provedores LLM."""
from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
import os


class LLMStrategy(ABC):
    """Contrato de providers LLM."""

    provider_name: str = "base"

    @abstractmethod
    def stream(self, prompt: str) -> Iterator[str]:
        """Gera resposta em streaming."""

    def complete(self, prompt: str) -> str:
        """Retorna resposta completa."""
        return "".join(self.stream(prompt))


class MockLLMStrategy(LLMStrategy):
    """Provider local para desenvolvimento sem API externa."""

    provider_name = "mock"

    def stream(self, prompt: str) -> Iterator[str]:
        normalized = (prompt or "").lower()
        clues: list[str] = []
        if "estoque" in normalized or "buffer" in normalized:
            clues.append("detectado buffer/estoque")
        if "motor" in normalized or "maquina" in normalized or "máquina" in normalized:
            clues.append("detectado motor/maquina")

        msg = (
            "Analise concluida. "
            + ("; ".join(clues) if clues else "sem palavras-chave criticas")
            + ". Modelo pronto para gerar Rede de Petri academica."
        )
        for token in msg.split(" "):
            yield token + " "


class OpenAIStrategy(LLMStrategy):
    """Provider OpenAI com fallback seguro quando SDK/API nao estiverem disponiveis."""

    provider_name = "openai"

    def __init__(self, model: str = "gpt-4o-mini", api_key: str | None = None) -> None:
        self.model = model
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")

    def stream(self, prompt: str) -> Iterator[str]:
        try:
            from openai import OpenAI  # type: ignore
        except Exception:
            yield "[OpenAI SDK nao instalada] "
            yield from MockLLMStrategy().stream(prompt)
            return

        if not self.api_key:
            yield "[OPENAI_API_KEY ausente] "
            yield from MockLLMStrategy().stream(prompt)
            return

        try:
            client = OpenAI(api_key=self.api_key)
            stream = client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                stream=True,
                temperature=0.1,
            )
            for chunk in stream:
                delta = chunk.choices[0].delta.content if chunk.choices else None
                if delta:
                    yield delta
        except Exception:
            yield "[Falha OpenAI, fallback local] "
            yield from MockLLMStrategy().stream(prompt)


class GeminiStrategy(LLMStrategy):
    """Provider Gemini com fallback seguro quando SDK/API nao estiverem disponiveis."""

    provider_name = "gemini"

    def __init__(self, model: str = "gemini-1.5-flash", api_key: str | None = None) -> None:
        self.model = model
        self.api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

    def stream(self, prompt: str) -> Iterator[str]:
        try:
            import google.generativeai as genai  # type: ignore
        except Exception:
            yield "[Gemini SDK nao instalada] "
            yield from MockLLMStrategy().stream(prompt)
            return

        if not self.api_key:
            yield "[GEMINI_API_KEY/GOOGLE_API_KEY ausente] "
            yield from MockLLMStrategy().stream(prompt)
            return

        try:
            genai.configure(api_key=self.api_key)
            model = genai.GenerativeModel(self.model)
            response = model.generate_content(prompt, stream=True)
            for chunk in response:
                txt = getattr(chunk, "text", None)
                if txt:
                    yield txt
        except Exception:
            yield "[Falha Gemini, fallback local] "
            yield from MockLLMStrategy().stream(prompt)


class LLMFactory:
    """Fabrica de providers concretos."""

    @staticmethod
    def create(provider: str, model: str | None = None, api_key: str | None = None) -> LLMStrategy:
        p = (provider or "mock").strip().lower()
        if p == "openai":
            return OpenAIStrategy(model=model or "gpt-4o-mini", api_key=api_key)
        if p == "gemini":
            return GeminiStrategy(model=model or "gemini-1.5-flash", api_key=api_key)
        return MockLLMStrategy()


class UnifiedLLMClient:
    """Cliente unificado para hot-swap de providers via factory."""

    def __init__(self, provider: str = "mock", model: str | None = None, api_key: str | None = None) -> None:
        self.provider = provider
        self.model = model
        self.api_key = api_key
        self._strategy = LLMFactory.create(provider=provider, model=model, api_key=api_key)

    def set_provider(self, provider: str, model: str | None = None, api_key: str | None = None) -> None:
        self.provider = provider
        self.model = model
        self.api_key = api_key
        self._strategy = LLMFactory.create(provider=provider, model=model, api_key=api_key)

    def stream_text(self, prompt: str) -> Iterator[str]:
        return self._strategy.stream(prompt)

    def complete_text(self, prompt: str) -> str:
        return self._strategy.complete(prompt)
