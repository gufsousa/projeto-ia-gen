"""Factory e strategies para hot-swapping de provedores LLM."""
from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover
    load_dotenv = None

from sed.secrets import get_secret

if load_dotenv:
    load_dotenv()


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
        self.api_key = api_key or get_secret("OPENAI_API_KEY")

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

    def __init__(self, model: str = "gemini-2.5-flash", api_key: str | None = None) -> None:
        self.model = model
        self.api_key = api_key or get_secret("GEMINI_API_KEY", "GOOGLE_API_KEY")

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
        except Exception as exc:
            yield f"[Falha Gemini: {type(exc).__name__}: {exc}] "
            yield "[Fallback local] "
            yield from MockLLMStrategy().stream(prompt)


class GroqStrategy(LLMStrategy):
    """Provider Groq com fallback seguro quando SDK/API nao estiverem disponiveis."""

    provider_name = "groq"

    def __init__(self, model: str = "llama-3.1-8b-instant", api_key: str | None = None) -> None:
        self.model = model
        self.api_key = api_key or get_secret("GROQ_API_KEY")

    def _completion_params(self, prompt: str) -> dict:
        """Build Groq completion params with gpt-oss friendly defaults."""
        params = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": True,
            "top_p": 1,
        }
        # Configuracao inspirada no exemplo oficial para openai/gpt-oss-*.
        if str(self.model).startswith("openai/gpt-oss"):
            params.update(
                {
                    "temperature": 1,
                    "max_completion_tokens": 8192,
                    "reasoning_effort": "medium",
                    "stop": None,
                }
            )
        else:
            params.update({"temperature": 0.1})
        return params

    def stream(self, prompt: str) -> Iterator[str]:
        try:
            from groq import Groq  # type: ignore
            use_native_sdk = True
        except Exception:
            use_native_sdk = False

        if not self.api_key:
            yield "[GROQ_API_KEY ausente] "
            yield from MockLLMStrategy().stream(prompt)
            return

        if use_native_sdk:
            try:
                client = Groq(api_key=self.api_key)
                stream = client.chat.completions.create(**self._completion_params(prompt))
                for chunk in stream:
                    delta = chunk.choices[0].delta.content if chunk.choices else None
                    if delta:
                        yield delta
                return
            except Exception:
                # Continua para fallback OpenAI-compatible.
                pass

        try:
            from openai import OpenAI  # type: ignore
        except Exception:
            yield "[Groq SDK e OpenAI SDK nao instaladas] "
            yield from MockLLMStrategy().stream(prompt)
            return

        try:
            client = OpenAI(api_key=self.api_key, base_url="https://api.groq.com/openai/v1")
            stream = client.chat.completions.create(**self._completion_params(prompt))
            for chunk in stream:
                delta = chunk.choices[0].delta.content if chunk.choices else None
                if delta:
                    yield delta
        except Exception:
            yield "[Falha Groq, fallback local] "
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
        if p == "groq":
            return GroqStrategy(model=model or "llama-3.1-8b-instant", api_key=api_key)
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
