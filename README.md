# Avaliacao Intermediaria - IA Generativa

Projeto individual para modelagem de Redes de Petri em Streamlit, com foco em interface, arquitetura e experiencia de uso.

## Escopo desta entrega

Esta versao segue a regra da avaliacao intermediaria:
- sem integracao de LLM real em producao;
- foco total em funcionalidade da interface;
- respostas de modo IA tratadas como mock/simulacao de comportamento futuro.

## Problema e solucao proposta

### Problema
Modelar sistemas de eventos discretos (SED) de forma visual e rapida, com dois fluxos:
- fluxo assistido por texto (modo IA);
- fluxo totalmente manual (modo simbolico).

### Solucao
Aplicacao Streamlit com:
- painel lateral de controle;
- canvas principal para renderizacao da Rede de Petri;
- persistencia de estado para evitar reset a cada interacao.

## Complexidade do prototipo

O prototipo foi desenhado para ir alem de um chat simples:
- dois modos completos de operacao (IA e Manual);
- historico de mensagens no modo IA;
- configuracao manual de lugares, transicoes, conexoes pre/post e tokens;
- configuracao visual de cores do DOT (fundo e desenho);
- exportacao PNG do grafo;
- preview adicional com componente de flow;
- barra lateral com acoes de sessao.

## Arquitetura e escolhas de design

### Stack
- Streamlit (UI e estado reativo)
- Graphviz (geracao e visualizacao DOT)
- streamlit-flow-component (preview tipo canvas)
- streamlit-float (painel flutuante)
- pydantic (tipagem para estruturas auxiliares)

### Estrutura modular
- `app.py`: entrypoint da aplicacao.
- `src/app_config.py`: configuracoes da sidebar e ciclo de vida do connector.
- `src/ui/petri_ui_connector.py`: orquestracao da UI, eventos e estado.
- `sed/grafo.py`: construcao das Redes de Petri (IA e manual).
- `sed/ui_css.py`: tema e ajustes visuais.

### Estado persistente
Uso de `st.session_state["petri_store"][workspace_id]` para manter:
- modo atual;
- historico de chat;
- DOT atual;
- parametros manuais;
- configuracao de cores.

Isso evita perda de dados durante reruns do Streamlit.

## O que foi implementado corretamente

- alternancia clara entre modo automatico e manual;
- modo manual com topologia fixa e topologia personalizada;
- renderizacao consistente do grafo no canvas principal;
- customizacao de cor aplicada em ambos os modos;
- download do grafo em PNG com fallback;
- organizacao da interface com barra lateral e canvas.

## O que funcionou bem no uso do agente de codificacao

- refatoracao da aplicacao para arquitetura modular;
- resolucao de problemas de `session_state` e rerender do Streamlit;
- ajuste iterativo de UX (chat lateral, botoes, modos, visibilidade condicional);
- evolucao do grafo (labels de lugares, tokens, estilos e tema).

Exemplos de direcionamentos que funcionaram:
- "separe UI principal e estado em um connector"
- "no modo manual, esconder itens de chat/sessao"
- "manter grafo no canvas e evitar reset ao interagir"

## O que nao funcionou bem / limitacoes

- conflito de estilos CSS em algumas iteracoes (layout alternando de forma inesperada);
- comportamento de widget em Streamlit com erros de chave/instanciacao;
- dependencia de ambiente para renderizacao Graphviz local;
- necessidade de varios ajustes finos para UX do chat em sidebar.

Como contornado:
- centralizacao da logica de estado no connector;
- tratamento de fallback para renderizacao e exportacao;
- simplificacao e estabilizacao de tema/CSS.

## Observacao importante sobre IA nesta etapa

Mesmo existindo estrutura de cliente/factory no codigo para evolucao futura, a avaliacao intermediaria considera apenas a interface e o fluxo mock.  
Nao ha dependencia obrigatoria de API externa para demonstrar o prototipo.

## Endpoint (preencher antes da entrega)

- Endpoint publico: `COLOCAR_LINK_AQUI`
- Repositorio GitHub: `COLOCAR_LINK_AQUI`

## Como executar localmente

1. Criar ambiente virtual:

```bash
python -m venv .venv
```

2. Ativar ambiente (PowerShell):

```powershell
.\.venv\Scripts\Activate.ps1
```

3. Instalar dependencias:

```bash
pip install -r requirements.txt
```

4. Rodar aplicacao:

```bash
streamlit run app.py
```

## Requisitos

Dependencias Python (`requirements.txt`):
- streamlit>=1.28
- graphviz
- pandas
- langgraph
- streamlit-flow-component
- streamlit-float
- pydantic

Dependencia de sistema (`packages.txt`):
- graphviz

## Estrutura de pastas

```text
.
|- app.py
|- README.md
|- requirements.txt
|- packages.txt
|- sed/
|  |- grafo.py
|  |- llm_factory.py
|  |- tags.py
|  |- ui_css.py
|- src/
   |- app_config.py
   |- ui/
      |- petri_ui_connector.py
```
