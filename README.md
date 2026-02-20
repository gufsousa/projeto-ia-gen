# Avaliação Intermediária - IA Generativa

Projeto individual para modelagem de Redes de Petri em Streamlit, com foco em interface, arquitetura e experiência de uso.

## Escopo desta entrega

Esta versão segue a regra da avaliação intermediária:
- sem integração de LLM real em produção;
- foco total em funcionalidade da interface;
- respostas do modo IA tratadas como mock/simulação de comportamento futuro.

## Problema e solução proposta

### Problema
Modelar sistemas de eventos discretos (SED) de forma visual e rápida, com dois fluxos:
- fluxo assistido por texto (modo IA);
- fluxo totalmente manual (modo simbólico).

### Solução
Aplicação Streamlit com:
- painel lateral de controle;
- canvas principal para renderização da Rede de Petri;
- persistência de estado para evitar reset a cada interação.

## Complexidade do protótipo

O protótipo foi desenhado para ir além de um chat simples:
- dois modos completos de operação (IA e Manual);
- histórico de mensagens no modo IA;
- configuração manual de lugares, transições, conexões pre/post e tokens;
- configuração visual de cores do DOT (fundo e desenho);
- exportação PNG do grafo;
- preview adicional com componente de flow;
- barra lateral com ações de sessão.

## Arquitetura e escolhas de design

### Stack
- Streamlit (UI e estado reativo)
- Graphviz (geração e visualização DOT)
- streamlit-flow-component (preview tipo canvas)
- streamlit-float (painel flutuante)
- pydantic (tipagem para estruturas auxiliares)

### Estrutura modular
- `app.py`: entrypoint da aplicação.
- `src/app_config.py`: configurações da sidebar e ciclo de vida do connector.
- `src/ui/petri_ui_connector.py`: orquestração da UI, eventos e estado.
- `sed/grafo.py`: construção das Redes de Petri (IA e manual).
- `sed/ui_css.py`: tema e ajustes visuais.

### Estado persistente
Uso de `st.session_state["petri_store"][workspace_id]` para manter:
- modo atual;
- histórico de chat;
- DOT atual;
- parâmetros manuais;
- configuração de cores.

Isso evita perda de dados durante reruns do Streamlit.

## O que foi implementado corretamente

- alternância clara entre modo automático e manual;
- modo manual com topologia fixa e topologia personalizada;
- renderização consistente do grafo no canvas principal;
- customização de cor aplicada em ambos os modos;
- download do grafo em PNG com fallback;
- organização da interface com barra lateral e canvas.

## O que funcionou bem no uso do agente de codificação

- refatoração da aplicação para arquitetura modular;
- resolução de problemas de `session_state` e rerender do Streamlit;
- ajuste iterativo de UX (chat lateral, botões, modos, visibilidade condicional);
- evolução do grafo (labels de lugares, tokens, estilos e tema).

Exemplos de direcionamentos que funcionaram:
- "separe UI principal e estado em um connector";
- "no modo manual, esconder itens de chat/sessão";
- "manter grafo no canvas e evitar reset ao interagir".

## O que não funcionou bem / limitações

- conflito de estilos CSS em algumas iterações (layout alternando de forma inesperada);
- comportamento de widget em Streamlit com erros de chave/instanciação;
- dependência de ambiente para renderização Graphviz local;
- necessidade de vários ajustes finos para UX do chat na sidebar.

Como foi contornado:
- centralização da lógica de estado no connector;
- tratamento de fallback para renderização e exportação;
- simplificação e estabilização de tema/CSS.

## Observação importante sobre IA nesta etapa

Mesmo existindo estrutura de cliente/factory no código para evolução futura, a avaliação intermediária considera apenas a interface e o fluxo mock.  
Não há dependência obrigatória de API externa para demonstrar o protótipo.

## Endpoint (preencher antes da entrega)

- Endpoint público: `COLOCAR_LINK_AQUI`
- Repositório GitHub: `COLOCAR_LINK_AQUI`

## Como executar localmente

1. Criar ambiente virtual:

```bash
python -m venv .venv
```

2. Ativar ambiente (PowerShell):

```powershell
.\.venv\Scripts\Activate.ps1
```

3. Instalar dependências:

```bash
pip install -r requirements.txt
```

4. Rodar aplicação:

```bash
streamlit run app.py
```

## Requisitos

Dependências Python (`requirements.txt`):
- streamlit>=1.28
- graphviz
- pandas
- langgraph
- streamlit-flow-component
- streamlit-float
- pydantic

Dependência de sistema (`packages.txt`):
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
