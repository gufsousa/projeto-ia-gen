#!/bin/bash
# Instala o Graphviz (binários do sistema) necessários para a biblioteca Python graphviz.
# No Streamlit Cloud, use packages.txt na raiz do projeto; este script serve para
# ambientes locais (Linux/WSL) ou Docker onde você executa o setup manualmente.
set -e
apt-get update
apt-get install -y graphviz
apt-get clean
rm -rf /var/lib/apt/lists/*
