# GestorFlow

Motor de classificação e auditoria de documentos para processos de repasse
Fundo a Fundo (saúde). Implementa a especificação em
[`docs/prompts_classificacao_auditoria_260918_164939.pdf`](docs/prompts_classificacao_auditoria_260918_164939.pdf):
17 tipos de documento, suas regras de conformidade e a cadeia cronológica
de dependência/invalidação em cascata entre eles.

## Módulos

- **`gestorflow_auditoria.py`** — estrutura principal: tipos de documento,
  regras de classificação, especificações de auditoria por tipo, validação
  da cadeia cronológica e orquestração do pipeline (`run_pipeline`). A
  extração real de campos fica a cargo de uma função `extract_fields_fn`
  plugada por quem integrar o módulo (ex.: uma chamada a um LLM).
- **`gestorflow_vision.py`** — extensão que usa a API da Anthropic (Claude)
  com visão para classificar e extrair campos diretamente de PDFs/imagens
  de documentos (fotos, scans, capturas de tela de sistemas como
  SAEP/SIOPS/SIOPE), reaproveitando as mesmas regras do módulo de
  auditoria como fonte única da verdade dos prompts.
- **`test_gestorflow_auditoria.py`** — testes unitários das regras de
  conformidade e da cadeia cronológica.

## Uso

```python
from gestorflow_auditoria import AuditContext, run_pipeline

contexto = AuditContext(
    cadastro_proposta={"municipio": "Exemplo/MA"},
    data_referencia_auditoria=date(2026, 9, 21),
    ano_referencia_auditoria=2026,
)

def extract_fields(texto, tipo):
    ...  # chamar um LLM com o prompt de auditoria do tipo correspondente

for linha in run_pipeline(documentos, extract_fields, contexto):
    print(linha)
```

Para classificar/extrair diretamente de PDF ou imagem com visão computacional,
use `gestorflow_vision.run_vision_pipeline` (requer `ANTHROPIC_API_KEY` no
ambiente e as dependências opcionais em `requirements.txt`).

## Testes

```bash
python -m unittest test_gestorflow_auditoria.py -v
```

## Instalação (extensão de visão)

```bash
pip install -r requirements.txt
```
