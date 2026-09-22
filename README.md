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
- **`gestorflow_vision.py`** — extensão com visão computacional para
  classificar e extrair campos diretamente de PDFs/imagens de documentos
  (fotos, scans, capturas de tela de sistemas como SAEP/SIOPS/SIOPE),
  reaproveitando as mesmas regras do módulo de auditoria como fonte
  única da verdade dos prompts. Suporta dois provedores por trás da
  mesma interface (`BaseVisionDocumentReader`):
  - **Claude** (`AnthropicVisionDocumentReader`) — requer
    `ANTHROPIC_API_KEY`, sem tier gratuito persistente.
  - **Gemini** (`GeminiVisionDocumentReader`) — requer
    `GEMINI_API_KEY`/`GOOGLE_API_KEY`, tem tier gratuito (crie uma chave
    em https://aistudio.google.com). Boa opção para testar o pipeline
    sem custo antes de decidir migrar para um provedor pago.

  `make_reader()` escolhe automaticamente o provedor pela API key
  disponível no ambiente (prioriza Anthropic se as duas existirem), ou
  aceita `provider="anthropic"|"gemini"` explicitamente.
- **`test_gestorflow_auditoria.py`** — testes unitários das regras de
  conformidade e da cadeia cronológica.
- **`gestorflow_cli.py`** — CLI de ponta a ponta: recebe arquivos
  (PDF/imagem) + um JSON de contexto e roda `gestorflow_vision` sobre
  eles, imprimindo/gravando a tabela final de conformidade.
- **`gestorflow_signature.py`** — reconhece os DOIS modelos de
  assinatura de um documento:
  1. **Digital** (`inspect_pdf`) — para PDFs nativos assinados
     eletronicamente (PAdES/CAdES): valida integridade, validade
     criptográfica e cobertura (detecta se o arquivo foi alterado após
     a assinatura). Com `--trust-roots` (bundle PEM de âncoras de
     confiança, ex.: cadeia da AC-Raiz ICP-Brasil publicada pelo ITI em
     https://www.iti.gov.br), também valida a cadeia de certificação;
     sem isso, a cadeia fica marcada como "não verificada".
  2. **Manuscrita** (`BaseVisionDocumentReader.assess_manual_signature`,
     em `gestorflow_vision.py` - funciona com qualquer provedor) — para
     fotos/scans de papel (ou PDF sem
     assinatura embutida): o modelo de visão avalia se há uma
     assinatura a caneta plausível, e sinaliza indícios de montagem
     (recorte/colagem) quando houver.

  `check_signature()` escolhe automaticamente o método certo por
  arquivo. **Escopo:** a verificação criptográfica só se aplica a PDF
  nativo assinado eletronicamente — não existe assinatura digital
  embutida em foto de papel. Sem `--trust-roots`, o PyHanko tenta (e
  falha) validar contra o trust store do sistema por padrão, o que
  imprime um traceback de log no console — isso é ruído esperado da
  biblioteca, não uma falha; o campo `trusted` do resultado fica `None`
  nesse caso, indicando "cadeia não verificada".

## Uso via CLI

```bash
# Gemini (tier gratuito) - crie uma chave em https://aistudio.google.com
export GEMINI_API_KEY=...

# ou Claude (pago)
# export ANTHROPIC_API_KEY=sk-ant-...

python gestorflow_cli.py \
  --context examples/context_exemplo.json \
  --files documentos/oficio.pdf documentos/rg_presidente.jpg \
  --output resultado.md
```

Sem `--provider`, o CLI detecta automaticamente pela API key disponível
no ambiente (Anthropic tem prioridade se as duas existirem). Para forçar
um provedor: `--provider gemini` ou `--provider anthropic`, com
`--model` para sobrescrever o modelo padrão de cada um.

O JSON de contexto segue o schema de `AuditContext`
(ver [`examples/context_exemplo.json`](examples/context_exemplo.json)):
cadastro da proposta, ficha do CNES, ato de nomeação do presidente do CMS,
documento de identificação já auditado, resoluções já usadas, contrato de
abertura de conta, certidão TCE, SIOPE bimestral e as datas de referência
da auditoria.

Use `--mode single_call` para uma chamada só por página (mais barato, mas
mistura classificação e extração) em vez do padrão `two_call` (classifica
e só então extrai com o schema certo).

Para também verificar assinaturas (digital + manuscrita), adicione:

```bash
python gestorflow_cli.py \
  --context examples/context_exemplo.json \
  --files documentos/resolucao_assinada.pdf documentos/oficio_escaneado.pdf \
  --verify-signatures \
  --trust-roots icp_brasil_raizes.pem
```

## Uso programático

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
use `gestorflow_vision.run_vision_pipeline` (sem `reader` explícito, usa
`make_reader()` para escolher o provedor pela API key disponível no
ambiente; requer as dependências opcionais em `requirements.txt`).

## Testes

```bash
python -m unittest test_gestorflow_auditoria.py test_gestorflow_cli.py test_gestorflow_signature.py test_gestorflow_vision.py -v
```

`test_gestorflow_signature.py` gera um certificado autoassinado e um PDF
assinado de verdade (via PyHanko) em um diretório temporário para testar a
inspeção contra uma assinatura digital real, não apenas mocks.

## Instalação

```bash
pip install -r requirements.txt
```

Inclui `pymupdf`/`pillow` (visão computacional), `anthropic` e
`google-genai` (provedores - instale pelo menos um) e `pyHanko`
(verificação de assinatura digital).
