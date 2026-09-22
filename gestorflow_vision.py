"""
GestorFlow - Extensão com Visão Computacional (PDF / Imagem)
==============================================================

Complementa `gestorflow_auditoria.py`: em vez de receber texto já extraído,
este módulo recebe PDFs ou imagens (fotos, scans, capturas de tela de
sistemas como SAEP/SIOPS/SIOPE) e usa um modelo com VISÃO para:

  1. Classificar o tipo de documento (com base nos mesmos indícios de
     CLASSIFICATION_RULES);
  2. Extrair os campos obrigatórios por FUNÇÃO SEMÂNTICA (mesma lógica dos
     prompts de auditoria do PDF original), devolvendo JSON estruturado.

Por que visão em vez de OCR tradicional (Tesseract, etc.)?
  - Muitos documentos têm layout variável entre municípios/bancos (o
    próprio PDF de origem diz isso explicitamente em vários prompts).
  - Há carimbos, assinaturas manuscritas, fotos (RG), tabelas complexas
    (extrato bancário, SIOPS) e telas de sistema (SAEP, validação TCE) -
    um modelo de visão lida melhor com isso do que OCR + regex posicional.
  - A extração "por função semântica, não por posição" pedida nos prompts
    originais é exatamente o que um modelo multimodal faz bem.

Dependências (instalar no ambiente onde este script for executado):
    pip install pymupdf anthropic pillow

Requer uma ANTHROPIC_API_KEY configurada no ambiente para as chamadas
reais ao modelo.
"""

from __future__ import annotations

import base64
import io
import json
import os
import re
from dataclasses import dataclass
from datetime import date
from typing import Any, Optional

from gestorflow_auditoria import (
    AUDIT_SPECS,
    CHAIN_GRAPH,
    CLASSIFICATION_RULES,
    DocumentType,
    DocumentRecord,
    audit_document,
    validate_chain,
    validate_declaracao_limites,
)

# Tipos que validate_chain()/validate_declaracao_limites() esperam
# encontrar no dict `documents` chaveado por DocumentType (um único
# registro por tipo, por design de gestorflow_auditoria.py). Qualquer
# outro tipo pode aparecer mais de uma vez no mesmo lote (ex.: os 6
# bimestres do SIOPS/SIOPE, várias páginas de identidade etc.) e por
# isso NÃO deve ser guardado nesse dict - guardá-lo lá faria cada nova
# ocorrência sobrescrever silenciosamente a anterior.
_TIPOS_DA_CADEIA = (
    set(CHAIN_GRAPH)
    | {dep for deps in CHAIN_GRAPH.values() for dep in deps}
    | {DocumentType.DECLARACAO_LIMITES}
)

# Dependências opcionais - import isolado para permitir rodar só a parte
# de montagem de prompts sem precisar delas instaladas.
try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None

try:
    import anthropic
except ImportError:
    anthropic = None


# ---------------------------------------------------------------------------
# 1. RASTERIZAÇÃO - PDF -> lista de imagens PNG (bytes)
# ---------------------------------------------------------------------------

def pdf_to_images(pdf_path: str, dpi: int = 200) -> list[bytes]:
    """Converte cada página de um PDF em PNG (bytes), pronto para visão."""
    if fitz is None:
        raise RuntimeError("Instale pymupdf: pip install pymupdf")

    images: list[bytes] = []
    doc = fitz.open(pdf_path)
    zoom = dpi / 72
    matrix = fitz.Matrix(zoom, zoom)
    for page in doc:
        pix = page.get_pixmap(matrix=matrix)
        images.append(pix.tobytes("png"))
    doc.close()
    return images


def load_image_bytes(path: str) -> bytes:
    with open(path, "rb") as f:
        return f.read()


def load_as_images(path: str) -> list[bytes]:
    """Entrada única: aceita .pdf (multi-página) ou imagem (.png/.jpg/...)."""
    if path.lower().endswith(".pdf"):
        return pdf_to_images(path)
    return [load_image_bytes(path)]


def _b64(image_bytes: bytes) -> str:
    return base64.b64encode(image_bytes).decode("utf-8")


def _media_type_for(image_bytes: bytes) -> str:
    # Assinatura simples de PNG vs JPEG
    if image_bytes[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    return "image/jpeg"


# ---------------------------------------------------------------------------
# 2. PROMPTS - montados dinamicamente a partir das mesmas regras usadas no
#    classificador/auditor "por texto" (uma única fonte da verdade)
# ---------------------------------------------------------------------------

def build_classification_prompt() -> str:
    tipos_txt = []
    for doc_type, rule in CLASSIFICATION_RULES.items():
        indicios = "\n     - ".join(rule.indicios)
        tipos_txt.append(f'  * "{doc_type.value}":\n     - {indicios}')
    catalogo = "\n".join(tipos_txt)

    return f"""Você é um classificador de documentos de auditoria de repasses
Fundo a Fundo (saúde). Olhe a imagem e identifique qual dos tipos abaixo ela
representa, com base nos indícios visuais e textuais de cada um.

Tipos possíveis e seus indícios:
{catalogo}

Responda APENAS em JSON, sem markdown, sem texto adicional, no formato:
{{"tipo": "<valor exato de um dos tipos acima>", "confianca": <0.0 a 1.0>,
  "indicios_encontrados": ["..."]}}

Se a imagem não corresponder a nenhum tipo com confiança razoável, use
"tipo": null.
"""


def build_combined_prompt() -> str:
    """
    Prompt de chamada única: classifica E extrai os campos no mesmo turno.
    Mais barato (1 chamada em vez de 2), mas mistura as duas tarefas -
    se a classificação estiver errada, os campos extraídos seguem o
    schema do tipo errado. Prefira `build_classification_prompt` +
    `build_extraction_prompt` (2 chamadas) quando precisão importar mais
    que custo.
    """
    catalogo_tipos = "\n".join(f'  - "{t.value}"' for t in CLASSIFICATION_RULES)
    schemas = []
    for doc_type, spec in AUDIT_SPECS.items():
        campos = ", ".join(f'"{c}"' for c in spec.campos_obrigatorios)
        schemas.append(f'  "{doc_type.value}": {{{campos}}}')
    schemas_txt = "\n".join(schemas)

    return f"""Você é um classificador e extrator de documentos de auditoria de
repasses Fundo a Fundo (saúde). Olhe a imagem, identifique o tipo de
documento entre:
{catalogo_tipos}

Depois, extraia os campos correspondentes ao tipo identificado, por
FUNÇÃO SEMÂNTICA (não por posição fixa), seguindo o schema de campos de
cada tipo:
{schemas_txt}

Regras de extração:
  - Campos ausentes na imagem: null (não invente valores).
  - Listas (ex.: unidades de saúde): lista de objetos com os subcampos
    relevantes (ex.: nome + CNES).
  - Datas em "YYYY-MM-DD"; valores monetários como número (float).

Responda APENAS em JSON, sem markdown, no formato:
{{"tipo": "<um dos tipos acima ou null>", "confianca": <0.0 a 1.0>,
  "campos": {{...os campos do schema do tipo identificado...}}}}
"""


def build_extraction_prompt(doc_type: DocumentType) -> str:
    spec = AUDIT_SPECS[doc_type]
    campos = "\n".join(f'  - "{c}"' for c in spec.campos_obrigatorios)

    return f"""Você está auditando um documento do tipo "{doc_type.value}".
Extraia os campos abaixo por FUNÇÃO SEMÂNTICA (o que o campo representa),
e não pela posição fixa no layout - a redação/diagramação varia entre
municípios, bancos ou sistemas.

Campos a extrair:
{campos}

Regras:
  - Se um campo for uma LISTA (ex.: unidades de saúde), retorne uma lista de
    objetos, cada um com os subcampos relevantes (ex.: nome + CNES).
  - Se um campo não estiver presente na imagem, retorne null para ele -
    NÃO invente valores.
  - Datas no formato "YYYY-MM-DD". Valores monetários como número (float).

Responda APENAS em JSON, sem markdown, sem texto adicional, com uma chave
por campo listado acima.
"""


def build_manual_signature_prompt() -> str:
    return """Você está avaliando se ESTE documento contém uma assinatura
MANUSCRITA (feita a mão, com caneta) de próprio punho, no campo onde uma
assinatura é esperada.

Responda APENAS em JSON, sem markdown, sem texto adicional, no formato:
{"assinatura_presente": true/false,
 "tipo": "manuscrita" | "digitalizada_sem_tinta" | "carimbo" | "ausente",
 "localizacao": "descricao breve de onde esta no documento, ou null",
 "aparenta_autentica": true/false/null,
 "observacoes": "indicios de rasura, corte, colagem ou montagem, ou null"}

Regras:
  - "manuscrita": tracos de caneta/tinta, variacao natural de pressao e
    espessura, assinatura unica e organica.
  - "digitalizada_sem_tinta": parece uma imagem/recorte colado (ex.: a
    mesma assinatura repetida de forma identica em varios documentos).
  - "carimbo": carimbo de "assinado digitalmente"/similar, sem
    assinatura manuscrita visivel.
  - "ausente": nao ha assinatura no campo esperado.
  - aparenta_autentica=false APENAS diante de indicio visual concreto de
    adulteracao (corte, colagem, proporcao distorcida, fundo
    inconsistente) - nao avalie caligrafia ou estilo pessoal, apenas
    indicios tecnicos de montagem de imagem.
  - aparenta_autentica=null quando nao houver como avaliar (ex.: campo
    "ausente").
"""


_ISO_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def _coerce_dates(obj: Any) -> Any:
    """
    Converte recursivamente strings no formato "YYYY-MM-DD" (o formato
    pedido nos prompts de extração) em `datetime.date`, para que os
    campos extraídos pela visão sejam compatíveis com as comparações de
    data feitas pelas funções `_check_*` de gestorflow_auditoria.py
    (que esperam `date`, não `str`).
    """
    if isinstance(obj, str):
        if _ISO_DATE_RE.match(obj):
            try:
                return date.fromisoformat(obj)
            except ValueError:
                return obj
        return obj
    if isinstance(obj, list):
        return [_coerce_dates(item) for item in obj]
    if isinstance(obj, dict):
        return {k: _coerce_dates(v) for k, v in obj.items()}
    return obj


def _safe_document_type(tipo_str: Optional[str]) -> Optional[DocumentType]:
    """
    Constrói um DocumentType a partir da saída (nao confiavel) do
    modelo. Um valor que nao corresponde a nenhum tipo conhecido
    (alucinacao, erro de digitacao, variacao de caixa) vira None em vez
    de lancar excecao - o chamador trata isso como "NAO CLASSIFICADO",
    igual ao caso de confianca baixa, em vez de derrubar o lote inteiro.
    """
    if not tipo_str:
        return None
    try:
        return DocumentType(tipo_str)
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# 3. CHAMADA AO MODELO COM VISÃO
# ---------------------------------------------------------------------------

@dataclass
class VisionClassification:
    document_type: Optional[DocumentType]
    confidence: float
    indicios_encontrados: list[str]


@dataclass
class ManualSignatureAssessment:
    """Avaliação de assinatura MANUSCRITA feita pelo modelo de visão,
    para documentos sem assinatura digital embutida (fotos/scans de
    papel). Complementa - não substitui - a verificação criptográfica
    de `gestorflow_signature.py`, que só se aplica a PDFs nativos
    assinados eletronicamente."""

    presente: bool
    tipo: str  # "manuscrita" | "digitalizada_sem_tinta" | "carimbo" | "ausente"
    localizacao: Optional[str]
    aparenta_autentica: Optional[bool]
    observacoes: Optional[str]

    @property
    def legitima(self) -> bool:
        """Presente, do tipo manuscrita e sem indício visual de adulteração."""
        return self.presente and self.tipo == "manuscrita" and self.aparenta_autentica is not False


class VisionDocumentReader:
    """
    Encapsula as chamadas ao modelo com visão. Usa a API da Anthropic
    (Claude) por padrão; troque `_call_vision` para usar outro provedor
    multimodal se preferir.
    """

    def __init__(self, api_key: Optional[str] = None, model: str = "claude-sonnet-4-6"):
        if anthropic is None:
            raise RuntimeError("Instale o SDK: pip install anthropic")
        self.client = anthropic.Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))
        self.model = model

    def _call_vision(self, image_bytes: bytes, prompt: str) -> str:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=1500,
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": _media_type_for(image_bytes),
                            "data": _b64(image_bytes),
                        },
                    },
                    {"type": "text", "text": prompt},
                ],
            }],
        )
        return "".join(block.text for block in response.content if block.type == "text")

    @staticmethod
    def _parse_json(raw: str) -> dict[str, Any]:
        cleaned = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        return json.loads(cleaned)

    def classify(self, image_bytes: bytes) -> VisionClassification:
        raw = self._call_vision(image_bytes, build_classification_prompt())
        data = self._parse_json(raw)
        doc_type = _safe_document_type(data.get("tipo"))
        return VisionClassification(
            document_type=doc_type,
            confidence=float(data.get("confianca", 0.0)),
            indicios_encontrados=data.get("indicios_encontrados", []),
        )

    def extract_fields(self, image_bytes: bytes, doc_type: DocumentType) -> dict[str, Any]:
        raw = self._call_vision(image_bytes, build_extraction_prompt(doc_type))
        return _coerce_dates(self._parse_json(raw))

    def assess_manual_signature(self, image_bytes: bytes) -> ManualSignatureAssessment:
        """Avalia se a imagem mostra uma assinatura manuscrita legítima.
        Use para documentos sem assinatura digital embutida (fotos/scans
        de papel) - ver `gestorflow_signature.check_signature`, que
        decide automaticamente entre este método e a verificação
        criptográfica conforme o tipo de arquivo."""
        raw = self._call_vision(image_bytes, build_manual_signature_prompt())
        data = self._parse_json(raw)
        return ManualSignatureAssessment(
            presente=bool(data.get("assinatura_presente", False)),
            tipo=data.get("tipo") or "ausente",
            localizacao=data.get("localizacao"),
            aparenta_autentica=data.get("aparenta_autentica"),
            observacoes=data.get("observacoes"),
        )

    def classify_and_extract(self, image_bytes: bytes) -> tuple[VisionClassification, dict[str, Any]]:
        """Chamada única: classifica e extrai os campos no mesmo turno."""
        raw = self._call_vision(image_bytes, build_combined_prompt())
        data = self._parse_json(raw)
        doc_type = _safe_document_type(data.get("tipo"))
        classification = VisionClassification(
            document_type=doc_type,
            confidence=float(data.get("confianca", 0.0)),
            indicios_encontrados=[],
        )
        return classification, _coerce_dates(data.get("campos", {}))


# ---------------------------------------------------------------------------
# 4. PIPELINE PONTA A PONTA - arquivos (PDF/imagem) -> tabela de auditoria
# ---------------------------------------------------------------------------

def run_vision_pipeline(
    file_paths: list[str],
    context: dict[str, Any],
    reader: Optional[VisionDocumentReader] = None,
    min_confidence: float = 0.5,
    mode: str = "two_call",
) -> list[tuple[str, str, str]]:
    """
    file_paths: lista de caminhos de PDF ou imagem (um documento por
                arquivo; PDFs multi-página são tratados página a página,
                cada página classificada individualmente).
    context: mesmo dicionário usado em `gestorflow_auditoria.run_pipeline`
             (cadastro da proposta, ficha do CNES, datas de referência etc.)
    mode: "two_call" (padrão, mais preciso: classifica e só então extrai
          com o schema certo) ou "single_call" (1 chamada só, mais barato,
          mas classificação e extração saem do mesmo turno do modelo).
    """
    if mode not in ("two_call", "single_call"):
        raise ValueError('mode deve ser "two_call" ou "single_call"')

    reader = reader or VisionDocumentReader()
    documents: dict[DocumentType, DocumentRecord] = {}
    avulsos: list[DocumentRecord] = []
    rows: list[tuple[str, str, str]] = []

    for path in file_paths:
        for page_num, image_bytes in enumerate(load_as_images(path), start=1):
            label_base = os.path.basename(path) + (f" (pág. {page_num})" if path.lower().endswith(".pdf") else "")

            if mode == "single_call":
                classification, fields_ = reader.classify_and_extract(image_bytes)
            else:
                classification = reader.classify(image_bytes)
                fields_ = {}

            if classification.document_type is None or classification.confidence < min_confidence:
                rows.append((label_base, "NAO CLASSIFICADO",
                             f"confianca={classification.confidence:.2f}"))
                continue

            doc_type = classification.document_type
            if mode == "two_call":
                fields_ = reader.extract_fields(image_bytes, doc_type)
            result = audit_document(doc_type, label_base, fields_, context)

            record = DocumentRecord(
                document_type=doc_type,
                label=result.document_label,
                data_documento=_parse_date_field(fields_),
                is_digital=True,
                conformity=result,
            )
            if doc_type in _TIPOS_DA_CADEIA:
                documents[doc_type] = record
            else:
                avulsos.append(record)

    validate_chain(documents)

    if DocumentType.DECLARACAO_LIMITES in documents and context.get("data_referencia_auditoria"):
        validate_declaracao_limites(
            documents[DocumentType.DECLARACAO_LIMITES],
            context["data_referencia_auditoria"],
        )

    for record in list(documents.values()) + avulsos:
        conforme = record.conformity.conforme if record.conformity else False
        pendencias = list(record.conformity.pendencias) if record.conformity else []
        if record.invalidado_em_cascata:
            conforme = False
            if record.motivo_invalidacao:
                pendencias.append(record.motivo_invalidacao)
        rows.append((record.label, "CONFORME" if conforme else "NAO CONFORME",
                     "; ".join(pendencias) if pendencias else "-"))

    return rows


def _parse_date_field(fields_: dict[str, Any]):
    """Tenta achar um campo de data plausível entre os extraídos, para uso
    na validação da cadeia cronológica (best-effort)."""
    from datetime import date
    for key in ("data", "data_emissao", "data_documento", "data_solicitacao",
                "data_assinatura", "data_publicacao"):
        val = fields_.get(key)
        if val:
            try:
                return date.fromisoformat(val)
            except (ValueError, TypeError):
                continue
    return None


# ---------------------------------------------------------------------------
# Exemplo de uso
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    arquivos = [
        "/caminho/oficio_processo123.pdf",
        "/caminho/rg_presidente_cms.jpg",
    ]
    contexto = {
        "cadastro_proposta": {"municipio": "Exemplo/MA"},
        "data_referencia_auditoria": "2026-09-21",
    }
    for linha in run_vision_pipeline(arquivos, contexto):
        print(linha)
