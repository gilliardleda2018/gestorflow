"""
GestorFlow - Verificação de Assinatura Digital em PDF (ICP-Brasil)
=====================================================================

Inspeciona PDFs NATIVOS em busca de assinaturas digitais embutidas
(padrão CAdES/PAdES, via PyHanko) e reporta, para cada assinatura:

  - Integridade: o conteúdo assinado não foi alterado.
  - Validade criptográfica: a assinatura confere com o certificado.
  - Cobertura: se a assinatura cobre o arquivo inteiro, ou se o
    documento sofreu alterações incrementais APÓS ser assinado
    (indício de adulteração).
  - Cadeia de confiança: se um bundle de âncoras de confiança em PEM
    for informado (--trust-roots), valida a cadeia de certificação até
    uma autoridade confiável (ex.: AC-Raiz ICP-Brasil, publicada pelo
    ITI em https://www.iti.gov.br). Sem o bundle, a cadeia fica como
    "não verificada" - a legitimidade fica restrita a integridade e
    validade criptográfica.

IMPORTANTE - escopo:
  - Só se aplica a PDFs nativos assinados eletronicamente (PAdES/CAdES
    embutido no arquivo). Fotos/scans de papel com assinatura
    manuscrita NÃO têm assinatura digital embutida - `inspect_pdf`
    retorna `is_native_pdf=False` nesse caso; use o campo "assinatura"
    já extraído pelo módulo de visão (gestorflow_vision.py) para
    presença visual nesses documentos.
  - A validação de cadeia (ICP-Brasil) depende do bundle de âncoras
    fornecido por quem integra o módulo - este projeto não embute
    nenhum certificado-raiz.

Dependências: pip install pyHanko
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from pyhanko.pdf_utils.reader import PdfFileReader
from pyhanko.sign.validation import validate_pdf_signature
from pyhanko.sign.validation.status import SignatureCoverageLevel
from pyhanko.keys import load_certs_from_pemder
from pyhanko_certvalidator import ValidationContext

from gestorflow_vision import ManualSignatureAssessment, VisionDocumentReader, load_as_images


@dataclass
class SignatureInfo:
    field_name: str
    signer: Optional[str]
    signing_time: Optional[datetime]
    intact: bool
    valid: bool
    covers_whole_document: bool
    trusted: Optional[bool]  # None = nenhum bundle de confianca informado
    erro: Optional[str] = None

    @property
    def legitima(self) -> bool:
        """Integra, criptograficamente valida, cobre o arquivo inteiro
        (nada foi alterado apos a assinatura) e - quando a cadeia foi
        verificada - confiavel."""
        if self.erro:
            return False
        base = self.intact and self.valid and self.covers_whole_document
        if self.trusted is False:
            return False
        return base


@dataclass
class PdfSignatureReport:
    path: str
    is_native_pdf: bool
    has_signatures: bool
    signatures: list[SignatureInfo] = field(default_factory=list)
    erro: Optional[str] = None

    @property
    def legitimo(self) -> bool:
        return (
            self.is_native_pdf
            and self.has_signatures
            and self.erro is None
            and all(s.legitima for s in self.signatures)
        )


def build_trust_context(trust_roots_path: Optional[str]) -> ValidationContext:
    """
    Carrega um bundle PEM com ancoras de confianca (ex.: cadeia da
    AC-Raiz ICP-Brasil) e monta um ValidationContext para validar a
    cadeia de certificacao das assinaturas encontradas.

    Sem `trust_roots_path`, retorna um ValidationContext com
    `trust_roots=[]` (nenhuma ancora) em vez de deixar o PyHanko cair
    no trust store do sistema operacional (comportamento padrao,
    obsoleto e nao confiavel para ICP-Brasil) - a cadeia simplesmente
    nao valida contra nada, o que e o esperado quando nenhum bundle foi
    informado.
    """
    if not trust_roots_path:
        return ValidationContext(trust_roots=[])
    if not os.path.exists(trust_roots_path):
        raise FileNotFoundError(f"bundle de ancoras de confianca nao encontrado: {trust_roots_path}")
    certs = list(load_certs_from_pemder([trust_roots_path]))
    if not certs:
        raise ValueError(f"nenhum certificado valido encontrado em {trust_roots_path}")
    return ValidationContext(trust_roots=certs)


def _signer_label(cert) -> str:
    try:
        native = cert.subject.native
        cn = native.get("common_name")
        org = native.get("organization_name")
        if cn and org:
            return f"{cn} ({org})"
        return cn or cert.subject.human_friendly
    except Exception:
        return cert.subject.human_friendly


def inspect_pdf(pdf_path: str, trust_roots_path: Optional[str] = None) -> PdfSignatureReport:
    """
    Inspeciona um PDF em busca de assinaturas digitais embutidas.
    Nunca lanca excecao - erros (arquivo corrompido, bundle de
    confianca invalido, etc.) sao reportados em `PdfSignatureReport.erro`.
    """
    if not pdf_path.lower().endswith(".pdf"):
        return PdfSignatureReport(
            path=pdf_path, is_native_pdf=False, has_signatures=False,
            erro="nao e um PDF - fotos/scans de papel nao tem assinatura digital embutida",
        )

    try:
        vc = build_trust_context(trust_roots_path)
    except (FileNotFoundError, ValueError) as exc:
        return PdfSignatureReport(path=pdf_path, is_native_pdf=True, has_signatures=False, erro=str(exc))
    verificar_cadeia = bool(trust_roots_path)

    try:
        with open(pdf_path, "rb") as f:
            reader = PdfFileReader(f)
            embedded = list(reader.embedded_signatures)
            if not embedded:
                return PdfSignatureReport(path=pdf_path, is_native_pdf=True, has_signatures=False)

            infos: list[SignatureInfo] = []
            for emb_sig in embedded:
                try:
                    status = validate_pdf_signature(emb_sig, signer_validation_context=vc)
                    infos.append(SignatureInfo(
                        field_name=emb_sig.field_name,
                        signer=_signer_label(status.signing_cert),
                        signing_time=status.signer_reported_dt,
                        intact=bool(status.intact),
                        valid=bool(status.valid),
                        covers_whole_document=(status.coverage == SignatureCoverageLevel.ENTIRE_FILE),
                        trusted=(bool(status.trusted) if verificar_cadeia else None),
                    ))
                except Exception as exc:
                    infos.append(SignatureInfo(
                        field_name=getattr(emb_sig, "field_name", "?"), signer=None, signing_time=None,
                        intact=False, valid=False, covers_whole_document=False,
                        trusted=None, erro=str(exc),
                    ))
            return PdfSignatureReport(path=pdf_path, is_native_pdf=True, has_signatures=True, signatures=infos)
    except Exception as exc:
        return PdfSignatureReport(path=pdf_path, is_native_pdf=True, has_signatures=False, erro=str(exc))


def format_report(report: PdfSignatureReport) -> str:
    linhas = [f"## {os.path.basename(report.path)}"]
    if report.erro:
        linhas.append(f"- ERRO: {report.erro}")
        return "\n".join(linhas)
    if not report.is_native_pdf:
        linhas.append("- Nao aplicavel (nao e PDF nativo assinado digitalmente)")
        return "\n".join(linhas)
    if not report.has_signatures:
        linhas.append("- Nenhuma assinatura digital embutida encontrada")
        return "\n".join(linhas)
    for sig in report.signatures:
        status = "LEGITIMA" if sig.legitima else "SUSPEITA/INVALIDA"
        linhas.append(f"- Campo '{sig.field_name}': {status}")
        linhas.append(f"  - Signatario: {sig.signer or '?'}")
        linhas.append(f"  - Assinado em: {sig.signing_time or '?'}")
        linhas.append(
            f"  - Integro: {sig.intact} | Criptograficamente valido: {sig.valid} "
            f"| Cobre o documento inteiro: {sig.covers_whole_document}"
        )
        if sig.trusted is None:
            linhas.append("  - Cadeia de confianca: NAO VERIFICADA (informe --trust-roots)")
        else:
            linhas.append(f"  - Cadeia de confianca: {'CONFIAVEL' if sig.trusted else 'NAO CONFIAVEL'}")
        if sig.erro:
            linhas.append(f"  - Erro na validacao: {sig.erro}")
    return "\n".join(linhas)


# ---------------------------------------------------------------------------
# Orquestrador - reconhece os dois modelos de assinatura (digital e manual)
# ---------------------------------------------------------------------------

@dataclass
class SignatureCheckResult:
    path: str
    metodo: str  # "digital" | "manual" | "sem_assinatura"
    legitima: Optional[bool]  # None quando nao ha assinatura para avaliar
    digital: Optional[PdfSignatureReport] = None
    manual: Optional[ManualSignatureAssessment] = None
    erro: Optional[str] = None


def check_signature(
    file_path: str,
    reader: Optional[VisionDocumentReader] = None,
    trust_roots_path: Optional[str] = None,
) -> SignatureCheckResult:
    """
    Reconhece os dois modelos de assinatura de um documento:

      1. Assinatura DIGITAL (criptográfica, PAdES/CAdES): se `file_path`
         for um PDF nativo com assinatura embutida, valida via
         `inspect_pdf` (integridade, cadeia de confiança se
         `trust_roots_path` for informado). Método = "digital".
      2. Assinatura MANUSCRITA: caso contrário (imagem, ou PDF sem
         assinatura digital embutida - ex.: scan de papel salvo como
         PDF), usa o modelo de visão para avaliar se há uma assinatura
         a caneta plausível na página. Método = "manual". Requer
         `reader` (VisionDocumentReader) - sem ele, retorna
         metodo="sem_assinatura" com erro explicando a limitação.

    Isso cobre tanto o processo 100% digital (documento gerado e
    assinado eletronicamente) quanto o processo com papel assinado à
    mão e depois digitalizado - os dois fluxos reais deste projeto.
    """
    if file_path.lower().endswith(".pdf"):
        digital_report = inspect_pdf(file_path, trust_roots_path)
        if digital_report.erro and digital_report.is_native_pdf:
            return SignatureCheckResult(path=file_path, metodo="digital", legitima=None, erro=digital_report.erro)
        if digital_report.has_signatures:
            return SignatureCheckResult(
                path=file_path, metodo="digital",
                legitima=digital_report.legitimo,
                digital=digital_report,
            )
        # PDF sem assinatura digital embutida -> pode ser um scan de
        # papel assinado a mao; cai para a avaliacao visual abaixo.

    if reader is None:
        return SignatureCheckResult(
            path=file_path, metodo="sem_assinatura", legitima=None,
            erro="nenhuma assinatura digital embutida encontrada e nenhum VisionDocumentReader "
                 "foi informado para avaliar assinatura manuscrita",
        )

    try:
        paginas = load_as_images(file_path)
    except Exception as exc:
        return SignatureCheckResult(path=file_path, metodo="sem_assinatura", legitima=None, erro=str(exc))

    if not paginas:
        return SignatureCheckResult(path=file_path, metodo="sem_assinatura", legitima=None, erro="documento vazio")

    assessment = reader.assess_manual_signature(paginas[0])
    return SignatureCheckResult(
        path=file_path, metodo="manual",
        legitima=(assessment.legitima if assessment.presente else None),
        manual=assessment,
    )


def format_check_result(result: SignatureCheckResult) -> str:
    if result.digital is not None:
        return format_report(result.digital)
    linhas = [f"## {os.path.basename(result.path)}"]
    if result.erro:
        linhas.append(f"- ERRO: {result.erro}")
    elif result.manual is not None:
        m = result.manual
        status = "LEGITIMA" if m.legitima else ("AUSENTE" if not m.presente else "SUSPEITA")
        linhas.append(f"- Assinatura manuscrita: {status}")
        linhas.append(f"  - Tipo: {m.tipo} | Localizacao: {m.localizacao or '?'}")
        linhas.append(f"  - Aparenta autentica: {m.aparenta_autentica}")
        if m.observacoes:
            linhas.append(f"  - Observacoes: {m.observacoes}")
    else:
        linhas.append("- Nenhuma assinatura encontrada")
    return "\n".join(linhas)
