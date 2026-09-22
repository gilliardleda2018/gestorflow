"""
GestorFlow - Módulo de Classificação e Auditoria de Documentos
================================================================

Estrutura em Python que reproduz o fluxo descrito no PDF
"Prompts de Classificação e Auditoria por Tipo de Documento":

1. Classificador (IA) -> identifica o tipo de documento a partir de indícios.
2. Auditor -> extrai campos por função semântica e aplica regras de
   conformidade específicas de cada tipo de documento.
3. Validador de Cadeia -> aplica a cadeia cronológica de dependência entre
   documentos e propaga invalidação em cascata quando necessário.

Este arquivo contém apenas a ESTRUTURA (dados de configuração + motor de
regras). A extração real de texto de PDFs/imagens e a chamada a um modelo
de linguagem para extrair os campos por função semântica ficam a cargo de
quem integrar este módulo (ex.: plugando uma função `llm_extract_fields`).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
from enum import Enum
from typing import Any, Callable, Optional


# ---------------------------------------------------------------------------
# 1. TIPOS DE DOCUMENTO
# ---------------------------------------------------------------------------

class DocumentType(str, Enum):
    OFICIO_SOLICITACAO_RECURSOS = "Oficio_Solicitacao_Recursos"
    COMPROVANTE_SAEP = "Comprovante_SAEP"
    PLANO_DE_APLICACAO = "Plano_de_Aplicacao"
    RESOLUCAO_PLEITO = "Resolucao_Pleito"
    LEI_CMS = "Lei_CMS"
    LEI_FUNDO_MUNICIPAL_SAUDE = "Lei_Fundo_Municipal_Saude"
    CONTRATO_ABERTURA_CONTA = "Contrato_Abertura_Conta"
    EXTRATO_BANCARIO = "Extrato_Bancario"
    CNPJ_FUNDO_MUNICIPAL_SAUDE = "CNPJ_Fundo_Municipal_Saude"
    CERTIDAO_TCE_LIMITES = "Certidao_TCE_Limites"
    VALIDACAO_CERTIDAO_TCE = "Validacao_Certidao_TCE"
    DECLARACAO_LIMITES = "Declaracao_Limites"
    SIOPS_BIMESTRAL = "SIOPS_Bimestral"
    SIOPE_BIMESTRAL = "SIOPE_Bimestral"
    VALIDACAO_RECIBO_SIOPE = "Validacao_Recibo_SIOPE"
    DECLARACAO_VERACIDADE = "Declaracao_Veracidade"
    DOC_IDENTIFICACAO_PRESIDENTE_CMS = "Documento_Identificacao_Presidente_CMS"
    # Documentos citados na cadeia do apêndice mas sem seção de prompt
    # própria no PDF (ficam com regras mínimas/placeholder):
    NOMEACAO_MEMBROS_CMS = "Nomeacao_Membros_CMS"
    NOMEACAO_PRESIDENTE_CMS = "Nomeacao_Presidente_CMS"
    RESOLUCAO_PAS = "Resolucao_PAS"


# ---------------------------------------------------------------------------
# 2. CLASSIFICADOR - indícios usados para identificar o tipo de documento
# ---------------------------------------------------------------------------

@dataclass
class ClassificationRule:
    document_type: DocumentType
    indicios: list[str]          # descrição textual dos 4 indícios esperados
    keywords: list[str]          # termos-chave usados no matching (heurística)


CLASSIFICATION_RULES: dict[DocumentType, ClassificationRule] = {
    DocumentType.OFICIO_SOLICITACAO_RECURSOS: ClassificationRule(
        DocumentType.OFICIO_SOLICITACAO_RECURSOS,
        indicios=[
            'Numeração no padrão "Ofício nº X/AAAA"',
            'Destinatário formal ("A Sua Excelência" / cargo de autoridade)',
            'Seção "Ref.:" ou assunto explícito de pedido/solicitação de recursos',
            'Fecho "Atenciosamente" + assinatura de autoridade municipal',
        ],
        keywords=["ofício", "a sua excelência", "ref:", "atenciosamente",
                   "solicit", "recursos fundo a fundo"],
    ),
    DocumentType.COMPROVANTE_SAEP: ClassificationRule(
        DocumentType.COMPROVANTE_SAEP,
        indicios=[
            'Cabeçalho/selo "SAEP - Sistema de Acompanhamento de Emendas Parlamentares"',
            'Título "Comprovante de Solicitação de Emenda Parlamentar"',
            "Estrutura de campos rotulados sequenciais",
            'Campo "Status" indicando situação da solicitação',
        ],
        keywords=["saep", "emenda parlamentar", "comprovante de solicitação", "status"],
    ),
    DocumentType.PLANO_DE_APLICACAO: ClassificationRule(
        DocumentType.PLANO_DE_APLICACAO,
        indicios=[
            'Título exato "PLANO DE APLICAÇÃO"',
            "Seções numeradas fixas (Identificação do Proponente, Unidade de Saúde, "
            "Fundamentação Legal, Objetivo, Utilização dos Recursos)",
            'Seção "UNIDADE DE SAÚDE" com lista de unidade(s) + CNES',
            'Fecho "DATA E ASSINATURA DO PROPONENTE" com assinatura de Prefeito(a)',
        ],
        keywords=["plano de aplicação", "unidade de saúde", "cnes", "proponente"],
    ),
    DocumentType.RESOLUCAO_PLEITO: ClassificationRule(
        DocumentType.RESOLUCAO_PLEITO,
        indicios=[
            'Emissão identificada como "Conselho Municipal de Saúde"',
            'Numeração "RESOLUÇÃO CMS Nº [X] de [data]"',
            'Objeto mencionando "pleito" + Secretaria de Estado da Saúde ou '
            '"liberação de recursos Fundo a Fundo"',
            'Fecho "PUBLIQUE-SE E CUMPRA-SE" + assinatura "Presidente do CMS"',
        ],
        keywords=["resolução cms", "pleito", "publique-se e cumpra-se",
                   "presidente do cms"],
    ),
    DocumentType.LEI_CMS: ClassificationRule(
        DocumentType.LEI_CMS,
        indicios=[
            'Numeração "LEI [nº X/AAAA]" ou referência a publicação em D.O.M.',
            'Objeto de instituir/regular/atualizar o "Conselho Municipal de Saúde"',
            "Estrutura legislativa típica (Capítulos/Artigos)",
            "Identificação do município emissor",
        ],
        keywords=["lei nº", "diário oficial", "conselho municipal de saúde",
                   "institui", "regulamenta"],
    ),
    DocumentType.LEI_FUNDO_MUNICIPAL_SAUDE: ClassificationRule(
        DocumentType.LEI_FUNDO_MUNICIPAL_SAUDE,
        indicios=[
            'Numeração "LEI [nº X/AAAA]" ou referência a publicação em D.O.M.',
            'Objeto de instituir/criar/regular o "Fundo Municipal de Saúde"',
            "Estrutura legislativa (vinculação, receitas, despesas, administração)",
            "Identificação do município emissor",
        ],
        keywords=["lei nº", "fundo municipal de saúde", "diário oficial"],
    ),
    DocumentType.CONTRATO_ABERTURA_CONTA: ClassificationRule(
        DocumentType.CONTRATO_ABERTURA_CONTA,
        indicios=[
            "Título de Proposta/Contrato de Abertura de Conta com identidade "
            "visual de instituição bancária",
            'Seções "Contratado" e "Proponente/Contratante"',
            'Tabela "Dirigente(s)" com nome + CPF',
            'Seção "Dados da conta" com agência/conta + assinatura do banco',
        ],
        keywords=["abertura de conta", "contratado", "proponente", "dirigentes",
                   "agência", "conta corrente"],
    ),
    DocumentType.EXTRATO_BANCARIO: ClassificationRule(
        DocumentType.EXTRATO_BANCARIO,
        indicios=[
            'Título "Extrato de Conta Corrente"',
            'Bloco "Cliente - Conta atual" com Agência, Conta e Período',
            'Tabela de "Lançamentos"',
            'Linha de saldo final ("Saldo")',
        ],
        keywords=["extrato de conta corrente", "lançamentos", "saldo"],
    ),
    DocumentType.CNPJ_FUNDO_MUNICIPAL_SAUDE: ClassificationRule(
        DocumentType.CNPJ_FUNDO_MUNICIPAL_SAUDE,
        indicios=[
            'Cabeçalho "REPÚBLICA FEDERATIVA DO BRASIL" + "CNPJ"',
            'Título "COMPROVANTE DE INSCRIÇÃO E DE SITUAÇÃO CADASTRAL"',
            'Campo "NOME EMPRESARIAL" contendo "FUNDO MUNICIPAL DE SAÚDE"',
            'Campo "SITUAÇÃO CADASTRAL"',
        ],
        keywords=["comprovante de inscrição", "situação cadastral",
                   "fundo municipal de saúde", "cnpj"],
    ),
    DocumentType.CERTIDAO_TCE_LIMITES: ClassificationRule(
        DocumentType.CERTIDAO_TCE_LIMITES,
        indicios=[
            'Cabeçalho "TRIBUNAL DE CONTAS DO ESTADO"',
            'Título "CERTIDÃO DE CUMPRIMENTO DOS LIMITES CONSTITUCIONAIS..."',
            "Lista de itens alfabéticos certificando percentuais/LC 101/00",
            'Campos "Data da emissão" e "Válida até"',
        ],
        keywords=["tribunal de contas", "limites constitucionais", "válida até",
                   "lc 101"],
    ),
    DocumentType.VALIDACAO_CERTIDAO_TCE: ClassificationRule(
        DocumentType.VALIDACAO_CERTIDAO_TCE,
        indicios=[
            'Título/contexto "VALIDAR CERTIDÃO DE CONVÊNIOS" ou similar',
            "Campo com código de validação numérico",
            'Mensagem de resultado ("A certidão é autêntica")',
            "Referência ao conteúdo da Certidão TCE validada",
        ],
        keywords=["validar certidão", "código de validação", "autêntica"],
    ),
    DocumentType.DECLARACAO_LIMITES: ClassificationRule(
        DocumentType.DECLARACAO_LIMITES,
        indicios=[
            'Título "DECLARAÇÃO" em cabeçalho de gabinete municipal',
            "Citação do art. 11, parágrafo único, LC 101/00",
            'Citação do art. 25, § 1º, IV, "b" e "c", LC 101/00',
            "Assinatura do Prefeito Municipal",
        ],
        keywords=["declaração", "art. 11", "art. 25", "lc 101", "prefeito"],
    ),
    DocumentType.SIOPS_BIMESTRAL: ClassificationRule(
        DocumentType.SIOPS_BIMESTRAL,
        indicios=[
            'Cabeçalho "Ministério da Saúde" + "SIOPS"',
            'Campo "Período: [ANO] / [Nº] Bimestre"',
            'Tabela "Demonstrativo da Aplicação de Recursos Próprios Municipais..."',
            'Assinatura do "SECRETÁRIO DA SAÚDE"',
        ],
        keywords=["siops", "ministério da saúde", "bimestre", "secretário da saúde"],
    ),
    DocumentType.SIOPE_BIMESTRAL: ClassificationRule(
        DocumentType.SIOPE_BIMESTRAL,
        indicios=[
            'Cabeçalho "Ministério da Educação" + "SIOPE"',
            'Título "RECIBO DE TRANSMISSÃO"',
            'Campo "Período: [ANO] [Nº] Bimestre"',
            'Tabela de "Indicadores constitucionais"',
        ],
        keywords=["siope", "ministério da educação", "recibo de transmissão"],
    ),
    DocumentType.VALIDACAO_RECIBO_SIOPE: ClassificationRule(
        DocumentType.VALIDACAO_RECIBO_SIOPE,
        indicios=[
            'Cabeçalho "FNDE" + logo "SIOPE"',
            'Título "Validar Recibos de Transmissão"',
            'Campos "Nº Recibo - Dígito Verificador" e "Código de Verificação"',
            'Botão/ação "Validar"',
        ],
        keywords=["fnde", "validar recibos", "código de verificação"],
    ),
    DocumentType.DECLARACAO_VERACIDADE: ClassificationRule(
        DocumentType.DECLARACAO_VERACIDADE,
        indicios=[
            'Título "DECLARAÇÃO DE VERACIDADE, LICITUDE E AUTENTICIDADE..."',
            "Corpo declarando veracidade/autenticidade no contexto de Fundo a Fundo",
            "Assunção expressa de responsabilidade legal",
            "Assinatura do Prefeito com CPF",
        ],
        keywords=["veracidade", "licitude", "autenticidade", "responsabilidade legal"],
    ),
    DocumentType.DOC_IDENTIFICACAO_PRESIDENTE_CMS: ClassificationRule(
        DocumentType.DOC_IDENTIFICACAO_PRESIDENTE_CMS,
        indicios=[
            "Foto do titular + campo de nome completo",
            "Número de documento (RG, Passaporte ou Carteira Profissional)",
            'Termo-chave ("Registro Geral"/"RG", "Passaporte", conselho de classe)',
            "Data de nascimento ou filiação",
        ],
        keywords=["registro geral", "passaporte", "crm", "coren", "oab", "crea"],
    ),
}


def classify_document(text: str) -> tuple[Optional[DocumentType], float, list[str]]:
    """
    Heurística simples de classificação por contagem de palavras-chave.
    Em produção, substituir/complementar por uma chamada a um modelo de
    linguagem usando os `indicios` de cada tipo como critério de decisão.

    Retorna (tipo_mais_provavel, score_normalizado, indicios_do_tipo).
    """
    text_low = text.lower()
    best_type: Optional[DocumentType] = None
    best_score = 0.0
    best_indicios: list[str] = []

    for doc_type, rule in CLASSIFICATION_RULES.items():
        hits = sum(1 for kw in rule.keywords if kw in text_low)
        score = hits / len(rule.keywords) if rule.keywords else 0
        if score > best_score:
            best_score = score
            best_type = doc_type
            best_indicios = rule.indicios

    return best_type, best_score, best_indicios


# ---------------------------------------------------------------------------
# 3. AUDITORIA - campos obrigatórios, dependências e regra de conformidade
# ---------------------------------------------------------------------------

@dataclass
class ConformityResult:
    document_label: str
    conforme: bool
    pendencias: list[str] = field(default_factory=list)

    def as_row(self) -> tuple[str, str, str]:
        return (
            self.document_label,
            "CONFORME" if self.conforme else "NAO CONFORME",
            "; ".join(self.pendencias) if self.pendencias else "-",
        )


@dataclass
class AuditSpec:
    document_type: DocumentType
    campos_obrigatorios: list[str]
    depende_de: list[DocumentType] = field(default_factory=list)
    participa_da_cascata: bool = True
    # função de checagem específica (campos extraídos, contexto) -> pendências
    check_fn: Optional[Callable[[dict[str, Any], dict[str, Any]], list[str]]] = None


@dataclass
class AuditContext:
    """
    Schema explícito do `context` usado pelas funções `_check_*`. Antes o
    contexto era um dict solto sem contrato definido; isso deixa claro o
    que precisa ser preenchido para auditar cada tipo de documento, e de
    onde essa informação normalmente vem (cadastro da proposta, atos de
    nomeação, documentos de referência já auditados).

    Todos os campos são opcionais porque cada tipo de documento usa só um
    subconjunto - preencha apenas o que for relevante para os documentos
    que você for auditar.
    """
    # Cadastro da proposta (fonte: sistema de cadastro do processo)
    cadastro_proposta: dict[str, Any] = field(default_factory=dict)
    # Ficha do CNES (fonte: base de referência de unidades de saúde)
    ficha_cnes: dict[str, Any] = field(default_factory=dict)
    # Ato de Nomeação do Presidente do CMS (nome, data_expedicao/publicacao, vigencia_anos)
    ato_nomeacao_presidente: dict[str, Any] = field(default_factory=dict)
    # Documento de Identificação do Presidente do CMS já auditado (nome)
    doc_identificacao_presidente: dict[str, Any] = field(default_factory=dict)
    # Números de Resolução já usados por este município em outras propostas
    resolucoes_ja_usadas: set = field(default_factory=set)
    # Contrato de Abertura de Conta já auditado (agencia, conta)
    contrato_abertura_conta: dict[str, Any] = field(default_factory=dict)
    # Certidão TCE já auditada (codigo_validacao)
    certidao_tce: dict[str, Any] = field(default_factory=dict)
    # SIOPE Bimestral já auditado (numero_recibo, codigo_verificacao)
    siope_bimestral: dict[str, Any] = field(default_factory=dict)
    # Datas de referência da auditoria em curso
    data_referencia_auditoria: Optional[date] = None
    ano_referencia_auditoria: Optional[int] = None

    def as_dict(self) -> dict[str, Any]:
        """Formato que as funções `_check_*` e `audit_document` esperam."""
        return {
            "cadastro_proposta": self.cadastro_proposta,
            "ficha_cnes": self.ficha_cnes,
            "ato_nomeacao_presidente": self.ato_nomeacao_presidente,
            "doc_identificacao_presidente": self.doc_identificacao_presidente,
            "resolucoes_ja_usadas": self.resolucoes_ja_usadas,
            "contrato_abertura_conta": self.contrato_abertura_conta,
            "certidao_tce": self.certidao_tce,
            "siope_bimestral": self.siope_bimestral,
            "data_referencia_auditoria": self.data_referencia_auditoria,
            "ano_referencia_auditoria": self.ano_referencia_auditoria,
        }

    @classmethod
    def from_json(cls, path: str) -> "AuditContext":
        """
        Carrega o contexto de um JSON. Datas devem vir como "YYYY-MM-DD"
        (convertidas automaticamente, inclusive as aninhadas em
        `ato_nomeacao_presidente`, usadas em comparacoes de data por
        `_check_resolucao_pleito`); `resolucoes_ja_usadas` como lista
        (convertida para set).
        """
        import json as _json
        import warnings

        with open(path, encoding="utf-8") as f:
            raw = _json.load(f)

        if raw.get("data_referencia_auditoria"):
            raw["data_referencia_auditoria"] = date.fromisoformat(raw["data_referencia_auditoria"])
        if "resolucoes_ja_usadas" in raw:
            raw["resolucoes_ja_usadas"] = set(raw["resolucoes_ja_usadas"])

        ato_nomeacao = raw.get("ato_nomeacao_presidente")
        if isinstance(ato_nomeacao, dict):
            for campo in ("data_expedicao", "data_publicacao"):
                if ato_nomeacao.get(campo):
                    ato_nomeacao[campo] = date.fromisoformat(ato_nomeacao[campo])

        campos_validos = {f_.name for f_ in cls.__dataclass_fields__.values()}
        desconhecidos = set(raw) - campos_validos
        if desconhecidos:
            warnings.warn(
                f"AuditContext.from_json: chave(s) desconhecida(s) no JSON de contexto, "
                f"ignorada(s) silenciosamente: {sorted(desconhecidos)} (verifique se nao e erro de digitacao)"
            )
        dados = {k: v for k, v in raw.items() if k in campos_validos}
        return cls(**dados)


def _missing_fields(fields_: dict[str, Any], required: list[str]) -> list[str]:
    # Usa `is None` / chave ausente, e não truthiness: valores como 0
    # (ex.: saldo zerado do extrato) ou "" são conteúdo válido, não ausência.
    return [f'campo ausente: "{f}"' for f in required
            if fields_.get(f) is None]


# --- funções de checagem específicas por tipo -------------------------------

def _check_plano_aplicacao(fields_: dict, ctx: dict) -> list[str]:
    pend = []
    cadastro = ctx.get("cadastro_proposta", {})
    for campo in ("valor_recurso", "municipio", "prefeito", "resolucao_cms"):
        if fields_.get(campo) and cadastro.get(campo) and fields_[campo] != cadastro[campo]:
            pend.append(f'"{campo}" divergente do cadastro da proposta')

    # `.get(chave, [])` nao basta: o prompt de extracao instrui o modelo a
    # devolver `null` (nao omitir a chave) quando o campo esta ausente, e
    # `.get()` so aplica o default para chave AUSENTE, nao para valor None
    # explicito - por isso o `or []` extra.
    unidades_doc = {u["cnes"]: u for u in (fields_.get("unidades") or [])}
    unidades_cad = {u["cnes"]: u for u in (cadastro.get("unidades") or [])}
    ficha_cnes = ctx.get("ficha_cnes", {})

    for cnes, unidade in unidades_doc.items():
        if cnes not in unidades_cad and cnes not in ficha_cnes:
            pend.append(f"CNES {cnes} não localizado na Ficha do CNES")
    for cnes in unidades_cad:
        if cnes not in unidades_doc:
            pend.append(f"Unidade {cnes} cadastrada na proposta mas ausente no Plano de Aplicação")
    return pend


def _check_resolucao_pleito(fields_: dict, ctx: dict) -> list[str]:
    pend = []
    cadastro = ctx.get("cadastro_proposta", {})
    for campo in ("numero_resolucao", "data", "municipio", "valor_aprovado"):
        if fields_.get(campo) and cadastro.get(campo) and fields_[campo] != cadastro[campo]:
            pend.append(f'"{campo}" divergente do cadastro da proposta')

    ato_nomeacao = ctx.get("ato_nomeacao_presidente", {})
    doc_identificacao = ctx.get("doc_identificacao_presidente", {})
    signatario = fields_.get("signatario")
    if signatario and ato_nomeacao.get("nome") and signatario != ato_nomeacao["nome"]:
        pend.append("signatario nao corresponde ao presidente nomeado/identificado")
    if signatario and doc_identificacao.get("nome") and signatario != doc_identificacao["nome"]:
        pend.append("signatario nao corresponde ao presidente nomeado/identificado")

    inicio = ato_nomeacao.get("data_expedicao") or ato_nomeacao.get("data_publicacao")
    vigencia_anos = ato_nomeacao.get("vigencia_anos", 2)
    if inicio and fields_.get("data"):
        fim = date(inicio.year + vigencia_anos, inicio.month, inicio.day)
        if not (inicio <= fields_["data"] <= fim):
            pend.append("mandato do presidente expirado/nao iniciado na data da resolucao")

    if fields_.get("numero_resolucao") in ctx.get("resolucoes_ja_usadas", set()):
        pend.append("numero de resolucao ja usado em outra proposta do municipio")
    return pend


def _check_extrato_bancario(fields_: dict, ctx: dict) -> list[str]:
    pend = []
    contrato = ctx.get("contrato_abertura_conta", {})
    if contrato.get("agencia") and fields_.get("agencia") != contrato.get("agencia"):
        pend.append("numero de agencia/conta divergente do Contrato de Abertura de Conta")
    if contrato.get("conta") and fields_.get("conta") != contrato.get("conta"):
        pend.append("numero de agencia/conta divergente do Contrato de Abertura de Conta")
    if fields_.get("saldo") not in (0, 0.0, "0,00", None) and fields_.get("saldo") != 0:
        pend.append("extrato nao apresenta saldo zerado")
    return pend


def _check_cnpj_fundo(fields_: dict, ctx: dict) -> list[str]:
    pend = []
    if fields_.get("situacao_cadastral") and fields_["situacao_cadastral"].upper() != "ATIVA":
        pend.append("CNPJ do Fundo nao esta com situacao cadastral ativa")
    return pend


def _check_certidao_tce(fields_: dict, ctx: dict) -> list[str]:
    pend = []
    ref_date = ctx.get("data_referencia_auditoria")
    if ref_date and fields_.get("valida_ate") and fields_["valida_ate"] < ref_date:
        pend.append("certidao TCE fora do prazo de validade")
    return pend


def _check_validacao_certidao_tce(fields_: dict, ctx: dict) -> list[str]:
    pend = []
    certidao = ctx.get("certidao_tce", {})
    if certidao.get("codigo_validacao") and fields_.get("codigo_validacao") != certidao["codigo_validacao"]:
        pend.append("codigo de validacao divergente da Certidao TCE")
    if fields_.get("resultado", "").lower() not in ("autentica", "autêntica"):
        pend.append("certidao nao confirmada como autentica na validacao")
    return pend


def _check_declaracao_limites(fields_: dict, ctx: dict) -> list[str]:
    pend = []
    ref_date = ctx.get("data_referencia_auditoria")
    if ref_date and fields_.get("data_emissao"):
        if (ref_date - fields_["data_emissao"]) > timedelta(days=30):
            pend.append("Declaracao de Limites emitida ha mais de 30 dias da data da auditoria")
    return pend


def _check_siops(fields_: dict, ctx: dict) -> list[str]:
    pend = []
    ano_ref = ctx.get("ano_referencia_auditoria")
    if ano_ref and fields_.get("ano") and fields_["ano"] != ano_ref:
        pend.append("SIOPS refere-se a exercicio diferente do ano vigente da auditoria")
    return pend


def _check_siope(fields_: dict, ctx: dict) -> list[str]:
    pend = []
    ano_ref = ctx.get("ano_referencia_auditoria")
    if ano_ref and fields_.get("ano") and fields_["ano"] != ano_ref:
        pend.append("SIOPE refere-se a exercicio diferente do ano vigente da auditoria")
    return pend


def _check_validacao_recibo_siope(fields_: dict, ctx: dict) -> list[str]:
    pend = []
    siope = ctx.get("siope_bimestral", {})
    for campo in ("numero_recibo", "codigo_verificacao"):
        if siope.get(campo) and fields_.get(campo) != siope.get(campo):
            pend.append("dados do recibo divergentes do SIOPE Bimestral correspondente")
    return pend


def _check_documento_identificacao_presidente(fields_: dict, ctx: dict) -> list[str]:
    pend = []
    ato = ctx.get("ato_nomeacao_presidente", {})
    cadastro = ctx.get("cadastro_proposta", {})
    nome = fields_.get("nome_titular")
    if nome and ato.get("nome") and nome != ato["nome"]:
        pend.append("titular do documento nao corresponde ao presidente nomeado (Ato de Nomeacao)")
    if nome and cadastro.get("presidente_cms") and nome != cadastro["presidente_cms"]:
        pend.append("titular do documento nao corresponde ao presidente cadastrado")
    return pend


AUDIT_SPECS: dict[DocumentType, AuditSpec] = {
    DocumentType.OFICIO_SOLICITACAO_RECURSOS: AuditSpec(
        DocumentType.OFICIO_SOLICITACAO_RECURSOS,
        campos_obrigatorios=[
            "numero_ano", "data_emissao", "autoridade_destinataria", "valor_solicitado",
            "finalidade", "unidades", "mencao_pas", "resolucao_cms", "assinatura",
        ],
        depende_de=[DocumentType.RESOLUCAO_PLEITO],
    ),
    DocumentType.COMPROVANTE_SAEP: AuditSpec(
        DocumentType.COMPROVANTE_SAEP,
        campos_obrigatorios=[
            "numero_solicitacao", "data_solicitacao", "acao", "municipio_beneficiario",
            "valor_solicitado", "detalhamento_objeto",
        ],
        participa_da_cascata=False,
    ),
    DocumentType.PLANO_DE_APLICACAO: AuditSpec(
        DocumentType.PLANO_DE_APLICACAO,
        campos_obrigatorios=[
            "municipio", "prefeito", "unidades", "resolucao_cms", "objetivo",
            "valor_total", "itens_despesa", "data_assinatura", "assinatura",
        ],
        depende_de=[DocumentType.RESOLUCAO_PLEITO],
        check_fn=_check_plano_aplicacao,
    ),
    DocumentType.RESOLUCAO_PLEITO: AuditSpec(
        DocumentType.RESOLUCAO_PLEITO,
        campos_obrigatorios=[
            "numero_resolucao", "data", "municipio", "objeto", "valor_aprovado",
            "unidades", "signatario", "data_publicacao",
        ],
        depende_de=[DocumentType.RESOLUCAO_PAS],
        check_fn=_check_resolucao_pleito,
    ),
    DocumentType.LEI_CMS: AuditSpec(
        DocumentType.LEI_CMS,
        campos_obrigatorios=["numero_ano_lei", "municipio", "objeto"],
        participa_da_cascata=False,
    ),
    DocumentType.LEI_FUNDO_MUNICIPAL_SAUDE: AuditSpec(
        DocumentType.LEI_FUNDO_MUNICIPAL_SAUDE,
        campos_obrigatorios=["numero_ano_lei", "municipio", "objeto"],
        participa_da_cascata=False,
    ),
    DocumentType.CONTRATO_ABERTURA_CONTA: AuditSpec(
        DocumentType.CONTRATO_ABERTURA_CONTA,
        campos_obrigatorios=["nome_titular", "cnpj_titular", "agencia", "conta", "assinatura_gerente"],
        participa_da_cascata=False,
    ),
    DocumentType.EXTRATO_BANCARIO: AuditSpec(
        DocumentType.EXTRATO_BANCARIO,
        campos_obrigatorios=["agencia", "conta", "saldo"],
        depende_de=[DocumentType.CONTRATO_ABERTURA_CONTA],
        check_fn=_check_extrato_bancario,
        participa_da_cascata=False,
    ),
    DocumentType.CNPJ_FUNDO_MUNICIPAL_SAUDE: AuditSpec(
        DocumentType.CNPJ_FUNDO_MUNICIPAL_SAUDE,
        campos_obrigatorios=["cnpj", "nome_empresarial", "municipio", "situacao_cadastral", "data_situacao"],
        check_fn=_check_cnpj_fundo,
        participa_da_cascata=False,
    ),
    DocumentType.CERTIDAO_TCE_LIMITES: AuditSpec(
        DocumentType.CERTIDAO_TCE_LIMITES,
        campos_obrigatorios=["orgao", "cnpj_orgao", "data_emissao", "valida_ate", "codigo_validacao"],
        check_fn=_check_certidao_tce,
        participa_da_cascata=False,
    ),
    DocumentType.VALIDACAO_CERTIDAO_TCE: AuditSpec(
        DocumentType.VALIDACAO_CERTIDAO_TCE,
        campos_obrigatorios=["codigo_validacao", "resultado", "orgao_cnpj"],
        depende_de=[DocumentType.CERTIDAO_TCE_LIMITES],
        check_fn=_check_validacao_certidao_tce,
        participa_da_cascata=False,
    ),
    DocumentType.DECLARACAO_LIMITES: AuditSpec(
        DocumentType.DECLARACAO_LIMITES,
        campos_obrigatorios=[
            "municipio", "prefeito", "art_11_lc101", "art_25_lc101", "data_emissao", "assinatura",
        ],
        check_fn=_check_declaracao_limites,
        participa_da_cascata=False,  # fora da cadeia, conforme apêndice
    ),
    DocumentType.SIOPS_BIMESTRAL: AuditSpec(
        DocumentType.SIOPS_BIMESTRAL,
        campos_obrigatorios=["municipio", "cnpj_secretaria", "periodo", "data_transmissao", "assinatura_secretario"],
        check_fn=_check_siops,
        participa_da_cascata=False,
    ),
    DocumentType.SIOPE_BIMESTRAL: AuditSpec(
        DocumentType.SIOPE_BIMESTRAL,
        campos_obrigatorios=["municipio", "cnpj", "periodo", "numero_recibo", "codigo_validacao", "data_transmissao"],
        check_fn=_check_siope,
        participa_da_cascata=False,
    ),
    DocumentType.VALIDACAO_RECIBO_SIOPE: AuditSpec(
        DocumentType.VALIDACAO_RECIBO_SIOPE,
        campos_obrigatorios=["municipio_cnpj", "periodo", "numero_recibo", "codigo_verificacao"],
        depende_de=[DocumentType.SIOPE_BIMESTRAL],
        check_fn=_check_validacao_recibo_siope,
        participa_da_cascata=False,
    ),
    DocumentType.DECLARACAO_VERACIDADE: AuditSpec(
        DocumentType.DECLARACAO_VERACIDADE,
        campos_obrigatorios=["municipio", "representante_legal", "trecho_veracidade", "data_emissao", "assinatura_cpf"],
        depende_de=[
            DocumentType.RESOLUCAO_PLEITO,
            DocumentType.PLANO_DE_APLICACAO,
            DocumentType.OFICIO_SOLICITACAO_RECURSOS,
        ],
    ),
    DocumentType.DOC_IDENTIFICACAO_PRESIDENTE_CMS: AuditSpec(
        DocumentType.DOC_IDENTIFICACAO_PRESIDENTE_CMS,
        campos_obrigatorios=["tipo_documento", "nome_titular", "numero_documento", "orgao_emissor"],
        check_fn=_check_documento_identificacao_presidente,
        participa_da_cascata=False,
    ),
    DocumentType.NOMEACAO_MEMBROS_CMS: AuditSpec(
        DocumentType.NOMEACAO_MEMBROS_CMS,
        campos_obrigatorios=["data_nomeacao", "membros", "vigencia"],
    ),
    DocumentType.NOMEACAO_PRESIDENTE_CMS: AuditSpec(
        DocumentType.NOMEACAO_PRESIDENTE_CMS,
        campos_obrigatorios=["nome_presidente", "data_nomeacao"],
        depende_de=[DocumentType.NOMEACAO_MEMBROS_CMS],
    ),
    DocumentType.RESOLUCAO_PAS: AuditSpec(
        DocumentType.RESOLUCAO_PAS,
        campos_obrigatorios=["numero_resolucao", "data"],
        depende_de=[DocumentType.NOMEACAO_PRESIDENTE_CMS],
    ),
}


def audit_document(
    document_type: DocumentType,
    document_label: str,
    extracted_fields: dict[str, Any],
    context: Optional[dict[str, Any]] = None,
) -> ConformityResult:
    """Aplica a regra de conformidade de um tipo de documento."""
    context = context or {}
    spec = AUDIT_SPECS[document_type]

    pendencias = _missing_fields(extracted_fields, spec.campos_obrigatorios)
    if spec.check_fn:
        pendencias += spec.check_fn(extracted_fields, context)

    return ConformityResult(
        document_label=document_label,
        conforme=len(pendencias) == 0,
        pendencias=pendencias,
    )


# ---------------------------------------------------------------------------
# 4. CADEIA CRONOLÓGICA DE VALIDAÇÃO (apêndice do PDF)
# ---------------------------------------------------------------------------

# Grafo de dependência (quem depende de quem), conforme apêndice:
CHAIN_GRAPH: dict[DocumentType, list[DocumentType]] = {
    DocumentType.NOMEACAO_MEMBROS_CMS: [],
    DocumentType.NOMEACAO_PRESIDENTE_CMS: [DocumentType.NOMEACAO_MEMBROS_CMS],
    DocumentType.RESOLUCAO_PAS: [DocumentType.NOMEACAO_PRESIDENTE_CMS],
    DocumentType.RESOLUCAO_PLEITO: [DocumentType.RESOLUCAO_PAS],
    DocumentType.PLANO_DE_APLICACAO: [DocumentType.RESOLUCAO_PLEITO],
    DocumentType.OFICIO_SOLICITACAO_RECURSOS: [DocumentType.RESOLUCAO_PLEITO],
    DocumentType.DECLARACAO_VERACIDADE: [
        DocumentType.NOMEACAO_PRESIDENTE_CMS,
        DocumentType.RESOLUCAO_PAS,
        DocumentType.RESOLUCAO_PLEITO,
        DocumentType.PLANO_DE_APLICACAO,
        DocumentType.OFICIO_SOLICITACAO_RECURSOS,
    ],
    # Declaracao_Limites é FORA da cadeia (documento independente).
}

# Documentos que NÃO participam da cascata mesmo se referenciados
# (ex.: Lei do CMS, Lei do Fundo, Declaração de Limites) — ver AuditSpec.participa_da_cascata.


@dataclass
class DocumentRecord:
    document_type: DocumentType
    label: str
    data_documento: Optional[date] = None
    is_digital: bool = True
    conformity: Optional[ConformityResult] = None
    invalidado_em_cascata: bool = False
    motivo_invalidacao: Optional[str] = None


def _regra_de_ouro(
    data_antecessor: date, digital_antecessor: bool,
    data_sucessor: date, digital_sucessor: bool,
) -> bool:
    """
    Compara datas entre antecessor e sucessor respeitando a "Regra de Ouro":
    - Ambos manuais ou ambos digitais: compara normalmente (igual é válido).
    - Antecessor digital + sucessor manual: antecessor perde 1 minuto (ancora
      no sucessor).
    - Antecessor manual + sucessor digital: sucessor ganha 1 minuto (ancora
      no antecessor).
    Como o modelo aqui trabalha em granularidade de dia (date), a regra dos
    "1 minuto" se traduz em: datas iguais são sempre aceitas como válidas
    (sucessor >= antecessor), independente da combinação manual/digital.
    """
    return data_sucessor >= data_antecessor


def validate_chain(documents: dict[DocumentType, DocumentRecord]) -> dict[DocumentType, DocumentRecord]:
    """
    Aplica, em ordem topológica, as checagens de data e conformidade da
    cadeia cronológica e propaga invalidação em cascata.

    `documents` deve conter os registros já auditados (campo `conformity`
    preenchido) para os tipos presentes no CHAIN_GRAPH.
    """
    ordem = [
        DocumentType.NOMEACAO_MEMBROS_CMS,
        DocumentType.NOMEACAO_PRESIDENTE_CMS,
        DocumentType.RESOLUCAO_PAS,
        DocumentType.RESOLUCAO_PLEITO,
        DocumentType.PLANO_DE_APLICACAO,
        DocumentType.OFICIO_SOLICITACAO_RECURSOS,
        DocumentType.DECLARACAO_VERACIDADE,
    ]

    for doc_type in ordem:
        record = documents.get(doc_type)
        if record is None:
            continue

        predecessores = CHAIN_GRAPH.get(doc_type, [])

        # 1) Se algum predecessor foi invalidado (não conforme ou já invalidado
        #    em cascata), propaga a invalidação.
        for pred_type in predecessores:
            pred = documents.get(pred_type)
            if pred is None:
                continue
            pred_invalido = pred.invalidado_em_cascata or (
                pred.conformity is not None and not pred.conformity.conforme
            )
            if pred_invalido:
                record.invalidado_em_cascata = True
                motivo_pred = pred.motivo_invalidacao or (
                    "; ".join(pred.conformity.pendencias) if pred.conformity else "motivo desconhecido"
                )
                record.motivo_invalidacao = (
                    f"invalidado por cascata: {pred.label} nao conforme - {motivo_pred}"
                )

        # 2) Checagem de data: sucessor >= predecessor (Regra de Ouro).
        if record.data_documento:
            for pred_type in predecessores:
                pred = documents.get(pred_type)
                if pred and pred.data_documento:
                    ok = _regra_de_ouro(
                        pred.data_documento, pred.is_digital,
                        record.data_documento, record.is_digital,
                    )
                    if not ok:
                        record.invalidado_em_cascata = True
                        record.motivo_invalidacao = (
                            f"data anterior à de {pred.label} (viola ordem da cadeia)"
                        )

    # Caso especial: Nomeação do Presidente fora do mandato dos membros ->
    # invalida TODA a cadeia subsequente (regra explícita do apêndice).
    membros = documents.get(DocumentType.NOMEACAO_MEMBROS_CMS)
    presidente = documents.get(DocumentType.NOMEACAO_PRESIDENTE_CMS)
    if membros and presidente and membros.data_documento and presidente.data_documento:
        vigencia = timedelta(days=365 * 2)  # 2 anos, se não especificado
        fim_mandato = membros.data_documento + vigencia
        if not (membros.data_documento <= presidente.data_documento <= fim_mandato):
            for doc_type in ordem[2:]:  # tudo a partir da Resolução da PAS
                rec = documents.get(doc_type)
                if rec:
                    rec.invalidado_em_cascata = True
                    rec.motivo_invalidacao = (
                        "processo integralmente invalidado: presidente do CMS "
                        "ilegítimo (mandato dos membros vencido na data da eleição)"
                    )

    return documents


# ---------------------------------------------------------------------------
# 5. REGRA INDEPENDENTE - Declaração de Limites (fora da cadeia)
# ---------------------------------------------------------------------------

def validate_declaracao_limites(
    record: DocumentRecord, data_referencia_auditoria: date
) -> DocumentRecord:
    """Declaração de Limites: única regra é vigência <= 30 dias, sem cadeia."""
    if record.data_documento and (data_referencia_auditoria - record.data_documento) > timedelta(days=30):
        record.conformity = record.conformity or ConformityResult(record.label, True, [])
        record.conformity.conforme = False
        record.conformity.pendencias.append(
            "Declaracao de Limites emitida ha mais de 30 dias da data da auditoria"
        )
    return record


# ---------------------------------------------------------------------------
# 6. ORQUESTRAÇÃO - pipeline de ponta a ponta
# ---------------------------------------------------------------------------

def run_pipeline(
    raw_documents: list[dict[str, Any]],
    extract_fields_fn: Callable[[str, DocumentType], dict[str, Any]],
    context: dict[str, Any] | AuditContext,
) -> list[tuple[str, str, str]]:
    """
    raw_documents: lista de {"texto": str, "label": str, "data": date|None,
                              "is_digital": bool}
    extract_fields_fn: função (texto, tipo) -> campos extraídos por função
                        semântica (tipicamente uma chamada a um LLM).
    context: dados auxiliares (cadastro da proposta, ficha do CNES, atos de
             nomeação, datas de referência da auditoria etc.)

    Retorna as linhas da tabela final | DOCUMENTO | CONFORMIDADE | PENDENCIAS |
    """
    if isinstance(context, AuditContext):
        context = context.as_dict()

    documents: dict[DocumentType, DocumentRecord] = {}
    rows: list[tuple[str, str, str]] = []

    for raw in raw_documents:
        doc_type, score, _ = classify_document(raw["texto"])
        if doc_type is None:
            rows.append((raw.get("label", "Desconhecido"), "NAO CLASSIFICADO", "-"))
            continue

        fields_ = extract_fields_fn(raw["texto"], doc_type)
        result = audit_document(doc_type, raw.get("label", doc_type.value), fields_, context)

        record = DocumentRecord(
            document_type=doc_type,
            label=result.document_label,
            data_documento=raw.get("data"),
            is_digital=raw.get("is_digital", True),
            conformity=result,
        )
        documents[doc_type] = record

    validate_chain(documents)

    # Nao chama validate_declaracao_limites aqui: a mesma regra dos 30 dias
    # ja roda dentro de audit_document() via _check_declaracao_limites
    # (AUDIT_SPECS[DECLARACAO_LIMITES].check_fn) - chamar as duas duplicava
    # a mesma pendencia duas vezes na mesma linha do relatorio.

    for record in documents.values():
        conforme = record.conformity.conforme if record.conformity else False
        pendencias = list(record.conformity.pendencias) if record.conformity else []
        if record.invalidado_em_cascata:
            conforme = False
            if record.motivo_invalidacao:
                pendencias.append(record.motivo_invalidacao)
        rows.append((
            record.label,
            "CONFORME" if conforme else "NAO CONFORME",
            "; ".join(pendencias) if pendencias else "-",
        ))

    return rows


# ---------------------------------------------------------------------------
# Exemplo mínimo de uso (execução direta do arquivo)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    def fake_extract(texto: str, tipo: DocumentType) -> dict[str, Any]:
        # Placeholder: em produção, chamar um LLM com o prompt de auditoria
        # correspondente ao `tipo` (ver docstrings/AUDIT_SPECS) para extrair
        # os campos por função semântica.
        return {}

    exemplo = [
        {"texto": "OFICIO no 12/2026 A Sua Excelencia... Ref: solicitacao de recursos",
         "label": "Oficio no 12/2026", "data": date(2026, 3, 10), "is_digital": True},
    ]

    contexto = {
        "cadastro_proposta": {"municipio": "Exemplo/MA"},
        "data_referencia_auditoria": date(2026, 9, 21),
        "ano_referencia_auditoria": 2026,
    }

    for linha in run_pipeline(exemplo, fake_extract, contexto):
        print(linha)
