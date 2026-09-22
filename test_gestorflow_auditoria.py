"""
Testes unitários das regras de conformidade (`_check_*`) e da cadeia
cronológica de validação do GestorFlow.

Rodar com:
    python3 -m unittest test_gestorflow_auditoria.py -v
"""

from __future__ import annotations

import os
import unittest
import warnings
from datetime import date, timedelta

from gestorflow_auditoria import (
    AuditContext,
    DocumentRecord,
    DocumentType,
    audit_document,
    validate_chain,
    validate_declaracao_limites,
)

EXEMPLO_CONTEXT = os.path.join(os.path.dirname(__file__), "examples", "context_exemplo.json")


class TestPlanoDeAplicacao(unittest.TestCase):
    def _ctx(self, **overrides):
        base = AuditContext(
            cadastro_proposta={
                "municipio": "Exemplo/MA", "prefeito": "Fulano", "valor_recurso": 10000.0,
                "resolucao_cms": "01/2026",
                "unidades": [{"cnes": "1111111"}, {"cnes": "2222222"}],
            },
            ficha_cnes={},
        )
        for k, v in overrides.items():
            setattr(base, k, v)
        return base.as_dict()

    def _fields(self, **overrides):
        base = dict(
            municipio="Exemplo/MA", prefeito="Fulano", resolucao_cms="01/2026",
            objetivo="Custeio", valor_total=10000.0, itens_despesa=["item"],
            data_assinatura="2026-04-01", assinatura="Fulano",
            unidades=[{"cnes": "1111111"}, {"cnes": "2222222"}],
        )
        base.update(overrides)
        return base

    def test_conforme_quando_tudo_bate_com_cadastro(self):
        r = audit_document(DocumentType.PLANO_DE_APLICACAO, "PA-1", self._fields(), self._ctx())
        self.assertTrue(r.conforme, r.pendencias)

    def test_nao_conforme_unidade_extra_sem_cnes_valido(self):
        fields_ = self._fields(unidades=[{"cnes": "1111111"}, {"cnes": "9999999"}])
        r = audit_document(DocumentType.PLANO_DE_APLICACAO, "PA-2", fields_, self._ctx())
        self.assertFalse(r.conforme)
        self.assertTrue(any("9999999" in p for p in r.pendencias))

    def test_nao_conforme_unidade_do_cadastro_ausente_no_documento(self):
        fields_ = self._fields(unidades=[{"cnes": "1111111"}])  # falta 2222222
        r = audit_document(DocumentType.PLANO_DE_APLICACAO, "PA-3", fields_, self._ctx())
        self.assertFalse(r.conforme)
        self.assertTrue(any("2222222" in p for p in r.pendencias))


class TestResolucaoPleito(unittest.TestCase):
    def _ctx(self):
        return AuditContext(
            cadastro_proposta={"numero_resolucao": "05/2026", "municipio": "Exemplo/MA", "valor_aprovado": 5000.0},
            ato_nomeacao_presidente={
                "nome": "Presidente X",
                "data_expedicao": date(2025, 1, 1),
                "vigencia_anos": 2,
            },
            doc_identificacao_presidente={"nome": "Presidente X"},
            resolucoes_ja_usadas=set(),
        ).as_dict()

    def _fields(self, **overrides):
        base = dict(
            numero_resolucao="05/2026", data=date(2026, 3, 1), municipio="Exemplo/MA",
            objeto="pleito", valor_aprovado=5000.0, unidades=[],
            signatario="Presidente X", data_publicacao="2026-03-02",
        )
        base.update(overrides)
        return base

    def test_conforme(self):
        r = audit_document(DocumentType.RESOLUCAO_PLEITO, "RP-1", self._fields(), self._ctx())
        self.assertTrue(r.conforme, r.pendencias)

    def test_nao_conforme_mandato_expirado(self):
        fields_ = self._fields(data=date(2027, 6, 1))  # fora dos 2 anos de vigência
        r = audit_document(DocumentType.RESOLUCAO_PLEITO, "RP-2", fields_, self._ctx())
        self.assertFalse(r.conforme)
        self.assertTrue(any("mandato" in p for p in r.pendencias))

    def test_nao_conforme_signatario_divergente(self):
        fields_ = self._fields(signatario="Outra Pessoa")
        r = audit_document(DocumentType.RESOLUCAO_PLEITO, "RP-3", fields_, self._ctx())
        self.assertFalse(r.conforme)
        self.assertTrue(any("signatario" in p for p in r.pendencias))

    def test_nao_conforme_numero_resolucao_repetido(self):
        ctx = AuditContext(
            cadastro_proposta={"numero_resolucao": "05/2026", "municipio": "Exemplo/MA", "valor_aprovado": 5000.0},
            resolucoes_ja_usadas={"05/2026"},
        ).as_dict()
        r = audit_document(DocumentType.RESOLUCAO_PLEITO, "RP-4", self._fields(), ctx)
        self.assertFalse(r.conforme)
        self.assertTrue(any("ja usado" in p for p in r.pendencias))


class TestExtratoBancario(unittest.TestCase):
    def _ctx(self):
        return AuditContext(
            contrato_abertura_conta={"agencia": "1234", "conta": "56789-0"}
        ).as_dict()

    def test_conforme_saldo_zerado_e_conta_igual(self):
        fields_ = dict(agencia="1234", conta="56789-0", saldo=0)
        r = audit_document(DocumentType.EXTRATO_BANCARIO, "EX-1", fields_, self._ctx())
        self.assertTrue(r.conforme, r.pendencias)

    def test_nao_conforme_saldo_nao_zerado(self):
        fields_ = dict(agencia="1234", conta="56789-0", saldo=150.0)
        r = audit_document(DocumentType.EXTRATO_BANCARIO, "EX-2", fields_, self._ctx())
        self.assertFalse(r.conforme)
        self.assertTrue(any("saldo zerado" in p for p in r.pendencias))

    def test_nao_conforme_agencia_divergente(self):
        fields_ = dict(agencia="0000", conta="56789-0", saldo=0)
        r = audit_document(DocumentType.EXTRATO_BANCARIO, "EX-3", fields_, self._ctx())
        self.assertFalse(r.conforme)
        self.assertTrue(any("agencia/conta divergente" in p for p in r.pendencias))


class TestCnpjFundo(unittest.TestCase):
    def _fields(self, situacao="ATIVA"):
        return dict(
            cnpj="00.000.000/0001-00", nome_empresarial="FUNDO MUNICIPAL DE SAUDE DE EXEMPLO",
            municipio="Exemplo/MA", situacao_cadastral=situacao, data_situacao="2026-01-01",
        )

    def test_conforme_ativa(self):
        r = audit_document(DocumentType.CNPJ_FUNDO_MUNICIPAL_SAUDE, "CNPJ-1", self._fields(), {})
        self.assertTrue(r.conforme, r.pendencias)

    def test_nao_conforme_suspensa(self):
        r = audit_document(DocumentType.CNPJ_FUNDO_MUNICIPAL_SAUDE, "CNPJ-2", self._fields("SUSPENSA"), {})
        self.assertFalse(r.conforme)
        self.assertTrue(any("ativa" in p for p in r.pendencias))


class TestCertidaoTCE(unittest.TestCase):
    def _fields(self, valida_ate):
        return dict(
            orgao="Prefeitura de Exemplo", cnpj_orgao="00.000.000/0001-00",
            data_emissao="2026-01-01", valida_ate=valida_ate, codigo_validacao="ABC123",
        )

    def test_conforme_dentro_da_validade(self):
        ctx = {"data_referencia_auditoria": date(2026, 6, 1)}
        r = audit_document(DocumentType.CERTIDAO_TCE_LIMITES, "TCE-1",
                            self._fields(date(2026, 12, 31)), ctx)
        self.assertTrue(r.conforme, r.pendencias)

    def test_nao_conforme_vencida(self):
        ctx = {"data_referencia_auditoria": date(2026, 6, 1)}
        r = audit_document(DocumentType.CERTIDAO_TCE_LIMITES, "TCE-2",
                            self._fields(date(2026, 1, 1)), ctx)
        self.assertFalse(r.conforme)
        self.assertTrue(any("fora do prazo" in p for p in r.pendencias))


class TestDeclaracaoLimites(unittest.TestCase):
    def _fields(self, data_emissao):
        return dict(
            municipio="Exemplo/MA", prefeito="Fulano",
            art_11_lc101="cumprido", art_25_lc101="cumprido",
            data_emissao=data_emissao, assinatura="Fulano",
        )

    def test_conforme_dentro_de_30_dias(self):
        ctx = {"data_referencia_auditoria": date(2026, 6, 15)}
        r = audit_document(DocumentType.DECLARACAO_LIMITES, "DL-1",
                            self._fields(date(2026, 6, 1)), ctx)
        self.assertTrue(r.conforme, r.pendencias)

    def test_nao_conforme_fora_de_30_dias(self):
        ctx = {"data_referencia_auditoria": date(2026, 6, 15)}
        r = audit_document(DocumentType.DECLARACAO_LIMITES, "DL-2",
                            self._fields(date(2026, 1, 1)), ctx)
        self.assertFalse(r.conforme)
        self.assertTrue(any("mais de 30 dias" in p for p in r.pendencias))


class TestCadeiaCronologica(unittest.TestCase):
    def test_cascata_invalida_dependentes(self):
        from gestorflow_auditoria import ConformityResult

        pleito_invalido = DocumentRecord(
            document_type=DocumentType.RESOLUCAO_PLEITO, label="Resolucao do Pleito 05/2026",
            data_documento=date(2026, 3, 1),
            conformity=ConformityResult("Resolucao do Pleito 05/2026", False, ["mandato expirado"]),
        )
        plano = DocumentRecord(
            document_type=DocumentType.PLANO_DE_APLICACAO, label="Plano de Aplicacao",
            data_documento=date(2026, 3, 5),
            conformity=ConformityResult("Plano de Aplicacao", True, []),
        )
        docs = {
            DocumentType.RESOLUCAO_PLEITO: pleito_invalido,
            DocumentType.PLANO_DE_APLICACAO: plano,
        }
        validate_chain(docs)
        self.assertTrue(plano.invalidado_em_cascata)
        self.assertIn("Resolucao do Pleito", plano.motivo_invalidacao)

    def test_data_fora_de_ordem_invalida(self):
        from gestorflow_auditoria import ConformityResult

        pleito = DocumentRecord(
            document_type=DocumentType.RESOLUCAO_PLEITO, label="Resolucao do Pleito",
            data_documento=date(2026, 5, 1),
            conformity=ConformityResult("Resolucao do Pleito", True, []),
        )
        oficio_antes = DocumentRecord(
            document_type=DocumentType.OFICIO_SOLICITACAO_RECURSOS, label="Oficio",
            data_documento=date(2026, 4, 1),  # anterior ao Pleito -> viola ordem
            conformity=ConformityResult("Oficio", True, []),
        )
        docs = {
            DocumentType.RESOLUCAO_PLEITO: pleito,
            DocumentType.OFICIO_SOLICITACAO_RECURSOS: oficio_antes,
        }
        validate_chain(docs)
        self.assertTrue(oficio_antes.invalidado_em_cascata)


class TestDeclaracaoLimitesForaDaCadeia(unittest.TestCase):
    def test_validate_declaracao_limites_marca_nao_conforme(self):
        from gestorflow_auditoria import ConformityResult

        record = DocumentRecord(
            document_type=DocumentType.DECLARACAO_LIMITES, label="Declaracao de Limites",
            data_documento=date(2026, 1, 1),
            conformity=ConformityResult("Declaracao de Limites", True, []),
        )
        validate_declaracao_limites(record, date(2026, 6, 15))
        self.assertFalse(record.conformity.conforme)


class TestAuditContextFromJson(unittest.TestCase):
    def test_datas_aninhadas_em_ato_nomeacao_viram_date(self):
        # examples/context_exemplo.json tem ato_nomeacao_presidente.data_expedicao
        # como string "YYYY-MM-DD" - from_json precisa converter para date,
        # senao _check_resolucao_pleito quebra em `inicio.year`.
        ctx = AuditContext.from_json(EXEMPLO_CONTEXT)
        self.assertIsInstance(ctx.ato_nomeacao_presidente["data_expedicao"], date)
        self.assertEqual(ctx.ato_nomeacao_presidente["data_expedicao"], date(2025, 1, 15))

    def test_resolucao_pleito_nao_quebra_com_contexto_de_exemplo(self):
        ctx = AuditContext.from_json(EXEMPLO_CONTEXT).as_dict()
        fields_ = dict(
            numero_resolucao="01/2026", data=date(2026, 3, 1), municipio="Exemplo/MA",
            objeto="pleito", valor_aprovado=10000.0, unidades=[],
            signatario="Ciclana da Silva", data_publicacao="2026-03-02",
        )
        r = audit_document(DocumentType.RESOLUCAO_PLEITO, "RP-1", fields_, ctx)
        self.assertTrue(r.conforme, r.pendencias)

    def test_chave_desconhecida_no_json_gera_aviso(self):
        import json
        import tempfile

        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as f:
            json.dump({"municipio_esperado": "Exemplo/MA"}, f)
            caminho = f.name
        try:
            with warnings.catch_warnings(record=True) as registrados:
                warnings.simplefilter("always")
                AuditContext.from_json(caminho)
            self.assertTrue(any("municipio_esperado" in str(w.message) for w in registrados))
        finally:
            os.remove(caminho)


if __name__ == "__main__":
    unittest.main()
