"""
Testes de `gestorflow_vision.py` que não fazem chamadas reais à API
(usam apenas as funções puras/helpers e um VisionDocumentReader mockado).

Cobrem especificamente os bugs de integração encontrados na revisão:
  - Datas extraídas pela visão como string quebrando comparações de
    `date` nas funções `_check_*` (gestorflow_auditoria.py).
  - `DocumentType(tipo_str)` derrubando o pipeline inteiro quando o
    modelo devolve um tipo que não bate com nenhum enum conhecido.
  - Documentos repetíveis do mesmo tipo (ex.: bimestres do SIOPS/SIOPE)
    se sobrescrevendo silenciosamente no dict interno do pipeline.
"""

from __future__ import annotations

import os
import unittest
from datetime import date
from unittest.mock import MagicMock, patch

from gestorflow_auditoria import DocumentType
from gestorflow_vision import (
    AnthropicVisionDocumentReader,
    GeminiVisionDocumentReader,
    _coerce_dates,
    _parse_date_field,
    _safe_document_type,
    make_reader,
    run_vision_pipeline,
)


class TestCoerceDates(unittest.TestCase):
    def test_converte_string_iso_para_date(self):
        self.assertEqual(_coerce_dates("2026-09-21"), date(2026, 9, 21))

    def test_ignora_string_que_nao_e_data(self):
        self.assertEqual(_coerce_dates("Fulano de Tal"), "Fulano de Tal")

    def test_converte_recursivamente_em_dict_e_lista(self):
        entrada = {
            "data_emissao": "2026-01-01",
            "unidades": [{"cnes": "111", "data_cadastro": "2025-05-10"}],
            "valor": 1000.0,
        }
        saida = _coerce_dates(entrada)
        self.assertEqual(saida["data_emissao"], date(2026, 1, 1))
        self.assertEqual(saida["unidades"][0]["data_cadastro"], date(2025, 5, 10))
        self.assertEqual(saida["valor"], 1000.0)

    def test_preserva_none_e_outros_tipos(self):
        self.assertIsNone(_coerce_dates(None))
        self.assertEqual(_coerce_dates(42), 42)
        self.assertEqual(_coerce_dates(True), True)


class TestSafeDocumentType(unittest.TestCase):
    def test_tipo_valido(self):
        self.assertEqual(
            _safe_document_type("Oficio_Solicitacao_Recursos"),
            DocumentType.OFICIO_SOLICITACAO_RECURSOS,
        )

    def test_tipo_invalido_vira_none_em_vez_de_lancar(self):
        self.assertIsNone(_safe_document_type("Tipo_Alucinado_Pelo_Modelo"))

    def test_tipo_none_vira_none(self):
        self.assertIsNone(_safe_document_type(None))

    def test_tipo_vazio_vira_none(self):
        self.assertIsNone(_safe_document_type(""))


class TestRunVisionPipelineNaoSobrescreveMesmoTipo(unittest.TestCase):
    def test_dois_documentos_repetiveis_do_mesmo_tipo_geram_duas_linhas(self):
        # SIOPS_BIMESTRAL nao participa da cadeia de cascata - cada
        # bimestre e um documento independente e nenhum pode sumir do
        # resultado so por compartilhar o mesmo DocumentType.
        reader = MagicMock()
        reader.classify.side_effect = [
            MagicMock(document_type=DocumentType.SIOPS_BIMESTRAL, confidence=0.9),
            MagicMock(document_type=DocumentType.SIOPS_BIMESTRAL, confidence=0.9),
        ]
        campos_base = {
            "municipio": "Exemplo/MA", "cnpj_secretaria": "1",
            "data_transmissao": "2026-03-01", "assinatura_secretario": "Fulano",
        }
        reader.extract_fields.side_effect = [
            {**campos_base, "periodo": "2026/1"},
            {**campos_base, "periodo": "2026/2"},
        ]
        with patch("gestorflow_vision.load_as_images", side_effect=[[b"pagina1"], [b"pagina2"]]):
            rows = run_vision_pipeline(
                ["siops_bim1.pdf", "siops_bim2.pdf"],
                context={},
                reader=reader,
                min_confidence=0.5,
            )
        self.assertEqual(len(rows), 2)
        labels = {r[0] for r in rows}
        self.assertTrue(any(l.startswith("siops_bim1.pdf") for l in labels))
        self.assertTrue(any(l.startswith("siops_bim2.pdf") for l in labels))

    def test_documento_da_cadeia_continua_unico_por_tipo(self):
        # RESOLUCAO_PLEITO participa da cadeia cronologica - o design de
        # validate_chain espera um unico registro por tipo, entao aqui o
        # comportamento de "1 por tipo" continua sendo o correto.
        reader = MagicMock()
        reader.classify.side_effect = [
            MagicMock(document_type=DocumentType.RESOLUCAO_PLEITO, confidence=0.9),
            MagicMock(document_type=DocumentType.RESOLUCAO_PLEITO, confidence=0.9),
        ]
        reader.extract_fields.side_effect = [{}, {}]
        with patch("gestorflow_vision.load_as_images", side_effect=[[b"p1"], [b"p2"]]):
            rows = run_vision_pipeline(
                ["resolucao_v1.pdf", "resolucao_v2.pdf"],
                context={},
                reader=reader,
                min_confidence=0.5,
            )
        self.assertEqual(len(rows), 1)


class TestParseDateField(unittest.TestCase):
    def test_aceita_date_ja_coagido_sem_lancar(self):
        # Bug de regressao: extract_fields()/classify_and_extract() agora
        # devolvem `date`, nao mais string, entao _parse_date_field
        # precisa aceitar os dois formatos.
        self.assertEqual(_parse_date_field({"data_emissao": date(2026, 3, 1)}), date(2026, 3, 1))

    def test_aceita_string_iso(self):
        self.assertEqual(_parse_date_field({"data": "2026-03-01"}), date(2026, 3, 1))

    def test_none_quando_nenhum_campo_de_data_presente(self):
        self.assertIsNone(_parse_date_field({"municipio": "Exemplo/MA"}))


class TestMakeReader(unittest.TestCase):
    def setUp(self):
        for var in ("ANTHROPIC_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY"):
            os.environ.pop(var, None)

    def test_autodetecta_anthropic_quando_so_essa_key_existe(self):
        os.environ["ANTHROPIC_API_KEY"] = "fake"
        with patch("gestorflow_vision.anthropic") as mock_anthropic:
            reader = make_reader()
        self.assertIsInstance(reader, AnthropicVisionDocumentReader)
        mock_anthropic.Anthropic.assert_called_once_with(api_key="fake")

    def test_autodetecta_gemini_quando_so_essa_key_existe(self):
        os.environ["GEMINI_API_KEY"] = "fake"
        with patch("gestorflow_vision.genai"):
            reader = make_reader()
        self.assertIsInstance(reader, GeminiVisionDocumentReader)

    def test_prefere_anthropic_quando_as_duas_keys_existem(self):
        os.environ["ANTHROPIC_API_KEY"] = "fake"
        os.environ["GEMINI_API_KEY"] = "fake"
        with patch("gestorflow_vision.anthropic"):
            reader = make_reader()
        self.assertIsInstance(reader, AnthropicVisionDocumentReader)

    def test_provider_explicito_tem_prioridade(self):
        os.environ["ANTHROPIC_API_KEY"] = "fake"
        reader = make_reader(provider="gemini", api_key="fake-gemini-key")
        self.assertIsInstance(reader, GeminiVisionDocumentReader)

    def test_sem_nenhuma_key_lanca_runtime_error(self):
        with self.assertRaises(RuntimeError):
            make_reader()

    def test_provider_desconhecido_lanca_value_error(self):
        with self.assertRaises(ValueError):
            make_reader(provider="gpt4o", api_key="fake")


class TestGeminiVisionDocumentReader(unittest.TestCase):
    def test_requer_api_key(self):
        for var in ("GEMINI_API_KEY", "GOOGLE_API_KEY"):
            os.environ.pop(var, None)
        with self.assertRaises(RuntimeError):
            GeminiVisionDocumentReader()

    def test_classify_usa_google_api_key_como_fallback(self):
        os.environ.pop("GEMINI_API_KEY", None)
        os.environ["GOOGLE_API_KEY"] = "fake"
        try:
            with patch("gestorflow_vision.genai") as mock_genai:
                GeminiVisionDocumentReader()
                mock_genai.Client.assert_called_once_with(api_key="fake")
        finally:
            del os.environ["GOOGLE_API_KEY"]

    def test_classify_parseia_resposta_do_modelo(self):
        with patch("gestorflow_vision.genai") as mock_genai:
            mock_response = MagicMock()
            mock_response.text = (
                '{"tipo": "Oficio_Solicitacao_Recursos", "confianca": 0.95, "indicios_encontrados": []}'
            )
            mock_genai.Client.return_value.models.generate_content.return_value = mock_response

            reader = GeminiVisionDocumentReader(api_key="fake")
            resultado = reader.classify(b"fake-image-bytes")

            self.assertEqual(resultado.document_type, DocumentType.OFICIO_SOLICITACAO_RECURSOS)
            self.assertEqual(resultado.confidence, 0.95)

    def test_extract_fields_coage_datas_da_resposta(self):
        with patch("gestorflow_vision.genai") as mock_genai:
            mock_response = MagicMock()
            mock_response.text = '{"data_emissao": "2026-03-01", "valor_solicitado": 1000.0}'
            mock_genai.Client.return_value.models.generate_content.return_value = mock_response

            reader = GeminiVisionDocumentReader(api_key="fake")
            campos = reader.extract_fields(b"fake-image-bytes", DocumentType.OFICIO_SOLICITACAO_RECURSOS)

            self.assertEqual(campos["data_emissao"], date(2026, 3, 1))
            self.assertEqual(campos["valor_solicitado"], 1000.0)


if __name__ == "__main__":
    unittest.main()
