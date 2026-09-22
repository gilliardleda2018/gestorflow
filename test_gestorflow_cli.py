"""
Testes de fumaça do CLI: cobrem formatação de tabela, parsing de
argumentos e os caminhos de erro (sem chamar a API da Anthropic).
"""

from __future__ import annotations

import os
import unittest

from gestorflow_auditoria import AuditContext
from gestorflow_cli import format_table, main, parse_args

EXEMPLO_CONTEXT = os.path.join(os.path.dirname(__file__), "examples", "context_exemplo.json")


class TestFormatTable(unittest.TestCase):
    def test_formata_linhas_em_markdown(self):
        rows = [("Oficio no 12/2026", "CONFORME", "-")]
        tabela = format_table(rows)
        self.assertIn("| DOCUMENTO | CONFORMIDADE | PENDENCIAS |", tabela)
        self.assertIn("| Oficio no 12/2026 | CONFORME | - |", tabela)


class TestParseArgs(unittest.TestCase):
    def test_defaults(self):
        args = parse_args(["--context", "ctx.json", "--files", "a.pdf", "b.jpg"])
        self.assertEqual(args.context, "ctx.json")
        self.assertEqual(args.files, ["a.pdf", "b.jpg"])
        self.assertEqual(args.mode, "two_call")
        self.assertEqual(args.min_confidence, 0.5)


class TestContextExemplo(unittest.TestCase):
    def test_context_exemplo_carrega(self):
        ctx = AuditContext.from_json(EXEMPLO_CONTEXT)
        self.assertEqual(ctx.cadastro_proposta["municipio"], "Exemplo/MA")
        self.assertEqual(ctx.ano_referencia_auditoria, 2026)
        self.assertIsInstance(ctx.resolucoes_ja_usadas, set)


class TestMainErrorPaths(unittest.TestCase):
    def test_erro_sem_api_key(self):
        env_backup = os.environ.pop("ANTHROPIC_API_KEY", None)
        try:
            rc = main(["--context", EXEMPLO_CONTEXT, "--files", "inexistente.pdf"])
            self.assertEqual(rc, 1)
        finally:
            if env_backup is not None:
                os.environ["ANTHROPIC_API_KEY"] = env_backup

    def test_erro_arquivo_nao_encontrado(self):
        os.environ["ANTHROPIC_API_KEY"] = "fake-key-para-teste"
        try:
            rc = main(["--context", EXEMPLO_CONTEXT, "--files", "nao_existe.pdf"])
            self.assertEqual(rc, 1)
        finally:
            del os.environ["ANTHROPIC_API_KEY"]

    def test_erro_contexto_nao_encontrado(self):
        os.environ["ANTHROPIC_API_KEY"] = "fake-key-para-teste"
        try:
            rc = main(["--context", "nao_existe.json", "--files", __file__])
            self.assertEqual(rc, 1)
        finally:
            del os.environ["ANTHROPIC_API_KEY"]


if __name__ == "__main__":
    unittest.main()
