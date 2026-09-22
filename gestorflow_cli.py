"""
GestorFlow - CLI de ponta a ponta
==================================

Roda o pipeline de classificação e auditoria sobre um conjunto de
arquivos (PDF ou imagem) usando o modelo de visão da Anthropic, com o
contexto de auditoria (cadastro da proposta, ficha do CNES, atos de
nomeação etc.) carregado de um JSON.

Uso:
    python gestorflow_cli.py --context examples/context_exemplo.json \
        --files doc1.pdf doc2.jpg doc3.png \
        [--mode two_call|single_call] [--min-confidence 0.5] \
        [--output resultado.md] [--model claude-sonnet-4-6]

Requer ANTHROPIC_API_KEY no ambiente (ou --api-key).
"""

from __future__ import annotations

import argparse
import os
import sys

from gestorflow_auditoria import AuditContext
from gestorflow_vision import VisionDocumentReader, run_vision_pipeline


def format_table(rows: list[tuple[str, str, str]]) -> str:
    linhas = ["| DOCUMENTO | CONFORMIDADE | PENDENCIAS |", "|---|---|---|"]
    for doc, conf, pend in rows:
        linhas.append(f"| {doc} | {conf} | {pend} |")
    return "\n".join(linhas)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pipeline de classificacao e auditoria do GestorFlow (PDF/imagem -> tabela de conformidade)"
    )
    parser.add_argument("--context", required=True, help="Caminho do JSON de contexto (AuditContext)")
    parser.add_argument("--files", nargs="+", required=True, help="PDFs/imagens dos documentos a auditar")
    parser.add_argument("--mode", choices=["two_call", "single_call"], default="two_call",
                         help="two_call (padrao, mais preciso) ou single_call (mais barato)")
    parser.add_argument("--min-confidence", type=float, default=0.5)
    parser.add_argument("--output", help="Se informado, grava a tabela em markdown neste caminho")
    parser.add_argument("--model", default="claude-sonnet-4-6")
    parser.add_argument("--api-key", default=None, help="Sobrescreve ANTHROPIC_API_KEY")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if not args.api_key and not os.environ.get("ANTHROPIC_API_KEY"):
        print("Erro: defina ANTHROPIC_API_KEY no ambiente ou use --api-key", file=sys.stderr)
        return 1

    faltando = [f for f in args.files if not os.path.exists(f)]
    if faltando:
        print(f"Erro: arquivo(s) nao encontrado(s): {', '.join(faltando)}", file=sys.stderr)
        return 1

    if not os.path.exists(args.context):
        print(f"Erro: arquivo de contexto nao encontrado: {args.context}", file=sys.stderr)
        return 1

    context = AuditContext.from_json(args.context)
    reader = VisionDocumentReader(api_key=args.api_key, model=args.model)

    rows = run_vision_pipeline(
        args.files,
        context.as_dict(),
        reader=reader,
        min_confidence=args.min_confidence,
        mode=args.mode,
    )

    tabela = format_table(rows)
    print(tabela)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(tabela + "\n")
        print(f"\nTabela gravada em {args.output}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
