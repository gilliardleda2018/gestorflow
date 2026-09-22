"""
GestorFlow - Interface web (Streamlit)
========================================

Front-end simples para rodar o pipeline de classificação/auditoria e a
verificação de assinatura (digital + manuscrita) sem usar o terminal:
upload de documentos, escolha do provedor de visão, execução e tabela
de conformidade na tela, com download do relatório em markdown.

Protegido por senha simples (segredo APP_PASSWORD) - sem ela definida,
a app roda sem gate (útil só para desenvolvimento local).

Rodar localmente:
    streamlit run app.py

Segredos necessários (nome igual, venham de onde vierem):
    APP_PASSWORD           - senha de acesso à interface (recomendado em produção)
    GEMINI_API_KEY          - provedor Gemini (tier gratuito)
    ANTHROPIC_API_KEY       - provedor Claude (pago)

Funciona tanto no Render (variável de ambiente de verdade) quanto no
Streamlit Community Cloud (segredo em `st.secrets`, formato TOML) -
`_sync_secrets_to_env()` copia o que estiver em `st.secrets` para
`os.environ` na subida, já que o resto do projeto (make_reader() etc.)
sempre lê variável de ambiente, para funcionar igual nos dois lugares.
"""

from __future__ import annotations

import json
import os
import tempfile
from datetime import date

import streamlit as st

from gestorflow_auditoria import AuditContext
from gestorflow_signature import check_signature, format_check_result
from gestorflow_vision import make_reader, run_vision_pipeline

st.set_page_config(page_title="GestorFlow", page_icon="📋", layout="wide")


def _sync_secrets_to_env() -> None:
    # st.secrets.get(...) lanca StreamlitSecretNotFoundError (nao KeyError)
    # quando nao existe NENHUM secrets.toml - por isso o try cobre o loop
    # inteiro, nao so o acesso inicial a st.secrets.
    try:
        for chave in ("APP_PASSWORD", "GEMINI_API_KEY", "GOOGLE_API_KEY", "ANTHROPIC_API_KEY"):
            valor = st.secrets.get(chave)
            if valor and not os.environ.get(chave):
                os.environ[chave] = valor
    except Exception:
        pass  # sem secrets.toml (uso local via variavel de ambiente) - tudo bem


def _check_password() -> bool:
    senha_esperada = os.environ.get("APP_PASSWORD")
    if not senha_esperada:
        return True  # sem senha configurada - roda aberto (uso local)

    if st.session_state.get("autenticado"):
        return True

    st.title("GestorFlow")
    senha = st.text_input("Senha de acesso", type="password")
    if st.button("Entrar"):
        if senha == senha_esperada:
            st.session_state["autenticado"] = True
            st.rerun()
        else:
            st.error("Senha incorreta.")
    return False


def _status_amigavel(status: str) -> str:
    return {
        "CONFORME": "✅ Está tudo certo",
        "NAO CONFORME": "❌ Tem algo pendente",
        "NAO CLASSIFICADO": "❓ Não reconhecido",
        "ERRO": "⚠️ Erro ao processar",
        "SOBRESCRITO": "⚠️ Duplicado - verificar",
    }.get(status, status)


def _metodo_amigavel(metodo: str) -> str:
    return {
        "digital": "🔏 Assinatura digital",
        "manual": "✍️ Assinatura manuscrita",
        "sem_assinatura": "⚠️ Não foi possível conferir a assinatura",
    }.get(metodo, metodo)


def _contexto_padrao() -> dict:
    return {
        "cadastro_proposta": {},
        "ficha_cnes": {},
        "ato_nomeacao_presidente": {},
        "doc_identificacao_presidente": {},
        "resolucoes_ja_usadas": [],
        "contrato_abertura_conta": {},
        "certidao_tce": {},
        "siope_bimestral": {},
        "data_referencia_auditoria": date.today().isoformat(),
        "ano_referencia_auditoria": date.today().year,
    }


def main() -> None:
    _sync_secrets_to_env()

    if not _check_password():
        return

    st.title("GestorFlow")
    st.caption(
        "Envie os documentos do processo e o sistema confere, um por um, se estão "
        "completos, corretos e assinados."
    )

    with st.sidebar:
        st.header("Opções")

        verificar_assinaturas = st.checkbox(
            "Também conferir as assinaturas",
            value=True,
            help="Verifica se cada documento tem uma assinatura válida - digital "
                 "(quando o PDF foi assinado eletronicamente) ou feita à mão. "
                 "Deixe marcado, a menos que você só queira conferir se os dados "
                 "estão completos.",
        )

        provider = None
        model = ""
        mode = "two_call"
        min_confidence = 0.5
        trust_roots_file = None

        with st.expander("Configurações avançadas (não é necessário mexer)"):
            st.caption(
                "Estas opções já vêm ajustadas para funcionar bem no dia a dia. "
                "Só mude alguma coisa aqui se um técnico pedir."
            )

            provider_label = st.selectbox(
                "Qual inteligência artificial vai ler os documentos",
                ["Escolher automaticamente", "Gemini (gratuito)", "Claude (pago)"],
                help="\"Escolher automaticamente\" usa a que já estiver configurada "
                     "no sistema - é o que a maioria das pessoas deve usar.",
            )
            provider = {
                "Escolher automaticamente": None,
                "Gemini (gratuito)": "gemini",
                "Claude (pago)": "anthropic",
            }[provider_label]

            model = st.text_input(
                "Nome de um modelo específico",
                help="Deixe em branco. Só preencha se um técnico te passar um nome específico.",
            )

            modo_label = st.radio(
                "Como ler os documentos",
                ["Mais cuidadoso (recomendado)", "Mais rápido"],
                help="\"Mais cuidadoso\" primeiro identifica o tipo do documento e só "
                     "depois lê os dados - erra menos. \"Mais rápido\" faz tudo de uma "
                     "vez só, mas com mais chance de erro.",
            )
            mode = {"Mais cuidadoso (recomendado)": "two_call", "Mais rápido": "single_call"}[modo_label]

            min_confidence = st.slider(
                "Nível de certeza exigido para aceitar um documento",
                0.0, 1.0, 0.5, 0.05,
                help="Quanto mais alto, mais rigoroso o sistema fica: documentos que "
                     "ele não reconhecer com segurança ficam marcados como "
                     "\"não identificado\" em vez de arriscar um palpite.",
            )

            trust_roots_file = st.file_uploader(
                "Arquivo de certificados oficiais para conferir assinatura digital",
                type=["pem"],
                help="Só necessário se você quiser confirmar que a assinatura digital "
                     "foi emitida por uma autoridade reconhecida (ICP-Brasil). Sem "
                     "isso, o sistema ainda confere se o arquivo não foi alterado "
                     "depois de assinado.",
            )

        contexto_upload = None
        with st.expander("Dados do processo para comparação (opcional)"):
            st.caption(
                "Se você tiver um arquivo com os dados já cadastrados do processo "
                "(município, valores, unidades de saúde etc.), o sistema também "
                "confere se os documentos batem entre si - não só se cada um está "
                "completo."
            )
            contexto_upload = st.file_uploader("Arquivo de dados do processo (.json)", type=["json"])

    st.subheader("1. Envie os documentos")
    st.caption("PDF, foto ou print do documento. Pode selecionar vários de uma vez.")
    arquivos = st.file_uploader(
        "Documentos do processo",
        type=["pdf", "png", "jpg", "jpeg"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    st.subheader("2. Analisar")
    rodar = st.button("Analisar documentos", type="primary", disabled=not arquivos)

    if rodar and arquivos:
        with tempfile.TemporaryDirectory() as tmpdir:
            caminhos = []
            for arquivo in arquivos:
                caminho = os.path.join(tmpdir, arquivo.name)
                with open(caminho, "wb") as f:
                    f.write(arquivo.getbuffer())
                caminhos.append(caminho)

            if contexto_upload is not None:
                dados_contexto = json.load(contexto_upload)
                contexto_path = os.path.join(tmpdir, "contexto.json")
                with open(contexto_path, "w", encoding="utf-8") as f:
                    json.dump(dados_contexto, f)
            else:
                contexto_path = os.path.join(tmpdir, "contexto.json")
                with open(contexto_path, "w", encoding="utf-8") as f:
                    json.dump(_contexto_padrao(), f)

            trust_roots_path = None
            if trust_roots_file is not None:
                trust_roots_path = os.path.join(tmpdir, "trust_roots.pem")
                with open(trust_roots_path, "wb") as f:
                    f.write(trust_roots_file.getbuffer())

            try:
                context = AuditContext.from_json(contexto_path)
            except (ValueError, OSError) as exc:
                st.error(f"Contexto inválido: {exc}")
                return

            try:
                reader = make_reader(provider=provider, model=model or None)
            except (RuntimeError, ValueError) as exc:
                st.error(str(exc))
                return

            with st.spinner("Lendo e conferindo os documentos... isso pode levar um minuto."):
                try:
                    rows = run_vision_pipeline(
                        caminhos, context.as_dict(), reader=reader,
                        min_confidence=min_confidence, mode=mode,
                    )
                except Exception as exc:
                    st.error(f"Não foi possível concluir a análise: {exc}")
                    return

            st.subheader("Resultado")
            st.table(
                [
                    {"Documento": d, "Situação": _status_amigavel(c), "O que falta ou está errado": p}
                    for d, c, p in rows
                ]
            )

            relatorio_md = "| DOCUMENTO | CONFORMIDADE | PENDENCIAS |\n|---|---|---|\n" + "\n".join(
                f"| {d} | {c} | {p} |" for d, c, p in rows
            )

            if verificar_assinaturas:
                st.subheader("Assinaturas")
                secoes = []
                with st.spinner("Conferindo assinaturas..."):
                    for caminho in caminhos:
                        resultado = check_signature(caminho, reader=reader, trust_roots_path=trust_roots_path)
                        texto = f"**{_metodo_amigavel(resultado.metodo)}**\n\n{format_check_result(resultado)}"
                        st.markdown(texto)
                        secoes.append(texto)
                relatorio_md += "\n\n## Verificação de assinatura\n\n" + "\n\n".join(secoes)

            st.download_button(
                "⬇️ Baixar relatório para guardar/enviar",
                data=relatorio_md,
                file_name="relatorio_gestorflow.md",
                mime="text/markdown",
            )


if __name__ == "__main__":
    main()
