"""
Testes de `gestorflow_signature.py`. Gera um certificado autoassinado
e um PDF assinado de verdade (via PyHanko) em um diretório temporário,
para validar a inspeção contra assinaturas digitais reais - não apenas
mocks.
"""

from __future__ import annotations

import datetime
import os
import tempfile
import unittest
from unittest.mock import MagicMock

import fitz  # pymupdf - só para montar um PDF em branco de teste
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import NameOID
from pyhanko.pdf_utils.incremental_writer import IncrementalPdfFileWriter
from pyhanko.sign import signers

from gestorflow_signature import (
    PdfSignatureReport,
    SignatureCheckResult,
    check_signature,
    inspect_pdf,
)
from gestorflow_vision import ManualSignatureAssessment


def _make_self_signed_cert(tmpdir: str) -> tuple[str, str]:
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    subject = issuer = x509.Name([
        x509.NameAttribute(NameOID.COMMON_NAME, "GestorFlow Teste"),
        x509.NameAttribute(NameOID.ORGANIZATION_NAME, "GestorFlow"),
    ])
    now = datetime.datetime.now(datetime.timezone.utc)
    cert = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(issuer)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - datetime.timedelta(days=1))
        .not_valid_after(now + datetime.timedelta(days=365))
        .sign(key, hashes.SHA256())
    )
    key_path = os.path.join(tmpdir, "key.pem")
    cert_path = os.path.join(tmpdir, "cert.pem")
    with open(key_path, "wb") as f:
        f.write(key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.TraditionalOpenSSL,
            serialization.NoEncryption(),
        ))
    with open(cert_path, "wb") as f:
        f.write(cert.public_bytes(serialization.Encoding.PEM))
    return key_path, cert_path


def _make_blank_pdf(path: str) -> None:
    doc = fitz.open()
    doc.new_page()
    doc.save(path)
    doc.close()


def _sign_pdf(src_path: str, dst_path: str, key_path: str, cert_path: str) -> None:
    signer = signers.SimpleSigner.load(key_path, cert_path)
    with open(src_path, "rb") as inf:
        writer = IncrementalPdfFileWriter(inf)
        with open(dst_path, "wb") as outf:
            signers.sign_pdf(
                writer,
                signers.PdfSignatureMetadata(field_name="Signature1"),
                signer=signer,
                output=outf,
            )


class TestInspectPdfComAssinaturaReal(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.key_path, self.cert_path = _make_self_signed_cert(self.tmp.name)
        self.blank_pdf = os.path.join(self.tmp.name, "blank.pdf")
        _make_blank_pdf(self.blank_pdf)
        self.signed_pdf = os.path.join(self.tmp.name, "assinado.pdf")
        _sign_pdf(self.blank_pdf, self.signed_pdf, self.key_path, self.cert_path)

    def test_assinatura_integra_e_valida_sem_trust_roots(self):
        report = inspect_pdf(self.signed_pdf)
        self.assertTrue(report.is_native_pdf)
        self.assertTrue(report.has_signatures)
        self.assertEqual(len(report.signatures), 1)
        sig = report.signatures[0]
        self.assertTrue(sig.intact)
        self.assertTrue(sig.valid)
        self.assertTrue(sig.covers_whole_document)
        self.assertIsNone(sig.trusted)  # sem bundle de confianca informado

    def test_cadeia_confiavel_quando_certificado_e_a_propria_ancora(self):
        report = inspect_pdf(self.signed_pdf, trust_roots_path=self.cert_path)
        sig = report.signatures[0]
        self.assertIsNotNone(sig.trusted)
        self.assertTrue(sig.trusted)
        self.assertTrue(sig.legitima)
        self.assertTrue(report.legitimo)

    def test_pdf_sem_assinatura_nao_tem_assinaturas(self):
        report = inspect_pdf(self.blank_pdf)
        self.assertTrue(report.is_native_pdf)
        self.assertFalse(report.has_signatures)
        self.assertFalse(report.legitimo)

    def test_arquivo_nao_pdf_marcado_nao_aplicavel(self):
        txt_path = os.path.join(self.tmp.name, "nota.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write("nao e pdf")
        report = inspect_pdf(txt_path)
        self.assertFalse(report.is_native_pdf)
        self.assertIsNotNone(report.erro)

    def test_trust_roots_inexistente_reporta_erro_sem_lancar_excecao(self):
        report = inspect_pdf(self.signed_pdf, trust_roots_path="nao_existe.pem")
        self.assertIsNotNone(report.erro)
        self.assertFalse(report.legitimo)


class TestCheckSignatureOrquestrador(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.key_path, self.cert_path = _make_self_signed_cert(self.tmp.name)
        self.blank_pdf = os.path.join(self.tmp.name, "blank.pdf")
        _make_blank_pdf(self.blank_pdf)
        self.signed_pdf = os.path.join(self.tmp.name, "assinado.pdf")
        _sign_pdf(self.blank_pdf, self.signed_pdf, self.key_path, self.cert_path)

    def test_pdf_com_assinatura_digital_usa_metodo_digital(self):
        resultado = check_signature(self.signed_pdf, reader=None, trust_roots_path=self.cert_path)
        self.assertEqual(resultado.metodo, "digital")
        self.assertTrue(resultado.legitima)
        self.assertIsInstance(resultado.digital, PdfSignatureReport)
        self.assertIsNone(resultado.manual)

    def test_pdf_sem_assinatura_digital_cai_para_avaliacao_manual(self):
        reader = MagicMock()
        reader.assess_manual_signature.return_value = ManualSignatureAssessment(
            presente=True, tipo="manuscrita", localizacao="rodape",
            aparenta_autentica=True, observacoes=None,
        )
        resultado = check_signature(self.blank_pdf, reader=reader)
        self.assertEqual(resultado.metodo, "manual")
        self.assertTrue(resultado.legitima)
        reader.assess_manual_signature.assert_called_once()

    def test_pdf_sem_assinatura_e_sem_reader_reporta_limitacao(self):
        resultado = check_signature(self.blank_pdf, reader=None)
        self.assertEqual(resultado.metodo, "sem_assinatura")
        self.assertIsNone(resultado.legitima)
        self.assertIsNotNone(resultado.erro)

    def test_assinatura_manuscrita_ausente_nao_e_legitima_nem_ilegitima(self):
        reader = MagicMock()
        reader.assess_manual_signature.return_value = ManualSignatureAssessment(
            presente=False, tipo="ausente", localizacao=None,
            aparenta_autentica=None, observacoes=None,
        )
        resultado = check_signature(self.blank_pdf, reader=reader)
        self.assertEqual(resultado.metodo, "manual")
        self.assertIsNone(resultado.legitima)  # ausente = nada para avaliar, nao "ilegitima"

    def test_assinatura_manuscrita_suspeita_de_montagem_nao_e_legitima(self):
        reader = MagicMock()
        reader.assess_manual_signature.return_value = ManualSignatureAssessment(
            presente=True, tipo="digitalizada_sem_tinta", localizacao="rodape",
            aparenta_autentica=False, observacoes="parece recorte colado",
        )
        resultado = check_signature(self.blank_pdf, reader=reader)
        self.assertFalse(resultado.legitima)


class TestManualSignatureAssessment(unittest.TestCase):
    def test_legitima_requer_presente_manuscrita_e_nao_suspeita(self):
        self.assertTrue(ManualSignatureAssessment(True, "manuscrita", None, True, None).legitima)
        self.assertTrue(ManualSignatureAssessment(True, "manuscrita", None, None, None).legitima)
        self.assertFalse(ManualSignatureAssessment(False, "ausente", None, None, None).legitima)
        self.assertFalse(ManualSignatureAssessment(True, "carimbo", None, True, None).legitima)
        self.assertFalse(ManualSignatureAssessment(True, "manuscrita", None, False, "colagem").legitima)


if __name__ == "__main__":
    unittest.main()
