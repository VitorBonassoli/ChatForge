"""
tests/test_atendente.py
Casos de teste CT01 ao CT07 conforme documento de estratégia.
"""

import sys, os, pytest, psycopg2
from unittest.mock import patch, MagicMock
from datetime import datetime

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../backend")))

from atendente import (
    chat, ModoIA, TipoPrompt,
    _tentar_registrar_pedido,
    buscar_produtos,
    extrair_quantidade,
)

# ── Fixtures ──────────────────────────────────────────────────────

@pytest.fixture
def produto():
    return {"id": 1, "nome": "Desentupidor X", "descricao": "Para pias e ralos",
            "categoria": "Limpeza", "preco": 49.90, "quantidade_estoque": 10, "ativo": True}

@pytest.fixture
def mock_conn(produto):
    """Cursor mockado que simula retornos padrão do banco."""
    pedido_row = {"id": 100, "cliente_id": 42, "conversa_id": 1, "produto_id": 1,
                  "quantidade": 2, "valor_total": 99.80, "status": "confirmado",
                  "criado_em": datetime(2025, 1, 1)}
    cur = MagicMock()
    cur.fetchone.side_effect = [produto, {"cliente_id": 42}, pedido_row]
    cur.__enter__ = lambda s: s
    cur.__exit__ = MagicMock(return_value=False)
    conn = MagicMock()
    conn.cursor.return_value = cur
    conn.__enter__ = lambda s: s
    conn.__exit__ = MagicMock(return_value=False)
    return conn, cur

# ── CT01 — Recomendação com produto existente ─────────────────────

@patch("atendente.buscar_produtos")
@patch("atendente.salvar_mensagem")
@patch("atendente.RunnableWithMessageHistory")
def test_CT01_recomenda_produto_existente(mock_rwh, mock_salvar, mock_buscar, produto):
    mock_buscar.return_value = [produto]
    chain = MagicMock()
    chain.invoke.return_value = MagicMock(content="Recomendo o Desentupidor X!")
    mock_rwh.return_value = chain

    resultado = chat(1, "Minha pia está entupida", ModoIA.DETALHADO, TipoPrompt.ESTRUTURADO)

    mock_buscar.assert_called_once()
    assert "Desentupidor X" in chain.invoke.call_args[0][0]["contexto_produtos"]
    assert resultado["bloqueado"] is False

# ── CT02 — Produto fora do catálogo ──────────────────────────────

@patch("atendente.buscar_produtos")
@patch("atendente.salvar_mensagem")
@patch("atendente.RunnableWithMessageHistory")
def test_CT02_sem_produto_no_catalogo(mock_rwh, mock_salvar, mock_buscar):
    mock_buscar.return_value = []
    chain = MagicMock()
    chain.invoke.return_value = MagicMock(content="Não temos esse produto.")
    mock_rwh.return_value = chain

    chat(1, "Meu carro quebrou", ModoIA.DETALHADO, TipoPrompt.ESTRUTURADO)

    assert "Nenhum produto relevante" in chain.invoke.call_args[0][0]["contexto_produtos"]

# ── CT03 — Pedido direto com cálculo de valor total ──────────────

def test_CT03_extrai_quantidade_e_calcula_valor(produto):
    assert extrair_quantidade("Quero comprar 2 unidades do Produto A") == 2
    assert produto["preco"] * 2 == pytest.approx(99.80, rel=1e-3)

@patch("atendente.get_connection")
def test_CT03_registra_pedido_com_valor_correto(mock_get_conn, mock_conn, produto):
    conn, cur = mock_conn
    mock_get_conn.return_value = conn

    resultado = _tentar_registrar_pedido(conversa_id=1, produto=produto, quantidade=2)

    assert resultado["sucesso"] is True
    assert resultado["pedido"]["valor_total"] == pytest.approx(99.80, rel=1e-3)

# ── CT04 — Estoque zero ───────────────────────────────────────────

@patch("atendente.get_connection")
def test_CT04_rejeita_pedido_sem_estoque(mock_get_conn, produto):
    produto_sem_estoque = {**produto, "quantidade_estoque": 0}
    cur = MagicMock()
    cur.fetchone.return_value = produto_sem_estoque
    cur.__enter__ = lambda s: s
    cur.__exit__ = MagicMock(return_value=False)
    conn = MagicMock()
    conn.cursor.return_value = cur
    conn.__enter__ = lambda s: s
    conn.__exit__ = MagicMock(return_value=False)
    mock_get_conn.return_value = conn

    resultado = _tentar_registrar_pedido(conversa_id=1, produto=produto_sem_estoque, quantidade=1)

    assert resultado["sucesso"] is False
    assert resultado["erro"] == "Estoque insuficiente"

# ── CT05 — Memória de contexto entre mensagens ───────────────────

@patch("atendente.buscar_produtos")
@patch("atendente.salvar_mensagem")
@patch("atendente.RunnableWithMessageHistory")
def test_CT05_memoria_entre_mensagens(mock_rwh, mock_salvar, mock_buscar, produto):
    mock_buscar.return_value = [produto]
    chain = MagicMock()
    chain.invoke.side_effect = [
        MagicMock(content="O Desentupidor X custa R$49,90."),
        MagicMock(content="Duas unidades saem por R$99,80."),
    ]
    mock_rwh.return_value = chain

    chat(99, "Qual o preço do Desentupidor X?", ModoIA.DETALHADO, TipoPrompt.ESTRUTURADO)
    chat(99, "E se eu levar duas unidades dele?", ModoIA.DETALHADO, TipoPrompt.ESTRUTURADO)

    assert chain.invoke.call_count == 2

# ── CT06 — Persistência do pedido no banco ───────────────────────

@patch("atendente.get_connection")
def test_CT06_grava_pedido_no_banco(mock_get_conn, mock_conn, produto):
    conn, cur = mock_conn
    mock_get_conn.return_value = conn

    resultado = _tentar_registrar_pedido(conversa_id=1, produto=produto, quantidade=2)

    queries = [str(c[0][0]) for c in cur.execute.call_args_list]
    assert any("INSERT INTO pedidos" in q for q in queries)
    assert any("UPDATE produtos" in q for q in queries)
    conn.commit.assert_called_once()
    assert resultado["sucesso"] is True

# ── CT07 — Falha de conexão com o banco ──────────────────────────

@patch("atendente.get_connection")
def test_CT07_falha_conexao_retorna_erro(mock_get_conn, produto):
    mock_get_conn.side_effect = psycopg2.OperationalError("connection refused")

    resultado = _tentar_registrar_pedido(conversa_id=1, produto=produto, quantidade=1)

    assert resultado["sucesso"] is False
    assert "Falha de conexão" in resultado["erro"]

@patch("atendente.get_connection")
def test_CT07_falha_loga_erro(mock_get_conn, produto):
    mock_get_conn.side_effect = psycopg2.OperationalError("connection refused")

    with patch("atendente.logger") as mock_logger:
        _tentar_registrar_pedido(conversa_id=1, produto=produto, quantidade=1)
        mock_logger.error.assert_called_once()
        assert "FALHA DE CONEXÃO" in mock_logger.error.call_args[0][0]
