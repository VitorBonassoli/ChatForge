from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr
from typing import Optional
from datetime import datetime, date

from atendente import (
    # Enums
    ModoIA,
    TipoPrompt,
    # Chat
    chat,
    # CRUD
    cadastrar_cliente,
    iniciar_conversa,
    encerrar_conversa,
    salvar_mensagem,
    cadastrar_produto,
    # Buscas
    buscar_produtos,
    buscar_historico,
    extrair_palavras_chave,
)

# ══════════════════════════════════════════════════════════════════
# APP
# ══════════════════════════════════════════════════════════════════

app = FastAPI(
    title="Atendente Virtual — Dual AI",
    description=(
        "API de atendimento com Gemini (vendas) + Groq (suporte técnico). "
        "Suporta 5 modos de IA e 3 tipos de prompt com proteções de segurança."
    ),
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ══════════════════════════════════════════════════════════════════
# UTILITÁRIO — serializa RealDictRow do psycopg2
# ══════════════════════════════════════════════════════════════════

def serializar(obj):
    """Converte RealDictRow/dict com datetime para dict JSON-serializável."""
    if obj is None:
        return None
    result = {}
    for k, v in dict(obj).items():
        if isinstance(v, (datetime, date)):
            result[k] = v.isoformat()
        else:
            result[k] = v
    return result

def serializar_lista(lista):
    return [serializar(item) for item in lista]


# ══════════════════════════════════════════════════════════════════
# SCHEMAS
# ══════════════════════════════════════════════════════════════════

class ClienteCreate(BaseModel):
    nome: str
    email: EmailStr
    senha_hash: str


class ConversaCreate(BaseModel):
    cliente_id: int


class MensagemCreate(BaseModel):
    mensagem: str
    modo: ModoIA = ModoIA.DETALHADO
    tipo_prompt: TipoPrompt = TipoPrompt.ESTRUTURADO


class ProdutoCreate(BaseModel):
    nome: str
    descricao: str
    categoria: str
    preco: float
    quantidade_estoque: int


class LoginBody(BaseModel):
    email: EmailStr
    senha_hash: str


# ══════════════════════════════════════════════════════════════════
# CLIENTES
# ══════════════════════════════════════════════════════════════════

@app.post("/clientes", summary="Cadastrar novo cliente")
def criar_cliente(body: ClienteCreate):
    try:
        cliente = cadastrar_cliente(body.nome, body.email, body.senha_hash)
        return {"sucesso": True, "cliente": serializar(cliente)}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Erro ao cadastrar cliente: {e or 'erro desconhecido'}")


@app.get("/clientes/{cliente_id}/historico", summary="Histórico de mensagens do cliente")
def historico_cliente(cliente_id: int):
    try:
        historico = buscar_historico(cliente_id)
        if historico is None:
            raise HTTPException(status_code=404, detail="Cliente não encontrado.")
        return {"sucesso": True, "historico": serializar_lista(historico)}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Erro ao buscar histórico do cliente {cliente_id}: {e or 'erro desconhecido'}")


# ══════════════════════════════════════════════════════════════════
# CONVERSAS
# ══════════════════════════════════════════════════════════════════

@app.post("/conversas", summary="Iniciar nova conversa")
def nova_conversa(body: ConversaCreate):
    try:
        conversa = iniciar_conversa(body.cliente_id)
        if conversa is None:
            raise HTTPException(status_code=404, detail="Cliente não encontrado.")
        return {"sucesso": True, "conversa": serializar(conversa)}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Erro ao iniciar conversa: {e or 'erro desconhecido'}")


@app.delete("/conversas/{conversa_id}", summary="Encerrar conversa")
def fechar_conversa(conversa_id: int):
    try:
        conversa = encerrar_conversa(conversa_id)
        if not conversa:
            raise HTTPException(status_code=404, detail="Conversa não encontrada.")
        return {"sucesso": True, "conversa": serializar(conversa)}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Erro ao encerrar conversa {conversa_id}: {e or 'erro desconhecido'}")


# ══════════════════════════════════════════════════════════════════
# CHAT — endpoint principal
# ══════════════════════════════════════════════════════════════════

@app.post("/conversas/{conversa_id}/mensagens", summary="Enviar mensagem ao atendente")
def enviar_mensagem(conversa_id: int, body: MensagemCreate):
    try:
        resultado = chat(
            conversa_id=conversa_id,
            mensagem_usuario=body.mensagem,
            modo=body.modo,
            tipo_prompt=body.tipo_prompt,
        )
        if resultado is None:
            raise HTTPException(status_code=404, detail="Conversa não encontrada.")
        return {"sucesso": True, **resultado}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao processar mensagem na conversa {conversa_id}: {e or 'erro desconhecido'}")


# ══════════════════════════════════════════════════════════════════
# PRODUTOS
# ══════════════════════════════════════════════════════════════════

@app.post("/produtos", summary="Cadastrar produto")
def criar_produto(body: ProdutoCreate):
    try:
        produto = cadastrar_produto(
            body.nome,
            body.descricao,
            body.categoria,
            body.preco,
            body.quantidade_estoque,
        )
        return {"sucesso": True, "produto": serializar(produto)}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Erro ao cadastrar produto: {e or 'erro desconhecido'}")


@app.get("/produtos/buscar", summary="Buscar produtos por palavras-chave")
def buscar(q: str = Query(..., description="Termos de busca separados por espaço")):
    try:
        palavras = extrair_palavras_chave(q)
        if not palavras:
            return {"sucesso": True, "produtos": [], "aviso": "Nenhuma palavra-chave relevante encontrada."}
        produtos = buscar_produtos(palavras)
        return {"sucesso": True, "total": len(produtos), "produtos": serializar_lista(produtos)}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Erro ao buscar produtos: {e or 'erro desconhecido'}")


# ══════════════════════════════════════════════════════════════════
# LOGIN
# ══════════════════════════════════════════════════════════════════

@app.post("/clientes/login", summary="Login do cliente")
def login_cliente(body: LoginBody):
    """
    Busca o cliente pelo email e senha. Se encontrar, inicia uma nova conversa.
    """
    from atendente import get_connection
    from psycopg2.extras import RealDictCursor

    try:
        with get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    "SELECT * FROM clientes WHERE email = %s AND senha = %s",
                    (body.email, body.senha_hash),
                )
                cliente = cur.fetchone()

        if not cliente:
            raise HTTPException(status_code=401, detail="Email ou senha inválidos.")

        conversa = iniciar_conversa(cliente["id"])
        return {
            "sucesso": True,
            "cliente": serializar(cliente),
            "conversa": serializar(conversa),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Erro ao realizar login: {e or 'erro desconhecido'}")