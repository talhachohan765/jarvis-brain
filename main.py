"""
Jarvis slim brain — chhota FastAPI relay (v1).

Client (Jarvis app, JarvisBrainClient.kt) bhejta hai:
  POST /v1/chat/completions  {model, stream:true, messages:[...]}
  Authorization: Bearer <BRAIN_KEY>
  X-BYOK-Gemini-Key / X-BYOK-Groq-Key (per-request, phone ki keys)

Server kaam:
  1. Bearer check (BRAIN_KEY env var se) — ghalat/missing -> 401.
  2. Gemini key mile to pehle Gemini REST (streamGenerateContent) try karo,
     key sirf `x-goog-api-key` header mein — kabhi URL query mein nahi.
  3. Gemini fail ho to Groq (OpenAI-compatible) try karo.
  4. Jawab OpenAI-compatible SSE chunks mein wapas karo, taake app ka
     parser (choices[0].delta.content) seedha chal jaye.

Security: koi key, URL ya header value kabhi log/print/file mein nahi.
Sirf method, path, provider naam aur provider ka HTTP status log hota hai.
Koi key yahan store nahi hoti — sab per-request headers mein aati hai.
"""

import json
import logging
import os
from typing import AsyncIterator, Optional

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("jarvis-brain")

BRAIN_KEY = os.environ.get("BRAIN_KEY", "")
GEMINI_BASE = "https://generativelanguage.googleapis.com/v1beta"
GROQ_BASE = "https://api.groq.com/openai/v1"
# Groq par Gemini model nahi chalte — fallback ke liye AKHRI CHAIN ka
# pehla Groq model istemal hota hai.
GROQ_DEFAULT_MODEL = "openai/gpt-oss-120b"
HTTP_TIMEOUT = 60.0

app = FastAPI(title="jarvis-brain", version="1.0.0")


def _auth_ok(request: Request) -> bool:
    """Bearer key check. BRAIN_KEY set na ho to fail-closed (koi bhi
    /v1 call allow nahi)."""
    if not BRAIN_KEY:
        return False
    auth = request.headers.get("authorization", "")
    return auth == "Bearer " + BRAIN_KEY


def _http_client() -> httpx.AsyncClient:
    """httpx client. Sandbox ka no_proxy list bracketed IPv6 entries
    ([::1]) rakhta hai jo httpx ke env parsing ko tor deta hai, is liye
    trust_env=False aur proxy URL seedha env se liya jata hai (koi value
    hardcode nahi, kabhi log nahi hoti). Render par koi proxy env nahi
    hota — wahan direct connection chalti hai."""
    proxy = (
        os.environ.get("https_proxy")
        or os.environ.get("HTTPS_PROXY")
        or os.environ.get("http_proxy")
        or os.environ.get("HTTP_PROXY")
    )
    # Sandbox ka egress proxy TLS intercept karta hai — uska CA bundle
    # SSL_CERT_FILE env mein hota hai. httpx/certifi use khud nahi parhta,
    # is liye verify mein seedha de do. Render par ye env nahi hota,
    # wahan default (certifi) verification chalti hai.
    ca_bundle = (
        os.environ.get("SSL_CERT_FILE")
        or os.environ.get("REQUESTS_CA_BUNDLE")
        or os.environ.get("CURL_CA_BUNDLE")
    )
    verify = ca_bundle if ca_bundle and os.path.exists(ca_bundle) else True
    if proxy:
        return httpx.AsyncClient(trust_env=False, proxy=proxy, verify=verify)
    return httpx.AsyncClient(trust_env=False, verify=verify)


def _openai_error(status: int, message: str) -> JSONResponse:
    return JSONResponse(
        status_code=status,
        content={"error": {"message": message, "type": "provider_error"}},
    )
    return JSONResponse(
        status_code=status,
        content={"error": {"message": message, "type": "provider_error"}},
    )


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/")
async def root():
    return {"service": "jarvis-brain", "status": "ok"}


@app.get("/agent")
async def agent_page():
    """Jarvis Agent Chat page — key wala chat page (static HTML).

    Page browser se seedha Gemini ko call karta hai (key sirf user ke
    browser ki localStorage mein). Brain sirf page ko serve karta hai;
    jawab ke liye brain istemal nahi hota."""
    path = os.path.join(os.path.dirname(__file__), "static", "agent-chat.html")
    return FileResponse(path, media_type="text/html")


@app.get("/v1/models")
async def models(request: Request):
    if not _auth_ok(request):
        return _openai_error(401, "unauthorized")
    return {
        "object": "list",
        "data": [
            {"id": "gemini-3-flash", "object": "model", "owned_by": "google"},
            {"id": "gemini-3.8-flash", "object": "model", "owned_by": "google"},
            {"id": "gemini-3.6-flash", "object": "model", "owned_by": "google"},
            {"id": "openai/gpt-oss-120b", "object": "model", "owned_by": "groq"},
            {"id": "openai/gpt-oss-20b", "object": "model", "owned_by": "groq"},
            {"id": "qwen/qwen3-8b-27b", "object": "model", "owned_by": "groq"},
        ],
    }


# ---------- OpenAI messages -> Gemini contents ----------

def _to_gemini(messages: list) -> tuple[Optional[dict], list]:
    system_parts: list[dict] = []
    contents: list[dict] = []
    for m in messages:
        role = m.get("role", "user")
        text = m.get("content", "") or ""
        if role == "system":
            system_parts.append({"text": text})
        elif role == "assistant":
            contents.append({"role": "model", "parts": [{"text": text}]})
        else:
            contents.append({"role": "user", "parts": [{"text": text}]})
    system_instruction = {"parts": system_parts} if system_parts else None
    return system_instruction, contents


def _gemini_text(payload: dict) -> str:
    """Ek Gemini streamGenerateContent chunk se text nikalo."""
    out = []
    for cand in payload.get("candidates", []) or []:
        content = cand.get("content") or {}
        for part in content.get("parts", []) or []:
            if part.get("text"):
                out.append(part["text"])
    return "".join(out)


def _openai_chunk(text: str, model: str, done: bool = False) -> str:
    chunk = {
        "id": "chatcmpl-jarvis",
        "object": "chat.completion.chunk",
        "created": 0,
        "model": model,
        "choices": [
            {
                "index": 0,
                "delta": {} if done else {"content": text},
                "finish_reason": "stop" if done else None,
            }
        ],
    }
    return "data: " + json.dumps(chunk) + "\n\n"


# ---------- Gemini ----------

async def _gemini_stream(
    client: httpx.AsyncClient,
    model: str,
    messages: list,
    api_key: str,
) -> AsyncIterator[str]:
    """Gemini SSE stream kholo, status check karo, phir OpenAI-compatible
    chunks ka generator wapas karo. Status ghalat ho to yahin raise karo
    (StreamingResponse banne se pehle), taake caller Groq par ja sake."""
    system_instruction, contents = _to_gemini(messages)
    body: dict = {"contents": contents}
    if system_instruction:
        body["systemInstruction"] = system_instruction
    url = GEMINI_BASE + "/models/" + model + ":streamGenerateContent?alt=sse"
    resp_ctx = client.stream(
        "POST",
        url,
        headers={"x-goog-api-key": api_key, "Content-Type": "application/json"},
        json=body,
        timeout=HTTP_TIMEOUT,
    )
    resp = await resp_ctx.__aenter__()
    if resp.status_code != 200:
        await resp_ctx.__aexit__(None, None, None)
        log.info("provider=gemini status=%s", resp.status_code)
        raise RuntimeError("gemini status %s" % resp.status_code)

    async def gen() -> AsyncIterator[str]:
        try:
            async for line in resp.aiter_lines():
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if not data:
                    continue
                try:
                    text = _gemini_text(json.loads(data))
                except Exception:
                    continue
                if text:
                    yield _openai_chunk(text, model)
            yield _openai_chunk("", model, done=True)
            yield "data: [DONE]\n\n"
        finally:
            await resp_ctx.__aexit__(None, None, None)

    return gen()


async def _gemini_once(
    client: httpx.AsyncClient,
    model: str,
    messages: list,
    api_key: str,
) -> dict:
    system_instruction, contents = _to_gemini(messages)
    body: dict = {"contents": contents}
    if system_instruction:
        body["systemInstruction"] = system_instruction
    url = GEMINI_BASE + "/models/" + model + ":generateContent"
    resp = await client.post(
        url,
        headers={"x-goog-api-key": api_key, "Content-Type": "application/json"},
        json=body,
        timeout=HTTP_TIMEOUT,
    )
    if resp.status_code != 200:
        log.info("provider=gemini status=%s", resp.status_code)
        raise RuntimeError("gemini status %s" % resp.status_code)
    text = _gemini_text(resp.json())
    return {
        "id": "chatcmpl-jarvis",
        "object": "chat.completion",
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
            }
        ],
    }


# ---------- Groq (OpenAI-compatible) ----------

def _groq_model(model: str) -> str:
    """Gemini model Groq par nahi chalta — us surat mein default Groq
    model istemal karo; warna client ka model pass-through."""
    if model.startswith("gemini") or model.startswith("google/"):
        return GROQ_DEFAULT_MODEL
    return model


async def _groq_stream(
    client: httpx.AsyncClient,
    model: str,
    messages: list,
    api_key: str,
) -> AsyncIterator[str]:
    """Groq SSE stream — Gemini wali hi two-phase shape: pehle status,
    phir chunks relay karo (Groq pehle se OpenAI-compatible hai)."""
    groq_model = _groq_model(model)
    resp_ctx = client.stream(
        "POST",
        GROQ_BASE + "/chat/completions",
        headers={
            "Authorization": "Bearer " + api_key,
            "Content-Type": "application/json",
        },
        json={"model": groq_model, "messages": messages, "stream": True},
        timeout=HTTP_TIMEOUT,
    )
    resp = await resp_ctx.__aenter__()
    if resp.status_code != 200:
        await resp_ctx.__aexit__(None, None, None)
        log.info("provider=groq status=%s", resp.status_code)
        raise RuntimeError("groq status %s" % resp.status_code)

    async def gen() -> AsyncIterator[str]:
        try:
            async for line in resp.aiter_lines():
                if line.startswith("data:"):
                    yield line + "\n\n"
        finally:
            await resp_ctx.__aexit__(None, None, None)

    return gen()


async def _groq_once(
    client: httpx.AsyncClient,
    model: str,
    messages: list,
    api_key: str,
) -> dict:
    groq_model = _groq_model(model)
    resp = await client.post(
        GROQ_BASE + "/chat/completions",
        headers={
            "Authorization": "Bearer " + api_key,
            "Content-Type": "application/json",
        },
        json={"model": groq_model, "messages": messages, "stream": False},
        timeout=HTTP_TIMEOUT,
    )
    if resp.status_code != 200:
        log.info("provider=groq status=%s", resp.status_code)
        raise RuntimeError("groq status %s" % resp.status_code)
    return resp.json()


# ---------- Chat endpoint ----------

@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    if not _auth_ok(request):
        return _openai_error(401, "unauthorized")
    try:
        body = await request.json()
    except Exception:
        return _openai_error(400, "invalid json body")
    model = body.get("model") or "gemini-3-flash"
    messages = body.get("messages") or []
    stream = bool(body.get("stream", False))
    if not messages:
        return _openai_error(400, "messages required")

    gemini_key = request.headers.get("x-byok-gemini-key", "")
    groq_key = request.headers.get("x-byok-groq-key", "")
    if not gemini_key and not groq_key:
        return _openai_error(400, "no provider key supplied")

    async with _http_client() as client:
        # 1) Gemini pehle (client ka model pass-through)
        if gemini_key:
            try:
                if stream:
                    gen = await _gemini_stream(client, model, messages, gemini_key)
                    return StreamingResponse(gen, media_type="text/event-stream")
                return JSONResponse(
                    await _gemini_once(client, model, messages, gemini_key)
                )
            except Exception as exc:
                # Gemini fail -> Groq fallback (neeche). Error detail
                # kabhi response mein nahi bhejte.
                log.info("gemini failed, trying groq: %s", type(exc).__name__)
        # 2) Groq fallback
        if groq_key:
            try:
                if stream:
                    gen = await _groq_stream(client, model, messages, groq_key)
                    return StreamingResponse(gen, media_type="text/event-stream")
                return JSONResponse(
                    await _groq_once(client, model, messages, groq_key)
                )
            except Exception as exc:
                log.info("groq failed: %s", type(exc).__name__)
        return _openai_error(502, "provider request failed")
