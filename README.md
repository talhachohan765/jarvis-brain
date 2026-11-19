# Jarvis Brain — slim relay server (v1)

Chhota FastAPI server jo Jarvis Android app ke liye Gemini/Groq ka
relay karta hai. App apni keys har request par header mein bhejti hai
(BYOK) — server par koi key save nahi hoti.

## Endpoints

- `GET /health` -> `{"status":"ok"}` (Render health check, no auth)
- `GET /v1/models` -> model list (Bearer auth)
- `POST /v1/chat/completions` -> OpenAI-compatible, `stream:true` SSE
  (Bearer auth + `X-BYOK-Gemini-Key` / `X-BYOK-Groq-Key`)

## Flow

1. Bearer key check (`BRAIN_KEY` env var) — ghalat/missing -> 401.
2. Gemini key mile to pehle Gemini REST (`streamGenerateContent`,
   key sirf `x-goog-api-key` header mein — kabhi URL mein nahi).
3. Gemini fail ho to Groq (`https://api.groq.com/openai/v1`) fallback.
   Gemini model Groq par nahi chalta, is liye fallback mein
   `openai/gpt-oss-120b` istemal hota hai.
4. Jawab OpenAI-compatible SSE chunks mein (`choices[0].delta.content`),
   aakhir mein `data: [DONE]`.

## Run (local)

```bash
BRAIN_KEY=test-secret uvicorn main:app --host 0.0.0.0 --port 8000
```

## Deploy (Render, free tier)

- Runtime: Python, region oregon
- Build: `pip install -r requirements.txt`
- Start: `uvicorn main:app --host 0.0.0.0 --port $PORT`
- Health check path: `/health`
- Env vars: `PORT=10000`, `BRAIN_KEY=<random, Render par set karo>`

Koi key, URL ya header value kabhi log/print/file mein nahi aati.
