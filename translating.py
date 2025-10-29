import os
import time
from pathlib import Path
import google.generativeai as genai
from dotenv import load_dotenv

start_time = time.time()

load_dotenv()

IN_DIR = Path("Letras")
OUT_DIR = Path("Lyrics")
MODEL_ID = "gemini-2.5-pro" 
TEMPERATURE = 0.0

PROMPT_HEADER = (
    "Translate the following text from Portuguese to English.\n\n"
    "Hard rules:\n"
    "- Do NOT soften language, do NOT paraphrase, do NOT adapt style.\n"
    "- Preserve punctuation and the exact line structure (one input line -> one output line, same order).\n"
    "- Keep slang and profanity literal when possible.\n"
    "- Output ONLY the translated text. No comments, no prefixes, no code fences.\n\n"
    "Text:\n"
)

# pra evitar que ele bloqueie minhas letras mais pesadas
SAFETY = [ 
    {"category":"HARM_CATEGORY_HATE_SPEECH","threshold":"BLOCK_NONE"},
    {"category":"HARM_CATEGORY_HARASSMENT","threshold":"BLOCK_NONE"},
    {"category":"HARM_CATEGORY_SEXUAL","threshold":"BLOCK_NONE"},
    {"category":"HARM_CATEGORY_DANGEROUS_CONTENT","threshold":"BLOCK_NONE"},
]

# extrai o texto da resposta
def _extract_text(resp):
    # tenta obter a lista de candidatos
    cands = getattr(resp, "candidates", None)
    if not cands:
        return None
    
    for cand in cands:
        content = getattr(cand, "content", None)
        if not content:
            continue

        parts = getattr(content, "parts", None)
        if not parts:
            continue

        out = []
        for p in parts:
            if isinstance(p, dict):
                t = p.get("text")
            else:
                t = getattr(p, "text", None)
            if t:
                out.append(t)

        if out:
            return "\n".join(out).strip()

    return None

# traduz o texto, dividindo se necessário
def translate_text(model, text: str) -> str:
    resp = model.generate_content(PROMPT_HEADER + text)
    out = _extract_text(resp)
    if out is not None:
        return out

    cand = resp.candidates[0] if getattr(resp, "candidates", None) else None
    if getattr(cand, "finish_reason", None) == 2:
        mid = len(text) // 2
        cut = text.rfind("\n", 0, mid)
        if cut == -1:
            cut = text.find("\n", mid)
            if cut == -1:
                cut = mid
        left = translate_text(model, text[:cut])
        right = translate_text(model, text[cut+1:])
        return (left + "\n" + right).strip()

    raise RuntimeError(f"Sem texto. finish_reason={getattr(cand, 'finish_reason', None)}")

def main():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("Falta a api key.")
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(MODEL_ID)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    files = sorted([p for p in IN_DIR.glob("*.txt") if p.is_file()])
    if not files:
        print("Nenhum .txt :(")
        return

    total = len(files)
    for i, src in enumerate(files, 1):
        print(f"[{i}/{total}] Traduzindo: {src.name}...", flush=True)
        text = src.read_text(encoding="utf-8", errors="ignore")
        if not text.strip():
            print(f"Vazio, pulando: {src.name}")
            continue

        try:
            en = translate_text(model, text)
            (OUT_DIR / src.name).write_text(en, encoding="utf-8")
            print(f"[{i}/{len(files)}] OK -> {src.name}")
        except Exception as e:
            print(f"[{i}/{len(files)}] ERRO em {src.name}: {e}")
            time.sleep(1)
        time.sleep(0.4)

if __name__ == "__main__":
    main()

end_time = time.time()
print(f"Tempo total de execução: {end_time - start_time:.2f} segundos")