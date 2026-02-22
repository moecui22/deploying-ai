import os
import re
import csv
import json
import math
import hashlib
import urllib.parse
import urllib.request

import gradio as gr
import chromadb


APP_TITLE = "Assignment 2 Chat — Helpful Course TA"

# Refusals: cats/dogs, zodiac/horoscopes
RESTRICTED_KEYWORDS = [
    "cat", "cats", "dog", "dogs",
    "horoscope", "horoscopes", "zodiac",
    "aries", "taurus", "gemini", "cancer", "leo", "virgo", "libra",
    "scorpio", "sagittarius", "capricorn", "aquarius", "pisces"
]

# Guardrails: do not reveal/modify system prompt or instructions
PROMPT_ATTACK_KEYWORDS = [
    "system prompt", "reveal prompt", "show prompt",
    "developer message", "hidden instructions",
    "ignore previous instructions", "jailbreak"
]


def guardrail_check(user_text: str):
    t = user_text.lower()
    for k in PROMPT_ATTACK_KEYWORDS:
        if k in t:
            return "I can not help with revealing or changing system instructions. Tell me your goal and I’ll help safely."
    for k in RESTRICTED_KEYWORDS:
        if re.search(r"\b" + re.escape(k) + r"\b", t):
            return "I can not help with that topic for this assignment. Try weather, FAQ search, or a tool (calculator/unit conversion)."
    return None


def get_json(url: str):
    req = urllib.request.Request(url, headers={"User-Agent": "assignment-chat/1.0"})
    with urllib.request.urlopen(req, timeout=10) as resp:
        return json.loads(resp.read().decode("utf-8"))


def weather_service(message: str):
    text = message.strip()

    m       = re.search(r"weather\s+in\s+(.+)", text, re.IGNORECASE)
    city    = m.group(1).strip() if m else text

    q       = urllib.parse.quote(city)
    geo_url = f"https://geocoding-api.open-meteo.com/v1/search?name={q}&count=1&language=en&format=json"
    geo     = get_json(geo_url)

    if not geo.get("results"):
        return f"I couldn’t find '{city}'. Try a nearby major city (e.g., Toronto, Vancouver)."

    r       = geo["results"][0]
    lat     = r["latitude"]
    lon     = r["longitude"]
    place   = f"{r.get('name','')}, {r.get('admin1','')}, {r.get('country','')}".strip(", ")

    w_url   = (
        "https://api.open-meteo.com/v1/forecast"
        f"?latitude={lat}&longitude={lon}"
        "&current=temperature_2m,wind_speed_10m"
        "&timezone=auto"
    )
    w       = get_json(w_url)

    cur     = w.get("current", {})
    temp    = cur.get("temperature_2m")
    wind    = cur.get("wind_speed_10m")
    tstamp  = cur.get("time", "")

    if temp is None or wind is None:
        return f"I found {place}, but the weather data was incomplete."

    return (
        f"Weather for {place}\n"
        f"- Time (local): {tstamp}\n"
        f"- Temperature: {temp}°C\n"
        f"- Wind: {wind} km/h"
    )


EMBED_DIM = 256


def simple_embed(text: str):
    vec     = [0.0] * EMBED_DIM
    tokens  = re.findall(r"[a-z0-9']+", text.lower())
    if not tokens:
        return vec

    for tok in tokens:
        h   = hashlib.md5(tok.encode("utf-8")).hexdigest()
        idx = int(h[:8], 16) % EMBED_DIM
        vec[idx] += 1.0
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


def load_faq(csv_path: str):
    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    return rows


def get_collection():
    base        = os.path.dirname(__file__)
    persist_dir = os.path.join(base, "chroma_db")
    csv_path    = os.path.join(base, "data", "faq.csv")
    client      = chromadb.PersistentClient(path=persist_dir)
    col         = client.get_or_create_collection("faq_collection")

    if col.count() > 0:
        return col

    rows = load_faq(csv_path)
    ids, docs, metas, embs = [], [], [], []

    for r in rows:
        rid     = (r.get("id") or "").strip()
        q       = (r.get("question") or "").strip()
        a       = (r.get("answer") or "").strip()
        if not rid or not q or not a:
            continue

        doc = f"Q: {q}\nA: {a}"
        ids.append(rid)
        docs.append(doc)
        metas.append({"question": q})
        embs.append(simple_embed(doc))

    col.add(ids=ids, documents=docs, metadatas=metas, embeddings=embs)
    return col


def semantic_service(question: str):
    col         = get_collection()
    q_emb       = simple_embed(question)

    res = col.query(
        query_embeddings=[q_emb],
        n_results=3,
        include=["documents", "metadatas"]
    )
    docs        = res["documents"][0]
    metas       = res["metadatas"][0]

    if not docs:
        return "I didn’t find anything relevant in the FAQ dataset. Try rephrasing or add more rows to `data/faq.csv`."

    top_doc         = docs[0]
    best_answer     = top_doc.split("\nA:", 1)[-1].strip() if "\nA:" in top_doc else top_doc

    matched_questions = []
    for m in metas:
        matched_questions.append("- " + m.get("question", "Relevant item"))

    return (
        "From the local FAQ dataset, the closest matches were:\n"
        + "\n".join(matched_questions)
        + "\n\nBest answer:\n"
        + best_answer
    )



def tool_service(message: str):
    t = message.lower().strip()

    if "calculate" in t or re.search(r"[0-9]\s*[\+\-\*\/]\s*[0-9]", t):
        exprs   = re.findall(r"[0-9\.\+\-\*\/\(\)\s]+", message)
        expr    = max(exprs, key=len).strip() if exprs else ""
        if not expr:
            return "Try: calculate (42*3)+2"
        if not re.fullmatch(r"[0-9\.\+\-\*\/\(\)\s]+", expr):
            return "I can only calculate basic numbers and + - * / ( )."
        try:
            val = eval(expr, {"__builtins__": {}}, {})
            return f"Tool used: calculate\nResult: {val}"
        except Exception:
            return "I couldn’t evaluate that. Try something like: calculate (17*3)+2"

    m = re.search(r"convert\s+([0-9\.]+)\s*([a-zA-Z]+)\s+to\s+([a-zA-Z]+)", t)
    if m:
        value   = float(m.group(1))
        u_from  = m.group(2).lower()
        u_to    = m.group(3).lower()

        if u_from == "km" and u_to in ["mile", "miles"]:
            out     = value * 0.621371
            return f"Tool used: convert\n{value} km = {round(out, 4)} miles"
        if u_from in ["mile", "miles"] and u_to == "km":
            out     = value / 0.621371
            return f"Tool used: convert\n{value} miles = {round(out, 4)} km"
        if u_from in ["c", "celsius"] and u_to in ["f", "fahrenheit"]:
            out     = (value * 9/5) + 32
            return f"Tool used: convert\n{value} C = {round(out, 4)} F"
        if u_from in ["f", "fahrenheit"] and u_to in ["c", "celsius"]:
            out     = (value - 32) * 5/9
            return f"Tool used: convert\n{value} F = {round(out, 4)} C"
        return "Supported conversions: km↔miles, C↔F. Example: convert 10 km to miles"
    return "Tool examples: calculate (17*3)+2  |  convert 10 km to miles"


def choose_service(user_text: str):
    t = user_text.lower()
    if "weather" in t or "forecast" in t or "temperature" in t:
        return "weather"
    if "faq" in t or "dataset" in t or "according to" in t:
        return "semantic"
    if "calculate" in t or "convert" in t or re.search(r"[0-9]\s*[\+\-\*\/]\s*[0-9]", t):
        return "tool"
    return "chat"


def normal_chat_reply():
    return (
        "I can help with:\n"
        "1) Weather (API): weather in Toronto\n"
        "2) FAQ search (vector DB): according to the FAQ, what are guardrails?\n"
        "3) Tools (function calling): calculate (17*3)+2  |  convert 10 km to miles\n\n"
        "What do you want to try?"
    )


def respond(user_text, history):
    blocked = guardrail_check(user_text)
    if blocked:
        history.append((user_text, blocked))
        return "", history
    which = choose_service(user_text)
    if which == "weather":
        answer = weather_service(user_text)
    elif which == "semantic":
        answer = semantic_service(user_text)
    elif which == "tool":
        answer = tool_service(user_text)
    else:
        answer = normal_chat_reply()
    history.append((user_text, answer))
    return "", history


def main():
    with gr.Blocks(title=APP_TITLE) as demo:
        gr.Markdown(f"# {APP_TITLE}")
        gr.Markdown(
            "Try:\n"
            "- weather in Toronto\n"
            "- according to the FAQ, what are guardrails?\n"
            "- calculate (17*3)+2\n"
            "- convert 10 km to miles"
        )

        chatbot = gr.Chatbot(height=420)
        state   = gr.State([])
        box     = gr.Textbox(label="Message", placeholder="Type here...", lines=1)
        btn     = gr.Button("Send")

        btn.click(respond, inputs=[box, state], outputs=[box, chatbot])
        btn.click(lambda h: h, inputs=[chatbot], outputs=[state])

        box.submit(respond, inputs=[box, state], outputs=[box, chatbot])
        box.submit(lambda h: h, inputs=[chatbot], outputs=[state])
    demo.launch()


if __name__ == "__main__":
    main()