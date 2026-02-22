# Assignment 2 — Conversational AI System: a helpful TA for a course

## Personality
This chat client behaves like a helpful TA:
- Direct answers with examples
- Practical suggestions
- Concise, logical responses

## Services

### Service 1 — API Calls (Open-Meteo Weather)
Users can ask for weather (e.g., What is the weather in Toronto).
The app calls Open-Meteo (no API key) and convert the JSON into a short natural-language summary.

### Service 2 — Semantic/Hybrid Query (ChromaDB)
Users can ask questions that are answered from a local dataset (`data/faq.csv`) using retrieval.
Implementation:
- Uses **ChromaDB with persistence** (`chroma_db/`)
- Embeddings are created with a simple hashing-based embedding and stored in Chroma.
- This is best described as hybrid retrieval, but it demonstrates vector search.

### Service 3 — Function Calling Tools (Calculator + Unit Conversion)
Tools:
- `calculate(expression)`
- `convert_units(value, unit_from, unit_to)`

## Chat UI + Memory
The UI is built with Gradio.
Conversation history is stored in `gr.State` and maintained throughout the session.

## Guardrails
The system refuses:
- Requests to reveal or modify system instructions (system prompt leakage attempts)
- Topics: cats/dogs, horoscopes/zodiac, Taylor Swift

## How to run
From the repository root:

```bash
cd 05_src/assignment_chat
python app.py