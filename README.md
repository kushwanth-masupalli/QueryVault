# QueryVault — README (First Edition)

> **QueryVault** is a document QA & semantic search system that lets you upload documents (PDF / DOCX / text), extract meaningful propositions, convert them into vector embeddings, store them in a Pinecone vector index, and answer natural-language queries by retrieving the most relevant chunks.

---

## 🔍 Project overview

QueryVault is an end-to-end pipeline for building a fast, accurate document question-answering system. This first edition focuses on:

* Extracting short "propositions" (meaningful sentences) from paragraphs using a prompt-based LLM workflow.
* Producing 384-dimensional embeddings via `sentence-transformers` (`all-MiniLM-L6-v2`).
* Storing embeddings + metadata (`content`) in Pinecone (serverless index).
* Offering query clients in Python and Node.js that return the original text alongside similarity scores.

This repo is ideal for students and engineers who want a practical, production-minded starting point for document retrieval and QA.

---

## ⚙️ Key features (First Edition)

* LLM-powered proposition extraction (LangChain / LangSmith / Google Gemini integration).
* Local embeddings with `sentence-transformers` (fast 384-D vectors).
* Pinecone (vector DB) storage with metadata support.
* Query clients for both Python and JavaScript (direct REST queries and SDK examples).
* Repair utilities to re-upsert metadata if it was accidentally overwritten.
* Helpful debugging & defensive patterns for working across SDKs/versions.

---

## 🧱 Tech stack

* Python (data processing, upsert scripts)

  * `sentence-transformers` (`all-MiniLM-L6-v2`) — 384-D embeddings
  * LangChain / LangSmith / Google Gemini (proposal extraction)
  * Pinecone new SDK (`Pinecone` class)
* Node.js (query client)

  * `axios` for direct REST queries
  * optional local embedding using `@xenova/transformers`
* Pinecone (vector database)
* Optional: ffmpeg / Whisper / other tools for advanced pipelines (future)

---

## 🔁 High-level architecture

1. Text input (PDF / DOCX / raw text) → chunk into paragraphs.
2. Run LLM prompt to extract concise propositions (short sentences).
3. Compute embeddings for each proposition with `sentence-transformers`.
4. Upsert vectors into Pinecone with metadata `{ content: "..." }`.
5. Query: convert user query → embedding → Pinecone query (includeMetadata: true).
6. Return best matches + metadata content → final answer / downstream LLM.

---

## 🚀 Quickstart (local)

### Prerequisites

* Python 3.10+ and pip
* Node.js 18+
* A Pinecone account + index (dimension 384)
* (Optional) Google Gemini API key if you use Gemini LLM flows

### Install (Python)

```bash
python -m venv .venv
source .venv/bin/activate    # or .\.venv\Scripts\activate on Windows
pip install -r requirements.txt
```

`requirements.txt` should include (example):

```
sentence-transformers
pinecone-client
langchain
langchain-google-genai
pydantic
langsmith
```

### Install (Node.js client)

```bash
npm install
# or
pnpm install
```

---

## 🔧 Configuration (.env)

Create a `.env` file for local development and add these values:

```
PINECONE_API_KEY=<your-pinecone-api-key>
PINECONE_INDEX_URL=https://<index>-<project>.svc.<region>.pinecone.io
PINECONE_NAMESPACE=          # optional
GOOGLE_API_KEY=<optional-google-gemini-key>
```

> **Security note:** Do not commit `.env` to git. Rotate any keys accidentally published.

---

## 🛠 Usage examples

### 1) Upsert (Python)

A minimal snippet (ref: `chunking.py`) — make sure to include `metadata` when building vectors:

```python
vectors = []
for i, sent in enumerate(propositions):
    embedding = embedding_model.encode(sent).tolist()
    vectors.append({
        "id": f"prop_{i}",
        "values": embedding,
        "metadata": {"content": sent}
    })

index.upsert(vectors=vectors)
```

**Important:** If you upsert the same `id` later **without** `metadata`, Pinecone will overwrite and drop the metadata. If you see missing text on query, re-upsert with metadata.

### 2) Fetch to verify metadata (Python)

```python
resp = index.fetch(ids=["prop_0","prop_1"])
# In the new SDK resp is a FetchResponse object — use resp.vectors
for vid, vec in resp.vectors.items():
    print(vid, vec.metadata)
```

### 3) Query (Node.js, direct REST)

Make sure your query body uses the correct param name: `includeMetadata: true` (NOT `includeMeta`):

```js
const body = {
  vector: queryEmbedding,
  topK: 3,
  includeMetadata: true,
  includeValues: false,
  namespace: process.env.PINECONE_NAMESPACE || undefined
}

// POST to `${PINECONE_INDEX_URL}/query` with Api-Key header
```

Then print defensively in JS:

```js
results.matches.forEach((match, i) => {
  console.log(`${i+1}. Score: ${match.score.toFixed(4)}`)
  console.log(`ID: ${match.id}`)
  console.log(`Text: ${match.metadata?.content ?? match.meta?.content ?? "⚠️ No metadata"}`)
})
```

---

## 🐞 Troubleshooting

**Problem:** Query returns only `id` and vectors, no text.

* ✅ Confirm Python upsert included `metadata` (see Upsert snippet).
* ✅ Confirm JavaScript query sets `includeMetadata: true`.
* ✅ Confirm both Python and JS use the **same** index URL / namespace / project.
* ✅ Use `index.fetch()` in Python to inspect stored metadata directly.

**Problem:** `FetchResponse` object raises `AttributeError: 'FetchResponse' object has no attribute 'get'`.

* Use `resp.vectors` (new SDK) instead of `resp.get(...)`.

**Problem:** Dimension mismatch on upsert/query.

* Verify your embedding length is 384 (all-MiniLM-L6-v2) and your Pinecone index dimension is 384.

---

## ♻️ Repairing missing metadata (safe re-upsert)

If metadata was overwritten, you can fetch existing vector values and re-upsert with `metadata` to restore text without recomputing embeddings:

```python
fetched = index.fetch(ids=all_ids)
fetched_vectors = fetched.vectors
vectors_to_upsert = []
for i, sent in enumerate(propositions):
    vid = f"prop_{i}"
    values = fetched_vectors.get(vid).values if vid in fetched_vectors else embedding_model.encode(sent).tolist()
    vectors_to_upsert.append({"id": vid, "values": values, "metadata": {"content": sent}})
index.upsert(vectors=vectors_to_upsert)
```

> For large datasets, upsert in batches (e.g. 200 vectors per request).

---

## ✅ Recommended best practices

* Always include `metadata` when upserting.
* Use explicit `namespace` if you separate datasets.
* Keep API keys in environment variables or secret stores.
* Add logging around upserts/queries to track environment and namespace.
* Re-run `fetch()` to verify stored metadata after major changes.

---

## 🛣 Roadmap (next features)

* Add a web UI for uploads, queries, and result inspection.
* Add support for larger embedding models and multimodal embeddings (images).
* Add an LLM-backed answer synthesis step that uses retrieved contexts.
* Implement paging / streaming for large result sets.
* Add unit tests and CI checks for end-to-end pipeline.

---

## 🙋 Contribution

Contributions, bug reports, and feature requests are welcome! Please open an issue describing the change and, if possible, include a minimal reproduction.

Suggested workflow:

1. Fork the repo
2. Create a branch: `feature/your-change`
3. Add tests and documentation
4. Open a pull request

---

## 📜 License

This project is released under the MIT License — modify as you need.

---

## 📬 Contact / Credits

Built by the QueryVault team. For help or questions, open an issue or contact the maintainers via the project repo.

---

