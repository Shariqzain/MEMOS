# MEMOS
### Memory Evolution and Management for Organized Semantic Storage

[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![React](https://img.shields.io/badge/React-18+-61DAFB?logo=react&logoColor=black)](https://react.dev)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.0+-3178C6?logo=typescript&logoColor=white)](https://www.typescriptlang.org)
[![MongoDB](https://img.shields.io/badge/MongoDB-6.0+-47A248?logo=mongodb&logoColor=white)](https://www.mongodb.com)
[![Neo4j](https://img.shields.io/badge/Neo4j-5.12+-008CC1?logo=neo4j&logoColor=white)](https://neo4j.com)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector_Store-FF6F00)](https://www.trychroma.com)
[![Tests](https://img.shields.io/badge/Tests-40%20Passed-brightgreen)](backend/tests)

---

## 🌟 Overview

**MEMOS (Memory Evolution and Management for Organized Semantic Storage)** is an end-to-end, state-of-the-art conversational AI system with persistent long-term memory. 

Unlike traditional RAG systems that rely solely on static vector search, MEMOS integrates:
- **Tri-Store Architecture**: Relational/Document storage (**MongoDB**), Dense Semantic Embeddings (**ChromaDB**), and Associative Entity Graphs (**Neo4j**).
- **Hybrid Reciprocal Rank Fusion (RRF)**: Merges dense cosine similarity rankings with multi-hop graph proximity scores into a unified ranking function.
- **Dynamic Memory Evolution**: Detects semantic drift, reconciles factual contradictions, supersedes obsolete statements, and provides complete version control with point-in-time restore.
- **Modern Interactive Web Interface**: High-performance React 18 / TypeScript frontend featuring a chat interface with memory transparency, real-time memory management, entity graph visualization, and an interactive evaluation dashboard.

---

## 📐 Architecture

```
                    ┌────────────────────────────────────────────────────────┐
                    │               Modern React 18 Dashboard                │
                    │   (Chat, Memory Graph, Timeline, Version Inspection)   │
                    └───────────────────────────┬────────────────────────────┘
                                                │ REST API / CORS
                                                ▼
                    ┌────────────────────────────────────────────────────────┐
                    │                   FastAPI Backend                      │
                    │  (Async Python, Pydantic V2, Sentence-Transformers)    │
                    └───────────┬───────────────────┬───────────────────┬────┘
                                │                   │                   │
                 ┌──────────────┘                   │                   └──────────────┐
                 ▼                                  ▼                                  ▼
      ┌────────────────────┐             ┌────────────────────┐             ┌────────────────────┐
      │   MongoDB Store    │             │  ChromaDB Vector   │             │   Neo4j Graph      │
      │ Canonical Memories │             │  384-d Embeddings  │             │ Multi-Hop Entities │
      │ Version History    │             │  Cosine / HNSW     │             │ Relationship Edges │
      │ Conversations      │             │  all-MiniLM-L6-v2  │             │ Cypher Traversals  │
      └────────────────────┘             └────────────────────┘             └────────────────────┘
```

For detailed architectural specifications, read [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

---

## 🚀 Quickstart

### Option 1: Docker Compose (Recommended)

Start the entire stack (MongoDB, Neo4j, backend, and frontend) with a single command:

```bash
docker-compose up --build
```

- **Frontend Application**: [http://localhost:3000](http://localhost:3000)
- **FastAPI Backend & Interactive Swagger Docs**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **Neo4j Browser**: [http://localhost:7474](http://localhost:7474) (User: `neo4j`, Password: `memos_password`)

---

### Option 2: Local Development Setup

#### 1. Backend Setup
```bash
# Navigate to backend
cd backend

# Create virtual environment & activate
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment variables
cp ../.env.example .env

# Run FastAPI server
uvicorn app.main:app --reload --port 8000
```

#### 2. Frontend Setup
```bash
# In a separate terminal, navigate to frontend
cd frontend

# Install npm dependencies
npm install

# Run Vite dev server
npm run dev
```
Open [http://localhost:5173](http://localhost:5173) in your browser.

---

## ⚙️ Environment Configuration

Copy `.env.example` to `.env` and configure credentials:

| Variable | Default Value | Description |
|---|---|---|
| `APP_ENV` | `development` | Runtime environment (`development` / `production`) |
| `BACKEND_PORT` | `8000` | Port for FastAPI server |
| `MONGODB_URI` | `mongodb://localhost:27017` | MongoDB connection connection string |
| `MONGODB_DATABASE` | `memos_db` | Main database name |
| `CHROMADB_PATH` | `./data/chroma` | Persistent directory for ChromaDB vector index |
| `NEO4J_URI` | `bolt://localhost:7687` | Neo4j Bolt protocol URI |
| `NEO4J_USER` | `neo4j` | Neo4j username |
| `NEO4J_PASSWORD` | `password` | Neo4j password |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama local API endpoint |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | SentenceTransformers embedding model |

---

## 🧪 Testing & Evaluation

### Running Unit & Integration Tests

The test suite validates RRF algorithms, extraction rules, candidate filtering, Pydantic schemas, and FastAPI endpoints:

```bash
cd backend
python -m pytest tests/
```
Output:
```
tests/test_integration.py .......... [ 25%]
tests/test_unit.py .............................. [100%]
======================= 40 passed in 23.48s =======================
```

### Running the Evaluation Benchmark

Run the automated benchmark on the 12 curated scenarios:

```bash
cd backend
python scripts/run_evaluation.py
```

Results:
- **Recall@1**: 1.0000
- **Recall@3**: 1.0000
- **Recall@5**: 1.0000
- **MRR**: 1.0000
- **nDCG@3**: 1.0000

See [docs/EVALUATION.md](docs/EVALUATION.md) for full benchmark documentation and formulas.

---

## 📖 API Documentation

Complete REST API documentation is available at [docs/API.md](docs/API.md) or interactively at `http://localhost:8000/docs`.

Key API Endpoints:
- `POST /api/chat` — Chat with hybrid memory retrieval and auto-extraction
- `GET /api/memories` — Filter, paginate, and list active memories
- `PATCH /api/memories/{id}` — In-place update memory content or importance
- `DELETE /api/memories/{id}` — Soft delete and de-index a memory
- `GET /api/memories/{id}/history` — Retrieve full version audit trail
- `POST /api/memories/{id}/restore` — Revert memory to any past version
- `GET /evaluation/summary` — Global statistics on memories, categories, and conversations

---

## 📜 License

MIT License. Designed and implemented for research and production conversational AI systems.
