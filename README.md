# Zero-Trust Enterprise RAG

**100% on-premise** enterprise RAG system with **Zero-Trust** security applied
at the chunk level (2-dimensional RBAC clearance × department), hierarchical semantic routing
via Key Semantic Points (KSP), asymmetric hybrid retrieval
(ChromaDB + BM25). All LLM (`llama3.2` via Ollama) and embeddings model
(`paraphrase-multilingual-MiniLM-L12-v2`) run locally — no data ever leaves
the perimeter.

📺 **Demo in a single command**: `docker compose up -d` → `http://localhost:8501`

> Final Degree Project (URV) by me. Complete repository with chapters, diagrams and experimentation notebooks in `docs/` and `strategy-analysis/`. Full Documentation `.pdf` PENDING.

---

## Clean Architecture, 6 layers

```
Frontend (Streamlit)           ── tab Ingesta + tab Consulta + disclaimer IA
        ↓ HTTP + X-User-Id
API (FastAPI)                  ── 7 endpoints, schemas Pydantic, error mapper
        ↓
Application (orchestrators)    ── IngestionApp (HITL gate) + QueryApp
        ↓
Domain (regles de negoci)      ── chunking + routing + retrieval + users
        ↓
Infrastructure (adapters)      ── ChromaDB + BM25 + Ollama + Embedder + UserStore
        ↓
Core (cross-cutting)           ── Settings + Exceptions + ClearanceLevel/Department
```

**Ingestion Pipeline** (two-phase Human-In-The-Loop):

```
Upload doc ─▶ Parser (9 formats) ─▶ LLM classifier (proposa metadata)
                                              │
                                  User revisa i confirma (HITL)
                                              ▼
        Chunker custom (PII isolation + clearance escalation)
                                              ▼
        Cross-dept regex rules (extenen allowed_departments per chunk)
                                              ▼
        ChromaDB (chunks_<dept>) + BM25 refit + KSP Router-Index
```

**Query Pipeline** (3 short-circuits, defense in depth):

```
Query + User ─▶ HierarchicalRouter (intent → α; KSP RBAC → target_depts)
                       │
                       └─ buit? ─▶ refusal no_routing_match (sense invocar LLM)
                       │
                       ▼
              AsymmetricEnsembleRetriever (RBAC abans de RRF)
                       │
                       └─ buit? ─▶ refusal no_accessible_context (sense LLM)
                       │
                       ▼
              Prompt amb context + cites ─▶ Ollama llama3.2
                       │
                       ▼
              QueryResponse(answer, citations, telemetry)
```

---

## Start-up

### Full Compose Mode (recommended)

```bash
git clone <repo-url>
cd ai-rag-context-auth-system
cp .env.example .env                 # PowerShell: Copy-Item .env.example .env
docker compose up -d                 # primera vegada ~10 min (build + pull llama3.2)
```

When the 4 containers (`chromadb`, `ollama`, `api`, `streamlit`) are `healthy`, open **http://localhost:8501** in your browser.

### Hybrid Dev Mode  (fast iteration with hot-reload)

```bash
docker compose up -d chromadb ollama           # només les deps al docker
python -m venv .venv && source .venv/Scripts/activate
pip install -r requirements.txt
python -m scripts.run_api --reload             # backend natiu, hot-reload
python -m scripts.run_frontend                 # frontend natiu, hot-reload
```

### Verification

```bash
curl http://localhost:8000/health
curl http://localhost:8000/users
```

---

## Demo users

8 predefined identities in `data/demo_users.json` covering the 7 departments
× 4 clearance levels, so that the **RBAC differential** can be demonstrated in
live (the same document returns different answers depending on who asks):

| ID | Username | Department | Clearance |
|---|---|---|---|
| u-001 | Anna Garcia | hr | PUBLIC (0) |
| u-002 | Pere Soler | hr | CONFIDENTIAL (2) |
| u-003 | Marta Vidal | finance | STRICT (3) |
| u-004 | Joan Pla | engineering | CONFIDENTIAL (2) |
| u-005 | Laia Roca | legal | STRICT (3) |
| u-006 | Pau Costa | sales | INTERNAL (1) |
| u-007 | Núria Tena | operations | CONFIDENTIAL (2) |
| u-008 | Roger Albert | marketing | INTERNAL (1) |

It is selected from the Streamlit sidebar dropdown; the API receives the identity
via `X-User-Id` header. There is no OAuth (FDP demo); the `UserStore(ABC)` abstraction
allows future swap without touching business code.

---

## Utility Scripts Provided

| Command | Purpose |
|---|---|
| `python -m scripts.run_api --reload` | Launches the native FastAPI backend (hybrid dev) |
| `python -m scripts.run_frontend` | Launches the native Streamlit frontend |
| `python -m scripts.healthcheck` | Smoke test fail-fast of the whole stack (5 steps) |
| `python -m scripts.inspect_stores` | Inspects ChromaDB collections and BM25 indexes |

### `inspect_stores.py` — what's indexed?

Main utility to see what the system knows about the ingested documents,
with the same adapter instance that the retriever uses:

```bash
# List all Chroma collections + all BM25 indices
python -m scripts.inspect_stores

# Show the first 10 chunks of a specific collection
python -m scripts.inspect_stores --collection chunks_legal --limit 10

# Search all collections (BM25 lexicon + Vector Chroma)
python -m scripts.inspect_stores --search "warranty clause"

# Restrict to one layer
python -m scripts.inspect_stores --no-chroma --search "IBAN"      # només BM25
python -m scripts.inspect_stores --no-bm25 --collection ksp_router_index

# Inside the container (Full Compose mode)
docker exec zero_trust_api python -m scripts.inspect_stores
```

Shows for each chunk the key audit fields: `parent_doc_id`,
`chunk_index`, `source_file`, `department`, `allowed_departments` (including
extensions for cross-dept rules), `clearance_level` (including escalations for PII),
`contains_PII`, `sensitivity_types`, `cross_dept_rules_fired`. For Chroma also
shows the `embedding_model` stamped on each collection (guard against embedder mismatch).

---

## Repository Structure

```
├── core/                       # config + exceptions + security enums
├── infrastructure/             # ChromaDB, BM25, Ollama, Embedder, UserStore adapters
├── domain/
│   ├── chunking/               # CustomRBACChunker, PII patterns, cross-dept rules
│   ├── routing/                # intent classifier, KSPRouterIndex, HierarchicalRouter
│   ├── retrieval/              # RBAC filter, RRF fusion, AsymmetricEnsembleRetriever
│   └── users.py                # User value object
├── application/                # IngestionApp + QueryApp + LLMDocumentClassifier
├── api/                        # FastAPI: schemas, dependencies, errors, main
├── frontend/                   # Streamlit: streamlit_app, components, api_client
├── scripts/                    # healthcheck, inspect_stores, run_api, run_frontend, ...
├── strategy-analysis/          # Notebooks Jupyter de Fase 1-3 + corpus experimental
│   ├── notebooks/ph1-*         # Chunking + metadata RBAC
│   ├── notebooks/ph2-*         # Retrieval pipeline (5 passos + augment del corpus)
│   ├── notebooks/ph3-*         # Routing multiagent
│   └── data/raw_docs/          # 23 documents del corpus experimental
├── data/                       # Estat runtime (gitignored; creat al primer up)
│   ├── chroma_db/              #   índex HNSW
│   ├── bm25_indexes/           #   pickle per col·lecció
│   ├── ollama/                 #   model llama3.2 descarregat
│   ├── raw_docs/               #   buit; els uploads vivien temporals
│   └── demo_users.json         #   8 identitats demo
├── docs/                       # Memòria del TFG.pdf & avaluacio.xlsx
├── Dockerfile, Dockerfile.frontend, docker-compose.yml, .dockerignore
├── requirements.txt, .env.example, .gitignore
└── README.md
```

---

## Endpoints API (FastAPI)

| Method | Path | Auth | Description |
|---|---|---|---|
| GET | `/health` | — | Status of ChromaDB + Ollama + Embedder |
| GET | `/users` | — | List 8 demo users (for the dropdown) |
| GET | `/users/{id}` | — | Resolve an identity (404 if unknown) |
| POST | `/ingestion/propose` | `X-User-Id` | Upload + LLM classifier → IngestionProposal                  |
| POST | `/ingestion/commit` | `X-User-Id` | Commit with HITL-confirmed metadata → IngestionReport |
| POST | `/query` | `X-User-Id` | Route + retrieve + LLM → QueryResponse (refusals = HTTP 200) |

Interactive documentation at `http://localhost:8000/docs` (Swagger UI generated by FastAPI).

---

## License and notice

**© 2026 [Arnau Faura i Ciré]. All rights reserved.**
This source code, architecture and documentation are part of a Final Degree Project (TFG) developed for the Universitat Rovira i Virgili (URV). Reproduction, distribution, public communication or total or partial transformation of this project is prohibited without the express written authorization of the author, except for strictly academic purposes and evaluation by the university itself.

**Disclaimer on Generative AI:**
This system uses a generative Extensive Language Model (LLM) (Llama 3.2 via Ollama) for text processing and generation, which implies that it may produce inaccuracies or unwanted responses. In order to guarantee transparency, all artifacts generated by Artificial Intelligence (including metadata proposals upon ingestion and final responses to queries) incorporate an explicit AI legal notice in the user interface. This measure aligns with the transparency requirements set out in the EU Artificial Intelligence Act (EU AI Act). The end user assumes the obligation and responsibility to review and validate any critical content emitted by the system before using it for decision-making.
