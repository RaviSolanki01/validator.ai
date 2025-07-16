# Validator-AI: Generative AI-Powered Compliance Auditor

## 🚀 Overview
**Validator-AI** is a modular, scalable MVP to analyze and audit organizational compliance against regulatory documents (e.g., ISO, GDPR, SOC 2). It uses generative AI + RAG (Retrieval-Augmented Generation) to:

- Parse and index regulatory documents
- Ingest company internal policy docs
- Analyze for compliance gaps
- Generate audit-ready reports and alerts

---

## 🔧 Features
- 📄 Document parsing (PDF, DOCX, scanned OCR)
- 🧠 RAG engine with vector-based semantic search
- 🤖 LLM integration (OpenAI, HuggingFace)
- 📊 Streamlit-based dashboard for gap visualization
- 🔐 OAuth and Vault-based credential security
- 📈 Monitoring with Prometheus + Grafana

---

## 🧱 Project Structure
```bash
Validator-AI/
│
├── config/                # Config files (env, logging, flags)
├── core/                  # Business logic (ingestion, RAG, compliance)
├── infrastructure/        # Vector DB, Redis, LLM APIs
├── api/                   # FastAPI-based interface
├── ui/                    # Streamlit frontend (chat + dashboards)
├── ops/                   # Monitoring, CI/CD, deploy scripts
├── tests/                 # Unit, integration, E2E tests
├── scripts/               # CLI scripts for bulk ops
├── Dockerfile             # API + worker build
├── docker-compose.yml     # Orchestration (API, DBs, etc.)
├── .env.example           # Secrets & keys template
└── pyproject.toml         # Dependency & tool management
```

---

## 🔄 Tech Stack
| Layer           | Tools Used                            |
|----------------|----------------------------------------|
| Backend        | FastAPI, Pydantic, Celery              |
| LLM Interface  | OpenAI API, HuggingFace Transformers   |
| Vector DB      | Qdrant / Milvus                        |
| Document IO    | Apache Tika, pdfplumber, Tesseract     |
| Embeddings     | SentenceTransformers, LangChain        |
| Frontend       | Streamlit / React                      |
| Security       | OAuth, Vault                           |
| Monitoring     | Prometheus, Grafana, ELK Stack         |

---

## 🛠️ Getting Started
1. Clone the repo: `git clone <repo-url>`
2. Create `.env` from `.env.example`
3. Start services: `make up`
4. Visit: `http://localhost:8000/docs` for API & `http://localhost:8501` for Streamlit UI

---

## 📌 Roadmap
- [ ] LLM-based reranking module
- [ ] Multi-regulation comparison view
- [ ] Role-based access control
- [ ] Auto-upload from Google Drive/S3

---

## 📄 License
MIT © 2025 Validator-AI Team
