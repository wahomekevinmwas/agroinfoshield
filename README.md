# 🌱 AgroInfoShield

**AI-Powered GMO Myth-Busting Dashboard for African Farmers and Policymakers**

> Submit a claim about GMO crops in English, Swahili, or Sheng — get a clear, evidence-backed verdict in seconds.

[![Python](https://img.shields.io/badge/Python-3.13-blue?logo=python)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.58-red?logo=streamlit)](https://streamlit.io)
[![Groq](https://img.shields.io/badge/Groq-Llama_3.3_70B-orange)](https://groq.com)
[![FAISS](https://img.shields.io/badge/FAISS-Vector_Search-purple)](https://faiss.ai)

---

## Dashboard Preview

![AgroInfoShield Chat Interface](assets/Screenshot%202026-06-08%20095952.png)
*Multilingual chat interface — accepts claims in English, Swahili, and Sheng*

![AgroInfoShield Verdict Result](assets/Screenshot%202026-06-08%20100055.png)
*Structured verdict with confidence score, Swahili explanation, sources, and related knowledge base entries*

---

## The Problem

GMO misinformation is a major barrier to food security in Kenya and East Africa. A 2023 study found that **40% of Kenyan media coverage** on GMOs contained unchallenged negative misinformation. Farmers, extension officers, and policymakers lack a fast, reliable, multilingual tool to verify claims they encounter daily.

AgroInfoShield closes that gap.

---

## What It Does

A user submits a claim or question in any language. The system:

1. **Detects the language** — English, Swahili, or Sheng automatically
2. **Retrieves evidence** — FAISS semantic search finds the most relevant entries in the verified knowledge base
3. **Generates a verdict** — Groq LLM (Llama 3.3 70B) returns a structured response grounded in retrieved evidence
4. **Returns a clear result** — Myth / Fact / Partially True with explanation, confidence score, and cited sources

---

## Example Queries

| Language | Input | Verdict |
|----------|-------|---------|
| English | `Are GMO crops safe for human consumption?` | ✅ Fact |
| Swahili | `Je, mazao ya GMO ni salama kula?` | ✅ Fact (in Swahili) |
| Sheng | `Hiyo GMO ni poa ama ni poison?` | ❌ Myth |
| Mixed | `gmo inacause cancer kweli?` | ❌ Myth (in Swahili) |

---

## Target Users

| User | Language | Use Case |
|------|----------|----------|
| Smallholder farmers | Swahili / Sheng | Verify claims from social media |
| Agricultural extension officers | Swahili / English | Farmer education |
| County agricultural officers | English / Swahili | Policy-grounded responses |
| Policymakers and researchers | English | Evidence-based decision support |
| Young farmers and market vendors | Sheng | Accessible myth-busting |

---

## Tech Stack

| Component | Technology | Why |
|-----------|-----------|-----|
| Dashboard | Python + Streamlit | Fast, clean, mobile-friendly |
| LLM | Groq API — Llama 3.3 70B | Free tier, fastest inference available |
| RAG Orchestration | LangChain | Standard for production RAG pipelines |
| Vector Search | FAISS | Local, fast, no cloud dependency |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) | Free, local, multilingual |
| Knowledge Base | myth_facts.json | 20 verified GMO entries, East Africa sourced |

---

## Architecture

```
User Query (English / Swahili / Sheng)
         │
         ▼
  language/detector.py        ← Detects query language
         │
         ▼
  rag/retriever.py             ← FAISS semantic search
         │
         ▼
  rag/pipeline.py              ← Formats context + calls Groq
         │
         ▼
  Groq LLM (Llama 3.3 70B)    ← Generates grounded verdict
         │
         ▼
  dashboard/chat.py            ← Renders verdict card
```

```
agroinfoshield/
├── app.py                  ← Streamlit entry point
├── rag/
│   ├── loader.py           ← Load and chunk myth_facts.json
│   ├── embedder.py         ← Generate embeddings + FAISS index
│   ├── retriever.py        ← Semantic search
│   └── pipeline.py         ← Full RAG orchestration
├── verdict/                ← Verdict engine (Phase 2)
├── language/               ← Language detection
├── dashboard/
│   └── chat.py             ← Chat interface
├── enrichment/             ← KB enrichment pipeline (Phase 2)
└── data/
    └── myth_facts.json     ← Verified GMO knowledge base
```

---

## Knowledge Base Sources

| Source | Type |
|--------|------|
| ISAAA AfriCenter (Nairobi) | African biotech authority |
| KALRO | Kenya national agricultural research |
| VIRCA Plus Project | GM cassava — Kenya & Uganda |
| SSSfA Project | Striga-resistant sorghum — Kenya & Ethiopia |
| Kenya National Biosafety Authority (NBA) | Regulatory body |
| WHO / FAO | International health and food safety |

---

## Setup

```bash
# 1. Clone the repo
git clone https://github.com/wahomekevinmwas/agroinfoshield.git
cd agroinfoshield

# 2. Create virtual environment
python -m venv venv
venv\Scripts\activate   # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set up environment variables
# Create .env file and add your Groq API key:
# GROQ_API_KEY=your_key_here
# Get free key at: console.groq.com

# 5. Run the dashboard
python -m streamlit run app.py
```

---

## Roadmap

**Phase 1 — Core RAG (complete)**
- [x] Knowledge base — 20 verified GMO entries
- [x] FAISS vector search
- [x] Groq LLM verdict generation
- [x] English + Swahili + Sheng support
- [x] Streamlit dashboard

**Phase 2 — Enrichment Pipeline**
- [ ] Google Gemini document ingestion from ISAAA/KALRO PDFs
- [ ] Web monitoring for new GMO myths in Kenyan media
- [ ] Human review queue for new KB entries

**Phase 3 — Production**
- [ ] Explore tab — browse full knowledge base
- [ ] About tab — sources and methodology
- [ ] Mobile-optimised layout
- [ ] WhatsApp integration for feature phone users

---

## Built By

**Kevin Wahome** — Python/Django Developer & Linux Systems Administrator
MSc Computer Science (Computational Intelligence), University of Nairobi

LinkedIn: www.linkedin.com/in/kevin-wahome-0340a5325 
GitHub: github.com/wahomekevinmwas

Also built: [LivestockAI](https://github.com/wahomekevinmwas/livestock-monitor) — Django livestock disease monitoring with Z-score anomaly detection
---

## License

MIT License — Open for use by African agricultural institutions and researchers.
