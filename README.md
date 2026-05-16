# ERP Market Intelligence — NLP Pipeline for Competitive Analysis

> **INST 664: Transforming Unstructured Content with AI**
> University of Maryland, iSchool — Spring 2026
> Priyanshu Gupta · Abinash Raj Rajarajan · Aravind A M

A multi-method NLP pipeline that transforms unstructured G2.com software reviews into structured competitive intelligence. The system collects 14,933 reviews across twelve ERP products, identifies recurring themes through topic modeling, scores aspect-level sentiment, and extracts product switching signals — producing outputs a product team can act on.

---

## Products Analyzed

| Product | Reviews | Switch % |
|---|---|---|
| Oracle NetSuite | 2,000 | 35.8% |
| SAP S/4HANA Cloud | 938 | 12.5% |
| Microsoft Dynamics 365 BC | 903 | 17.9% |
| QuickBooks Enterprise | 1,055 | 12.1% |
| Intuit Enterprise Suite | 21 | 28.6% |
| Sage Intacct | 2,000 | 32.5% |
| Workday Financial Management | 226 | 18.0% |
| Acumatica | 2,000 | 8.7% |
| Xero | 1,589 | 20.0% |
| Certinia | 437 | 42.5% |
| QuickBooks Online | 2,000 | — |
| QuickBooks Desktop Pro | 1,764 | — |

---

## Methodology

The pipeline runs in six sequential stages:

### Stage 1 — Data Collection
Reviews are scraped from G2.com via the [Apify](https://apify.com) `jupri/g2-explorer` actor. Each review has four structured text fields: `likes`, `dislikes`, `use_case`, and an optional `switched_from` field containing prior product name and reason. Reviewer metadata (company size, industry, region, date, incentive type) is captured alongside the text.

### Stage 2 — Normalization
`src/g2_normalizer.py` maps actor-specific field names to a unified schema. Key steps: extracting `switched_from` product names and reason from nested JSON, flagging incentivized reviews (`g2gives`, `g2_incentivized`), normalizing dates, and expanding 37 ERP domain abbreviations before embedding (e.g. `ap` → `accounts payable`, `gl` → `general ledger`). Reviews under 50 characters are dropped. A systematic G2 title-leakage artifact affecting 37.8% of `use_case` records is stripped via regex.

### Stage 3 — Topic Modeling (BERTopic)
`src/nlp_bertopic.py` runs three separate BERTopic models — one each for `likes`, `dislikes`, and `use_case`. Keeping fields separate is essential: the same dimension (e.g. reporting) can be a strength for one product and a pain point for another.

**Embedding:** `BAAI/bge-base-en-v1.5` (768-dim) — chosen over `all-MiniLM-L6-v2` for stronger domain-specific clustering on ERP terminology.

**Clustering:** HDBSCAN with `cluster_selection_method=leaf` (handles uneven product densities), followed by `reduce_outliers(strategy="embeddings")` to reassign noise documents. `update_topics()` is called with the original `CountVectorizer` instance explicitly — omitting this causes silent stopword loss.

**Stopwords:** sklearn English stopwords + all product/vendor names + meta-commentary phrases (e.g. "no downsides", "nothing to dislike").

Each model produces ~59 fine-grained topics per field with 0% outlier rate.

### Stage 4 — LLM Group Assignment
`src/nlp_topic_grouping.py` uses a three-stage Claude (`claude-sonnet-4-20250514`) pipeline to organize the 59 fine-grained topics into interpretable groups without any predefined taxonomy:

- **Stage 1** — Per-topic enrichment: resolves abbreviations, writes a plain-English description, identifies candidate functional categories, flags noise topics (meta-commentary with no feature content). Results cached to disk.
- **Stage 2** — Global clustering: all enriched topics sent in one prompt; model proposes groups that emerge from the data.
- **Stage 3** — Self-critique: model checks for oversized groups, misplacements, and remaining noise. Final assignments used in all downstream analysis.

Output: 14 groups for `likes`, 16 for `dislikes` (e.g. *Financial Reporting & Business Intelligence*, *Implementation, Setup & Cost Management*, *Customer Support & Service Quality*).

### Stage 5 — Aspect-Based Sentiment Analysis
`src/nlp_absa.py` scores sentiment per aspect using `yangheng/deberta-v3-base-absa-v1.1` — a model fine-tuned specifically for aspect-aware classification. Each sentence receives a `(sentence, aspect_term)` pair as input, where `aspect_term` is the group label cleaned and lowercased.

Sentence scores (positive / neutral / negative probabilities) are converted to a 1–5 rating:

```
rating = positive × 5 + neutral × 3 + negative × 1
```

Scores aggregate from sentence → review → product × aspect level. Separate aggregates are computed for `likes` and `dislikes` fields, with a net score combining both.

Across 54,382 scored sentences: 43.2% positive, 34.6% neutral, 22.2% negative. Mean rating 3.43/5.

### Stage 6 — Data Joining
`src/join_enriched.py` produces two clean analysis-ready datasets:

- **`reviews_meta.csv`** — one row per review, metadata only (ratings, reviewer segment, region, incentive type, switched_from)
- **`reviews_text_classified.csv`** — long format, one row per review × field, with full BERTopic + LLM classification (topic_id, top_keywords, macro_group, group_label, plain_description, ABSA rating)

---

## Key Findings

- **Customer support** scores lowest in dislikes across all twelve products (avg 1.92–2.83/5), with 73–88% negative sentences — corroborated independently by both topic frequency and ABSA.
- **Financial Reporting** is the strongest competitive differentiator: likes-side scores are uniformly high (4.42–4.76) but dislikes-side scores vary widely (MS Dynamics 1.46 vs Workday 2.95).
- **Implementation complexity** is universally negative in dislikes (QB Enterprise 1.20, Workday 1.47, Certinia 1.63, NetSuite 2.47 with 192 reviews).
- **14.1%** of reviews contain switching signals; most common origins are QuickBooks Online, QuickBooks Desktop, and Sage 100.

---

## Project Structure

```
market_intelligence_AI_agent/
│
├── src/
│   ├── g2_config.py              # Product list, Apify actor config, field mappings
│   ├── g2_normalizer.py          # Schema normalization, bonus_data extraction
│   ├── apify_loader.py           # Apify actor runner and raw data saver
│   ├── nlp_bertopic.py           # BERTopic pipeline (3 separate models)
│   ├── nlp_topic_grouping.py     # 3-stage LLM group assignment
│   ├── nlp_absa.py               # DeBERTa ABSA scoring pipeline
│   ├── test_absa_model.py        # 8-pair ABSA model validation test
│   └── join_enriched.py          # Produces reviews_meta.csv + reviews_text_classified.csv
│
├── data/
│   ├── g2/
│   │   ├── raw/                  # Raw Apify JSON per product
│   │   └── normalized/           # all_products_normalized.csv
│   ├── bertopic/
│   │   └── v5/                   # Topic model outputs per field
│   ├── absa/                     # ABSA sentence, review, product-aspect outputs
│   └── enriched/                 # Final joined datasets
│
├── main.py                       # FastAPI app entry point
├── requirements.txt
├── set_up.md
└── .gitignore
```

---

## Setup

**Prerequisites:** Python 3.11+, pip, Windows/Mac/Linux

```bash
# 1. Clone
git clone https://github.com/priyanshu124/market_intelligence_AI_agent.git
cd market_intelligence_AI_agent

# 2. Virtual environment
python -m venv venv
source venv/bin/activate          # Mac/Linux
venv\Scripts\Activate.ps1         # Windows PowerShell

# 3. Install dependencies
pip install -r requirements.txt
```

**Environment variables** — create a `.env` file:

```env
ANTHROPIC_API_KEY=your_key_here
APIFY_API_TOKEN=your_token_here
```

---

## Running the Pipeline

Run each stage in order:

```bash
# Stage 1+2: Scrape and normalize (requires Apify token)
python -m src.apify_loader

# Stage 3: Topic modeling — ~60 min CPU
python -m src.nlp_bertopic
python -m src.nlp_bertopic --fields likes dislikes   # specific fields only

# Stage 4: LLM group assignment (requires Anthropic key)
python -m src.nlp_topic_grouping
python -m src.nlp_topic_grouping --skip-stage1       # use cached enrichments

# Stage 5: ABSA scoring — ~60 min CPU with batch-size 128
python -m src.test_absa_model                        # validate model first
python -m src.nlp_absa --batch-size 128
python -m src.nlp_absa --fields dislikes             # directional run

# Stage 6: Join into analysis datasets
python -m src.join_enriched
```

---

## Output Files

| File | Description |
|---|---|
| `data/enriched/reviews_meta.csv` | 14,933 reviews × 19 metadata columns |
| `data/enriched/reviews_text_classified.csv` | ~40K rows — one per review × field, with all NLP labels |
| `data/absa/absa_heatmap.csv` | Products × aspects net rating pivot (dashboard input) |
| `data/absa/absa_net_scores.csv` | Likes vs dislikes ratings side by side per product-aspect |
| `data/absa/absa_product_aspect.csv` | Full product × aspect × field breakdown with % positive/negative |
| `data/absa/absa_sentences.csv` | Sentence-level scores — full audit trail |
| `data/bertopic/v5/grouping_reasoning_*.json` | Full LLM reasoning log per field (Stage 1–3) |

---

## Dependencies

| Package | Purpose |
|---|---|
| `bertopic` | Topic modeling |
| `sentence-transformers` | BAAI/bge-base-en-v1.5 embeddings |
| `hdbscan` | Density-based clustering |
| `umap-learn` | Dimensionality reduction |
| `transformers` | DeBERTa ABSA model |
| `torch` | Model inference |
| `anthropic` | Claude API for LLM grouping |
| `pandas` / `numpy` | Data processing |
| `apify-client` | G2 data collection |
| `fastapi` / `uvicorn` | API layer |
| `tqdm` | Progress bars |

Full list in `requirements.txt`.

---

## References

- Grootendorst, M. (2022). BERTopic. *arXiv:2203.05794*
- Pontiki et al. (2014). SemEval-2014 Task 4: ABSA. *SemEval 2014*
- Yang & Li (2023). PyABSA. *CIKM 2023*
- Xiao et al. (2023). C-Pack / BGE embeddings. *arXiv:2309.07597*
- Yao et al. (2023). ReAct. *ICLR 2023*
- Blei et al. (2003). Latent Dirichlet Allocation. *JMLR*

---

## License

For academic use — University of Maryland INST 664 course project.
