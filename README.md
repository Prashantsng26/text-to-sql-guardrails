# Text-to-SQL System: Phase 1 — Schema-Aware Prompt Engine

A modular Python system for Phase 1 of Text-to-SQL translation:
- Programmatic live database schema introspection via SQLAlchemy Inspector and DuckDB catalog.
- Automatic categorical column detection and sample value extraction.
- Semantic schema filtering using local sentence embeddings (`sentence-transformers`).
- Business-level query ambiguity detection with concrete SQL interpretations.
- Schema-aware prompt construction with filtered DDL, active foreign keys, valid categorical values, and curated DuckDB few-shot examples.

---

## 📁 Repository Structure

- [`db_setup.py`](file:///Users/prashantsingh/Desktop/Major%20Project/db_setup.py) — Initializes and seeds `ecommerce.duckdb` with 5 relational tables (`customers`, `products`, `orders`, `order_items`, `support_tickets`), foreign keys, and realistic synthetic data (~50-265 rows per table).
- [`schema_extractor.py`](file:///Users/prashantsingh/Desktop/Major%20Project/schema_extractor.py) — Introspects DuckDB via SQLAlchemy `inspect(engine)`; extracts table definitions, column types, PKs, FK relationships, and categorical columns with up to 6 distinct sample values. Exports to `schema.json`.
- [`schema_filter.py`](file:///Users/prashantsingh/Desktop/Major%20Project/schema_filter.py) — Embeds user question and table representations using `all-MiniLM-L6-v2` locally (no API key required), computes cosine similarity, prints score rankings, and filters relevant tables.
- [`ambiguity_handler.py`](file:///Users/prashantsingh/Desktop/Major%20Project/ambiguity_handler.py) — Detects domain-ambiguous terms (e.g. `revenue`, `active users`, `top selling`) and maps them to business interpretations with DuckDB SQL templates.
- [`prompt_constructor.py`](file:///Users/prashantsingh/Desktop/Major%20Project/prompt_constructor.py) — Assembles system directives, filtered schema DDL, foreign key relationships, categorical constraints, few-shot examples, and the target question into a final prompt string.
- [`main.py`](file:///Users/prashantsingh/Desktop/Major%20Project/main.py) — End-to-end CLI orchestrating all 4 stages across sample and custom queries.
- [`requirements.txt`](file:///Users/prashantsingh/Desktop/Major%20Project/requirements.txt) — Dependency list (`duckdb`, `duckdb-engine`, `sqlalchemy`, `sentence-transformers`).
- [`schema.json`](file:///Users/prashantsingh/Desktop/Major%20Project/schema.json) — Programmatically extracted database schema metadata.

---

## 🚀 Quick Start

### 1. Installation

```bash
# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install requirements
pip install -r requirements.txt
```

### 2. Running End-to-End Pipeline

Run all test cases (simple lookup, multi-table join, ambiguous query, domain-specific filter):
```bash
python main.py
```

Run with a custom question:
```bash
python main.py --question "Which customer tier generated the highest average order value?"
```

Force database re-seeding:
```bash
python main.py --reseed
```

---

## 🧪 Independent Module Testing

Each component is independently executable and testable:

### 1. Seed Sample Database
```bash
python db_setup.py
```

### 2. Introspect Schema & Generate `schema.json`
```bash
python schema_extractor.py
```

### 3. Test Semantic Table Filtering
```bash
python schema_filter.py
```

### 4. Test Ambiguity Detection
```bash
python ambiguity_handler.py
```

### 5. Test Prompt Assembly
```bash
python prompt_constructor.py
```
