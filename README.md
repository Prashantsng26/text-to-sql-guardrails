# Text-to-SQL System: Phase 1 & Phase 2

An enterprise-grade Text-to-SQL pipeline with schema-aware prompt engineering, LLM structured generation (`instructor`), AST-based SQL syntax validation (`sqlparse`), multi-rule guardrail safety verification, and dual-isolated sandbox execution.

---

## 🏗️ Architecture & Pipeline Overview

```
                                  [User Natural Language Query]
                                                │
                                                ▼
                        ┌────────────────────────────────────────────────┐
                        │   STAGE 1: Ambiguity Detection & Disambiguation│
                        │   (ambiguity_handler.py)                       │
                        └───────────────────────┬────────────────────────┘
                                                │
                                                ▼
                        ┌────────────────────────────────────────────────┐
                        │   STAGE 2: Semantic Schema Filtering (MiniLM)  │
                        │   (schema_filter.py)                           │
                        └───────────────────────┬────────────────────────┘
                                                │
                                                ▼
                        ┌────────────────────────────────────────────────┐
                        │   STAGE 3: Schema-Aware Prompt Construction    │
                        │   (prompt_constructor.py)                      │
                        └───────────────────────┬────────────────────────┘
                                                │
                                                ▼
                        ┌────────────────────────────────────────────────┐
                        │   STAGE 4: Structured SQL Generation           │
                        │   (sql_generator.py - instructor + sqlparse)   │
                        └───────────────────────┬────────────────────────┘
                                                │
                                                ▼
                        ┌────────────────────────────────────────────────┐
                        │   STAGE 5: Guardrail Safety Layer              │
                        │   (guardrails.py - DDL/DML/LIMIT/Subquery/Scan)│
                        └───────────────┬────────────────┬───────────────┘
                                        │                │
                             [Allowed]  │                │ [Blocked]
                                        ▼                ▼
                     ┌───────────────────────┐   ┌───────────────────────┐
                     │ STAGE 6: Sandbox Exec │   │ Log to JSONL          │
                     │ (sandbox_executor.py) │   │ (guardrail_log.jsonl) │
                     └───────────────────────┘   └───────────────────────┘
```

---

## 📁 Repository Structure

- [`db_setup.py`](file:///Users/prashantsingh/Desktop/Major%20Project/db_setup.py) — Initializes and seeds `ecommerce.duckdb` with 5 relational tables (`customers`, `products`, `orders`, `order_items`, `support_tickets`), foreign keys, and realistic synthetic data.
- [`schema_extractor.py`](file:///Users/prashantsingh/Desktop/Major%20Project/schema_extractor.py) — Introspects DuckDB via SQLAlchemy `inspect(engine)`; extracts column types, PKs, FK relationships, and categorical sample values into `schema.json`.
- [`schema_filter.py`](file:///Users/prashantsingh/Desktop/Major%20Project/schema_filter.py) — Local sentence-transformers table embedding & similarity ranking.
- [`ambiguity_handler.py`](file:///Users/prashantsingh/Desktop/Major%20Project/ambiguity_handler.py) — Detects domain ambiguity (e.g. `revenue`, `active users`, `churned`) and provides concrete SQL templates.
- [`prompt_constructor.py`](file:///Users/prashantsingh/Desktop/Major%20Project/prompt_constructor.py) — Assembles filtered schema DDL, foreign keys, categorical constraints, dynamically filtered few-shots, and the question.
- [`sql_generator.py`](file:///Users/prashantsingh/Desktop/Major%20Project/sql_generator.py) — Uses `instructor` with Pydantic for structured generation (`GeneratedSQL`: `sql`, `explanation`, `confidence`, `tables_used`, `columns_used`). Validates SQL syntax with `sqlparse` and DuckDB parser, retrying once on syntax error before failing.
- [`guardrails.py`](file:///Users/prashantsingh/Desktop/Major%20Project/guardrails.py) — Configurable multi-rule safety engine: blocks DDL, blocks DML writes, auto-injects `LIMIT 1000`, enforces max subquery depth ($\le 3$), and validates row scan estimations via `EXPLAIN`. Logs blocked queries to `guardrail_log.jsonl`.
- [`sandbox_executor.py`](file:///Users/prashantsingh/Desktop/Major%20Project/sandbox_executor.py) — Executes queries within a read-only DuckDB connection (`read_only=True`, `PRAGMA enable_external_access = false;`) inside an explicit transaction rolled back immediately after result retrieval.
- [`main.py`](file:///Users/prashantsingh/Desktop/Major%20Project/main.py) — End-to-end CLI orchestrating the full 6-stage pipeline.
- [`guardrail_log.jsonl`](file:///Users/prashantsingh/Desktop/Major%20Project/guardrail_log.jsonl) — Audit log of all guardrail violations with timestamps and rules triggered.

---

## 🔒 Defense-in-Depth Sandbox Security Architecture

The sandbox executor implements two independent layers of security isolation:

1. **Storage & Engine Level Read-Only Isolation**:
   - The connection is opened with `duckdb.connect(database=db_path, read_only=True)`. DuckDB's storage engine enforces that no write locks or mutating operations (`INSERT`, `UPDATE`, `DELETE`, `DROP`, `ALTER`) can execute, raising `PermissionException` immediately if attempted.
   - Sets `PRAGMA enable_external_access = false;` to block all file-system access (e.g., `read_csv('/etc/passwd')`) and remote HTTP requests.
2. **Transaction Rollback Isolation**:
   - The query executes inside `BEGIN TRANSACTION;` and unconditionally issues `ROLLBACK;` in a `finally` block.

---

## 🚀 Quick Start & CLI Usage

### 1. Environment Setup

```bash
# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install requirements
pip install -r requirements.txt
```

### 2. Optional: Configure LLM API Keys

Set your API key as an environment variable (the system will use `gpt-4o` or `claude-3-5-sonnet` if set, and automatically falls back to an offline rule-based generator if no key is provided):

```bash
export OPENAI_API_KEY="sk-..."
# or
export ANTHROPIC_API_KEY="sk-ant-..."
```

### 3. Run Full Pipeline

```bash
# Run all sample queries (including guardrail violation tests)
python main.py

# Run a custom natural language query
python main.py --question "Which product categories generated the most sales in completed orders?"
```

---

## 🧪 Independent Module Testing

Each Phase 2 component is independently testable:

```bash
# 1. Test Structured SQL Generation & Syntax Retry
python sql_generator.py

# 2. Test Guardrails (DDL/DML block, LIMIT injection, subquery depth)
python guardrails.py

# 3. Test Sandbox Execution & Transaction Rollback
python sandbox_executor.py
```
