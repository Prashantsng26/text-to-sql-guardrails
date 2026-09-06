"""
schema_filter.py
Semantic table filtering using local sentence embeddings (sentence-transformers).
Computes cosine similarity between natural language user question and table representations
(table name, column names, descriptions), returning only relevant tables above a threshold.
"""

from typing import Any, Dict, List, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer

# Load a lightweight, fast, local embedding model
MODEL_NAME = "all-MiniLM-L6-v2"
_model = None

def get_embedding_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model

def _cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    dot = np.dot(vec_a, vec_b)
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(dot / (norm_a * norm_b))

def _generate_table_representation(table_name: str, table_info: Dict[str, Any]) -> str:
    """Generates a rich semantic text representation for a table."""
    col_names = [col["name"] for col in table_info.get("columns", [])]
    col_str = ", ".join(col_names)
    desc = table_info.get("description", "")
    
    # Include foreign key reference context if available
    fk_contexts = []
    for fk in table_info.get("foreign_keys", []):
        ref_table = fk.get("referred_table")
        if ref_table:
            fk_contexts.append(f"relates to {ref_table}")
    fk_str = f" ({'; '.join(fk_contexts)})" if fk_contexts else ""

    return f"Table '{table_name}': {desc} Columns: {col_str}.{fk_str}"

def filter_relevant_tables(
    question: str,
    schema: Dict[str, Any],
    threshold: float = 0.30,
    min_tables: int = 1
) -> List[str]:
    """
    Embeds the user question and table representations.
    Computes cosine similarities, prints the ranked scores, and returns tables above threshold.

    Args:
        question: User's natural language question
        schema: Structured database schema dict (from schema_extractor.py)
        threshold: Minimum cosine similarity score required to include a table
        min_tables: Minimum number of top tables to return if none meet threshold

    Returns:
        List of relevant table names
    """
    model = get_embedding_model()
    tables = schema.get("tables", {})
    if not tables:
        return []

    # 1. Prepare table textual representations
    table_names = list(tables.keys())
    table_texts = [_generate_table_representation(name, tables[name]) for name in table_names]

    # 2. Compute embeddings
    question_embedding = model.encode(question, convert_to_numpy=True)
    table_embeddings = model.encode(table_texts, convert_to_numpy=True)

    # 3. Calculate cosine similarities
    scored_tables: List[Tuple[str, float, str]] = []
    for i, name in enumerate(table_names):
        sim = _cosine_similarity(question_embedding, table_embeddings[i])
        scored_tables.append((name, sim, table_texts[i]))

    # 4. Sort descending by similarity score
    scored_tables.sort(key=lambda x: x[1], reverse=True)

    # 5. Print ranked similarity scores for transparency
    print(f"\n[Schema Filter] Semantic Similarity Ranking for: '{question}'")
    print(f"{'-'*75}")
    print(f"{'Table':<20} | {'Score':<8} | {'Status':<10} | {'Semantic Summary'}")
    print(f"{'-'*75}")

    relevant_tables: List[str] = []
    for name, score, repr_text in scored_tables:
        is_selected = score >= threshold
        if is_selected:
            relevant_tables.append(name)
            status = "SELECTED"
        else:
            status = "EXCLUDED"
        summary_clip = (repr_text[:50] + "...") if len(repr_text) > 50 else repr_text
        print(f"{name:<20} | {score:0.4f}   | {status:<10} | {summary_clip}")
    print(f"{'-'*75}")

    # Fallback to top-k if threshold excludes all tables
    if not relevant_tables and scored_tables:
        fallback = [t[0] for t in scored_tables[:min_tables]]
        print(f"[Schema Filter] Notice: No tables met threshold {threshold:0.2f}. Falling back to top {min_tables} table(s): {fallback}")
        relevant_tables = fallback

    return relevant_tables


if __name__ == "__main__":
    from schema_extractor import extract_schema
    schema = extract_schema()

    sample_questions = [
        "What are the emails of all Gold tier customers living in Canada?",
        "Which product categories generated the most revenue from completed orders?",
        "Are there any unresolved support tickets with urgent priority?"
    ]

    for q in sample_questions:
        selected = filter_relevant_tables(q, schema, threshold=0.30)
        print(f"=> Final Relevant Tables: {selected}\n")
