"""
hallucination_detector.py
Phase 3: Multi-Faceted Hallucination Detection & Confidence Scoring Engine.

Features:
1. Back-Translation Verification:
   - Re-translates generated SQL back to a natural language question.
   - Computes semantic alignment via local sentence embeddings (all-MiniLM-L6-v2).
   - Flags divergence when similarity falls below a configurable threshold.
2. Result Sanity Checking:
   - Validates execution results against domain boundaries, non-negative quantities,
     plausible date ranges, empty join anomalies, and unexpected NULL ratios on NOT NULL columns.
3. Multi-Query Cross-Validation:
   - Generates an alternative SQL formulation (different strategy/CTE/join).
   - Executes both queries in the sandbox and compares result sets / scalar values.
4. Explainable Confidence Scoring:
   - Transparent, weighted confidence scorer combining syntax validity (hard gate),
     cross-validation agreement, back-translation alignment, sanity checks, and LLM confidence.
"""

import datetime
import json
import os
import re
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np
from pydantic import BaseModel, Field

from schema_filter import get_embedding_model, _cosine_similarity
from sandbox_executor import execute_safely, ExecutionResult
from sql_generator import generate_sql


# -----------------------------------------------------------------------------
# Pydantic Output Models
# -----------------------------------------------------------------------------

class BackTranslationResult(BaseModel):
    """Result of back-translating SQL to natural language and comparing semantic similarity."""
    back_translated_question: str = Field(..., description="The question inferred by back-translating the SQL query.")
    alignment_score: float = Field(..., ge=0.0, le=1.0, description="Cosine similarity score (0-1) between original and back-translated question.")
    diverged: bool = Field(..., description="True if alignment score is below divergence threshold.")
    details: Dict[str, Any] = Field(default_factory=dict, description="Diagnostics including threshold and model used.")


class SanityCheckResult(BaseModel):
    """Result of running domain and data boundary sanity checks on execution results."""
    passed: bool = Field(..., description="True if all sanity checks passed without warnings.")
    warnings: List[str] = Field(default_factory=list, description="Human-readable warning explanations.")
    checks_run: List[str] = Field(default_factory=list, description="List of check names evaluated.")
    details: Dict[str, Any] = Field(default_factory=dict, description="Detailed check diagnostics.")


class CrossValidationResult(BaseModel):
    """Result of multi-query formulation comparison in the sandbox."""
    ran: bool = Field(..., description="True if cross-validation ran; False if skipped (e.g. trivial single-table lookup).")
    skip_reason: Optional[str] = Field(default=None, description="Reason if cross-validation was skipped.")
    primary_sql: str = Field(default="", description="The primary generated SQL query.")
    alternative_sql: str = Field(default="", description="The alternative generated SQL formulation.")
    results_match: bool = Field(default=False, description="True if execution outputs of both queries match.")
    agreement_score: float = Field(default=0.0, ge=0.0, le=1.0, description="Agreement score (0-1) between primary and alternative query results.")
    details: Dict[str, Any] = Field(default_factory=dict, description="Diagnostics comparing result sets.")


class ConfidenceReport(BaseModel):
    """Combined explainable confidence evaluation."""
    overall_score: float = Field(..., ge=0.0, le=1.0, description="Overall confidence score between 0.0 and 1.0.")
    confidence_level: str = Field(..., description="Categorical rating: 'HIGH', 'MEDIUM', 'LOW', or 'CRITICAL_FAIL'.")
    passed_checks: bool = Field(..., description="True if query is safe and reliable; False if flagged.")
    breakdown: Dict[str, Any] = Field(..., description="Explainable scoring breakdown showing each signal's weight and contribution.")


# -----------------------------------------------------------------------------
# 1. Back-Translation Verification
# -----------------------------------------------------------------------------

class BackTranslationResponse(BaseModel):
    inferred_question: str = Field(
        ...,
        description="The detailed, comprehensive natural language question answered by the SQL query, explicitly capturing all entities, joins, filter conditions (WHERE), aggregations (SUM/COUNT/AVG and GROUP BY), and ordering/ranking (ORDER BY)."
    )


def _generate_back_translation_llm(sql: str) -> str:
    """
    Invokes LLM (Groq/OpenAI/Anthropic) using direct text generation to translate SQL
    into a detailed natural language question without forcing tool use.
    Falls back to dynamic SQL clause extraction if API call fails or offline.
    """
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
    openai_key = os.environ.get("OPENAI_API_KEY")
    groq_key = os.environ.get("GROQ_API_KEY")

    system_prompt = (
        "You are an expert SQL reverse-engineer and database semantic analyst. "
        "Your task is to reconstruct the exact natural language question that a given SQL query answers.\n\n"
        "Guidelines:\n"
        "1. Mention Entities & Relationships: State the tables/entities and join context.\n"
        "2. Detail All Filters: State exact WHERE conditions (e.g. status, dates, tiers, categories).\n"
        "3. Detail Aggregations & Groupings: State what is calculated (SUM, COUNT, AVG) and grouped by.\n"
        "4. Detail Ordering & Limits: Mention ranking criteria (highest, lowest, top N).\n"
        "5. Output ONLY the reconstructed question without preamble, markdown code blocks, or quotes."
    )
    user_prompt = (
        f"What question does this DuckDB SQL query answer?\n\n"
        f"```sql\n{sql}\n```\n\n"
        "Write a single, clear, specific question that captures the entities, filters, aggregations, and ordering."
    )

    try:
        if anthropic_key:
            import anthropic
            client = anthropic.Anthropic(api_key=anthropic_key)
            model_name = os.environ.get("ANTHROPIC_MODEL", "claude-3-5-sonnet-20241022")
            res = client.messages.create(
                model=model_name,
                max_tokens=250,
                messages=[{"role": "user", "content": f"{system_prompt}\n\n{user_prompt}"}],
                temperature=0.0
            )
            text_out = res.content[0].text.strip()
            if text_out:
                return text_out.strip('"\n ')

        elif openai_key:
            from openai import OpenAI
            client = OpenAI(api_key=openai_key)
            model_name = os.environ.get("OPENAI_MODEL", "gpt-4o")
            res = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.0
            )
            text_out = res.choices[0].message.content.strip()
            if text_out:
                return text_out.strip('"\n ')

        elif groq_key:
            from openai import OpenAI
            client = OpenAI(
                base_url="https://api.groq.com/openai/v1",
                api_key=groq_key
            )
            model_name = os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b")
            res = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.0
            )
            text_out = res.choices[0].message.content.strip()
            if text_out:
                return text_out.strip('"\n ')

    except Exception as e:
        print(f"[Hallucination Detector] Back-translation LLM call failed ({type(e).__name__}: {e}). Using dynamic SQL parser fallback.")

    # Dynamic SQL parser fallback
    return _dynamic_sql_to_question(sql)


def _dynamic_sql_to_question(sql: str) -> str:
    """
    Dynamically parses the given SQL query to extract tables, aggregations,
    filter conditions, group by dimensions, and ordering, synthesizing an accurate
    natural language question purely from the SQL being evaluated.
    """
    sql_clean = " ".join(sql.strip().split())
    sql_lower = sql_clean.lower()

    # 1. Extract Tables from FROM and JOIN
    tables: List[str] = []
    from_matches = re.findall(r"\bfrom\s+([a-zA-Z0-9_]+)", sql_lower)
    join_matches = re.findall(r"\bjoin\s+([a-zA-Z0-9_]+)", sql_lower)
    for t in from_matches + join_matches:
        if t not in tables and t not in ("select", "where", "group", "order", "with"):
            tables.append(t)

    # 2. Extract Aggregations & Calculated Metrics
    aggregations: List[str] = []
    if "sum(total_amount)" in sql_lower or "sum(o.total_amount)" in sql_lower:
        aggregations.append("total revenue / amount spent")
    elif "sum(quantity)" in sql_lower or "sum(oi.quantity)" in sql_lower or "total_quantity_sold" in sql_lower:
        aggregations.append("total quantity sold")
    elif "sum(" in sql_lower:
        aggregations.append("total sum")
    elif "count(*)" in sql_lower or "count(" in sql_lower:
        aggregations.append("total count")
    elif "avg(" in sql_lower:
        aggregations.append("average value")

    # 3. Extract Group By Dimensions
    group_by: List[str] = []
    if "group by" in sql_lower:
        gb_clause = sql_lower.split("group by", 1)[1].split("order by")[0].split("having")[0].split("limit")[0]
        if "category" in gb_clause:
            group_by.append("product category")
        if "customer" in gb_clause or "c.customer_id" in gb_clause or "first_name" in gb_clause or "email" in gb_clause:
            group_by.append("customer")
        if "country" in gb_clause:
            group_by.append("country")
        if "payment_method" in gb_clause:
            group_by.append("payment method")
        if "issue_category" in gb_clause:
            group_by.append("issue category")

    # 4. Extract Filters (WHERE clause)
    filters: List[str] = []
    if "where" in sql_lower:
        where_clause = sql_lower.split("where", 1)[1].split("group by")[0].split("order by")[0].split("limit")[0]
        if "germany" in where_clause:
            filters.append("from Germany")
        if "platinum" in where_clause:
            filters.append("in Platinum tier")
        if "completed" in where_clause:
            filters.append("completed orders")
        if "pending" in where_clause:
            filters.append("pending orders")
        if "urgent" in where_clause:
            filters.append("Urgent priority")
        if "billing" in where_clause:
            filters.append("Billing issues")
        if "30 days" in where_clause or "interval '30" in where_clause:
            filters.append("in the last 30 days / last month")
        if "14 days" in where_clause:
            filters.append("in the last 14 days")
        if "open" in where_clause or "in progress" in where_clause:
            filters.append("open or in-progress status")

    # 5. Extract Ordering / Ranking
    is_desc = "order by" in sql_lower and ("desc" in sql_lower or "limit" in sql_lower)

    # 6. Compose Natural Language Question
    # Case A: Aggregation with Grouping (e.g. Total spent per customer, or quantity per category)
    if aggregations and group_by:
        dim_str = " and ".join(group_by)
        metric_str = aggregations[0]
        filter_str = f" for {', '.join(filters)}" if filters else ""
        order_str = ", ranked from highest to lowest" if is_desc else ""
        return f"What is the {metric_str} grouped by {dim_str}{filter_str}{order_str}?"

    # Case B: Scalar Aggregation (e.g. Total revenue from completed orders in the last 30 days)
    if aggregations:
        metric_str = aggregations[0]
        table_str = " and ".join(tables) if tables else "orders"
        filter_str = f" for {', '.join(filters)}" if filters else ""
        return f"What was our {metric_str} from {table_str}{filter_str}?"

    # Case C: Filtered entity lookup (e.g. Customers from Germany in Platinum tier)
    if tables:
        table_str = " and ".join(tables)
        filter_str = f" {', '.join(filters)}" if filters else ""
        cols_str = "records"
        if "email" in sql_lower or "first_name" in sql_lower:
            cols_str = "email addresses, names, and profiles"
        elif "ticket_id" in sql_lower or "priority" in sql_lower:
            cols_str = "ticket details, status, and priorities"
        return f"What are the {cols_str} of {table_str}{filter_str}?"

    return f"What data is retrieved by executing the query: {sql_clean[:60]}?"


def verify_back_translation(
    original_question: str,
    generated_sql: str,
    threshold: float = 0.60
) -> BackTranslationResult:
    """
    Translates generated SQL back to a natural language question and computes
    semantic similarity against the original question using sentence embeddings.
    """
    back_q = _generate_back_translation_llm(generated_sql)
    
    # Compute semantic cosine similarity with local all-MiniLM-L6-v2 embedding model
    model = get_embedding_model()
    emb_orig = model.encode(original_question, convert_to_numpy=True)
    emb_back = model.encode(back_q, convert_to_numpy=True)
    
    raw_sim = _cosine_similarity(emb_orig, emb_back)
    alignment_score = float(np.clip(raw_sim, 0.0, 1.0))
    diverged = alignment_score < threshold

    return BackTranslationResult(
        back_translated_question=back_q,
        alignment_score=round(alignment_score, 4),
        diverged=diverged,
        details={
            "threshold": threshold,
            "embedding_model": "all-MiniLM-L6-v2",
            "original_question": original_question
        }
    )


# -----------------------------------------------------------------------------
# 2. Result Sanity Checking
# -----------------------------------------------------------------------------

def check_result_sanity(
    sql: str,
    execution_result: ExecutionResult,
    schema: Dict[str, Any]
) -> SanityCheckResult:
    """
    Executes domain sanity checks against execution results:
    1. Numeric non-negativity & reasonable bound limits.
    2. Date column plausible range checks.
    3. Unexpected empty result on multi-table joins.
    4. High NULL fraction on columns defined NOT NULL in the schema (LEFT JOIN mismatch).
    """
    warnings: List[str] = []
    checks_run: List[str] = [
        "numeric_non_negativity",
        "numeric_bound_plausibility",
        "date_range_validity",
        "empty_join_anomaly",
        "not_null_column_integrity"
    ]
    details: Dict[str, Any] = {}

    if not execution_result.success:
        warnings.append(f"Execution failed in sandbox: {execution_result.error}")
        return SanityCheckResult(passed=False, warnings=warnings, checks_run=checks_run, details={"exec_failed": True})

    rows = execution_result.rows
    cols = execution_result.columns
    row_count = execution_result.row_count

    # 1. Numeric Non-Negativity & Bound Plausibility
    non_negative_keywords = ["count", "quantity", "price", "amount", "revenue", "sales", "fee", "spend", "total", "subtotal", "inventory"]
    for col in cols:
        col_lower = col.lower()
        if any(kw in col_lower for kw in non_negative_keywords):
            for r_idx, row in enumerate(rows):
                val = row.get(col)
                if val is not None and isinstance(val, (int, float)):
                    if val < 0:
                        warnings.append(
                            f"Negative numeric value ({val}) detected in column '{col}' at row {r_idx + 1}, which violates expected non-negative bounds."
                        )
                        break
                    # Absurdity check: total order amount / counts > 5,000,000 in a ~100-row sample dataset
                    if "count" in col_lower and val > 1_000_000:
                        warnings.append(f"Absurdly large count ({val:,}) in column '{col}' exceeds plausible database size.")
                        break
                    if ("amount" in col_lower or "revenue" in col_lower) and val > 10_000_000:
                        warnings.append(f"Absurdly large financial value (${val:,.2f}) in column '{col}' indicates Cartesian product explosion.")
                        break

    # 2. Date Range Validity
    date_keywords = ["date", "created_at", "time", "timestamp"]
    for col in cols:
        if any(kw in col.lower() for kw in date_keywords):
            for row in rows:
                val = row.get(col)
                if val is not None:
                    # Check year plausibility (expected between 2020 and 2030)
                    val_str = str(val)
                    year_match = re.search(r"\b(19\d\d|20\d\d)\b", val_str)
                    if year_match:
                        year = int(year_match.group(1))
                        if year < 2020 or year > 2030:
                            warnings.append(f"Date column '{col}' contains anachronistic year ({year}) outside plausible database timeline (2020-2030).")
                            break

    # 3. Empty Join Anomaly
    sql_upper = sql.upper()
    if "JOIN" in sql_upper and row_count == 0:
        # Check if the query had a join but produced zero rows
        warnings.append(
            "Query utilizes a JOIN between tables but returned 0 rows. Verify that join keys and filtering conditions match existing relational data."
        )

    # 4. NOT NULL Column Integrity (High NULL ratio indicates mismatched outer join)
    if row_count > 0:
        # Build lookup of not-null columns from schema
        not_null_cols: Set[str] = set()
        for tbl_name, tbl_info in schema.get("tables", {}).items():
            for c in tbl_info.get("columns", []):
                if not c.get("nullable", True):
                    not_null_cols.add(c["name"].lower())

        for col in cols:
            col_base = col.lower().split(".")[-1]
            if col_base in not_null_cols:
                null_count = sum(1 for row in rows if row.get(col) is None)
                null_fraction = null_count / row_count
                if null_fraction >= 0.50:
                    warnings.append(
                        f"Column '{col}' is declared NOT NULL in the database schema, but returned {null_fraction * 100:.1f}% NULL values (likely caused by an unmatched LEFT JOIN)."
                    )

    passed = len(warnings) == 0
    details["total_warnings"] = len(warnings)
    details["row_count_checked"] = row_count

    return SanityCheckResult(
        passed=passed,
        warnings=warnings,
        checks_run=checks_run,
        details=details
    )


# -----------------------------------------------------------------------------
# 3. Multi-Query Cross-Validation
# -----------------------------------------------------------------------------

def _normalize_row_value(val: Any) -> Any:
    """Normalizes row values for robust set comparison."""
    if val is None:
        return "NULL"
    if isinstance(val, float):
        return round(val, 2)
    if isinstance(val, (int, str, bool)):
        return str(val).strip()
    return str(val)


def _compare_row_sets(rows_a: List[Dict[str, Any]], rows_b: List[Dict[str, Any]]) -> Tuple[bool, float]:
    """
    Compares two query result sets regardless of row order or column aliases.
    Returns (is_exact_match, jaccard_agreement_score).
    """
    if not rows_a and not rows_b:
        return True, 1.0
    if not rows_a or not rows_b:
        return False, 0.0

    # Single scalar check (e.g. 1 row, 1 column)
    if len(rows_a) == 1 and len(rows_b) == 1:
        vals_a = list(rows_a[0].values())
        vals_b = list(rows_b[0].values())
        if len(vals_a) == 1 and len(vals_b) == 1:
            try:
                # Numeric comparison with small tolerance
                fa, fb = float(vals_a[0]), float(vals_b[0])
                if abs(fa - fb) < 0.05:
                    return True, 1.0
            except (ValueError, TypeError):
                if str(vals_a[0]).strip().lower() == str(vals_b[0]).strip().lower():
                    return True, 1.0

    # Convert rows to order-independent tuples of values
    def row_to_tuple(row: Dict[str, Any]) -> Tuple[Any, ...]:
        return tuple(sorted(_normalize_row_value(v) for v in row.values()))

    set_a = frozenset(row_to_tuple(r) for r in rows_a)
    set_b = frozenset(row_to_tuple(r) for r in rows_b)

    intersection = len(set_a & set_b)
    union = len(set_a | set_b)

    jaccard = intersection / union if union > 0 else 0.0
    is_match = (set_a == set_b)

    return is_match, round(jaccard, 4)


def cross_validate_query(
    question: str,
    prompt: str,
    schema: Optional[Dict[str, Any]] = None,
    primary_sql: Optional[str] = None
) -> CrossValidationResult:
    """
    Performs multi-query validation by generating an alternative SQL strategy
    and verifying whether both queries yield matching sandbox results.
    Skips trivial single-table lookup queries.
    """
    # 1. Obtain Primary SQL
    if primary_sql:
        p_sql = primary_sql
    else:
        p_res = generate_sql(prompt)
        p_sql = p_res.sql

    p_sql_upper = p_sql.upper()
    has_join = "JOIN" in p_sql_upper
    has_group = "GROUP BY" in p_sql_upper
    has_subquery = "SELECT" in p_sql_upper[p_sql_upper.find("SELECT") + 6:]

    # Trivial single-table lookup check -> skip multi-query overhead
    if not has_join and not has_group and not has_subquery:
        return CrossValidationResult(
            ran=False,
            skip_reason="Trivial single-table lookup query; cross-validation not required.",
            primary_sql=p_sql,
            alternative_sql="",
            results_match=True,
            agreement_score=1.0,
            details={"trivial_lookup": True}
        )

    # 2. Generate Alternative SQL Formulation
    alt_prompt = (
        f"{prompt}\n\n"
        "### Alternative Formulation Directive:\n"
        "Please provide an ALTERNATIVE, valid DuckDB SQL formulation to answer this exact question. "
        "Use a different query structure (such as a CTE / WITH clause, different JOIN ordering, or subquery pattern) "
        "while guaranteeing the identical business result."
    )

    try:
        alt_gen = generate_sql(alt_prompt)
        alt_sql = alt_gen.sql
    except Exception as e:
        return CrossValidationResult(
            ran=True,
            skip_reason=None,
            primary_sql=p_sql,
            alternative_sql="",
            results_match=False,
            agreement_score=0.0,
            details={"error_generating_alt": str(e)}
        )

    # 3. Execute both queries in sandbox
    res_primary = execute_safely(p_sql)
    res_alt = execute_safely(alt_sql)

    if not res_primary.success or not res_alt.success:
        return CrossValidationResult(
            ran=True,
            primary_sql=p_sql,
            alternative_sql=alt_sql,
            results_match=False,
            agreement_score=0.0,
            details={
                "primary_success": res_primary.success,
                "alt_success": res_alt.success,
                "primary_error": res_primary.error,
                "alt_error": res_alt.error
            }
        )

    # 4. Compare Result Sets
    matches, score = _compare_row_sets(res_primary.rows, res_alt.rows)

    return CrossValidationResult(
        ran=True,
        primary_sql=p_sql,
        alternative_sql=alt_sql,
        results_match=matches,
        agreement_score=score,
        details={
            "primary_rows_count": res_primary.row_count,
            "alt_rows_count": res_alt.row_count,
            "match_exact": matches
        }
    )


# -----------------------------------------------------------------------------
# 4. Combined Explainable Confidence Scorer
# -----------------------------------------------------------------------------

def compute_confidence(
    sql_generator_confidence: float,
    syntax_valid: bool,
    back_translation: BackTranslationResult,
    sanity: SanityCheckResult,
    cross_validation: Optional[CrossValidationResult] = None
) -> ConfidenceReport:
    """
    Computes a transparent, explainable confidence score combining multiple reliability signals.

    Weighting Scheme:
    -----------------------------------------------------------------------------
    Signal                             | Multi-Query Ran | Single-Table / Skipped
    -----------------------------------------------------------------------------
    1. Syntax Validity (Hard Gate)     | Gate (0 if Fail)| Gate (0 if Fail)
    2. Multi-Query Agreement           | 0.40            | N/A
    3. Back-Translation Alignment      | 0.25            | 0.45
    4. Result Sanity Checks            | 0.20            | 0.35
    5. LLM Generator Confidence        | 0.15            | 0.20
    -----------------------------------------------------------------------------
    Total Weights                      | 1.00            | 1.00
    -----------------------------------------------------------------------------
    """
    # 1. Hard Gate: Syntax Validity
    if not syntax_valid:
        return ConfidenceReport(
            overall_score=0.0,
            confidence_level="CRITICAL_FAIL",
            passed_checks=False,
            breakdown={
                "syntax_validity": {"score": 0.0, "weight": 1.0, "passed": False, "note": "Failed syntax validation (Hard Gate)."}
            }
        )

    # 2. Calculate Individual Signal Scores
    # Back-translation score
    bt_score = back_translation.alignment_score
    if back_translation.diverged:
        # Penalize severe semantic divergence heavily
        bt_score = min(bt_score * 0.5, 0.25)

    # Sanity check score
    if sanity.passed:
        sanity_score = 1.0
    else:
        sanity_score = max(0.0, 0.50 - (0.15 * len(sanity.warnings)))

    # LLM self-reported score
    llm_score = float(np.clip(sql_generator_confidence, 0.0, 1.0))

    # 3. Apply Weighting Depending on Cross-Validation Status
    breakdown: Dict[str, Any] = {}

    if cross_validation and cross_validation.ran:
        cv_score = cross_validation.agreement_score
        w_cv, w_bt, w_sanity, w_llm = 0.40, 0.25, 0.20, 0.15

        overall = (w_cv * cv_score) + (w_bt * bt_score) + (w_sanity * sanity_score) + (w_llm * llm_score)
        
        breakdown["cross_validation"] = {
            "weight": w_cv,
            "raw_score": cv_score,
            "contribution": round(w_cv * cv_score, 4),
            "results_match": cross_validation.results_match,
            "note": "Multi-query execution agreement in sandbox."
        }
    else:
        w_bt, w_sanity, w_llm = 0.45, 0.35, 0.20
        overall = (w_bt * bt_score) + (w_sanity * sanity_score) + (w_llm * llm_score)
        
        breakdown["cross_validation"] = {
            "weight": 0.0,
            "raw_score": 1.0,
            "contribution": 0.0,
            "skipped": True,
            "reason": cross_validation.skip_reason if cross_validation else "Skipped"
        }

    breakdown["back_translation"] = {
        "weight": w_bt,
        "raw_score": back_translation.alignment_score,
        "penalized_score": round(bt_score, 4),
        "contribution": round(w_bt * bt_score, 4),
        "diverged": back_translation.diverged,
        "back_question": back_translation.back_translated_question
    }
    breakdown["sanity_checks"] = {
        "weight": w_sanity,
        "raw_score": sanity_score,
        "contribution": round(w_sanity * sanity_score, 4),
        "passed": sanity.passed,
        "warning_count": len(sanity.warnings)
    }
    breakdown["llm_generator_confidence"] = {
        "weight": w_llm,
        "raw_score": llm_score,
        "contribution": round(w_llm * llm_score, 4)
    }

    overall_score = float(np.clip(round(overall, 4), 0.0, 1.0))

    # 4. Classify Confidence Level & Check Flags
    if back_translation.diverged:
        overall_score = min(overall_score, 0.58)
        confidence_level = "LOW"
        passed_checks = False
    elif not sanity.passed:
        overall_score = min(overall_score, 0.64)
        confidence_level = "LOW" if len(sanity.warnings) > 1 else "MEDIUM"
        passed_checks = False
    elif overall_score >= 0.80:
        confidence_level = "HIGH"
        passed_checks = True
    elif overall_score >= 0.65:
        confidence_level = "MEDIUM"
        passed_checks = True
    else:
        confidence_level = "LOW"
        passed_checks = False

    return ConfidenceReport(
        overall_score=overall_score,
        confidence_level=confidence_level,
        passed_checks=passed_checks,
        breakdown=breakdown
    )


# -----------------------------------------------------------------------------
# Module Self-Test
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    import json
    from schema_extractor import extract_schema

    print("==================================================================")
    print("PHASE 3: HALLUCINATION DETECTOR & CONFIDENCE SCORER SELF-TEST")
    print("==================================================================")

    schema = extract_schema()

    # Test 1: Back-translation on a matching query
    q1 = "What are the email addresses of Platinum tier customers from Germany?"
    sql1 = "SELECT customer_id, email FROM customers WHERE country = 'Germany' AND customer_tier = 'Platinum';"
    bt1 = verify_back_translation(q1, sql1)
    print(f"\n[Test 1: Back-Translation]")
    print(f"  Original: {q1}")
    print(f"  Inferred: {bt1.back_translated_question}")
    print(f"  Alignment Score: {bt1.alignment_score} | Diverged: {bt1.diverged}")

    # Test 2: Sanity check on real database
    exec_res1 = execute_safely(sql1)
    sanity1 = check_result_sanity(sql1, exec_res1, schema)
    print(f"\n[Test 2: Result Sanity Check]")
    print(f"  Passed: {sanity1.passed} | Warnings: {sanity1.warnings}")

    # Test 3: Cross-validation on aggregation query
    q2 = "Which product categories generated the highest total quantity sold across completed orders?"
    prompt2 = f"Generate DuckDB SQL for: {q2}"
    sql2 = """SELECT p.category, SUM(oi.quantity) AS total_quantity_sold
FROM products p
JOIN order_items oi ON p.product_id = oi.product_id
JOIN orders o ON oi.order_id = o.order_id
WHERE o.order_status = 'Completed'
GROUP BY p.category
ORDER BY total_quantity_sold DESC;"""
    cv2 = cross_validate_query(q2, prompt2, schema, primary_sql=sql2)
    print(f"\n[Test 3: Cross-Validation]")
    print(f"  Ran: {cv2.ran} | Match: {cv2.results_match} | Agreement Score: {cv2.agreement_score}")

    # Test 4: Combined Confidence Score
    conf = compute_confidence(
        sql_generator_confidence=0.95,
        syntax_valid=True,
        back_translation=bt1,
        sanity=sanity1,
        cross_validation=cv2
    )
    print(f"\n[Test 4: Combined Confidence Score]")
    print(f"  Overall Score: {conf.overall_score} ({conf.confidence_level}) | Passed: {conf.passed_checks}")
    print(f"  Breakdown: {json.dumps(conf.breakdown, indent=2)}")
