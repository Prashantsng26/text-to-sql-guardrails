"""
guardrails.py
Multi-layered safety and guardrail engine for validating and sanitizing SQL queries before execution.
Enforces independently toggleable safety rules:
1. Block DDL statements (CREATE, ALTER, DROP, TRUNCATE, etc.)
2. Block DML write operations (INSERT, UPDATE, DELETE, MERGE, etc.)
3. Auto-inject row limit (default LIMIT 1000) if no limit is present in outer query
4. Reject queries with subquery nesting depth > 3
5. Reject queries whose EXPLAIN plan estimates scanning > N rows
Logs all blocked queries with rule violations and timestamps to guardrail_log.jsonl.
"""

from datetime import datetime, timezone
import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple
import duckdb
from pydantic import BaseModel, Field
import sqlparse
from sqlparse.sql import Identifier, IdentifierList, Parenthesis, Statement, Token, TokenList
from sqlparse.tokens import DDL, DML, Keyword

GUARDRAIL_LOG_PATH = "guardrail_log.jsonl"
DEFAULT_DB_PATH = "ecommerce.duckdb"

DEFAULT_GUARDRAIL_CONFIG: Dict[str, Any] = {
    "block_ddl": True,
    "block_dml": True,
    "enforce_limit": True,
    "max_limit": 1000,
    "check_subquery_depth": True,
    "max_subquery_depth": 3,
    "check_scan_rows": True,
    "max_estimated_scan_rows": 100000,
    "db_path": DEFAULT_DB_PATH,
    "log_file": GUARDRAIL_LOG_PATH
}


class GuardrailResult(BaseModel):
    """Structured result returned by the guardrail check."""
    allowed: bool = Field(..., description="True if query passed all enabled guardrails; False if blocked.")
    violations: List[str] = Field(default_factory=list, description="List of specific rule violations.")
    original_sql: str = Field(..., description="Original input SQL query.")
    sanitized_sql: str = Field(..., description="Sanitized SQL (e.g., with LIMIT injected if required).")
    details: Dict[str, Any] = Field(default_factory=dict, description="Diagnostic details from rule evaluations.")


def _has_ddl(parsed_statements: List[Statement]) -> Tuple[bool, List[str]]:
    """Checks if any statement contains DDL operations."""
    ddl_keywords = {"CREATE", "ALTER", "DROP", "TRUNCATE", "RENAME", "COMMENT", "GRANT", "REVOKE", "PRAGMA"}
    violations = []
    
    for stmt in parsed_statements:
        stmt_type = stmt.get_type()
        if stmt_type in ddl_keywords:
            violations.append(f"DDL statement type detected: {stmt_type}.")
        
        for token in stmt.flatten():
            if token.ttype in (Keyword, DDL, Keyword.DDL):
                token_val = token.value.upper()
                if token_val in ddl_keywords:
                    violations.append(f"DDL keyword '{token_val}' detected.")
                    break

    return len(violations) > 0, list(set(violations))


def _has_dml_writes(parsed_statements: List[Statement]) -> Tuple[bool, List[str]]:
    """Checks if any statement contains mutating DML write operations."""
    dml_write_keywords = {"INSERT", "UPDATE", "DELETE", "MERGE", "REPLACE", "UPSERT", "CALL", "COPY", "EXPORT"}
    violations = []

    for stmt in parsed_statements:
        stmt_type = stmt.get_type()
        if stmt_type in dml_write_keywords:
            violations.append(f"DML write statement type detected: {stmt_type}.")
        
        for token in stmt.flatten():
            if token.ttype in (Keyword, DML, Keyword.DML):
                token_val = token.value.upper()
                if token_val in dml_write_keywords:
                    violations.append(f"Mutating DML keyword '{token_val}' detected.")
                    break

    return len(violations) > 0, list(set(violations))


def _calculate_subquery_depth(sql: str) -> int:
    """
    Calculates the maximum depth of nested subqueries (nested SELECT statements).
    """
    parsed = sqlparse.parse(sql)
    if not parsed:
        return 0

    max_depth = 0

    def walk_tokens(token_list: TokenList, current_depth: int):
        nonlocal max_depth
        for token in token_list.tokens:
            if isinstance(token, Parenthesis):
                # Check if this parenthesis contains a SELECT statement
                token_str = token.value.upper()
                if "SELECT" in token_str:
                    new_depth = current_depth + 1
                    if new_depth > max_depth:
                        max_depth = new_depth
                    walk_tokens(token, new_depth)
                else:
                    walk_tokens(token, current_depth)
            elif isinstance(token, TokenList):
                walk_tokens(token, current_depth)

    walk_tokens(parsed[0], 0)
    return max_depth


def _check_and_inject_limit(sql: str, max_limit: int = 1000) -> Tuple[str, bool]:
    """
    Checks if the outer query has a LIMIT clause. If not, injects LIMIT {max_limit}.
    Returns (updated_sql, was_injected).
    """
    cleaned_sql = sql.strip().rstrip(";")
    parsed = sqlparse.parse(cleaned_sql)
    if not parsed:
        return sql, False

    stmt = parsed[0]
    has_outer_limit = False

    # Check top-level tokens for LIMIT (ignoring inside parentheses)
    for token in stmt.tokens:
        if token.ttype is Keyword and token.value.upper() == "LIMIT":
            has_outer_limit = True
            break

    # Regex check as secondary verification for outer query LIMIT
    # Match LIMIT <num> at the end of the query string not inside parens
    if not has_outer_limit:
        # Check if regex matches LIMIT \d+ at the end of query
        limit_match = re.search(r"\bLIMIT\s+(\d+)\s*$", cleaned_sql, flags=re.IGNORECASE)
        if limit_match:
            has_outer_limit = True

    if not has_outer_limit:
        injected_sql = f"{cleaned_sql}\nLIMIT {max_limit};"
        return injected_sql, True

    # If limit already present, ensure trailing semicolon
    return f"{cleaned_sql};", False


def _check_explain_scan_rows(sql: str, db_path: str, max_rows: int) -> Tuple[bool, Optional[str], Dict[str, Any]]:
    """
    Runs EXPLAIN on the read-only database to estimate query scan / cardinality size.
    Returns (is_exceeded, violation_message, explain_stats).
    Fails closed (blocks query) if database connection or EXPLAIN fails.
    """
    if not os.path.exists(db_path):
        return (
            True,
            f"Database file '{db_path}' not found for EXPLAIN row scan verification.",
            {"error": "db_not_found", "failed_closed": True}
        )

    con = None
    cleaned_sql = sql.strip().rstrip(";")
    try:
        con = duckdb.connect(db_path, read_only=True)
        max_estimated_cardinality = 0
        total_scan_rows = 0

        # Primary method: EXPLAIN (FORMAT JSON)
        try:
            res_json = con.execute(f"EXPLAIN (FORMAT JSON) {cleaned_sql};").fetchall()
            if res_json and len(res_json[0]) > 1:
                plan_data = json.loads(res_json[0][1])

                def traverse(node):
                    nonlocal max_estimated_cardinality, total_scan_rows
                    if isinstance(node, dict):
                        extra = node.get("extra_info", {})
                        card_str = extra.get("Estimated Cardinality")
                        if card_str is not None:
                            try:
                                card_val = int(card_str)
                                max_estimated_cardinality = max(max_estimated_cardinality, card_val)
                                if "SCAN" in node.get("name", "").upper():
                                    total_scan_rows += card_val
                            except (ValueError, TypeError):
                                pass
                        for child in node.get("children", []):
                            traverse(child)
                    elif isinstance(node, list):
                        for item in node:
                            traverse(item)

                traverse(plan_data)
        except Exception:
            # Fallback method: Text format EXPLAIN
            res_text = con.execute(f"EXPLAIN {cleaned_sql};").fetchall()
            explain_text = "\n".join(str(r[1] if len(r) > 1 else r[0]) for r in res_text)

            # Match ~265 rows, 265 rows, or EC: 265
            row_matches = re.findall(r"(?:~|\bEC:\s*)(\d+)\s*(?:rows)?", explain_text, flags=re.IGNORECASE)
            if row_matches:
                numbers = [int(m) for m in row_matches]
                max_estimated_cardinality = max(numbers)
                total_scan_rows = sum(numbers)

        effective_estimated_rows = max_estimated_cardinality if max_estimated_cardinality > 0 else total_scan_rows

        if effective_estimated_rows > max_rows:
            return (
                True,
                f"Estimated row scan ({effective_estimated_rows:,}) exceeds maximum allowed threshold of {max_rows:,} rows.",
                {"estimated_cardinality": effective_estimated_rows, "max_threshold": max_rows}
            )

        return False, None, {"estimated_cardinality": effective_estimated_rows, "max_threshold": max_rows}

    except Exception as e:
        # Fail closed on any exception during explain
        return (
            True,
            f"Row scan verification failed during EXPLAIN execution: {str(e)}",
            {"explain_error": str(e), "failed_closed": True}
        )
    finally:
        if con:
            try:
                con.close()
            except Exception:
                pass


def _log_guardrail_violation(
    sql: str,
    violations: List[str],
    config: Dict[str, Any],
    log_file: str = GUARDRAIL_LOG_PATH
) -> None:
    """Logs blocked queries with timestamp and violation details to a JSONL file."""
    log_entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "blocked_sql": sql,
        "violations": violations,
        "config_used": {k: v for k, v in config.items() if k != "db_path"}
    }
    
    try:
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry) + "\n")
    except Exception as e:
        print(f"[Guardrails] Warning: Failed to write to {log_file}: {e}")


def check_guardrails(
    sql: str,
    config: Optional[Dict[str, Any]] = None
) -> GuardrailResult:
    """
    Validates a SQL query against configured guardrail rules.
    
    Args:
        sql: The input SQL query string.
        config: Optional dictionary overriding default guardrail settings.

    Returns:
        GuardrailResult with allowed status, sanitized SQL, and violations list.
    """
    active_config = dict(DEFAULT_GUARDRAIL_CONFIG)
    if config:
        active_config.update(config)

    violations: List[str] = []
    details: Dict[str, Any] = {}
    sanitized_sql = sql

    parsed_statements = sqlparse.parse(sql)
    if not parsed_statements:
        violations.append("Empty or unparseable SQL statement.")
        _log_guardrail_violation(sql, violations, active_config, active_config["log_file"])
        return GuardrailResult(
            allowed=False,
            violations=violations,
            original_sql=sql,
            sanitized_sql=sql,
            details=details
        )

    # 1. Rule: Block DDL
    if active_config.get("block_ddl", True):
        has_ddl_ops, ddl_violations = _has_ddl(parsed_statements)
        if has_ddl_ops:
            violations.extend(ddl_violations)
            details["ddl_blocked"] = True

    # 2. Rule: Block DML Writes
    if active_config.get("block_dml", True):
        has_dml_ops, dml_violations = _has_dml_writes(parsed_statements)
        if has_dml_ops:
            violations.extend(dml_violations)
            details["dml_blocked"] = True

    # 3. Rule: Check Subquery Nesting Depth
    if active_config.get("check_subquery_depth", True):
        max_depth_allowed = active_config.get("max_subquery_depth", 3)
        actual_depth = _calculate_subquery_depth(sql)
        details["subquery_depth"] = actual_depth
        if actual_depth > max_depth_allowed:
            violations.append(
                f"Subquery nesting depth {actual_depth} exceeds maximum allowed depth of {max_depth_allowed}."
            )

    # 4. Rule: Enforce Row Limit (Auto-injection)
    if active_config.get("enforce_limit", True) and not violations:
        max_limit = active_config.get("max_limit", 1000)
        sanitized_sql, injected = _check_and_inject_limit(sql, max_limit)
        details["limit_injected"] = injected
        details["effective_limit"] = max_limit

    # 5. Rule: Max Estimated Row Scan (EXPLAIN plan)
    if active_config.get("check_scan_rows", True) and not violations:
        max_scan = active_config.get("max_scan_rows") or active_config.get("max_estimated_scan_rows", 100000)
        db_path = active_config.get("db_path", DEFAULT_DB_PATH)
        is_exceeded, scan_violation, scan_stats = _check_explain_scan_rows(sanitized_sql, db_path, max_scan)
        details["scan_stats"] = scan_stats
        if is_exceeded and scan_violation:
            violations.append(scan_violation)

    allowed = len(violations) == 0

    # 6. Log if blocked
    if not allowed:
        _log_guardrail_violation(sql, violations, active_config, active_config["log_file"])

    return GuardrailResult(
        allowed=allowed,
        violations=violations,
        original_sql=sql,
        sanitized_sql=sanitized_sql,
        details=details
    )


if __name__ == "__main__":
    print("--- Testing Guardrails ---")

    # Test 1: Valid SELECT without LIMIT -> should inject LIMIT 1000
    q1 = "SELECT * FROM customers WHERE country = 'USA'"
    res1 = check_guardrails(q1)
    print(f"\nTest 1 (SELECT without LIMIT): Allowed={res1.allowed}")
    print(f"Sanitized SQL:\n{res1.sanitized_sql}")

    # Test 2: Block DML DELETE
    q2 = "DELETE FROM support_tickets WHERE ticket_status = 'Closed';"
    res2 = check_guardrails(q2)
    print(f"\nTest 2 (DELETE Block): Allowed={res2.allowed}")
    print(f"Violations: {res2.violations}")

    # Test 3: Block DDL DROP
    q3 = "DROP TABLE customers;"
    res3 = check_guardrails(q3)
    print(f"\nTest 3 (DROP Block): Allowed={res3.allowed}")
    print(f"Violations: {res3.violations}")

    # Test 4: Deeply nested subquery (depth 4 > 3)
    q4 = """
    SELECT * FROM customers WHERE customer_id IN (
        SELECT customer_id FROM orders WHERE order_id IN (
            SELECT order_id FROM order_items WHERE product_id IN (
                SELECT product_id FROM products WHERE category IN (
                    SELECT category FROM products WHERE unit_price > 500
                )
            )
        )
    )
    """
    res4 = check_guardrails(q4)
    print(f"\nTest 4 (Subquery Depth > 3): Allowed={res4.allowed}")
    print(f"Violations: {res4.violations}")

    # Test 5: Row scan threshold exceeded
    q5 = "SELECT * FROM order_items;"
    res5 = check_guardrails(q5, {"max_scan_rows": 10})
    print(f"\nTest 5 (Row Scan > 10 rows): Allowed={res5.allowed}")
    print(f"Violations: {res5.violations}")
    print(f"Scan Stats: {res5.details.get('scan_stats')}")

    print(f"\nCheck {GUARDRAIL_LOG_PATH} for logged violations.")
