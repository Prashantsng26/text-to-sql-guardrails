"""
main.py
End-to-End Orchestrator for Text-to-SQL Pipeline (Phases 1, 2 & 3).
Executes the full 7-stage lifecycle for natural language questions:
  Stage 1: Domain Ambiguity Detection (ambiguity_handler.py)
  Stage 2: Semantic Schema Filtering (schema_filter.py)
  Stage 3: Schema-Aware Prompt Construction (prompt_constructor.py)
  Stage 4: Structured SQL Generation & Syntax Validation (sql_generator.py)
  Stage 5: Multi-Rule Guardrail Safety Verification & Logging (guardrails.py)
  Stage 6: Isolated Read-Only Sandbox Execution (sandbox_executor.py)
  Stage 7: Hallucination Detection & Confidence Scoring (hallucination_detector.py)
"""

import argparse
import json
import os
import sys
import traceback
from typing import Any, Dict, List, Optional

from db_setup import seed_database
from schema_extractor import extract_schema
from ambiguity_handler import check_ambiguity
from schema_filter import filter_relevant_tables
from prompt_constructor import build_prompt
from sql_generator import generate_sql, GeneratedSQL
from guardrails import check_guardrails
from sandbox_executor import execute_safely
from hallucination_detector import (
    verify_back_translation,
    check_result_sanity,
    cross_validate_query,
    compute_confidence,
    BackTranslationResult,
    SanityCheckResult,
    CrossValidationResult,
    ConfidenceReport
)

DB_FILE = "ecommerce.duckdb"

SAMPLE_QUESTIONS = [
    {
        "type": "1. Simple Single-Table Lookup",
        "question": "What are the email addresses and sign-up dates of all Platinum tier customers from Germany?"
    },
    {
        "type": "2. Multi-Table Aggregation & Join",
        "question": "Which product categories generated the highest total quantity sold across all completed orders?"
    },
    {
        "type": "3. Deliberately Ambiguous Query",
        "question": "What was our total revenue from active customers last month?"
    },
    {
        "type": "4. Domain-Specific Filter (Customer Support)",
        "question": "List all open or in-progress support tickets with Urgent priority concerning Billing issues."
    },
    {
        "type": "5. Guardrail Violation Test: Mutating DML (Delete Attempt)",
        "question": "Find all churned customers and also delete their old support tickets.",
        "direct_sql": "DELETE FROM support_tickets WHERE ticket_status = 'Closed';"
    },
    {
        "type": "6. Guardrail Violation Test: Nested Subquery Depth > 3",
        "question": "Deeply nested lookup of customers whose ordered products belong to top categories.",
        "direct_sql": """SELECT * FROM customers WHERE customer_id IN (
    SELECT customer_id FROM orders WHERE order_id IN (
        SELECT order_id FROM order_items WHERE product_id IN (
            SELECT product_id FROM products WHERE category IN (
                SELECT category FROM products WHERE unit_price > 500
            )
        )
    )
);"""
    },
    {
        "type": "7. Hallucination Detection Test: Semantic Divergence & Sanity Anomaly",
        "question": "Show all affiliate reward points, customer review ratings, and coupon rebate codes for orders in Asia.",
        "direct_sql": "SELECT customer_id, email, customer_tier FROM customers WHERE country = 'Germany' LIMIT 5;"
    }
]


def print_banner(title: str, char: str = "=", width: int = 95) -> None:
    print("\n" + char * width)
    print(f" {title.center(width - 2)} ")
    print(char * width)


def print_stage_header(stage_num: int, title: str) -> None:
    print(f"\n{'#' * 95}")
    print(f"  STAGE {stage_num}: {title.upper()}")
    print(f"{'#' * 95}")


def run_pipeline_for_question(
    question: str,
    schema: dict,
    question_index: Optional[int] = None,
    question_type: str = "",
    direct_sql: Optional[str] = None
) -> Dict[str, Any]:
    """
    Executes the 7-stage Text-to-SQL pipeline for a single question.
    Catches errors at each stage gracefully to ensure remaining test cases continue.
    
    Returns:
        Dict with execution summary (status, confidence_score, confidence_level, details)
    """
    header = f"TEST CASE {question_index}: {question_type}" if question_index else "TEST CASE"
    print_banner(header, char="=")
    print(f"👉 INPUT QUESTION: \"{question}\"")

    # -------------------------------------------------------------
    # STAGE 1: Ambiguity Detection
    # -------------------------------------------------------------
    print_stage_header(1, "Domain Ambiguity Detection (ambiguity_handler.py)")
    try:
        ambiguity_result = check_ambiguity(question, schema)
        if ambiguity_result:
            print(f"⚠️  AMBIGUITY DETECTED: Term '{ambiguity_result['term']}' (matched: '{ambiguity_result['matched_phrase']}')")
            print("   Potential business interpretations & SQL templates:")
            for idx, interp in enumerate(ambiguity_result["interpretations"], 1):
                print(f"   [{idx}] {interp['meaning']}")
                print(f"       SQL: {interp['example_sql']}")
        else:
            print("✅ NO AMBIGUITY DETECTED: Query terminology maps unambiguously to schema concepts.")
    except Exception as e:
        print(f"❌ STAGE 1 FAILED (Ambiguity Check): {type(e).__name__}: {e}")

    # -------------------------------------------------------------
    # STAGE 2: Semantic Schema Filtering
    # -------------------------------------------------------------
    print_stage_header(2, "Semantic Schema Filtering (schema_filter.py)")
    try:
        relevant_tables = filter_relevant_tables(question, schema, threshold=0.28)
        print(f"🎯 Selected Relevant Tables for Prompt Context: {relevant_tables}")
    except Exception as e:
        print(f"❌ STAGE 2 FAILED (Schema Filter): {type(e).__name__}: {e}")
        relevant_tables = list(schema.get("tables", {}).keys())
        print(f"   Falling back to all tables: {relevant_tables}")

    # -------------------------------------------------------------
    # STAGE 3: Prompt Construction
    # -------------------------------------------------------------
    print_stage_header(3, "Prompt Construction (prompt_constructor.py)")
    try:
        final_prompt = build_prompt(question, schema, relevant_tables)
        print("📜 Filtered Schema Prompt Preview (first 300 chars):")
        print(f"{final_prompt[:300]}...\n[Prompt fully constructed with {len(relevant_tables)} tables]")
    except Exception as e:
        print(f"❌ STAGE 3 FAILED (Prompt Construction): {type(e).__name__}: {e}")
        return {"status": "failed", "stage": "Stage 3: Prompt Construction", "error": str(e)}

    # -------------------------------------------------------------
    # STAGE 4: Structured SQL Generation & Syntax Validation
    # -------------------------------------------------------------
    print_stage_header(4, "Structured SQL Generation & Syntax Validation (sql_generator.py)")
    
    if direct_sql:
        print("⚙️  [Test Harness Direct SQL Mode]: Using test SQL directly to verify guardrails.")
        sql_to_validate = direct_sql.strip()
        generated_info = {
            "sql": sql_to_validate,
            "explanation": "Hardcoded test query for guardrail verification.",
            "confidence": 1.0,
            "tables_used": relevant_tables,
            "columns_used": ["*"]
        }
        print(f"🤖 TEST SQL:\n{sql_to_validate}")
    else:
        try:
            generated = generate_sql(final_prompt)
            sql_to_validate = generated.sql
            generated_info = {
                "sql": generated.sql,
                "explanation": generated.explanation,
                "confidence": generated.confidence,
                "tables_used": generated.tables_used,
                "columns_used": generated.columns_used
            }
            print(f"🤖 GENERATED SQL:\n{generated.sql}")
            print(f"💡 Explanation: {generated.explanation}")
            print(f"📊 Generator Confidence: {generated.confidence:0.2f}")
            print(f"📋 Referenced Tables: {generated.tables_used} | Columns: {generated.columns_used}")
            print("✅ SQL Syntax Validation: Passed (sqlparse & DuckDB dry-run parser)")
        except Exception as e:
            print(f"❌ STAGE 4 FAILED: Structured SQL Generation failed.")
            print(f"   Error: {type(e).__name__}: {e}")
            return {"status": "failed", "stage": "Stage 4: SQL Generation", "error": str(e)}

    # -------------------------------------------------------------
    # STAGE 5: Multi-Rule Guardrail Safety Verification
    # -------------------------------------------------------------
    print_stage_header(5, "Guardrail Safety Verification (guardrails.py)")
    try:
        guardrail_res = check_guardrails(sql_to_validate)

        if not guardrail_res.allowed:
            print("🚫 QUERY BLOCKED BY GUARDRAILS!")
            print("   Violations Triggered:")
            for v in guardrail_res.violations:
                print(f"   - {v}")
            print("📝 Violation logged to 'guardrail_log.jsonl'.")
            print("🛑 Skipping sandbox execution & hallucination checks due to safety policy violation.")
            return {"status": "blocked", "violations": guardrail_res.violations}

        print("🛡️  GUARDRAIL CHECKS PASSED:")
        print("   - DDL Block: Allowed")
        print("   - DML Mutation Block: Allowed")
        print(f"   - Subquery Depth Check: Allowed (Depth: {guardrail_res.details.get('subquery_depth', 0)})")
        if guardrail_res.details.get("limit_injected"):
            print(f"   - Row Limit Enforcement: Injected LIMIT {guardrail_res.details.get('effective_limit')}")
        else:
            print("   - Row Limit Enforcement: LIMIT already present")
        print(f"✨ Sanitized SQL for Execution:\n{guardrail_res.sanitized_sql}")
        sanitized_sql = guardrail_res.sanitized_sql
    except Exception as e:
        print(f"❌ STAGE 5 FAILED (Guardrails): {type(e).__name__}: {e}")
        return {"status": "failed", "stage": "Stage 5: Guardrails", "error": str(e)}

    # -------------------------------------------------------------
    # STAGE 6: Isolated Read-Only Sandbox Execution
    # -------------------------------------------------------------
    print_stage_header(6, "Sandbox Execution (sandbox_executor.py)")
    try:
        exec_res = execute_safely(sanitized_sql, db_path=DB_FILE)

        if not exec_res.success:
            print(f"❌ Execution Error in Sandbox: {exec_res.error}")
            return {"status": "exec_error", "error": exec_res.error}

        print(f"⚡ Execution Succeeded in {exec_res.execution_time_ms} ms")
        print(f"📈 Total Rows Returned: {exec_res.row_count}")
        print(f"📑 Columns: {exec_res.columns}")
        
        if exec_res.rows:
            print("\n🔎 Sample Output Rows (up to 3):")
            for r_idx, row in enumerate(exec_res.rows[:3], 1):
                print(f"   Row {r_idx}: {row}")
        else:
            print("   (Query returned 0 matching rows)")

        print("\n🌲 EXPLAIN Plan Preview:")
        plan_preview = exec_res.explain_plan[:250].strip() if exec_res.explain_plan else "N/A"
        print(f"{plan_preview}...")

    except Exception as e:
        print(f"❌ STAGE 6 FAILED (Sandbox Execution): {type(e).__name__}: {e}")
        return {"status": "failed", "stage": "Stage 6: Sandbox Execution", "error": str(e)}

    # -------------------------------------------------------------
    # STAGE 7: Hallucination Detection & Confidence Scoring
    # -------------------------------------------------------------
    print_stage_header(7, "Hallucination Detection & Confidence Scoring (hallucination_detector.py)")
    try:
        # 1. Back-Translation Verification
        print("🔄 [1/4] Running Back-Translation Verification...")
        bt_res = verify_back_translation(question, sanitized_sql)
        print(f"   Inferred Intent: \"{bt_res.back_translated_question}\"")
        diverge_str = "⚠️ YES (Semantics Diverged)" if bt_res.diverged else "✅ NO (Semantics Aligned)"
        print(f"   Semantic Alignment Score: {bt_res.alignment_score:.4f} | Diverged: {diverge_str}")

        # 2. Result Sanity Checking
        print("\n🩺 [2/4] Running Domain Result Sanity Checks...")
        sanity_res = check_result_sanity(sanitized_sql, exec_res, schema)
        if sanity_res.passed:
            print(f"   ✅ Sanity Checks Passed ({len(sanity_res.checks_run)} checks evaluated without warnings)")
        else:
            print(f"   ⚠️  Sanity Warnings Detected ({len(sanity_res.warnings)}):")
            for w in sanity_res.warnings:
                print(f"      - {w}")

        # 3. Multi-Query Cross-Validation
        print("\n🔀 [3/4] Running Multi-Query Sandbox Cross-Validation...")
        cv_res = cross_validate_query(question, final_prompt, schema, primary_sql=sanitized_sql)
        if not cv_res.ran:
            print(f"   ℹ️  Skipped: {cv_res.skip_reason}")
        else:
            match_str = "✅ EXACT MATCH" if cv_res.results_match else "⚠️ MISMATCH"
            print(f"   Alternative SQL Generated:\n   {cv_res.alternative_sql.replace(chr(10), chr(10) + '   ')}")
            print(f"   Result Agreement: {match_str} (Jaccard Score: {cv_res.agreement_score:.2f})")

        # 4. Combined Explainable Confidence Scorer
        print("\n📊 [4/4] Computing Explainable Confidence Score...")
        generator_conf = generated_info.get("confidence", 0.90) if 'generated_info' in locals() else 0.90
        conf_report = compute_confidence(
            sql_generator_confidence=generator_conf,
            syntax_valid=True,
            back_translation=bt_res,
            sanity=sanity_res,
            cross_validation=cv_res
        )

        level_icon = "🟢" if conf_report.confidence_level == "HIGH" else ("🟡" if conf_report.confidence_level == "MEDIUM" else "🔴")
        print(f"\n🎯 FINAL CONFIDENCE SCORE: {conf_report.overall_score:.4f} [{level_icon} {conf_report.confidence_level}]")
        print(f"   Passed Reliability Checks: {'✅ PASSED' if conf_report.passed_checks else '⚠️ FLAGGED (Potential Hallucination/Low Quality)'}")
        print("   Signal Contributions Breakdown:")
        for signal_name, info in conf_report.breakdown.items():
            wt = info.get("weight", 0.0)
            contrib = info.get("contribution", 0.0)
            raw = info.get("raw_score", 0.0)
            print(f"     • {signal_name:<25} : Weight = {wt:.2f} | Raw Score = {raw:.2f} | Contrib = +{contrib:.4f}")

        return {
            "status": "success",
            "generated_sql": sanitized_sql,
            "row_count": exec_res.row_count,
            "execution_time_ms": exec_res.execution_time_ms,
            "confidence_score": conf_report.overall_score,
            "confidence_level": conf_report.confidence_level,
            "passed_checks": conf_report.passed_checks,
            "confidence_report": conf_report
        }

    except Exception as e:
        print(f"❌ STAGE 7 FAILED (Hallucination Detection): {type(e).__name__}: {e}")
        traceback.print_exc()
        return {
            "status": "success",
            "generated_sql": sanitized_sql if 'sanitized_sql' in locals() else "N/A",
            "row_count": exec_res.row_count,
            "execution_time_ms": exec_res.execution_time_ms,
            "stage7_warning": str(e)
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Text-to-SQL Engine: Phase 1, 2 & 3")
    parser.add_argument("--reseed", action="store_true", help="Force re-creation and re-seeding of DuckDB database")
    parser.add_argument("--question", type=str, help="Run the full pipeline on a single custom question")
    args = parser.parse_args()

    print_banner("PHASE 1, 2 & 3: TEXT-TO-SQL PROMPT ENGINE, SAFETY LAYER & HALLUCINATION DETECTOR", char="*")

    # Step 0: Ensure database exists
    if args.reseed or not os.path.exists(DB_FILE):
        print("\n[Init] Initializing and seeding DuckDB database...")
        seed_database(DB_FILE)

    # Step 1: Programmatic Schema Introspection
    print_stage_header(0, "Live Database Introspection (schema_extractor.py)")
    schema = extract_schema(DB_FILE, output_json="schema.json")
    print(f"✅ Successfully introspected DuckDB database '{DB_FILE}'.")
    print(f"   Found {len(schema['tables'])} tables: {list(schema['tables'].keys())}")
    print("   Extracted detailed column types, foreign keys, and categorical values to 'schema.json'.")

    # Step 2: Run questions
    run_records: List[tuple] = []

    if args.question:
        res = run_pipeline_for_question(args.question, schema, question_index=1, question_type="Custom Query")
        run_records.append(({"type": "Custom Query", "question": args.question}, res))
    else:
        for idx, item in enumerate(SAMPLE_QUESTIONS, 1):
            res = run_pipeline_for_question(
                item["question"],
                schema,
                question_index=idx,
                question_type=item["type"],
                direct_sql=item.get("direct_sql")
            )
            run_records.append((item, res))

    # Step 3: Print Final Execution Summary
    print_banner("PIPELINE TEST EXECUTION SUMMARY", char="*")
    total = len(run_records)
    succeeded = sum(1 for _, r in run_records if r.get("status") == "success" and r.get("confidence_level") in ("HIGH", "MEDIUM", None))
    low_conf = sum(1 for _, r in run_records if r.get("status") == "success" and r.get("confidence_level") == "LOW")
    blocked = sum(1 for _, r in run_records if r.get("status") == "blocked")
    failed = sum(1 for _, r in run_records if r.get("status") in ("failed", "exec_error"))

    print(f"Total Test Cases Executed: {total}")
    print(f"  🟢 Executed High/Medium Confidence : {succeeded}")
    print(f"  🔴 Flagged Low Confidence / Anomaly: {low_conf}")
    print(f"  🛡️  Safely Blocked by Guardrails     : {blocked}")
    print(f"  ❌ Failed / Errored Test Cases     : {failed}")
    print("-" * 95)
    for idx, (item, res) in enumerate(run_records, 1):
        status = res.get("status", "unknown").upper()
        conf_lvl = res.get("confidence_level")
        conf_score = res.get("confidence_score")

        if status == "SUCCESS":
            if conf_lvl == "HIGH":
                status_display = f"🟢 HIGH ({conf_score:.2f})"
            elif conf_lvl == "MEDIUM":
                status_display = f"🟡 MED  ({conf_score:.2f})"
            elif conf_lvl == "LOW":
                status_display = f"🔴 LOW  ({conf_score:.2f})"
            else:
                status_display = "✅ SUCCESS"
        elif status == "BLOCKED":
            status_display = "🛡️  BLOCKED"
        else:
            status_display = f"❌ {status}"
        
        q_clip = item["question"][:42] + "..." if len(item["question"]) > 42 else item["question"]
        print(f"[{idx}] {status_display:<16} | {item['type']:<45} | {q_clip}")
    print("-" * 95)

    print("\n" + "=" * 95)
    print("DETAILED TEST CASE QUESTIONS & GENERATED SQL PAIRINGS")
    print("=" * 95)
    for idx, (item, res) in enumerate(run_records, 1):
        status = res.get("status", "unknown").upper()
        conf_lvl = res.get("confidence_level", "N/A")
        conf_score = res.get("confidence_score")
        score_str = f"{conf_score:.2f}" if conf_score is not None else "N/A"
        raw_sql = res.get("generated_sql", item.get("direct_sql", "None (Stage Failed)"))
        clean_sql = " ".join(raw_sql.split()) if raw_sql else "N/A"
        if len(clean_sql) > 90:
            clean_sql = clean_sql[:87] + "..."

        print(f"[{idx}] {item['type']}")
        print(f"    ❓ Question      : \"{item['question']}\"")
        print(f"    🤖 Generated SQL : {clean_sql}")
        print(f"    🎯 Result Status : {status} (Level: {conf_lvl}, Score: {score_str})")
        print("-" * 95)

    print_banner("PIPELINE RUN COMPLETE", char="*")


if __name__ == "__main__":
    main()
