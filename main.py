"""
main.py
End-to-End Orchestrator for Phase 1 Text-to-SQL: Schema-Aware Prompt Engine.
Executes all 4 stages in sequence with clearly formatted stage outputs:
  Stage 1: Ambiguity Check & Disambiguation Mapping
  Stage 2: Programmatic Schema Introspection
  Stage 3: Semantic Schema Filtering (Ranking & Cutoff)
  Stage 4: Schema-Aware Prompt Construction
"""

import argparse
import json
import os
import sys
from typing import Optional

from db_setup import seed_database
from schema_extractor import extract_schema
from ambiguity_handler import check_ambiguity
from schema_filter import filter_relevant_tables
from prompt_constructor import build_prompt

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
    }
]


def print_banner(title: str, char: str = "=", width: int = 85) -> None:
    print("\n" + char * width)
    print(f" {title.center(width - 2)} ")
    print(char * width)


def print_stage_header(stage_num: int, title: str) -> None:
    print(f"\n{'#' * 85}")
    print(f"  STAGE {stage_num}: {title.upper()}")
    print(f"{'#' * 85}")


def run_pipeline_for_question(question: str, schema: dict, question_index: Optional[int] = None, question_type: str = "") -> None:
    header = f"TEST CASE {question_index}: {question_type}" if question_index else "TEST CASE"
    print_banner(header, char="=")
    print(f"👉 INPUT QUESTION: \"{question}\"")

    # -------------------------------------------------------------
    # STAGE 1: Ambiguity Detection
    # -------------------------------------------------------------
    print_stage_header(1, "Domain Ambiguity Detection (ambiguity_handler.py)")
    ambiguity_result = check_ambiguity(question, schema)
    if ambiguity_result:
        print(f"⚠️  AMBIGUITY DETECTED: Term '{ambiguity_result['term']}' (matched: '{ambiguity_result['matched_phrase']}')")
        print("   Potential business interpretations & SQL templates:")
        for idx, interp in enumerate(ambiguity_result["interpretations"], 1):
            print(f"   [{idx}] {interp['meaning']}")
            print(f"       SQL: {interp['example_sql']}")
    else:
        print("✅ NO AMBIGUITY DETECTED: Query terminology maps unambiguously to schema concepts.")

    # -------------------------------------------------------------
    # STAGE 2: Semantic Schema Filtering
    # -------------------------------------------------------------
    print_stage_header(2, "Semantic Schema Filtering (schema_filter.py)")
    relevant_tables = filter_relevant_tables(question, schema, threshold=0.28)
    print(f"🎯 Selected Relevant Tables for Prompt Context: {relevant_tables}")

    # -------------------------------------------------------------
    # STAGE 3: Prompt Construction
    # -------------------------------------------------------------
    print_stage_header(3, "Prompt Construction (prompt_constructor.py)")
    final_prompt = build_prompt(question, schema, relevant_tables)
    
    print("📜 FULL ASSEMBLED PROMPT STRING (Ready for LLM):")
    print("-" * 85)
    print(final_prompt)
    print("-" * 85)


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 1 Text-to-SQL Schema-Aware Prompt Engine")
    parser.add_argument("--reseed", action="store_true", help="Force re-creation and re-seeding of DuckDB database")
    parser.add_argument("--question", type=str, help="Run the pipeline on a single custom question")
    args = parser.parse_args()

    print_banner("PHASE 1: TEXT-TO-SQL SCHEMA-AWARE PROMPT ENGINE", char="*")

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
    if args.question:
        run_pipeline_for_question(args.question, schema, question_index=1, question_type="Custom Query")
    else:
        for idx, item in enumerate(SAMPLE_QUESTIONS, 1):
            run_pipeline_for_question(item["question"], schema, question_index=idx, question_type=item["type"])

    print_banner("PIPELINE EXECUTION COMPLETED SUCCESSFULLY", char="*")


if __name__ == "__main__":
    main()
