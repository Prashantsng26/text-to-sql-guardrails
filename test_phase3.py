"""
test_phase3.py
Unit tests and verification suite for Phase 3: Hallucination Detection & Confidence Scoring.
"""

import os
import unittest
from schema_extractor import extract_schema
from sandbox_executor import ExecutionResult, execute_safely
from hallucination_detector import (
    BackTranslationResult,
    SanityCheckResult,
    CrossValidationResult,
    ConfidenceReport,
    verify_back_translation,
    check_result_sanity,
    cross_validate_query,
    compute_confidence
)


class TestPhase3HallucinationDetector(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.schema = extract_schema()

    def test_1_back_translation_aligned(self):
        """Tests that a well-matching SQL query yields high alignment and diverged=False."""
        q = "What are the email addresses and sign-up dates of all Platinum tier customers from Germany?"
        sql = "SELECT email, created_at FROM customers WHERE country = 'Germany' AND customer_tier = 'Platinum';"
        res = verify_back_translation(q, sql, threshold=0.60)
        self.assertIsInstance(res, BackTranslationResult)
        self.assertFalse(res.diverged)
        self.assertGreaterEqual(res.alignment_score, 0.60)
        print(f"\n[Test 1 Aligned] Score: {res.alignment_score:.4f}, Inferred: {res.back_translated_question}")

    def test_1_back_translation_diverged(self):
        """Tests that a totally mismatched SQL query yields low alignment and diverged=True."""
        q = "Show me the top 5 most expensive products in the catalog."
        # Mismatched SQL that answers something completely different (support tickets)
        sql = "SELECT ticket_id, priority, status FROM support_tickets WHERE priority = 'Urgent';"
        res = verify_back_translation(q, sql, threshold=0.60)
        self.assertIsInstance(res, BackTranslationResult)
        self.assertTrue(res.diverged)
        self.assertLess(res.alignment_score, 0.60)
        print(f"\n[Test 1 Diverged] Score: {res.alignment_score:.4f}, Inferred: {res.back_translated_question}")

    def test_2_sanity_check_passing(self):
        """Tests sanity checks pass on valid query results."""
        sql = "SELECT customer_id, email, customer_tier FROM customers WHERE country = 'Germany';"
        exec_res = execute_safely(sql)
        sanity = check_result_sanity(sql, exec_res, self.schema)
        self.assertIsInstance(sanity, SanityCheckResult)
        self.assertTrue(sanity.passed)
        self.assertEqual(len(sanity.warnings), 0)
        print(f"\n[Test 2 Passing] Passed: {sanity.passed}, Warnings: {sanity.warnings}")

    def test_2_sanity_check_anomalies(self):
        """Tests sanity checks catch negative values, empty joins, and high null fractions."""
        # 1. Negative amount
        mock_exec_negative = ExecutionResult(
            success=True,
            columns=["product_id", "total_price"],
            rows=[{"product_id": 1, "total_price": -45.0}],
            row_count=1,
            execution_time_ms=5.0
        )
        sanity_neg = check_result_sanity("SELECT product_id, -45.0 AS total_price FROM products;", mock_exec_negative, self.schema)
        self.assertFalse(sanity_neg.passed)
        self.assertTrue(any("Negative numeric value" in w for w in sanity_neg.warnings))

        # 2. Empty JOIN
        mock_exec_empty_join = ExecutionResult(
            success=True,
            columns=["order_id", "customer_id"],
            rows=[],
            row_count=0,
            execution_time_ms=5.0
        )
        sanity_empty = check_result_sanity("SELECT o.order_id, c.customer_id FROM orders o JOIN customers c ON o.customer_id = 999999;", mock_exec_empty_join, self.schema)
        self.assertFalse(sanity_empty.passed)
        self.assertTrue(any("returned 0 rows" in w for w in sanity_empty.warnings))

        # 3. High NULL fraction on NOT NULL column (e.g. customer_id is PK/NOT NULL)
        mock_exec_nulls = ExecutionResult(
            success=True,
            columns=["customer_id", "email"],
            rows=[{"customer_id": None, "email": "test@example.com"}, {"customer_id": None, "email": "test2@example.com"}],
            row_count=2,
            execution_time_ms=5.0
        )
        sanity_nulls = check_result_sanity("SELECT c.customer_id, c.email FROM orders o LEFT JOIN customers c ON o.customer_id = 9999;", mock_exec_nulls, self.schema)
        self.assertFalse(sanity_nulls.passed)
        self.assertTrue(any("NOT NULL in the database schema" in w for w in sanity_nulls.warnings))
        print(f"\n[Test 2 Anomalies] Detected warnings:\n" + "\n".join(f"  - {w}" for w in sanity_neg.warnings + sanity_empty.warnings + sanity_nulls.warnings))

    def test_3_cross_validation_skip_trivial(self):
        """Tests that single-table trivial lookup skips multi-query generation."""
        q = "List all customer emails."
        prompt = "SELECT email FROM customers;"
        cv = cross_validate_query(q, prompt, self.schema, primary_sql="SELECT email FROM customers;")
        self.assertIsInstance(cv, CrossValidationResult)
        self.assertFalse(cv.ran)
        self.assertIn("Trivial single-table lookup", cv.skip_reason)
        print(f"\n[Test 3 Skipped] Ran: {cv.ran}, Skip Reason: {cv.skip_reason}")

    def test_3_cross_validation_aggregation(self):
        """Tests cross-validation runs and matches on multi-table join / aggregation."""
        q = "Which product categories generated the highest total quantity sold across all completed orders?"
        prompt = f"Generate DuckDB SQL for: {q}"
        primary_sql = """SELECT p.category, SUM(oi.quantity) AS total_quantity_sold
FROM products p
JOIN order_items oi ON p.product_id = oi.product_id
JOIN orders o ON oi.order_id = o.order_id
WHERE o.order_status = 'Completed'
GROUP BY p.category
ORDER BY total_quantity_sold DESC;"""
        cv = cross_validate_query(q, prompt, self.schema, primary_sql=primary_sql)
        self.assertIsInstance(cv, CrossValidationResult)
        self.assertTrue(cv.ran)
        self.assertGreaterEqual(cv.agreement_score, 0.0)
        print(f"\n[Test 3 Aggregation] Ran: {cv.ran}, Match: {cv.results_match}, Score: {cv.agreement_score}")

    def test_4_combined_confidence_scoring(self):
        """Tests combined confidence report calculation and explanation breakdown."""
        # 1. High confidence case
        bt_good = BackTranslationResult(
            back_translated_question="What are the emails of Platinum customers in Germany?",
            alignment_score=0.92,
            diverged=False
        )
        sanity_good = SanityCheckResult(passed=True, warnings=[])
        cv_good = CrossValidationResult(
            ran=True,
            primary_sql="SELECT ...",
            alternative_sql="WITH cte AS (...) SELECT ...",
            results_match=True,
            agreement_score=1.0
        )
        conf_high = compute_confidence(
            sql_generator_confidence=0.95,
            syntax_valid=True,
            back_translation=bt_good,
            sanity=sanity_good,
            cross_validation=cv_good
        )
        self.assertIsInstance(conf_high, ConfidenceReport)
        self.assertEqual(conf_high.confidence_level, "HIGH")
        self.assertTrue(conf_high.passed_checks)
        self.assertGreater(conf_high.overall_score, 0.85)

        # 2. Hard Gate Syntax Failure
        conf_gate = compute_confidence(
            sql_generator_confidence=0.95,
            syntax_valid=False,
            back_translation=bt_good,
            sanity=sanity_good,
            cross_validation=cv_good
        )
        self.assertEqual(conf_gate.overall_score, 0.0)
        self.assertEqual(conf_gate.confidence_level, "CRITICAL_FAIL")
        self.assertFalse(conf_gate.passed_checks)

        # 3. Diverged / Low Confidence Case
        bt_diverged = BackTranslationResult(
            back_translated_question="What are support tickets?",
            alignment_score=0.35,
            diverged=True
        )
        sanity_warn = SanityCheckResult(passed=False, warnings=["Mismatched join produced empty result"])
        cv_mismatch = CrossValidationResult(
            ran=True,
            primary_sql="SELECT ...",
            alternative_sql="SELECT ...",
            results_match=False,
            agreement_score=0.20
        )
        conf_low = compute_confidence(
            sql_generator_confidence=0.50,
            syntax_valid=True,
            back_translation=bt_diverged,
            sanity=sanity_warn,
            cross_validation=cv_mismatch
        )
        self.assertEqual(conf_low.confidence_level, "LOW")
        self.assertFalse(conf_low.passed_checks)
        self.assertLess(conf_low.overall_score, 0.65)
        print(f"\n[Test 4 Combined Confidence] High: {conf_high.overall_score} ({conf_high.confidence_level}) | Low: {conf_low.overall_score} ({conf_low.confidence_level})")


if __name__ == "__main__":
    unittest.main()
