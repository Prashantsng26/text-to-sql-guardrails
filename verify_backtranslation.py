from hallucination_detector import verify_back_translation

# Case A: SQL that clearly matches the question -> should NOT diverge
q1 = "Which customers spent the most on completed orders?"
sql1 = """SELECT c.customer_id, c.first_name, c.last_name, SUM(o.total_amount) AS total_spent
FROM customers c JOIN orders o ON c.customer_id = o.customer_id
WHERE o.order_status = 'Completed'
GROUP BY c.customer_id, c.first_name, c.last_name
ORDER BY total_spent DESC;"""
r1 = verify_back_translation(q1, sql1)
print("\nCase A (should align):")
print("  Inferred:", r1.back_translated_question)
print("  Score:", r1.alignment_score, "| Diverged:", r1.diverged)
assert not r1.diverged, "Case A should not diverge!"

# Case B: SQL that clearly does NOT match the question -> should diverge
q2 = "How many support tickets are urgent?"
sql2 = "SELECT product_name, unit_price FROM products WHERE category = 'Electronics';"
r2 = verify_back_translation(q2, sql2)
print("\nCase B (should diverge):")
print("  Inferred:", r2.back_translated_question)
print("  Score:", r2.alignment_score, "| Diverged:", r2.diverged)
assert r2.diverged, "Case B should diverge!"

# Case C: Test Case 3 Revenue query
q3 = "What was our total revenue from active customers last month?"
sql3 = """SELECT ROUND(SUM(total_amount), 2) AS total_revenue
FROM orders
WHERE order_status = 'Completed'
  AND order_date >= CURRENT_DATE - INTERVAL '30 days';"""
r3 = verify_back_translation(q3, sql3)
print("\nCase C (Test Case 3 - should align with revenue intent):")
print("  Inferred:", r3.back_translated_question)
print("  Score:", r3.alignment_score, "| Diverged:", r3.diverged)
assert not r3.diverged, "Case C should not diverge!"

print("\n🎉 ALL BACK-TRANSLATION TESTS PASSED!")
