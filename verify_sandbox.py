from sandbox_executor import execute_safely

# Test 1: Normal SELECT should work and return real data
r1 = execute_safely("SELECT * FROM customers LIMIT 3;")
print("Test 1 (normal SELECT):")
print("  rows:", r1.rows)
print("  row_count:", r1.row_count)
print("  execution_time_ms:", r1.execution_time_ms)
print("  explain_plan (first 200 chars):", str(r1.explain_plan)[:200])
print()

# Test 2: Try to sneak a write through — even if guardrails were bypassed,
# the sandbox's read-only connection should block it
try:
    r2 = execute_safely("DELETE FROM customers WHERE customer_id = 1;")
    print("Test 2 (DELETE attempt): DID NOT RAISE - allowed=", r2)
except Exception as e:
    print("Test 2 (DELETE attempt): correctly raised an error ->", type(e).__name__, str(e)[:200])
