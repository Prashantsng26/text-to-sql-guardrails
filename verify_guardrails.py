from guardrails import check_guardrails

# Test 1: DDL should be blocked
r1 = check_guardrails("DROP TABLE customers;", {})
print("Test 1 (DROP TABLE):", r1)

# Test 2: DML write should be blocked
r2 = check_guardrails("DELETE FROM support_tickets WHERE ticket_id = 1;", {})
print("Test 2 (DELETE):", r2)

# Test 3: Valid SELECT without LIMIT -> should auto-inject
r3 = check_guardrails("SELECT * FROM customers;", {})
print("Test 3 (no LIMIT):", r3)

# Test 4: Valid SELECT with LIMIT already -> should pass unchanged
r4 = check_guardrails("SELECT * FROM customers LIMIT 10;", {})
print("Test 4 (has LIMIT):", r4)

# Test 5: Deeply nested subquery -> should be blocked
deep_query = "SELECT * FROM (SELECT * FROM (SELECT * FROM (SELECT * FROM (SELECT * FROM customers))));"
r5 = check_guardrails(deep_query, {})
print("Test 5 (deep subquery):", r5)
