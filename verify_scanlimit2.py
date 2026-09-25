from guardrails import check_guardrails

# High limit — should pass since order_items only has 265 rows
r = check_guardrails("SELECT * FROM order_items;", {"max_scan_rows": 10000})
print("Test (high limit, 265 rows):", r)

# Different table to make sure it's not hardcoded to order_items
r2 = check_guardrails("SELECT * FROM customers;", {"max_scan_rows": 10})
print("Test (customers, 60 rows, low limit):", r2)
