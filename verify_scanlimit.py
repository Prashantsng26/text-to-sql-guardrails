from guardrails import check_guardrails

# Test with a very low scan limit to force a block
config = {"max_scan_rows": 10}
r = check_guardrails("SELECT * FROM order_items;", config)
print("Test (low scan limit, 265 rows in table):", r)
