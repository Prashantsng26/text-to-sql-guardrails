from ambiguity_handler import check_ambiguity
import json
schema = json.load(open('schema.json'))
print("Q1 (revenue):", check_ambiguity('What is our total revenue?', schema))
print()
print("Q2 (top-selling):", check_ambiguity('Show me the top-selling products', schema))
print()
print("Q3 (clear):", check_ambiguity('List all customers in Delhi', schema))
