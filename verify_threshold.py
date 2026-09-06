from schema_filter import filter_relevant_tables
import json
schema = json.load(open('schema.json'))
q = 'Which product categories generated the most revenue from completed orders?'
print("Threshold 0.3:", filter_relevant_tables(q, schema, 0.3))
print("Threshold 0.35:", filter_relevant_tables(q, schema, 0.35))
print("Threshold 0.5:", filter_relevant_tables(q, schema, 0.5))
print("Threshold 0.5:", filter_relevant_tables(q, schema, 0.5))
