from prompt_constructor import build_prompt
from schema_filter import filter_relevant_tables
import json
schema = json.load(open('schema.json'))
q = 'Which product categories generated the most revenue from completed orders?'
tables = filter_relevant_tables(q, schema, 0.3)
prompt = build_prompt(q, schema, tables)
print(prompt)
