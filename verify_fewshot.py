from prompt_constructor import build_prompt
from schema_filter import filter_relevant_tables
import json

schema = json.load(open('schema.json'))
q = "Which products are running low on inventory?"
tables = filter_relevant_tables(q, schema, 0.3)
print("Relevant tables:", tables)
prompt = build_prompt(q, schema, tables)
print(prompt)