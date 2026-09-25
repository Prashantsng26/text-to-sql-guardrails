from sql_generator import generate_sql
from prompt_constructor import build_prompt
from schema_filter import filter_relevant_tables
import json

schema = json.load(open("schema.json"))
q = "Which customers have spent the most on completed orders?"
tables = filter_relevant_tables(q, schema, 0.3)
prompt = build_prompt(q, schema, tables)
result = generate_sql(prompt)
print("SQL:", result.sql)
print("Explanation:", result.explanation)
print("Confidence:", result.confidence)
print("Tables used:", result.tables_used)
print("Columns used:", result.columns_used)
