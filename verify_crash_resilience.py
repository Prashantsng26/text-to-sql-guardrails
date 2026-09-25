from main import run_pipeline_for_question
import json

schema = json.load(open("schema.json"))

try:
    result = run_pipeline_for_question(
        "Explain the meaning of life using only the database schema.",
        schema,
        question_index=99,
        question_type="Stress Test"
    )
    print("Pipeline handled it gracefully. Result:", result)
except Exception as e:
    print("PIPELINE STILL CRASHED:", type(e).__name__, str(e)[:300])
