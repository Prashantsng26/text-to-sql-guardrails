"""
prompt_constructor.py
Assembles a rich, schema-aware prompt for a downstream LLM.
Includes filtered table schemas, active foreign key join paths,
categorical column values, and dynamically filtered DuckDB few-shot examples
whose SQL queries only reference the relevant tables.
"""

from typing import Any, Dict, List, Optional, Set

# Curated pool of few-shot Question -> SQL pairs, tagged with the tables they reference
FEW_SHOT_POOL = [
    # --- Single-Table Examples ---
    {
        "tables": ["customers"],
        "question": "List all Platinum customers from Canada who signed up in 2025.",
        "sql": """SELECT customer_id, first_name, last_name, email
FROM customers
WHERE customer_tier = 'Platinum'
  AND country = 'Canada'
  AND EXTRACT(YEAR FROM created_at) = 2025;"""
    },
    {
        "tables": ["customers"],
        "question": "How many customers are registered in each country?",
        "sql": """SELECT country, COUNT(*) AS customer_count
FROM customers
GROUP BY country
ORDER BY customer_count DESC;"""
    },
    {
        "tables": ["products"],
        "question": "List all active products in the Electronics category with price under $100.",
        "sql": """SELECT product_id, product_name, unit_price, inventory_count
FROM products
WHERE category = 'Electronics'
  AND status = 'Active'
  AND unit_price < 100.00;"""
    },
    {
        "tables": ["products"],
        "question": "What is the average price of products by category?",
        "sql": """SELECT category, ROUND(AVG(unit_price), 2) AS avg_price
FROM products
GROUP BY category
ORDER BY avg_price DESC;"""
    },
    {
        "tables": ["orders"],
        "question": "What is the total revenue and count of Completed orders by payment method?",
        "sql": """SELECT payment_method,
       COUNT(*) AS order_count,
       ROUND(SUM(total_amount), 2) AS total_revenue
FROM orders
WHERE order_status = 'Completed'
GROUP BY payment_method
ORDER BY total_revenue DESC;"""
    },
    {
        "tables": ["orders"],
        "question": "Find all orders placed in the last 30 days that have a Pending status.",
        "sql": """SELECT order_id, customer_id, order_date, total_amount
FROM orders
WHERE order_status = 'Pending'
  AND order_date >= CURRENT_DATE - INTERVAL '30 days';"""
    },
    {
        "tables": ["support_tickets"],
        "question": "How many open support tickets with 'Urgent' priority exist per issue category?",
        "sql": """SELECT issue_category,
       COUNT(*) AS urgent_ticket_count
FROM support_tickets
WHERE ticket_status = 'Open'
  AND priority = 'Urgent'
GROUP BY issue_category
ORDER BY urgent_ticket_count DESC;"""
    },
    {
        "tables": ["support_tickets"],
        "question": "List all support tickets resolved in the last 14 days.",
        "sql": """SELECT ticket_id, customer_id, priority, issue_category
FROM support_tickets
WHERE ticket_status = 'Resolved'
  AND created_at >= CURRENT_DATE - INTERVAL '14 days';"""
    },

    # --- Two-Table Examples ---
    {
        "tables": ["customers", "orders"],
        "question": "Find the top 3 customers who spent the most on completed orders, including their country.",
        "sql": """SELECT c.customer_id,
       c.first_name || ' ' || c.last_name AS customer_name,
       c.country,
       ROUND(SUM(o.total_amount), 2) AS total_spend
FROM customers c
JOIN orders o ON c.customer_id = o.customer_id
WHERE o.order_status = 'Completed'
GROUP BY c.customer_id, c.first_name, c.last_name, c.country
ORDER BY total_spend DESC
LIMIT 3;"""
    },
    {
        "tables": ["customers", "orders"],
        "question": "What is the email of the customer with the most orders?",
        "sql": """SELECT c.email,
       COUNT(o.order_id) AS total_orders
FROM customers c
JOIN orders o ON c.customer_id = o.customer_id
GROUP BY c.customer_id, c.email
ORDER BY total_orders DESC
LIMIT 1;"""
    },
    {
        "tables": ["customers", "orders"],
        "question": "Which customers from Germany have never placed an order?",
        "sql": """SELECT c.customer_id, c.first_name, c.last_name, c.email
FROM customers c
LEFT JOIN orders o ON c.customer_id = o.customer_id
WHERE c.country = 'Germany'
  AND o.order_id IS NULL;"""
    },
    {
        "tables": ["products", "order_items"],
        "question": "List the top 5 best selling products by total quantity sold.",
        "sql": """SELECT p.product_id,
       p.product_name,
       SUM(oi.quantity) AS total_units_sold
FROM products p
JOIN order_items oi ON p.product_id = oi.product_id
GROUP BY p.product_id, p.product_name
ORDER BY total_units_sold DESC
LIMIT 5;"""
    },
    {
        "tables": ["orders", "order_items"],
        "question": "What is the average number of items per completed order?",
        "sql": """SELECT ROUND(AVG(item_count), 2) AS avg_items_per_order
FROM (
    SELECT o.order_id, SUM(oi.quantity) AS item_count
    FROM orders o
    JOIN order_items oi ON o.order_id = oi.order_id
    WHERE o.order_status = 'Completed'
    GROUP BY o.order_id
);"""
    },
    {
        "tables": ["customers", "support_tickets"],
        "question": "Which Platinum tier customers have filed more than 2 support tickets?",
        "sql": """SELECT c.customer_id,
       c.first_name || ' ' || c.last_name AS customer_name,
       COUNT(st.ticket_id) AS ticket_count
FROM customers c
JOIN support_tickets st ON c.customer_id = st.customer_id
WHERE c.customer_tier = 'Platinum'
GROUP BY c.customer_id, c.first_name, c.last_name
HAVING COUNT(st.ticket_id) > 2;"""
    },

    # --- Multi-Table Examples (3+ Tables) ---
    {
        "tables": ["products", "order_items", "orders"],
        "question": "Which product category generated the highest total sales in completed orders?",
        "sql": """SELECT p.category,
       ROUND(SUM((oi.quantity * oi.unit_price) - oi.discount_amount), 2) AS total_sales
FROM products p
JOIN order_items oi ON p.product_id = oi.product_id
JOIN orders o ON oi.order_id = o.order_id
WHERE o.order_status = 'Completed'
GROUP BY p.category
ORDER BY total_sales DESC
LIMIT 1;"""
    },
    {
        "tables": ["customers", "orders", "order_items", "products"],
        "question": "What are the total sales generated by Gold customers for Electronics products?",
        "sql": """SELECT ROUND(SUM((oi.quantity * oi.unit_price) - oi.discount_amount), 2) AS total_gold_electronics_sales
FROM customers c
JOIN orders o ON c.customer_id = o.customer_id
JOIN order_items oi ON o.order_id = oi.order_id
JOIN products p ON oi.product_id = p.product_id
WHERE c.customer_tier = 'Gold'
  AND p.category = 'Electronics'
  AND o.order_status = 'Completed';"""
    }
]


def get_relevant_few_shot_examples(
    relevant_tables: List[str],
    pool: Optional[List[Dict[str, Any]]] = None,
    min_examples: int = 2,
    max_examples: int = 4
) -> List[Dict[str, Any]]:
    """
    Selects 2 to 4 few-shot examples whose SQL queries strictly reference a subset of relevant_tables.
    Prioritizes examples that maximize coverage of relevant_tables and table diversity.
    If no examples match as an exact subset, returns an empty list to prevent referencing out-of-scope tables.
    """
    if pool is None:
        pool = FEW_SHOT_POOL

    rel_set = set(relevant_tables)
    
    # 1. Filter examples where all referenced tables are within relevant_tables
    exact_matches = []
    for ex in pool:
        ex_tables = set(ex.get("tables", []))
        if ex_tables and ex_tables.issubset(rel_set):
            exact_matches.append(ex)

    if not exact_matches:
        return []

    # 2. Sort matches: prioritize examples that cover more of the relevant_tables, then by table count
    def score_example(ex: Dict[str, Any]) -> tuple:
        ex_tables = set(ex.get("tables", []))
        overlap_size = len(ex_tables & rel_set)
        return (overlap_size, len(ex_tables))

    exact_matches.sort(key=score_example, reverse=True)

    # 3. Select up to max_examples
    return exact_matches[:max_examples]


def _format_table_schema(table_name: str, table_info: Dict[str, Any]) -> str:
    """Formats a single table's columns and primary keys in a clean DDL-like structure."""
    col_lines = []
    for col in table_info.get("columns", []):
        col_def = f"    {col['name']} {col['type']}"
        if not col.get("nullable", True):
            col_def += " NOT NULL"
        if col.get("is_primary_key", False):
            col_def += " PRIMARY KEY"
        col_lines.append(col_def)

    cols_str = ",\n".join(col_lines)
    desc = table_info.get("description", "")
    desc_comment = f"-- Description: {desc}\n" if desc else ""
    return f"{desc_comment}CREATE TABLE {table_name} (\n{cols_str}\n);"


def _extract_active_foreign_keys(schema: Dict[str, Any], relevant_tables: List[str]) -> List[str]:
    """Finds foreign key relationships where BOTH tables are in relevant_tables."""
    rel_set = set(relevant_tables)
    fk_lines = []

    for tbl_name in relevant_tables:
        tbl_info = schema.get("tables", {}).get(tbl_name, {})
        for fk in tbl_info.get("foreign_keys", []):
            ref_table = fk.get("referred_table")
            if ref_table in rel_set:
                col = ", ".join(fk.get("constrained_columns", []))
                ref_col = ", ".join(fk.get("referred_columns", []))
                fk_lines.append(f"- {tbl_name}.{col} -> {ref_table}.{ref_col} (JOIN ON {tbl_name}.{col} = {ref_table}.{ref_col})")

    return fk_lines


def _extract_categorical_values(schema: Dict[str, Any], relevant_tables: List[str]) -> List[str]:
    """Extracts known categorical column values for the relevant tables."""
    cat_lines = []
    for tbl_name in relevant_tables:
        tbl_info = schema.get("tables", {}).get(tbl_name, {})
        for col in tbl_info.get("columns", []):
            if col.get("is_categorical") and col.get("sample_values"):
                vals_str = ", ".join(repr(v) for v in col["sample_values"])
                cat_lines.append(f"- `{tbl_name}.{col['name']}`: [{vals_str}]")

    return cat_lines


def build_prompt(
    user_question: str,
    schema: Dict[str, Any],
    relevant_tables: List[str],
    few_shot_examples: Optional[List[Dict[str, str]]] = None
) -> str:
    """
    Constructs a comprehensive, schema-filtered prompt string for DuckDB Text-to-SQL translation.
    Only includes schema DDL, foreign keys, categorical values, and few-shot examples
    strictly corresponding to the relevant_tables subset.

    Args:
        user_question: The natural language question to convert to SQL.
        schema: The full database schema dictionary.
        relevant_tables: List of tables deemed relevant by the schema filter.
        few_shot_examples: Optional custom few-shot examples. If None, dynamically selected from FEW_SHOT_POOL.

    Returns:
        The formatted prompt ready to be sent to an LLM.
    """
    # Ensure relevant_tables exist in schema
    all_tables = schema.get("tables", {})
    active_tables = [t for t in relevant_tables if t in all_tables]
    if not active_tables:
        active_tables = list(all_tables.keys())

    # 1. Format Schema Section
    schema_sections = [_format_table_schema(t, all_tables[t]) for t in active_tables]
    schema_block = "\n\n".join(schema_sections)

    # 2. Foreign Keys Section
    fk_relationships = _extract_active_foreign_keys(schema, active_tables)
    if fk_relationships:
        fk_block = "### Foreign Key Relationships:\n" + "\n".join(fk_relationships)
    else:
        fk_block = "### Foreign Key Relationships:\nNone among the filtered tables."

    # 3. Categorical Values Section
    cat_values = _extract_categorical_values(schema, active_tables)
    if cat_values:
        cat_block = "### Valid Categorical Column Values:\n" + "\n".join(cat_values)
    else:
        cat_block = "### Valid Categorical Column Values:\nNone specified."

    # 4. Dynamically Select and Format Few-Shot Examples
    if few_shot_examples is None:
        selected_examples = get_relevant_few_shot_examples(active_tables)
    else:
        # If user passed custom examples, filter if tagged, or use as is
        rel_set = set(active_tables)
        selected_examples = [
            ex for ex in few_shot_examples
            if "tables" not in ex or set(ex["tables"]).issubset(rel_set)
        ]

    if selected_examples:
        examples_formatted = []
        for idx, ex in enumerate(selected_examples, 1):
            examples_formatted.append(f"Example {idx}:\nQuestion: {ex['question']}\nDuckDB SQL:\n```sql\n{ex['sql']}\n```")
        few_shot_block = "\n\n".join(examples_formatted)
    else:
        few_shot_block = "None available for the selected table combination. Adhere strictly to the provided schema."

    # 5. Assemble the Master Prompt
    prompt = f"""You are an expert DuckDB SQL translator. Your task is to generate a single, syntactically correct DuckDB SQL query to answer the user's question based strictly on the provided database schema.

### Database Dialect Guidelines (DuckDB):
1. Use standard DuckDB SQL syntax.
2. For case-insensitive text matching, use `ILIKE` or `LOWER(col) = LOWER('val')`.
3. For date manipulation, use functions like `CURRENT_DATE`, `EXTRACT(YEAR FROM col)`, `INTERVAL 'X days'`.
4. Only reference the tables and columns provided in the schema below. Do NOT invent columns or tables.
5. Use the exact categorical string values provided in the schema.
6. Return only the SQL query enclosed within a markdown code block: ```sql ... ```.

### Relevant Database Schema:
{schema_block}

{fk_block}

{cat_block}

### Few-Shot Examples:
{few_shot_block}

### User Question:
{user_question}

### DuckDB SQL Query:
"""
    return prompt.strip()


if __name__ == "__main__":
    from schema_extractor import extract_schema
    schema = extract_schema()

    sample_question = "What are the names and emails of Gold customers who spent over $500 in total completed orders?"
    sample_relevant_tables = ["customers", "orders"]

    constructed_prompt = build_prompt(sample_question, schema, sample_relevant_tables)

    print("=" * 80)
    print("ASSEMBLED PROMPT PREVIEW (Relevant Tables: customers, orders)")
    print("=" * 80)
    print(constructed_prompt)
    print("=" * 80)
