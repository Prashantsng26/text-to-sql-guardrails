"""
ambiguity_handler.py
Detects business-level and domain ambiguities in natural language queries
and provides concrete schema-aligned SQL interpretations.
"""

import re
from typing import Any, Dict, List, Optional

# Known ambiguous domain definitions tailored to the e-commerce schema
AMBIGUITY_RULES = [
    {
        "term": "revenue",
        "patterns": [r"\brevenue\b", r"\btotal sales\b", r"\bearnings\b"],
        "interpretations": [
            {
                "meaning": "Gross Revenue: Total billed amount across all non-cancelled orders (including shipping).",
                "example_sql": "SELECT SUM(total_amount) AS gross_revenue FROM orders WHERE order_status != 'Cancelled';"
            },
            {
                "meaning": "Net Revenue: Total order amount excluding shipping fees for Completed orders only.",
                "example_sql": "SELECT SUM(total_amount - shipping_fee) AS net_revenue FROM orders WHERE order_status = 'Completed';"
            },
            {
                "meaning": "Product Line Revenue: Subtotal after item-level discounts, excluding shipping fees.",
                "example_sql": "SELECT SUM((unit_price * quantity) - discount_amount) AS line_item_revenue FROM order_items;"
            }
        ]
    },
    {
        "term": "active customer",
        "patterns": [r"\bactive (?:customers?|users?)\b", r"\bactive accounts?\b"],
        "interpretations": [
            {
                "meaning": "Recent Buyers: Customers who have placed at least one completed order in the last 90 days.",
                "example_sql": "SELECT DISTINCT c.* FROM customers c JOIN orders o ON c.customer_id = o.customer_id WHERE o.order_status = 'Completed' AND o.order_date >= CURRENT_DATE - INTERVAL '90 days';"
            },
            {
                "meaning": "Loyalty Tier: Customers with an active premium membership status ('Gold' or 'Platinum').",
                "example_sql": "SELECT * FROM customers WHERE customer_tier IN ('Gold', 'Platinum');"
            }
        ]
    },
    {
        "term": "top selling products",
        "patterns": [r"\btop[- ]selling\b", r"\bbest[- ]sellers?\b", r"\bmost popular products?\b"],
        "interpretations": [
            {
                "meaning": "By Unit Volume: Products with the highest total quantity of items sold.",
                "example_sql": "SELECT p.product_name, SUM(oi.quantity) AS total_units_sold FROM products p JOIN order_items oi ON p.product_id = oi.product_id GROUP BY p.product_id, p.product_name ORDER BY total_units_sold DESC LIMIT 5;"
            },
            {
                "meaning": "By Sales Revenue: Products generating the highest dollar revenue.",
                "example_sql": "SELECT p.product_name, SUM((oi.unit_price * oi.quantity) - oi.discount_amount) AS total_product_revenue FROM products p JOIN order_items oi ON p.product_id = oi.product_id GROUP BY p.product_id, p.product_name ORDER BY total_product_revenue DESC LIMIT 5;"
            }
        ]
    },
    {
        "term": "average order value",
        "patterns": [r"\baverage order value\b", r"\baov\b", r"\bavg order amount\b"],
        "interpretations": [
            {
                "meaning": "Completed AOV: Average total amount for completed orders only.",
                "example_sql": "SELECT AVG(total_amount) AS aov_completed FROM orders WHERE order_status = 'Completed';"
            },
            {
                "meaning": "All Orders AOV: Average total amount across all orders including Pending and Shipped.",
                "example_sql": "SELECT AVG(total_amount) AS aov_all FROM orders WHERE order_status NOT IN ('Cancelled', 'Refunded');"
            }
        ]
    },
    {
        "term": "inactive customers",
        "patterns": [r"\binactive (?:customers?|users?)\b", r"\bdormant (?:customers?|users?)\b", r"\bchurned\b"],
        "interpretations": [
            {
                "meaning": "No Orders in 180 Days: Customers who have not placed any order in the last 180 days.",
                "example_sql": "SELECT c.* FROM customers c WHERE c.customer_id NOT IN (SELECT customer_id FROM orders WHERE order_date >= CURRENT_DATE - INTERVAL '180 days');"
            },
            {
                "meaning": "Zero Lifetime Orders: Customers who have registered but never placed an order.",
                "example_sql": "SELECT c.* FROM customers c LEFT JOIN orders o ON c.customer_id = o.customer_id WHERE o.order_id IS NULL;"
            }
        ]
    }
]


def check_ambiguity(question: str, schema: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
    """
    Checks if a user question contains known ambiguous business terms.

    Args:
        question: The natural language user query.
        schema: Optional schema dictionary for context-dependent disambiguation.

    Returns:
        A structured ambiguity payload if an ambiguous term is matched:
        {
            "ambiguous": True,
            "term": "revenue",
            "matched_phrase": "total sales",
            "interpretations": [
                {"meaning": ..., "example_sql": ...},
                ...
            ]
        }
        Or None if no ambiguity is detected.
    """
    question_lower = question.lower()

    for rule in AMBIGUITY_RULES:
        for pattern in rule["patterns"]:
            match = re.search(pattern, question_lower, flags=re.IGNORECASE)
            if match:
                return {
                    "ambiguous": True,
                    "term": rule["term"],
                    "matched_phrase": match.group(0),
                    "interpretations": rule["interpretations"]
                }

    return None


if __name__ == "__main__":
    test_queries = [
        "What was our total revenue last quarter?",
        "List all active users from Germany.",
        "Show me the top selling products this year.",
        "What are the emails of customers with Gold tier?",
        "Show all open support tickets."
    ]

    for q in test_queries:
        result = check_ambiguity(q)
        print(f"\nQuestion: '{q}'")
        if result:
            print(f"  [AMBIGUITY DETECTED]: Term '{result['term']}' (matched: '{result['matched_phrase']}')")
            for i, interp in enumerate(result["interpretations"], 1):
                print(f"    Option {i}: {interp['meaning']}")
                print(f"      SQL: {interp['example_sql']}")
        else:
            print("  [CLEAR / UNAMBIGUOUS]")
