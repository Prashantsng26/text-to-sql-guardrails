"""
db_setup.py
Initializes and seeds a realistic file-based DuckDB database with 5 interconnected tables,
explicit Primary Key & Foreign Key constraints, and categorical columns.
"""

import os
import random
from datetime import datetime, timedelta
import duckdb

DB_PATH = "ecommerce.duckdb"

def seed_database(db_path: str = DB_PATH, num_customers: int = 60, num_orders: int = 100, num_products: int = 50, num_tickets: int = 50) -> None:
    """Create tables with foreign keys and seed realistic synthetic data."""
    if os.path.exists(db_path):
        os.remove(db_path)
        print(f"Removed existing database at {db_path}")

    conn = duckdb.connect(db_path)
    random.seed(42)

    # 1. Create Schema DDL with explicit PK and FK constraints
    print("Creating database schema...")
    conn.execute("""
    CREATE TABLE customers (
        customer_id INTEGER PRIMARY KEY,
        first_name VARCHAR NOT NULL,
        last_name VARCHAR NOT NULL,
        email VARCHAR NOT NULL,
        country VARCHAR NOT NULL,
        customer_tier VARCHAR NOT NULL,
        created_at TIMESTAMP NOT NULL
    );
    """)

    conn.execute("""
    CREATE TABLE products (
        product_id INTEGER PRIMARY KEY,
        product_name VARCHAR NOT NULL,
        category VARCHAR NOT NULL,
        unit_price DECIMAL(10, 2) NOT NULL,
        inventory_count INTEGER NOT NULL,
        status VARCHAR NOT NULL
    );
    """)

    conn.execute("""
    CREATE TABLE orders (
        order_id INTEGER PRIMARY KEY,
        customer_id INTEGER NOT NULL REFERENCES customers(customer_id),
        order_date TIMESTAMP NOT NULL,
        order_status VARCHAR NOT NULL,
        payment_method VARCHAR NOT NULL,
        total_amount DECIMAL(10, 2) NOT NULL,
        shipping_fee DECIMAL(10, 2) NOT NULL
    );
    """)

    conn.execute("""
    CREATE TABLE order_items (
        item_id INTEGER PRIMARY KEY,
        order_id INTEGER NOT NULL REFERENCES orders(order_id),
        product_id INTEGER NOT NULL REFERENCES products(product_id),
        quantity INTEGER NOT NULL,
        unit_price DECIMAL(10, 2) NOT NULL,
        discount_amount DECIMAL(10, 2) NOT NULL
    );
    """)

    conn.execute("""
    CREATE TABLE support_tickets (
        ticket_id INTEGER PRIMARY KEY,
        customer_id INTEGER NOT NULL REFERENCES customers(customer_id),
        priority VARCHAR NOT NULL,
        ticket_status VARCHAR NOT NULL,
        issue_category VARCHAR NOT NULL,
        created_at TIMESTAMP NOT NULL
    );
    """)

    # 2. Synthetic Data Generators
    first_names = [
        "James", "Mary", "Robert", "Patricia", "John", "Jennifer", "Michael", "Linda",
        "David", "Elizabeth", "William", "Barbara", "Richard", "Susan", "Joseph", "Jessica",
        "Thomas", "Sarah", "Charles", "Karen", "Christopher", "Nancy", "Daniel", "Lisa",
        "Matthew", "Betty", "Anthony", "Margaret", "Mark", "Sandra"
    ]
    last_names = [
        "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis",
        "Rodriguez", "Martinez", "Hernandez", "Lopez", "Gonzalez", "Wilson", "Anderson",
        "Thomas", "Taylor", "Moore", "Jackson", "Martin", "Lee", "Perez", "Thompson", "White"
    ]
    countries = ["USA", "UK", "Germany", "Canada", "France", "India"]
    customer_tiers = ["Bronze", "Silver", "Gold", "Platinum"]

    categories = ["Electronics", "Footwear", "Apparel", "Home & Kitchen", "Books", "Sports"]
    product_prefixes = {
        "Electronics": ["Wireless Noise-Cancelling", "Smart 4K", "Ultra-Slim", "Pro Gaming", "Bluetooth 5.3"],
        "Footwear": ["Running Pro", "Classic Leather", "Trail Hiking", "Slip-on Comfort", "Air Mesh"],
        "Apparel": ["Organic Cotton", "Waterproof Tech", "Merino Wool", "Casual Denim", "Thermal Fleece"],
        "Home & Kitchen": ["Stainless Steel", "Smart Barista", "Ergonomic Ceramic", "Digital Air", "Bamboo Fiber"],
        "Books": ["Mastering Python", "Data Engineering at Scale", "Designing ML Systems", "SQL Deep Dive", "Cloud Architecture Guide"],
        "Sports": ["Carbon Fiber", "All-Weather Grip", "Hydro Flask Pro", "Adjustable Dumbbell", "Resistance Heavy"]
    }
    product_nouns = {
        "Electronics": ["Headphones", "Monitor", "Power Bank", "Mechanical Keyboard", "Smartwatch"],
        "Footwear": ["Sneakers", "Boots", "Loafers", "Sandals", "Trainers"],
        "Apparel": ["T-Shirt", "Jacket", "Sweater", "Jeans", "Hoodie"],
        "Home & Kitchen": ["Chef Knife", "Espresso Machine", "Pan Set", "Fryer", "Cutting Board"],
        "Books": ["Handbook", "Principles", "Cookbook", "Volume 1", "Case Studies"],
        "Sports": ["Racket", "Yoga Mat", "Bottle", "Weights", "Band"]
    }
    product_statuses = ["Active", "Discontinued", "Out of Stock", "Backorder"]

    order_statuses = ["Completed", "Pending", "Shipped", "Cancelled", "Refunded"]
    payment_methods = ["Credit Card", "PayPal", "Debit Card", "Bank Transfer", "Gift Card"]

    ticket_priorities = ["Low", "Medium", "High", "Urgent"]
    ticket_statuses = ["Open", "In Progress", "Resolved", "Closed"]
    ticket_categories = ["Billing", "Shipping", "Product Defect", "Return Request", "Account Access"]

    base_date = datetime(2025, 1, 1, 10, 0, 0)

    # 3. Seed Customers
    print(f"Seeding {num_customers} customers...")
    customers_data = []
    for cid in range(1, num_customers + 1):
        fn = random.choice(first_names)
        ln = random.choice(last_names)
        email = f"{fn.lower()}.{ln.lower()}{cid}@example.com"
        country = random.choice(countries)
        tier = random.choices(customer_tiers, weights=[40, 30, 20, 10])[0]
        created_at = base_date + timedelta(days=random.randint(0, 300), hours=random.randint(0, 23))
        customers_data.append((cid, fn, ln, email, country, tier, created_at))

    conn.executemany(
        "INSERT INTO customers VALUES (?, ?, ?, ?, ?, ?, ?)",
        customers_data
    )

    # 4. Seed Products
    print(f"Seeding {num_products} products...")
    products_data = []
    product_prices = {}
    for pid in range(1, num_products + 1):
        cat = random.choice(categories)
        p_prefix = random.choice(product_prefixes[cat])
        p_noun = random.choice(product_nouns[cat])
        name = f"{p_prefix} {p_noun} #{pid}"
        if cat == "Electronics":
            price = round(random.uniform(79.99, 899.99), 2)
        elif cat == "Books":
            price = round(random.uniform(19.99, 69.99), 2)
        elif cat == "Home & Kitchen":
            price = round(random.uniform(29.99, 349.99), 2)
        else:
            price = round(random.uniform(25.00, 199.99), 2)
        product_prices[pid] = price
        inventory = random.randint(0, 250)
        status = random.choices(product_statuses, weights=[70, 10, 10, 10])[0]
        products_data.append((pid, name, cat, price, inventory, status))

    conn.executemany(
        "INSERT INTO products VALUES (?, ?, ?, ?, ?, ?)",
        products_data
    )

    # 5. Seed Orders & Order Items
    print(f"Seeding {num_orders} orders and related order items...")
    orders_data = []
    order_items_data = []
    item_id_counter = 1

    for oid in range(1, num_orders + 1):
        cid = random.randint(1, num_customers)
        order_date = base_date + timedelta(days=random.randint(50, 420), hours=random.randint(0, 23))
        order_status = random.choices(order_statuses, weights=[60, 10, 15, 10, 5])[0]
        payment_method = random.choice(payment_methods)
        shipping_fee = 0.00 if random.random() < 0.3 else round(random.uniform(4.99, 14.99), 2)

        # Generate 1 to 4 items per order
        num_items = random.randint(1, 4)
        selected_pids = random.sample(range(1, num_products + 1), num_items)
        order_subtotal = 0.0

        for pid in selected_pids:
            qty = random.randint(1, 3)
            unit_price = product_prices[pid]
            discount = round(unit_price * qty * 0.1, 2) if random.random() < 0.25 else 0.0
            order_subtotal += (unit_price * qty) - discount
            order_items_data.append((item_id_counter, oid, pid, qty, unit_price, discount))
            item_id_counter += 1

        total_amount = round(order_subtotal + shipping_fee, 2)
        orders_data.append((oid, cid, order_date, order_status, payment_method, total_amount, shipping_fee))

    conn.executemany(
        "INSERT INTO orders VALUES (?, ?, ?, ?, ?, ?, ?)",
        orders_data
    )

    conn.executemany(
        "INSERT INTO order_items VALUES (?, ?, ?, ?, ?, ?)",
        order_items_data
    )

    # 6. Seed Support Tickets
    print(f"Seeding {num_tickets} support tickets...")
    tickets_data = []
    for tid in range(1, num_tickets + 1):
        cid = random.randint(1, num_customers)
        priority = random.choices(ticket_priorities, weights=[30, 40, 20, 10])[0]
        status = random.choices(ticket_statuses, weights=[25, 25, 35, 15])[0]
        category = random.choice(ticket_categories)
        created_at = base_date + timedelta(days=random.randint(60, 430), hours=random.randint(0, 23))
        tickets_data.append((tid, cid, priority, status, category, created_at))

    conn.executemany(
        "INSERT INTO support_tickets VALUES (?, ?, ?, ?, ?, ?)",
        tickets_data
    )

    conn.close()
    print(f"Successfully created and seeded database '{db_path}' with:")
    print(f"  - customers: {len(customers_data)} rows")
    print(f"  - products: {len(products_data)} rows")
    print(f"  - orders: {len(orders_data)} rows")
    print(f"  - order_items: {len(order_items_data)} rows")
    print(f"  - support_tickets: {len(tickets_data)} rows")

if __name__ == "__main__":
    seed_database()
