"""
schema_extractor.py
Introspects a DuckDB database via SQLAlchemy Inspector.
Extracts tables, columns (name, type, nullable), primary keys, foreign key relationships,
and automatically discovers categorical columns along with sample distinct values.
Dumps the structured result to schema.json and returns a structured dictionary / dataclass.
"""

import json
import os
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional
from sqlalchemy import create_engine, inspect, text

DB_PATH = "ecommerce.duckdb"
SCHEMA_JSON_PATH = "schema.json"

TABLE_DESCRIPTIONS = {
    "customers": "Customer profiles, account creation date, geographic country, and loyalty tier.",
    "products": "Product catalog including categories, unit prices, stock inventory, and product active status.",
    "orders": "Customer purchase orders with timestamps, order statuses, payment methods, total amounts, and shipping fees.",
    "order_items": "Line items for orders containing ordered product IDs, quantities, unit prices, and discount amounts.",
    "support_tickets": "Customer support tickets with priority levels, current ticket statuses, and issue categories."
}


@dataclass
class ColumnInfo:
    name: str
    type: str
    nullable: bool
    is_primary_key: bool
    is_categorical: bool = False
    sample_values: List[Any] = field(default_factory=list)


@dataclass
class ForeignKeyInfo:
    constrained_columns: List[str]
    referred_table: str
    referred_columns: List[str]


@dataclass
class TableInfo:
    name: str
    description: str
    row_count: int
    columns: List[ColumnInfo]
    primary_keys: List[str]
    foreign_keys: List[ForeignKeyInfo]


@dataclass
class DatabaseSchema:
    database_type: str
    database_path: str
    tables: Dict[str, TableInfo]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def extract_schema(db_path: str = DB_PATH, output_json: Optional[str] = SCHEMA_JSON_PATH) -> Dict[str, Any]:
    """
    Introspects the DuckDB database via SQLAlchemy Inspector and extracts:
    - Tables
    - Columns (name, type, nullable, primary key)
    - Foreign keys (referencing column -> referenced table & column)
    - Categorical columns (with sample distinct values up to 6)
    """
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Database file '{db_path}' not found. Please run db_setup.py first.")

    engine = create_engine(f"duckdb:///{db_path}")
    inspector = inspect(engine)

    tables_data: Dict[str, TableInfo] = {}

    with engine.connect() as conn:
        table_names = inspector.get_table_names()

        for table_name in table_names:
            # 1. Row count
            row_count_res = conn.execute(text(f'SELECT COUNT(*) FROM "{table_name}"')).scalar()
            row_count = int(row_count_res) if row_count_res is not None else 0

            # 2. Primary Keys via SQLAlchemy Inspector (with DuckDB catalog fallback)
            pk_constraint = inspector.get_pk_constraint(table_name)
            pk_cols = pk_constraint.get("constrained_columns", []) if pk_constraint else []

            if not pk_cols:
                try:
                    pk_query = text(f"""
                        SELECT constraint_column_names
                        FROM duckdb_constraints()
                        WHERE table_name = '{table_name}' AND constraint_type = 'PRIMARY KEY';
                    """)
                    pk_rows = conn.execute(pk_query).fetchall()
                    for pk_row in pk_rows:
                        cols = pk_row[0]
                        if isinstance(cols, (list, tuple)):
                            pk_cols.extend(cols)
                        elif cols:
                            pk_cols.append(str(cols))
                except Exception:
                    pass

            # 3. Foreign Keys via SQLAlchemy Inspector (with DuckDB catalog fallback)
            raw_fks = inspector.get_foreign_keys(table_name)
            fk_list: List[ForeignKeyInfo] = []

            if raw_fks:
                for fk in raw_fks:
                    fk_list.append(ForeignKeyInfo(
                        constrained_columns=fk.get("constrained_columns", []),
                        referred_table=fk.get("referred_table", ""),
                        referred_columns=fk.get("referred_columns", [])
                    ))
            else:
                try:
                    constraint_query = text(f"""
                        SELECT constraint_column_names, referenced_table, referenced_column_names
                        FROM duckdb_constraints()
                        WHERE table_name = '{table_name}' AND constraint_type = 'FOREIGN KEY';
                    """)
                    fk_rows = conn.execute(constraint_query).fetchall()
                    for fk_row in fk_rows:
                        c_cols = list(fk_row[0]) if isinstance(fk_row[0], (list, tuple)) else [str(fk_row[0])]
                        ref_tbl = str(fk_row[1])
                        ref_cols = list(fk_row[2]) if isinstance(fk_row[2], (list, tuple)) else [str(fk_row[2])]
                        fk_list.append(ForeignKeyInfo(
                            constrained_columns=c_cols,
                            referred_table=ref_tbl,
                            referred_columns=ref_cols
                        ))
                except Exception:
                    pass

            # 4. Columns & Categorical Detection
            raw_columns = inspector.get_columns(table_name)
            column_objects: List[ColumnInfo] = []
            non_categorical_types = ("INT", "DECIMAL", "NUMERIC", "FLOAT", "DOUBLE", "DATE", "TIMESTAMP", "TIME")
            numeric_or_id_keywords = ("id", "count", "quantity", "price", "amount", "fee", "total", "subtotal", "rate")

            for col in raw_columns:
                col_name = col["name"]
                col_type = str(col["type"])
                col_nullable = bool(col.get("nullable", True))
                is_pk = col_name in pk_cols

                is_categorical = False
                sample_values: List[Any] = []

                # Categorical heuristic:
                # Target text/varchar/categorical columns with low cardinality (<= 10 distinct values)
                col_type_upper = col_type.upper()
                is_numeric_type = any(t in col_type_upper for t in non_categorical_types)
                is_numeric_name = any(kw in col_name.lower().split("_") for kw in numeric_or_id_keywords)

                if row_count > 0 and not is_pk and not (is_numeric_type and is_numeric_name):
                    try:
                        distinct_res = conn.execute(
                            text(f'SELECT COUNT(DISTINCT "{col_name}") FROM "{table_name}" WHERE "{col_name}" IS NOT NULL')
                        ).scalar()
                        distinct_count = int(distinct_res) if distinct_res is not None else 0

                        if 2 <= distinct_count <= 10:
                            val_rows = conn.execute(
                                text(f'SELECT DISTINCT "{col_name}" FROM "{table_name}" WHERE "{col_name}" IS NOT NULL ORDER BY 1 LIMIT 6')
                            ).fetchall()
                            sample_values = [r[0] for r in val_rows if r[0] is not None]
                            if len(sample_values) >= 2:
                                is_categorical = True
                    except Exception:
                        pass

                column_objects.append(ColumnInfo(
                    name=col_name,
                    type=col_type,
                    nullable=col_nullable,
                    is_primary_key=is_pk,
                    is_categorical=is_categorical,
                    sample_values=sample_values
                ))

            table_desc = TABLE_DESCRIPTIONS.get(table_name, f"Table containing {table_name} data.")
            tables_data[table_name] = TableInfo(
                name=table_name,
                description=table_desc,
                row_count=row_count,
                columns=column_objects,
                primary_keys=pk_cols,
                foreign_keys=fk_list
            )

    schema_obj = DatabaseSchema(
        database_type="duckdb",
        database_path=db_path,
        tables=tables_data
    )

    schema_dict = schema_obj.to_dict()

    if output_json:
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(schema_dict, f, indent=2, default=str)
        print(f"Extracted schema dumped to '{output_json}'")

    return schema_dict


if __name__ == "__main__":
    schema = extract_schema()
    print("\n--- Introspected Schema Summary ---")
    for tbl_name, tbl_data in schema["tables"].items():
        print(f"\nTable: {tbl_name} ({tbl_data['row_count']} rows)")
        print(f"  Description: {tbl_data['description']}")
        print(f"  Primary Keys: {tbl_data['primary_keys']}")
        if tbl_data["foreign_keys"]:
            print("  Foreign Keys:")
            for fk in tbl_data["foreign_keys"]:
                print(f"    {fk['constrained_columns']} -> {fk['referred_table']}.{fk['referred_columns']}")
        print("  Columns:")
        for col in tbl_data["columns"]:
            cat_info = f" [Categorical: {col['sample_values']}]" if col["is_categorical"] else ""
            pk_info = " [PK]" if col["is_primary_key"] else ""
            print(f"    - {col['name']} ({col['type']}){pk_info}{cat_info}")
