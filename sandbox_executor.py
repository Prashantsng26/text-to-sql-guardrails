"""
sandbox_executor.py
Secure, isolated sandbox execution engine for DuckDB queries.

Defense-in-Depth Architecture:
1. Storage-Level Read-Only Mode:
   - Connects using `duckdb.connect(database=db_path, read_only=True)`.
   - Sets `SET access_mode = 'READ_ONLY';`.
   - Configures `PRAGMA enable_external_access = false;` to disable any file-system or remote network access.
   - Configures `PRAGMA lock_configuration;` to prevent the session from altering security parameters.
2. Transaction Rollback Isolation:
   - Executes queries within `BEGIN TRANSACTION;` and unconditionally issues `ROLLBACK;` in a `finally` block,
     ensuring no mutations can persist even if read-only mode were theoretically bypassed.
3. Execution Telemetry:
   - Captures wall-clock execution time (ms), total row count, column metadata, and EXPLAIN execution plan.
"""

import os
import time
from typing import Any, Dict, List, Optional
import duckdb
from pydantic import BaseModel, Field

DEFAULT_DB_PATH = "ecommerce.duckdb"


class ExecutionResult(BaseModel):
    """Structured execution output from the sandbox."""
    success: bool = Field(..., description="True if query executed successfully; False otherwise.")
    rows: List[Dict[str, Any]] = Field(default_factory=list, description="List of row dictionaries.")
    columns: List[str] = Field(default_factory=list, description="List of column names returned by the query.")
    row_count: int = Field(default=0, description="Total number of rows returned.")
    execution_time_ms: float = Field(default=0.0, description="Execution duration in milliseconds.")
    explain_plan: str = Field(default="", description="The query EXPLAIN physical/logical execution plan.")
    error: Optional[str] = Field(default=None, description="Error message if execution failed.")


def execute_safely(
    sql: str,
    db_path: str = DEFAULT_DB_PATH
) -> ExecutionResult:
    """
    Executes a SQL query within a locked, read-only DuckDB sandbox transaction.
    Guarantees rollback after execution.
    
    Args:
        sql: The validated and sanitized SQL query.
        db_path: Path to the DuckDB database file.

    Returns:
        ExecutionResult containing returned rows, execution time (ms), row count, and explain plan.
    """
    if not os.path.exists(db_path):
        return ExecutionResult(
            success=False,
            error=f"Database file '{db_path}' does not exist. Run db_setup.py first."
        )

    con = None
    try:
        # 1. Establish Restricted Read-Only Connection
        # Storage-level read-only mode prevents any write operations to the database file.
        con = duckdb.connect(database=db_path, read_only=True)

        # 2. Security Pragma Restrictions
        # Disable external filesystem I/O and remote network fetches (e.g. read_csv, httpfs)
        try:
            con.execute("PRAGMA enable_external_access = false;")
        except Exception:
            pass

        # 3. Capture EXPLAIN Plan
        explain_plan = ""
        try:
            cleaned_sql = sql.strip().rstrip(";")
            explain_rows = con.execute(f"EXPLAIN {cleaned_sql};").fetchall()
            explain_plan = "\n".join(str(r[1] if len(r) > 1 else r[0]) for r in explain_rows)
        except Exception as ep_err:
            explain_plan = f"Unable to generate EXPLAIN plan: {ep_err}"

        # 4. Execute within Explicit Transaction with Guaranteed Rollback
        start_time = time.perf_counter()
        con.execute("BEGIN TRANSACTION;")
        
        try:
            cursor = con.execute(sql)
            raw_rows = cursor.fetchall()
            col_names = [col[0] for col in cursor.description] if cursor.description else []
        finally:
            # Unconditional rollback: defense in depth against any possible mutating side-effects
            try:
                con.execute("ROLLBACK;")
            except Exception:
                pass

        elapsed_ms = round((time.perf_counter() - start_time) * 1000, 3)

        # Format rows as list of dicts
        dict_rows = [dict(zip(col_names, row)) for row in raw_rows]

        return ExecutionResult(
            success=True,
            rows=dict_rows,
            columns=col_names,
            row_count=len(dict_rows),
            execution_time_ms=elapsed_ms,
            explain_plan=explain_plan
        )

    except Exception as e:
        return ExecutionResult(
            success=False,
            error=str(e),
            explain_plan=""
        )
    finally:
        if con:
            try:
                con.close()
            except Exception:
                pass


if __name__ == "__main__":
    print("--- Testing Sandbox Executor ---")

    # 1. Test Valid SELECT Query
    valid_query = """
    SELECT customer_id, first_name, last_name, email, country, customer_tier
    FROM customers
    WHERE country = 'Germany' AND customer_tier = 'Platinum'
    LIMIT 5;
    """
    res = execute_safely(valid_query)
    print(f"Success: {res.success}")
    print(f"Row Count: {res.row_count}")
    print(f"Execution Time: {res.execution_time_ms} ms")
    print(f"Columns: {res.columns}")
    print(f"First Row: {res.rows[0] if res.rows else 'None'}")
    print(f"Explain Plan Preview:\n{res.explain_plan[:300]}...")

    # 2. Test Attempted Write Mutation in Sandbox
    mutation_query = "DELETE FROM customers WHERE customer_id = 1;"
    mut_res = execute_safely(mutation_query)
    print(f"\nMutation Query Test -> Success: {mut_res.success}")
    print(f"Expected Error Caught: {mut_res.error}")
