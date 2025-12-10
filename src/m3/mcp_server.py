"""
M3 MCP Server - MIMIC-IV + MCP + Models
Provides MCP tools for querying MIMIC-IV data via SQLite or BigQuery.
"""

import os
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Tuple
import time
import requests
import re  

import pandas as pd
import sqlparse
from fastmcp import FastMCP

from m3.auth import init_oauth2, require_oauth2
from m3.config import get_default_database_path

# Create FastMCP server instance
mcp = FastMCP("m3")

# Global variables for backend configuration
_backend = None
_db_path = None
_bq_client = None
_project_id = None

# ==========================================
# PREFLIGHT HELPERS FOR FEEDING-TUBE TASK
# ==========================================

_FEWSHOT_LINE_RE = re.compile(
    r'-\s*note:\s*"(.*?)"\s*;\s*label:\s*"(Yes|No|Unclear)"',
    re.IGNORECASE | re.DOTALL,
)


def _parse_ftube_fewshots(raw: str) -> List[Tuple[str, str]]:
    """Simply parse few-shot example patterns from user text."""
    pairs: List[Tuple[str, str]] = []
    for m in _FEWSHOT_LINE_RE.finditer(raw):
        note = m.group(1).strip()
        label = m.group(2).title().strip()
        if note and label:
            pairs.append((note, label))
    return pairs


def _extract_ftube_context_and_where(user_text: str) -> Tuple[str, str]:
    """
    Extract Context: ... / WHERE: ... blocks from user input if present.
    - Treat the line after prefixes like Context:, 배경:, 정의: as the context
    - Treat the line after prefixes like WHERE:, 조건: as the where_clause hint
    """
    ctx = ""
    where = ""

    # Context / background / definition section
    m_ctx = re.search(
        r"(?:Context|context|Background|background|Definition|definition)\s*:\s*(.+)",
        user_text,
        re.IGNORECASE | re.DOTALL,
    )
    if m_ctx:
        ctx = m_ctx.group(1).strip()

    # WHERE / condition section
    m_where = re.search(
        r"(?:WHERE|where|Condition|condition)\s*:\s*(.+)",
        user_text,
        re.IGNORECASE,
    )
    if m_where:
        where = m_where.group(1).strip()

    return ctx, where


def _validate_limit(limit: int) -> bool:
    """Validate limit parameter to prevent resource exhaustion."""
    return isinstance(limit, int) and 0 < limit <= 1000

def _contains_raw_note_text(sql_query: str) -> bool:
    """
    Detect direct accesses to note text columns to prevent PHI exposure
    within execute_mimic_query.

    Examples of blocked patterns:
      - ' text', 'note_text', '.text'
      - lower(text), substr(text,...)
      - LIKE '%ng tube%', etc.
    """
    q = sql_query.lower()

    # Reading raw text from note tables
    if " from note_" in q or "from note " in q:
        if " text" in q or "note_text" in q or ".text" in q:
            return True
        if "substr(text" in q or "lower(text" in q:
            return True

    text_patterns = [
        " text ",          # SELECT text, ...
        " text,",          # SELECT text, ...
        ".text",           # t.text
        "substr(text", 
        "lower(text",
        " like '%ng tube%'", 
        " like '%peg%'", 
        " like '%tube feed%'", 
        " like '%enteral%'", 
    ]
    for pat in text_patterns:
        if pat in q:
            return True

    return False


def _is_safe_query(sql_query: str, internal_tool: bool = False) -> tuple[bool, str]:
    """Secure SQL validation - blocks injection attacks, allows legitimate queries."""
    try:
        if not sql_query or not sql_query.strip():
            return False, "Empty query"

        # Parse SQL to validate structure
        parsed = sqlparse.parse(sql_query.strip())
        if not parsed:
            return False, "Invalid SQL syntax"

        # Block multiple statements (main injection vector)
        if len(parsed) > 1:
            return False, "Multiple statements not allowed"

        statement = parsed[0]
        statement_type = statement.get_type()

        # Allow SELECT and PRAGMA (PRAGMA is needed for schema exploration)
        if statement_type not in (
            "SELECT",
            "UNKNOWN",
        ):  # PRAGMA shows as UNKNOWN in sqlparse
            return False, "Only SELECT and PRAGMA queries allowed"

        # Check if it's a PRAGMA statement (these are safe for schema exploration)
        sql_upper = sql_query.strip().upper()
        if sql_upper.startswith("PRAGMA"):
            return True, "Safe PRAGMA statement"

        # For SELECT statements, block dangerous injection patterns
        if statement_type == "SELECT":
            # Block dangerous write operations within SELECT
            dangerous_keywords = {
                "INSERT",
                "UPDATE",
                "DELETE",
                "DROP",
                "CREATE",
                "ALTER",
                "TRUNCATE",
                "REPLACE",
                "MERGE",
                "EXEC",
                "EXECUTE",
            }

            for keyword in dangerous_keywords:
                if f" {keyword} " in f" {sql_upper} ":
                    return False, f"Write operation not allowed: {keyword}"

            # Block common injection patterns that are rarely used in legitimate analytics
            injection_patterns = [
                # Classic SQL injection patterns
                ("1=1", "Classic injection pattern"),
                ("OR 1=1", "Boolean injection pattern"),
                ("AND 1=1", "Boolean injection pattern"),
                ("OR '1'='1'", "String injection pattern"),
                ("AND '1'='1'", "String injection pattern"),
                ("WAITFOR", "Time-based injection"),
                ("SLEEP(", "Time-based injection"),
                ("BENCHMARK(", "Time-based injection"),
                ("LOAD_FILE(", "File access injection"),
                ("INTO OUTFILE", "File write injection"),
                ("INTO DUMPFILE", "File write injection"),
            ]

            for pattern, description in injection_patterns:
                if pattern in sql_upper:
                    return False, f"Injection pattern detected: {description}"

            # Context-aware protection: Block suspicious table/column names not in medical databases
            suspicious_names = [
                "PASSWORD",
                "ADMIN",
                "USER",
                "LOGIN",
                "AUTH",
                "TOKEN",
                "CREDENTIAL",
                "SECRET",
                "KEY",
                "HASH",
                "SALT",
                "SESSION",
                "COOKIE",
            ]

            for name in suspicious_names:
                if name in sql_upper:
                    return (
                        False,
                        f"Suspicious identifier detected: {name} (not medical data)",
                    )

        return True, "Safe"

    except Exception as e:
        return False, f"Validation error: {e}"


def _init_backend():
    """Initialize the backend based on environment variables."""
    global _backend, _db_path, _bq_client, _project_id

    # Initialize OAuth2 authentication
    init_oauth2()

    _backend = os.getenv("M3_BACKEND", "sqlite")

    if _backend == "sqlite":
        _db_path = os.getenv("M3_DB_PATH")
        if not _db_path:
            # Use default database path
            _db_path = get_default_database_path("mimic-iv-demo")

        # Ensure the database exists
        if not Path(_db_path).exists():
            raise FileNotFoundError(f"SQLite database not found: {_db_path}")

    elif _backend == "bigquery":
        try:
            from google.cloud import bigquery
        except ImportError:
            raise ImportError(
                "BigQuery dependencies not found. Install with: pip install google-cloud-bigquery"
            )

        # User's GCP project ID for authentication and billing
        # MIMIC-IV data resides in the public 'physionet-data' project
        _project_id = os.getenv("M3_PROJECT_ID", "physionet-data")
        try:
            _bq_client = bigquery.Client(project=_project_id)
        except Exception as e:
            raise RuntimeError(f"Failed to initialize BigQuery client: {e}")

    else:
        raise ValueError(f"Unsupported backend: {_backend}")


# Initialize backend when module is imported
_init_backend()


def _get_backend_info() -> str:
    """Get current backend information for display in responses."""
    if _backend == "sqlite":
        return f"🔧 **Current Backend:** SQLite (local database)\n📁 **Database Path:** {_db_path}\n"
    else:
        return f"🔧 **Current Backend:** BigQuery (cloud database)\n☁️ **Project ID:** {_project_id}\n"


# ==========================================
# INTERNAL QUERY EXECUTION FUNCTIONS
# ==========================================
# These functions perform the actual database operations
# and are called by the MCP tools. This prevents MCP tools
# from calling other MCP tools, which violates the MCP protocol.


def _execute_sqlite_query(sql_query: str) -> str:
    """Execute SQLite query - internal function."""
    try:
        conn = sqlite3.connect(_db_path)
        try:
            df = pd.read_sql_query(sql_query, conn)

            if df.empty:
                return "No results found"

            # Limit output size
            if len(df) > 50:
                result = df.head(50).to_string(index=False)
                result += f"\n... ({len(df)} total rows, showing first 50)"
            else:
                result = df.to_string(index=False)

            return result
        finally:
            conn.close()
    except Exception as e:
        # Re-raise the exception so the calling function can handle it with enhanced guidance
        raise e


def _execute_bigquery_query(sql_query: str) -> str:
    """Execute BigQuery query - internal function."""
    try:
        from google.cloud import bigquery

        job_config = bigquery.QueryJobConfig()
        query_job = _bq_client.query(sql_query, job_config=job_config)
        df = query_job.to_dataframe()

        if df.empty:
            return "No results found"

        # Limit output size
        if len(df) > 50:
            result = df.head(50).to_string(index=False)
            result += f"\n... ({len(df)} total rows, showing first 50)"
        else:
            result = df.to_string(index=False)

        return result

    except Exception as e:
        # Re-raise the exception so the calling function can handle it with enhanced guidance
        raise e


def _execute_query_internal(sql_query: str) -> str:
    """Internal query execution function that handles backend routing."""
    # Security check
    is_safe, message = _is_safe_query(sql_query)
    if not is_safe:
        if "describe" in sql_query.lower() or "show" in sql_query.lower():
            return f"""❌ **Security Error:** {message}

        🔍 **For table structure:** Use `get_table_info('table_name')` instead of DESCRIBE
        📋 **Why this is better:** Shows columns, types, AND sample data to understand the actual data

        💡 **Recommended workflow:**
        1. `get_database_schema()` ← See available tables
        2. `get_table_info('table_name')` ← Explore structure
        3. `execute_mimic_query('SELECT ...')` ← Run your analysis"""

        return f"❌ **Security Error:** {message}\n\n💡 **Tip:** Only SELECT statements are allowed for data analysis."

    try:
        if _backend == "sqlite":
            return _execute_sqlite_query(sql_query)
        else:  # bigquery
            return _execute_bigquery_query(sql_query)
    except Exception as e:
        error_msg = str(e).lower()

        # Provide specific, actionable error guidance
        suggestions = []

        if "no such table" in error_msg or "table not found" in error_msg:
            suggestions.append(
                "🔍 **Table name issue:** Use `get_database_schema()` to see exact table names"
            )
            suggestions.append(
                f"📋 **Backend-specific naming:** {_backend} has specific table naming conventions"
            )
            suggestions.append(
                "💡 **Quick fix:** Check if the table name matches exactly (case-sensitive)"
            )

        if "no such column" in error_msg or "column not found" in error_msg:
            suggestions.append(
                "🔍 **Column name issue:** Use `get_table_info('table_name')` to see available columns"
            )
            suggestions.append(
                "📝 **Common issue:** Column might be named differently (e.g., 'anchor_age' not 'age')"
            )
            suggestions.append(
                "👀 **Check sample data:** `get_table_info()` shows actual column names and sample values"
            )

        if "syntax error" in error_msg:
            suggestions.append(
                "📝 **SQL syntax issue:** Check quotes, commas, and parentheses"
            )
            suggestions.append(
                f"🎯 **Backend syntax:** Verify your SQL works with {_backend}"
            )
            suggestions.append(
                "💭 **Try simpler:** Start with `SELECT * FROM table_name LIMIT 5`"
            )

        if "describe" in error_msg.lower() or "show" in error_msg.lower():
            suggestions.append(
                "🔍 **Schema exploration:** Use `get_table_info('table_name')` instead of DESCRIBE"
            )
            suggestions.append(
                "📋 **Better approach:** `get_table_info()` shows columns AND sample data"
            )

        if not suggestions:
            suggestions.append(
                "🔍 **Start exploration:** Use `get_database_schema()` to see available tables"
            )
            suggestions.append(
                "📋 **Check structure:** Use `get_table_info('table_name')` to understand the data"
            )

        suggestion_text = "\n".join(f"   {s}" for s in suggestions)

        return f"""❌ **Query Failed:** {e}

🛠️ **How to fix this:**
{suggestion_text}

🎯 **Quick Recovery Steps:**
1. `get_database_schema()` ← See what tables exist
2. `get_table_info('your_table')` ← Check exact column names
3. Retry your query with correct names

📚 **Current Backend:** {_backend} - table names and syntax are backend-specific"""

# ==========================================
# INTERNAL HELPERS FOR GEMINI FTUBE TOOL
# ==========================================

def _normalize_ftube_label(text: str) -> str:
    if not text:
        return "Unclear"
    tok = text.split()[0].rstrip(".,").title()
    return tok if tok in {"Yes", "No", "Unclear"} else "Unclear"


def _fetch_notes_for_gemini(
    table_name: str,
    id_column: str,
    text_column: str,
    where_clause: str = "",
    limit: int = 50,
) -> List[Tuple[Any, str]]:
    """
    NOTE:
      - Claude never receives note_text directly.
      - This function is only used inside m3, and its return value is passed
        directly to the Gemini proxy.
    """
    if limit <= 0 or limit > 1000:
        raise ValueError("limit must be between 1 and 1000")

    # Base SELECT query
    sql = f"SELECT {id_column} AS note_id, {text_column} AS note_text FROM {table_name}"
    if where_clause:
        sql += f" WHERE {where_clause}"
    sql += f" LIMIT {int(limit)}"

    # Security validation (SELECT only)
    is_safe, msg = _is_safe_query(sql)
    if not is_safe:
        raise ValueError(f"Unsafe query for Gemini helper: {msg}")

    if _backend == "sqlite":
        conn = sqlite3.connect(_db_path)
        try:
            df = pd.read_sql_query(sql, conn)
        finally:
            conn.close()
    else:  # bigquery
        from google.cloud import bigquery

        job_config = bigquery.QueryJobConfig()
        query_job = _bq_client.query(sql, job_config=job_config)
        df = query_job.to_dataframe()

    # Return (note_id, note_text) list
    results: List[Tuple[Any, str]] = []
    if not df.empty:
        for _, row in df.iterrows():
            results.append((row["note_id"], row["note_text"]))
    return results


# ==========================================
# MCP TOOLS - PUBLIC API
# ==========================================
# These are the tools exposed via MCP protocol.
# They should NEVER call other MCP tools - only internal functions.

@mcp.tool()
@require_oauth2
def plan_feeding_tube_gemini(
    user_text: str,
    table_name: str = "note_discharge",
    id_column: str = "note_id",
    text_column: str = "text",
    limit: int = 20,
) -> Dict[str, Any]:
    """
    🔍 Before running the feeding-tube classification workflow, check whether the user's
    description includes:
    - context (definition/policy)
    - few-shot examples
    - WHERE clause (target note subset)

    If any of these are missing, return questions to request clarification.
    If all elements are present, return a plan containing arguments for
    classify_feeding_tube_gemini_batch.
    """
    # 1) Extract context / few-shot pairs / WHERE candidates
    context_suggest, where_suggest = _extract_ftube_context_and_where(user_text)
    fewshot_pairs = _parse_ftube_fewshots(user_text)

    has_context = bool(context_suggest)
    has_fewshots = len(fewshot_pairs) > 0

    questions: List[Dict[str, Any]] = []

    # Missing context → ask user for policy definition
    if not has_context:
        questions.append(
            {
                "id": "context",
                "type": "textarea",
                "label": (
                    "Please provide the classification policy/definition.\n"
                    "Example: 'Mark Yes only if the patient actually had an active feeding tube "
                    "(NG/PEG/J-tube/etc.) during the hospital stay. "
                    "Mentions of possible tube placement, recommendations, or negations should be marked No.'"
                ),
            }
        )

    # Missing few-shot examples → ask optionally
    if not has_fewshots:
        questions.append(
            {
                "id": "fewshots",
                "type": "textarea",
                "optional": True,
                "placeholder": (
                    'Optional few-shot examples (if you want tighter control):\n'
                    '- note: "example sentence 1"; label: "Yes"\n'
                    '- note: "example sentence 2"; label: "No"'
                ),
            }
        )

    # Missing WHERE clause → ask optionally
    if not where_suggest:
        questions.append(
            {
                "id": "where_clause",
                "type": "text",
                "optional": True,
                "placeholder": (
                    "Specify the WHERE clause if you want to limit which notes to classify. Example: "
                    "\"dischtime >= '2110-01-01'\". Leave empty to use all notes."
                ),
            }
        )

    # If missing information, return questions
    if questions:
        return {
            "needs_more_info": True,
            "questions": questions,
            "detected": {
                "has_context": has_context,
                "has_fewshots": has_fewshots,
                "context_suggest": context_suggest,
                "fewshot_pairs": fewshot_pairs,
                "where_suggest": where_suggest,
            },
        }

    # 2) All required information is present → return plan for batch classification
    plan = {
        "tool": "classify_feeding_tube_gemini_batch",
        "args": {
            "table_name": table_name,
            "id_column": id_column,
            "text_column": text_column,
            "where_clause": where_suggest,
            "limit": limit,
            "context": context_suggest,
            # few-shot pairs are preserved for potential future use
            "fewshot_pairs": fewshot_pairs,
        },
    }

    return {
        "needs_more_info": False,
        "questions": [],
        "detected": {
            "has_context": has_context,
            "has_fewshots": has_fewshots,
            "context_suggest": context_suggest,
            "fewshot_pairs": fewshot_pairs,
            "where_suggest": where_suggest,
        },
        "plan": plan,
    }


@mcp.tool()
@require_oauth2
def get_database_schema() -> str:
    """🔍 Discover what data is available in the MIMIC-IV database.

    **When to use:** Start here when you need to understand what tables exist, or when someone asks about data that might be in multiple tables.

    **What this does:** Shows all available tables so you can identify which ones contain the data you need.

    **Next steps after using this:**
    - If you see relevant tables, use `get_table_info(table_name)` to explore their structure
    - Common tables: `patients` (demographics), `admissions` (hospital stays), `icustays` (ICU data), `labevents` (lab results)

    Returns:
        List of all available tables in the database with current backend info
    """
    if _backend == "sqlite":
        query = "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        result = _execute_query_internal(query)
        return f"{_get_backend_info()}\n📋 **Available Tables:**\n{result}"
    else:  # bigquery
        # Show fully qualified table names that are ready to copy-paste into queries
        query = """
        SELECT CONCAT('`physionet-data.mimiciv_3_1_hosp.', table_name, '`') as query_ready_table_name
        FROM `physionet-data.mimiciv_3_1_hosp.INFORMATION_SCHEMA.TABLES`
        UNION ALL
        SELECT CONCAT('`physionet-data.mimiciv_3_1_icu.', table_name, '`') as query_ready_table_name
        FROM `physionet-data.mimiciv_3_1_icu.INFORMATION_SCHEMA.TABLES`
        ORDER BY query_ready_table_name
        """
        result = _execute_query_internal(query)
        return f"{_get_backend_info()}\n📋 **Available Tables (query-ready names):**\n{result}\n\n💡 **Copy-paste ready:** These table names can be used directly in your SQL queries!"


@mcp.tool()
@require_oauth2
def get_table_info(table_name: str, show_sample: bool = True) -> str:
    """📋 Explore a specific table's structure and see sample data.

    **When to use:** After you know which table you need (from `get_database_schema()`), use this to understand the columns and data format.

    **What this does:**
    - Shows column names, types, and constraints
    - Displays sample rows so you understand the actual data format
    - Helps you write accurate SQL queries

    **Pro tip:** Always look at sample data! It shows you the actual values, date formats, and data patterns.

    Args:
        table_name: Exact table name from the schema (case-sensitive). Can be simple name or fully qualified BigQuery name.
        show_sample: Whether to include sample rows (default: True, recommended)

    Returns:
        Complete table structure with sample data to help you write queries
    """
    backend_info = _get_backend_info()

    if _backend == "sqlite":
        # Get column information
        pragma_query = f"PRAGMA table_info({table_name})"
        try:
            result = _execute_sqlite_query(pragma_query)
            if "error" in result.lower():
                return f"{backend_info}❌ Table '{table_name}' not found. Use get_database_schema() to see available tables."

            info_result = f"{backend_info}📋 **Table:** {table_name}\n\n**Column Information:**\n{result}"

            if show_sample:
                sample_query = f"SELECT * FROM {table_name} LIMIT 3"
                sample_result = _execute_sqlite_query(sample_query)
                info_result += (
                    f"\n\n📊 **Sample Data (first 3 rows):**\n{sample_result}"
                )

            return info_result
        except Exception as e:
            return f"{backend_info}❌ Error examining table '{table_name}': {e}\n\n💡 Use get_database_schema() to see available tables."

    else:  # bigquery
        # Handle both simple names (patients) and fully qualified names (`physionet-data.mimiciv_3_1_hosp.patients`)
        # Detect qualified names by content: dots + physionet pattern
        if "." in table_name and "physionet-data" in table_name:
            # Qualified name (format-agnostic: works with or without backticks)
            clean_name = table_name.strip("`")
            full_table_name = f"`{clean_name}`"
            parts = clean_name.split(".")

            # Validate BigQuery qualified name format: project.dataset.table
            if len(parts) != 3:
                error_msg = (
                    f"{_get_backend_info()}❌ **Invalid qualified table name:** `{table_name}`\n\n"
                    "**Expected format:** `project.dataset.table`\n"
                    "**Example:** `physionet-data.mimiciv_3_1_hosp.diagnoses_icd`\n\n"
                    "**Available MIMIC-IV datasets:**\n"
                    "- `physionet-data.mimiciv_3_1_hosp.*` (hospital module)\n"
                    "- `physionet-data.mimiciv_3_1_icu.*` (ICU module)"
                )
                return error_msg

            simple_table_name = parts[2]  # table name
            dataset = f"{parts[0]}.{parts[1]}"  # project.dataset
        else:
            # Simple name - try both datasets to find the table
            simple_table_name = table_name
            full_table_name = None
            dataset = None

        # If we have a fully qualified name, try that first
        if full_table_name:
            try:
                # Get column information using the dataset from the full name
                dataset_parts = dataset.split(".")
                if len(dataset_parts) >= 2:
                    project_dataset = f"`{dataset_parts[0]}.{dataset_parts[1]}`"
                    info_query = f"""
                    SELECT column_name, data_type, is_nullable
                    FROM {project_dataset}.INFORMATION_SCHEMA.COLUMNS
                    WHERE table_name = '{simple_table_name}'
                    ORDER BY ordinal_position
                    """

                    info_result = _execute_bigquery_query(info_query)
                    if "No results found" not in info_result:
                        result = f"{backend_info}📋 **Table:** {full_table_name}\n\n**Column Information:**\n{info_result}"

                        if show_sample:
                            sample_query = f"SELECT * FROM {full_table_name} LIMIT 3"
                            sample_result = _execute_bigquery_query(sample_query)
                            result += f"\n\n📊 **Sample Data (first 3 rows):**\n{sample_result}"

                        return result
            except Exception:
                pass  # Fall through to try simple name approach

        # Try both datasets with simple name (fallback or original approach)
        for dataset in ["mimiciv_3_1_hosp", "mimiciv_3_1_icu"]:
            try:
                full_table_name = f"`physionet-data.{dataset}.{simple_table_name}`"

                # Get column information
                info_query = f"""
                SELECT column_name, data_type, is_nullable
                FROM `physionet-data.{dataset}.INFORMATION_SCHEMA.COLUMNS`
                WHERE table_name = '{simple_table_name}'
                ORDER BY ordinal_position
                """

                info_result = _execute_bigquery_query(info_query)
                if "No results found" not in info_result:
                    result = f"{backend_info}📋 **Table:** {full_table_name}\n\n**Column Information:**\n{info_result}"

                    if show_sample:
                        sample_query = f"SELECT * FROM {full_table_name} LIMIT 3"
                        sample_result = _execute_bigquery_query(sample_query)
                        result += (
                            f"\n\n📊 **Sample Data (first 3 rows):**\n{sample_result}"
                        )

                    return result
            except Exception:
                continue

        return f"{backend_info}❌ Table '{table_name}' not found in any dataset. Use get_database_schema() to see available tables."


@mcp.tool()
@require_oauth2
def execute_mimic_query(sql_query: str) -> str:
    """🚀 Execute SQL queries to analyze MIMIC-IV data.

    **💡 Pro tip:** For best results, explore the database structure first!

    **Recommended workflow (especially for smaller models):**
    1. **See available tables:** Use `get_database_schema()` to list all tables
    2. **Examine table structure:** Use `get_table_info('table_name')` to see columns and sample data
    3. **Write your SQL query:** Use exact table/column names from exploration

    **Why exploration helps:**
    - Table names vary between backends (SQLite vs BigQuery)
    - Column names may be unexpected (e.g., age might be 'anchor_age')
    - Sample data shows actual formats and constraints

    Args:
        sql_query: Your SQL SELECT query (must be SELECT only)

    Returns:
        Query results or helpful error messages with next steps
    """
    # 🔒 Extra protection: prevent local note text exposure
    if os.getenv("M3_BLOCK_NOTE_TEXT", "1") == "1" and _contains_raw_note_text(sql_query):
        return (
            "❌ **Security Policy:** Raw note text columns are blocked in this environment.\n\n"
            "In this MCP environment, raw discharge notes and similar free-text fields cannot be directly exposed to the assistant.\n"
            "- Columns like `text`, `note_text`, or expressions such as `LOWER(text)` and `SUBSTR(text, ...)` cannot be queried via `execute_mimic_query`.\n"
            "- Instead, please use a workflow such as:\n"
            "  1. Run `classify_feeding_tube_gemini_batch` to classify note_ids using Gemini.\n"
            "  2. Bring back only labels/counts/ID lists (Yes/No/Unclear) into this environment.\n"
            "  3. If needed, inspect the raw text only in a secure local environment (Python, SQL) without sending it to the model.\n\n"
            "💡 If you really need to query text safely, you can create a 'safe view' (without PHI) and query that instead."
        )

    return _execute_query_internal(sql_query)


@mcp.tool()
@require_oauth2
def get_icu_stays(patient_id: int | None = None, limit: int = 10) -> str:
    """🏥 Get ICU stay information and length of stay data.

    **⚠️ Note:** This is a convenience function that assumes standard MIMIC-IV table structure.
    **For reliable queries:** Use `get_database_schema()` → `get_table_info()` → `execute_mimic_query()` workflow.

    **What you'll get:** Patient IDs, admission times, length of stay, and ICU details.

    Args:
        patient_id: Specific patient ID to query (optional)
        limit: Maximum number of records to return (default: 10)

    Returns:
        ICU stay data as formatted text or guidance if table not found
    """
    # Security validation
    if not _validate_limit(limit):
        return "Error: Invalid limit. Must be a positive integer between 1 and 10000."

    # Try common ICU table names based on backend
    if _backend == "sqlite":
        icustays_table = "icu_icustays"
    else:  # bigquery
        icustays_table = "`physionet-data.mimiciv_3_1_icu.icustays`"

    if patient_id:
        query = f"SELECT * FROM {icustays_table} WHERE subject_id = {patient_id}"
    else:
        query = f"SELECT * FROM {icustays_table} LIMIT {limit}"

    # Execute with error handling that suggests proper workflow
    result = _execute_query_internal(query)
    if "error" in result.lower() or "not found" in result.lower():
        return f"""❌ **Convenience function failed:** {result}

💡 **For reliable results, use the proper workflow:**
1. `get_database_schema()` ← See actual table names
2. `get_table_info('table_name')` ← Understand structure
3. `execute_mimic_query('your_sql')` ← Use exact names

This ensures compatibility across different MIMIC-IV setups."""

    return result


@mcp.tool()
@require_oauth2
def get_lab_results(
    patient_id: int | None = None, lab_item: str | None = None, limit: int = 20
) -> str:
    """🧪 Get laboratory test results quickly.

    **⚠️ Note:** This is a convenience function that assumes standard MIMIC-IV table structure.
    **For reliable queries:** Use `get_database_schema()` → `get_table_info()` → `execute_mimic_query()` workflow.

    **What you'll get:** Lab values, timestamps, patient IDs, and test details.

    Args:
        patient_id: Specific patient ID to query (optional)
        lab_item: Lab item to search for in the value field (optional)
        limit: Maximum number of records to return (default: 20)

    Returns:
        Lab results as formatted text or guidance if table not found
    """
    # Security validation
    if not _validate_limit(limit):
        return "Error: Invalid limit. Must be a positive integer between 1 and 10000."

    # Try common lab table names based on backend
    if _backend == "sqlite":
        labevents_table = "hosp_labevents"
    else:  # bigquery
        labevents_table = "`physionet-data.mimiciv_3_1_hosp.labevents`"

    # Build query conditions
    conditions = []
    if patient_id:
        conditions.append(f"subject_id = {patient_id}")
    if lab_item:
        # Escape single quotes for safety in LIKE clause
        escaped_lab_item = lab_item.replace("'", "''")
        conditions.append(f"value LIKE '%{escaped_lab_item}%'")

    base_query = f"SELECT * FROM {labevents_table}"
    if conditions:
        base_query += " WHERE " + " AND ".join(conditions)
    base_query += f" LIMIT {limit}"

    # Execute with error handling that suggests proper workflow
    result = _execute_query_internal(base_query)
    if "error" in result.lower() or "not found" in result.lower():
        return f"""❌ **Convenience function failed:** {result}

💡 **For reliable results, use the proper workflow:**
1. `get_database_schema()` ← See actual table names
2. `get_table_info('table_name')` ← Understand structure
3. `execute_mimic_query('your_sql')` ← Use exact names

This ensures compatibility across different MIMIC-IV setups."""

    return result


@mcp.tool()
@require_oauth2
def get_race_distribution(limit: int = 10) -> str:
    """📊 Get race distribution from hospital admissions.

    **⚠️ Note:** This is a convenience function that assumes standard MIMIC-IV table structure.
    **For reliable queries:** Use `get_database_schema()` → `get_table_info()` → `execute_mimic_query()` workflow.

    **What you'll get:** Count of patients by race category, ordered by frequency.

    Args:
        limit: Maximum number of race categories to return (default: 10)

    Returns:
        Race distribution as formatted text or guidance if table not found
    """
    # Security validation
    if not _validate_limit(limit):
        return "Error: Invalid limit. Must be a positive integer between 1 and 10000."

    # Try common admissions table names based on backend
    if _backend == "sqlite":
        admissions_table = "hosp_admissions"
    else:  # bigquery
        admissions_table = "`physionet-data.mimiciv_3_1_hosp.admissions`"

    query = f"SELECT race, COUNT(*) as count FROM {admissions_table} GROUP BY race ORDER BY count DESC LIMIT {limit}"

    # Execute with error handling that suggests proper workflow
    result = _execute_query_internal(query)
    if "error" in result.lower() or "not found" in result.lower():
        return f"""❌ **Convenience function failed:** {result}

💡 **For reliable results, use the proper workflow:**
1. `get_database_schema()` ← See actual table names
2. `get_table_info('table_name')` ← Understand structure
3. `execute_mimic_query('your_sql')` ← Use exact names

This ensures compatibility across different MIMIC-IV setups."""

    return result

# ==========================================
# GEMINI FEEDING-TUBE CLASSIFIER TOOL
# ==========================================

def _normalize_ftube_label(text: str) -> str:
    if not text:
        return "Unclear"
    tok = text.split()[0].rstrip(".,").title()
    return tok if tok in {"Yes", "No", "Unclear"} else "Unclear"


def _call_gemini_proxy_ftube(
    snippet: str,
    context: str = "",
    fewshots: List[Tuple[str, str]] | None = None,
    temperature: float = 0.0,
) -> Dict[str, Any]:
    """
    Call the FastAPI server running on a Vertex AI JupyterLab VM
    (e.g., http://<VM_IP>:8081/ftube/classify) to receive
    feeding tube classification (Yes/No/Unclear).
    """
    url = os.getenv("GEMINI_PROXY_URL")
    if not url:
        raise RuntimeError(
            "Environment variable GEMINI_PROXY_URL is not set. "
            "Example: GEMINI_PROXY_URL='http://35.224.111.104:8081/ftube/classify'"
        )

    fs_payload = [{"note": n, "label": l} for (n, l) in (fewshots or [])] if fewshots else None

    payload: Dict[str, Any] = {
        "snippet": snippet,
        "context": context or "",
        "fewshots": fs_payload,
        "temperature": float(temperature),
    }

    t0 = time.time()
    resp = requests.post(url, json=payload, timeout=60)
    latency = time.time() - t0
    resp.raise_for_status()

    data = resp.json()
    prediction = data.get("prediction")
    raw_output = data.get("raw_output", "")
    latency_sec = data.get("latency_sec", latency)

    if not prediction:
        prediction = _normalize_ftube_label(raw_output)

    return {
        "prediction": prediction,
        "raw_output": raw_output,
        "latency_sec": float(latency_sec),
    }


@mcp.tool()
@require_oauth2
def classify_feeding_tube_gemini_by_id(
    note_id: int,
    table_name: str,
    id_column: str = "note_id",
    text_column: str = "note_text",
    context: str = "",
    temperature: float = 0.0,
) -> Dict[str, Any]:
    """
    NOTE:
      - The assistant only sees note_id and the label.
      - The note text flows only from DB → Gemini proxy inside the m3 server.
        It is never exposed directly to the assistant.

    Args:
        note_id: ID of the note to classify (integer)
        table_name: Name of the note table (e.g., 'note_discharge')
        id_column: Name of the note_id column (default: 'note_id')
        text_column: Name of the text column (default: 'note_text')
        context: Classification policy/definition text (optional)
        temperature: Gemini temperature (default 0.0)

    Returns:
        {
          "note_id": <int>,
          "prediction": "Yes" | "No" | "Unclear",
          "latency_sec": <float>,
          "backend": "gemini_proxy"
        }
    """
    where_clause = f"{id_column} = {int(note_id)}"
    rows = _fetch_notes_for_gemini(
        table_name=table_name,
        id_column=id_column,
        text_column=text_column,
        where_clause=where_clause,
        limit=1,
    )
    if not rows:
        return {
            "note_id": note_id,
            "prediction": "Unclear",
            "latency_sec": 0.0,
            "backend": "gemini_proxy",
            "message": "No note found for the given ID",
        }

    _, note_text = rows[0]
    result = _call_gemini_proxy_ftube(
        snippet=note_text,
        context=context,
        fewshots=None,
        temperature=temperature,
    )
    return {
        "note_id": note_id,
        "prediction": result["prediction"],
        "latency_sec": result["latency_sec"],
        "backend": "gemini_proxy",
    }

@mcp.tool()
@require_oauth2
def classify_feeding_tube_gemini_batch(
    table_name: str,
    id_column: str = "note_id",
    text_column: str = "note_text",
    where_clause: str = "",
    limit: int = 20,
    context: str = "",
    temperature: float = 0.0,
) -> Dict[str, Any]:
    """
    Classify multiple notes in a batch (note text is never exposed to the assistant).

    Args:
        table_name: Note table name (e.g., 'note_discharge')
        id_column: ID column name (default 'note_id')
        text_column: Text column name (default 'note_text')
        where_clause: SQL WHERE clause (e.g., "dischtime >= '2110-01-01'")
        limit: Maximum number of notes (default 20)
        context: Classification policy/definition text
        temperature: Gemini temperature (default 0.0)

    Returns:
        {
          "backend": "gemini_proxy",
          "n_notes": <int>,
          "counts": {"Yes": int, "No": int, "Unclear": int},
          "predictions": [
            {"note_id": ..., "prediction": "Yes" | "No" | "Unclear"},
            ...
          ]
        }
    """
    rows = _fetch_notes_for_gemini(
        table_name=table_name,
        id_column=id_column,
        text_column=text_column,
        where_clause=where_clause,
        limit=limit,
    )
    if not rows:
        return {
            "backend": "gemini_proxy",
            "n_notes": 0,
            "counts": {"Yes": 0, "No": 0, "Unclear": 0},
            "predictions": [],
            "message": "No notes matched the given condition",
        }

    predictions: List[Dict[str, Any]] = []
    counts = {"Yes": 0, "No": 0, "Unclear": 0}

    for nid, note_text in rows:
        result = _call_gemini_proxy_ftube(
            snippet=note_text,
            context=context,
            fewshots=None,
            temperature=temperature,
        )
        label = result["prediction"]
        if label not in counts:
            label = "Unclear"
        counts[label] += 1
        predictions.append({"note_id": nid, "prediction": label})

    return {
        "backend": "gemini_proxy",
        "n_notes": len(predictions),
        "counts": counts,
        "predictions": predictions,
    }


def main():
    """Main entry point for MCP server."""
    # Run the FastMCP server
    mcp.run()



if __name__ == "__main__":
    main()
