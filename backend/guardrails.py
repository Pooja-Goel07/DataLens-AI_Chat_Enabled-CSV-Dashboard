# guardrails.py - Safety checks for LLM-generated SQL before it is executed.
#
# The chat pipeline lets an LLM write SQL. Without guardrails that SQL runs as-is, which is
# both a security hole (destructive or cross-user queries) and a reliability risk (unbounded
# result sets, runaway queries). Every generated query must pass `guard()` before execution.
import re

# Row cap applied when the model doesn't specify its own LIMIT.
DEFAULT_ROW_LIMIT = 1000
# Server-side per-query timeout (MySQL optimizer hint), in milliseconds.
QUERY_TIMEOUT_MS = 8000

# Write / DDL keywords that must never appear in a query we execute.
# NOTE: REPLACE, TRUNCATE, and INSERT are intentionally NOT listed — they are also legitimate
# SQL *functions* (e.g. REPLACE(str,...), TRUNCATE(num,d), INSERT(str,...)) that appear in
# ordinary SELECTs (the prompt asks the model to strip currency symbols with REPLACE()).
# Their dangerous *statement* forms (REPLACE INTO / TRUNCATE TABLE / INSERT INTO) start the
# statement and are already rejected by the "must start with SELECT/WITH" check below.
_DANGEROUS = re.compile(
    r'\b(DROP|DELETE|UPDATE|ALTER|CREATE|GRANT|REVOKE|'
    r'RENAME|LOCK|LOAD|HANDLER|INTO|OUTFILE|DUMPFILE)\b',
    re.IGNORECASE,
)
# Table references: the identifier following FROM or JOIN (optionally backtick-quoted).
_TABLE_REF = re.compile(r'\b(?:FROM|JOIN)\s+`?([a-zA-Z0-9_]+)`?', re.IGNORECASE)
# CTE names defined via `WITH name AS (` / `, name AS (` — these are valid internal references.
_CTE_DEF = re.compile(r'([a-zA-Z0-9_]+)\s+AS\s*\(', re.IGNORECASE)


class SQLGuardrailError(Exception):
    """Raised when a query fails a safety check and must not be executed."""


def _strip(sql: str) -> str:
    return sql.strip().rstrip(';').strip()


def enforce_read_only(sql: str) -> str:
    """Allow only a single read-only SELECT/WITH statement. Returns the cleaned SQL."""
    s = _strip(sql)
    if not s:
        raise SQLGuardrailError("Empty query.")
    # No statement chaining — a lone SELECT can't smuggle a second statement.
    if ';' in s:
        raise SQLGuardrailError("Multiple SQL statements are not allowed.")
    head = s.lstrip('(').strip().lower()
    if not (head.startswith('select') or head.startswith('with')):
        raise SQLGuardrailError("Only read-only SELECT queries are allowed.")
    if _DANGEROUS.search(s):
        raise SQLGuardrailError("Query contains a disallowed (write/DDL) operation.")
    return s


def referenced_tables(sql: str) -> set:
    """Base tables referenced after FROM/JOIN (excludes CTE names and derived subqueries)."""
    cte_names = {m.lower() for m in _CTE_DEF.findall(sql)}
    return {t.lower() for t in _TABLE_REF.findall(sql)} - cte_names


def enforce_table_allowlist(sql: str, allowed_tables) -> None:
    """Ensure every referenced base table belongs to the current user."""
    allowed = {t.lower() for t in allowed_tables}
    unauthorized = [t for t in referenced_tables(sql) if t not in allowed]
    if unauthorized:
        raise SQLGuardrailError(
            f"Query references tables you don't have access to: {', '.join(sorted(unauthorized))}"
        )


def inject_limit(sql: str, limit: int = DEFAULT_ROW_LIMIT) -> str:
    """Append a row LIMIT if the query doesn't already have one."""
    if re.search(r'\bLIMIT\b', sql, re.IGNORECASE):
        return sql
    return f"{sql} LIMIT {limit}"


def inject_timeout_hint(sql: str, ms: int = QUERY_TIMEOUT_MS) -> str:
    """Add a MySQL MAX_EXECUTION_TIME hint to plain SELECT queries."""
    if re.match(r'(?i)^\s*SELECT\b', sql):
        return re.sub(r'(?i)^(\s*SELECT)\b',
                      rf'\1 /*+ MAX_EXECUTION_TIME({ms}) */', sql, count=1)
    return sql  # CTE (WITH ...) queries: skip the hint rather than place it wrongly


def guard(sql: str, allowed_tables) -> str:
    """Full pipeline: validate read-only + ownership, then bound rows and time.

    Returns the safe SQL to execute. Raises SQLGuardrailError if the query is not allowed.
    """
    sql = enforce_read_only(sql)
    enforce_table_allowlist(sql, allowed_tables)
    sql = inject_limit(sql)
    sql = inject_timeout_hint(sql)
    return sql
