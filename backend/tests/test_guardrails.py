"""Unit tests for the SQL guardrails (no DB or LLM required).

These cover the security-critical layer: only read-only, single-statement, own-table queries
may run, and safe queries are bounded with a LIMIT and a timeout hint.
"""
import pytest

from guardrails import guard, enforce_read_only, referenced_tables, SQLGuardrailError

ALLOWED = ["alice_sales", "alice_orders"]


def _guard(sql):
    return guard(sql, ALLOWED)


# --- Blocked: writes / DDL -------------------------------------------------
@pytest.mark.parametrize("sql", [
    "DROP TABLE alice_sales",
    "DELETE FROM alice_sales",
    "UPDATE alice_sales SET price = 0",
    "INSERT INTO alice_sales VALUES (1)",
    "TRUNCATE TABLE alice_sales",
    "REPLACE INTO alice_sales VALUES (1)",
    "ALTER TABLE alice_sales ADD COLUMN x INT",
    "CREATE TABLE t (id INT)",
])
def test_write_statements_are_blocked(sql):
    with pytest.raises(SQLGuardrailError):
        _guard(sql)


# --- Blocked: injection / exfiltration / scope -----------------------------
def test_stacked_statements_blocked():
    with pytest.raises(SQLGuardrailError):
        _guard("SELECT * FROM alice_sales; DROP TABLE alice_sales")


def test_into_outfile_blocked():
    with pytest.raises(SQLGuardrailError):
        _guard("SELECT * FROM alice_sales INTO OUTFILE '/tmp/x'")


def test_other_users_table_blocked():
    with pytest.raises(SQLGuardrailError):
        _guard("SELECT * FROM bob_secret")


def test_system_table_blocked():
    with pytest.raises(SQLGuardrailError):
        _guard("SELECT * FROM users")


# --- Allowed: read-only queries are bounded --------------------------------
def test_plain_select_gets_limit_and_timeout():
    out = _guard("SELECT * FROM alice_sales")
    assert "MAX_EXECUTION_TIME" in out
    assert "LIMIT 1000" in out


def test_existing_limit_not_doubled():
    out = _guard("SELECT price FROM alice_sales LIMIT 5")
    assert out.lower().count("limit") == 1


def test_join_across_own_tables_allowed():
    out = _guard("SELECT a.id FROM alice_sales a JOIN alice_orders o ON a.id = o.id")
    assert "alice_orders" in out


def test_cte_referencing_own_table_allowed():
    out = _guard("WITH t AS (SELECT * FROM alice_sales) SELECT COUNT(*) FROM t")
    assert out.startswith("WITH")


# --- Regression: legit SQL *functions* must not be mistaken for statements --
@pytest.mark.parametrize("sql", [
    "SELECT AVG(REPLACE(REPLACE(price,'$',''),',','')) FROM alice_sales",
    "SELECT TRUNCATE(AVG(price), 2) FROM alice_sales",
    "SELECT INSERT(product, 1, 0, 'X') FROM alice_sales",
])
def test_function_forms_are_allowed(sql):
    # Should not raise — these are string/number functions, not write statements.
    assert _guard(sql)


# --- Helpers ---------------------------------------------------------------
def test_referenced_tables_excludes_cte_names():
    refs = referenced_tables("WITH t AS (SELECT * FROM alice_sales) SELECT * FROM t")
    assert refs == {"alice_sales"}


def test_enforce_read_only_strips_trailing_semicolon():
    assert enforce_read_only("SELECT 1;") == "SELECT 1"
