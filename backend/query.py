# query.py - Natural-language -> SQL chat pipeline (LangChain + Gemini).
#
# Active-table state is resolved per-request from the database (db.resolve_active_table).
# All LLM-generated SQL is passed through guardrails.guard() before execution, the model
# only ever sees the requesting user's own tables, failed queries get one self-correction
# retry, and the final narration is checked for groundedness against the query results.
import os
from fastapi import APIRouter, HTTPException, Depends
from langchain_community.utilities import SQLDatabase
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from dotenv import load_dotenv
import re
from pydantic import BaseModel

from auth import get_current_user
from db import resolve_active_table, list_user_table_names
from guardrails import guard, SQLGuardrailError

load_dotenv()

ask_question = APIRouter()

MYSQL_URI = os.getenv("MYSQL_URI")
if not MYSQL_URI:
    raise ValueError("MYSQL_URI not found in environment variables.")

# Groundedness checking can be disabled (e.g. to save LLM calls) via env.
ENABLE_GROUNDEDNESS_CHECK = os.getenv("ENABLE_GROUNDEDNESS_CHECK", "true").lower() != "false"

SQL_ANALYST_PROMPT = """You are a SQL analyst. Generate SQL queries for data analysis and provide statistical insights.

**Current Active Dataset Context:**
{active_table_context}

**Available Database Schema (only your own tables):**
{schema_info}

**Available Tables:**
{table_names}

**Rules:**
- PRIORITIZE the active dataset (table) when generating queries unless the user specifically mentions another table
- ONLY use tables listed above. Never reference any other table.
- Generate read-only SELECT queries only. Never write, update, or delete data.
- For data analysis: Generate comprehensive SQL with clear column aliases
- For summaries: Include COUNT, AVG, MIN, MAX, STDDEV where relevant
- Handle numeric columns that might have formatting (currency symbols, commas, etc.)
- Use backticks for column names with spaces or special characters
- Generate working MySQL queries only
- Adapt to the actual column names and data types in the schema

**User Question:** {question}

Generate a single read-only SELECT query focusing on the active dataset. Return it in a ```sql code block."""

ANALYSIS_PROMPT = """You are a data analyst. Interpret the following SQL query results and provide meaningful insights.

**Dataset Context:** {active_table_context}
**Original Question:** {question}
**SQL Query:** {sql_query}
**Query Results:** {results}

Provide a clear, human-readable analysis of what these results mean. Base every statement strictly on the
results shown above — do not invent numbers. Focus on:
- What the numbers tell us about the data
- Key insights and patterns
- Practical interpretation of statistics
- Any notable findings

Keep the analysis concise but informative."""

CONCEPTUAL_PROMPT = """You are a statistics and data analysis expert. Answer the following question clearly and concisely:

**Question:** {question}

Provide a clear explanation that helps someone understand the concept."""

# Prompt used to repair a query that failed to execute (one retry).
SQL_FIX_PROMPT = """The following MySQL query failed to execute. Fix it so it runs correctly.

**Available Schema (only your own tables):**
{schema_info}

**Dataset Context:** {active_table_context}
**Original Question:** {question}
**Broken SQL:** {sql_query}
**Database Error:** {error}

Return only the corrected, single read-only SELECT query in a ```sql code block."""

# Prompt used to fact-check the narration against the actual results.
GROUNDEDNESS_PROMPT = """You are a strict fact-checker. Decide whether the proposed answer is fully supported
by the SQL results (no invented or contradicted numbers).

**Question:** {question}
**SQL Results:** {results}
**Proposed Answer:** {analysis}

Reply with exactly one line:
- "GROUNDED" if every claim in the answer is supported by the results.
- "NOT GROUNDED: <short reason>" otherwise."""

try:
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.1, max_tokens=1500)
    db = SQLDatabase.from_uri(MYSQL_URI)

    sql_chain = LLMChain(llm=llm, prompt=PromptTemplate(
        input_variables=["active_table_context", "schema_info", "table_names", "question"],
        template=SQL_ANALYST_PROMPT))

    analysis_chain = LLMChain(llm=llm, prompt=PromptTemplate(
        input_variables=["active_table_context", "question", "sql_query", "results"],
        template=ANALYSIS_PROMPT))

    conceptual_chain = LLMChain(llm=llm, prompt=PromptTemplate(
        input_variables=["question"], template=CONCEPTUAL_PROMPT))

    sql_fix_chain = LLMChain(llm=llm, prompt=PromptTemplate(
        input_variables=["schema_info", "active_table_context", "question", "sql_query", "error"],
        template=SQL_FIX_PROMPT))

    # Lower temperature judge for the fact-check.
    judge_llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.0, max_tokens=200)
    groundedness_chain = LLMChain(llm=judge_llm, prompt=PromptTemplate(
        input_variables=["question", "results", "analysis"], template=GROUNDEDNESS_PROMPT))

except Exception as e:
    raise RuntimeError(f"Failed to initialize SQL components: {e}")


def build_active_table_context(ctx):
    """Build a human-readable context string from a resolved active-table dict."""
    if not ctx["active_table"]:
        return "No active dataset. Please upload a CSV file first."
    table_info = ctx["table_info"]
    context = f"Active Dataset: '{ctx['active_table']}'"
    if "original_name" in table_info:
        context += f" (originally '{table_info['original_name']}')"
    if table_info.get("file_name"):
        context += f" from file '{table_info['file_name']}'"
    if "columns" in table_info:
        context += f"\nColumns in active dataset: {', '.join(table_info['columns'])}"
    if "rows" in table_info:
        context += f"\nTotal rows: {table_info['rows']}"
    return context


def extract_sql_from_response(response_text):
    """Extract a SQL query from an LLM response."""
    sql_pattern = r'```sql\s*(.*?)\s*```'
    matches = re.findall(sql_pattern, response_text, re.DOTALL | re.IGNORECASE)
    if matches:
        return matches[0].strip()

    select_pattern = r'(SELECT.*?(?=\n\n|\n[A-Z]|\n\*\*|$))'
    select_matches = re.findall(select_pattern, response_text, re.DOTALL | re.IGNORECASE)
    if select_matches:
        return select_matches[0].strip()

    return None


def get_schema_info(allowed_tables=None):
    """Get schema info, restricted to `allowed_tables` when provided (privacy scoping)."""
    try:
        tables = db.get_usable_table_names()
        if allowed_tables is not None:
            allowed = {t.lower() for t in allowed_tables}
            tables = [t for t in tables if t.lower() in allowed]
        schema_info = {}
        for table in tables:
            try:
                schema_info[table] = db.get_table_info([table])
            except Exception as e:
                schema_info[table] = f"Error: {e}"
        return tables, schema_info
    except Exception as e:
        return [], {"error": str(e)}


def get_question_type(question):
    """Determine question type by keyword routing."""
    question_lower = question.lower().strip()

    greeting_keywords = ['hi', 'hello', 'hey', 'hy', 'good morning', 'good afternoon', 'good evening', 'how are you', 'what\'s up', 'whats up']
    for keyword in greeting_keywords:
        if question_lower == keyword or question_lower.startswith(keyword + ' ') or question_lower.startswith(keyword + ','):
            return "greeting"

    data_indicators = ['price', 'column', 'table', 'data', 'average', 'sum', 'count', 'min', 'max', 'total']
    has_data_context = any(indicator in question_lower for indicator in data_indicators)

    if not has_data_context:
        conceptual_keywords = ['what is', 'define', 'explain', 'meaning of', 'concept of', 'how does', 'what does']
        if any(keyword in question_lower for keyword in conceptual_keywords):
            return "conceptual"

    if any(phrase in question_lower for phrase in ['column names', 'list columns', 'show columns', 'what columns']):
        return "column_names"

    analysis_keywords = ['analyze', 'analysis', 'insight', 'pattern', 'trend', 'summary', 'statistics', 'compare']
    simple_list_keywords = ['list', 'show all', 'get all', 'display all']

    if any(keyword in question_lower for keyword in analysis_keywords):
        return "data_analysis_with_insights"
    elif any(keyword in question_lower for keyword in simple_list_keywords):
        return "simple_data_query"
    else:
        return "data_analysis_with_insights"


def needs_analysis(question, sql_query):
    """Determine if results need AI analysis."""
    question_lower = question.lower()

    if any(phrase in question_lower for phrase in ['list', 'show all', 'get all', 'display all']):
        return False

    if sql_query and 'SELECT' in sql_query.upper():
        select_part = sql_query.upper().split('FROM')[0]
        if (select_part.count(',') == 0 and
                not any(func in select_part for func in ['COUNT', 'AVG', 'SUM', 'MIN', 'MAX', 'STDDEV'])):
            return False

    return True


def check_groundedness(question, results, analysis):
    """Fact-check the narration against the results. Fails open (returns grounded) on error."""
    if not ENABLE_GROUNDEDNESS_CHECK:
        return True, ""
    try:
        verdict = groundedness_chain.run({
            "question": question,
            "results": str(results),
            "analysis": analysis,
        }).strip()
        if verdict.upper().startswith("GROUNDED"):
            return True, ""
        return False, verdict
    except Exception as e:
        print(f"Groundedness check failed (failing open): {e}")
        return True, ""


def run_guarded_sql(sql_query, allowed_tables, question, schema_text, active_context):
    """Guard + execute the SQL, with one self-correction retry on execution error.

    Returns (safe_sql, results, error). Exactly one of results/error is meaningful.
    A guardrail rejection returns immediately (no retry).
    """
    current_sql = sql_query
    last_error = None

    for attempt in range(2):
        # Safety gate — a rejection here is terminal (do not retry unsafe SQL).
        try:
            safe_sql = guard(current_sql, allowed_tables)
        except SQLGuardrailError as ge:
            return current_sql, None, f"__BLOCKED__:{ge}"

        try:
            results = db.run(safe_sql)
            return safe_sql, results, None
        except Exception as db_error:
            last_error = str(db_error)
            if attempt == 0:
                # One self-correction attempt: hand the error back to the model.
                try:
                    fix_resp = sql_fix_chain.run({
                        "schema_info": schema_text,
                        "active_table_context": active_context,
                        "question": question,
                        "sql_query": safe_sql,
                        "error": last_error,
                    })
                    fixed = extract_sql_from_response(fix_resp)
                    if fixed:
                        current_sql = fixed
                        continue
                except Exception as fix_error:
                    last_error = f"{last_error} (self-correction failed: {fix_error})"
            return safe_sql, None, last_error

    return current_sql, None, last_error


class AskRequest(BaseModel):
    message: str
    table_name: str | None = None


@ask_question.post("/ask")
def ask_post(payload: AskRequest, current_user: str = Depends(get_current_user)):
    """Ask a question about the data - requires authentication."""
    q = payload.message
    try:
        ctx = resolve_active_table(current_user, payload.table_name)
        active_context = build_active_table_context(ctx)

        if not ctx["active_table"]:
            return {
                "answer": "Please upload a CSV file first. I need a dataset to work with before I can answer questions about your data."
            }

        table_info = ctx["table_info"]
        question_type = get_question_type(q)

        if question_type == "greeting":
            session_info = f" I can see you have '{table_info.get('original_name', 'a dataset')}' loaded and ready to analyze!"
            return {
                "answer": f"Hello! I'm your data analysis assistant.{session_info} Feel free to ask me questions about your data, request summaries, or ask for specific analyses."
            }

        if question_type == "conceptual":
            try:
                response = conceptual_chain.run({"question": q})
                return {"answer": response.strip()}
            except Exception as e:
                return {"answer": f"I'm having trouble accessing the AI model right now. Error: {str(e)}"}

        if question_type == "column_names":
            columns = table_info.get("columns", [])
            table_name = table_info.get("original_name", ctx["active_table"])
            return {"answer": f"Columns in your active dataset '{table_name}':\n{', '.join(columns)}"}

        # --- Data query path: scope schema to the user's own tables ---
        allowed_tables = list_user_table_names(current_user)
        table_names, schema_info = get_schema_info(allowed_tables)
        schema_text = ""
        for table, info in schema_info.items():
            schema_text += f"\n**Table: {table}**\n{info}\n"

        try:
            response = sql_chain.run({
                "question": q,
                "table_names": ", ".join(table_names),
                "schema_info": schema_text,
                "active_table_context": active_context,
            })
        except Exception as sql_error:
            return {"error": f"Failed to generate SQL query: {str(sql_error)}"}

        sql_query = extract_sql_from_response(response)
        if not sql_query:
            return {"answer": "Could not generate appropriate SQL query"}

        # Guardrails + execution (+ one self-correction retry)
        safe_sql, query_result, error = run_guarded_sql(
            sql_query, allowed_tables, q, schema_text, active_context)

        if error is not None:
            if error.startswith("__BLOCKED__:"):
                return {
                    "answer": "I couldn't run that request because it was blocked by the safety guardrails "
                              f"({error.split('__BLOCKED__:', 1)[1]}). I can only run read-only queries on your own datasets.",
                    "blocked": True,
                }
            return {"sql_query": safe_sql, "error": f"Query execution failed: {error}"}

        active_dataset = table_info.get("original_name", ctx["active_table"])

        if needs_analysis(q, safe_sql):
            try:
                analysis = analysis_chain.run({
                    "question": q,
                    "sql_query": safe_sql,
                    "results": str(query_result),
                    "active_table_context": active_context,
                }).strip()
            except Exception as analysis_error:
                return {
                    "sql_query": safe_sql,
                    "results": query_result,
                    "analysis": f"Query executed successfully. Analysis unavailable due to: {str(analysis_error)}",
                    "active_dataset": active_dataset,
                }

            grounded, reason = check_groundedness(q, query_result, analysis)
            if not grounded:
                analysis += ("\n\n⚠️ Note: parts of this interpretation may not be fully supported "
                             "by the query results — please verify before relying on it.")

            return {
                "sql_query": safe_sql,
                "results": query_result,
                "analysis": analysis,
                "active_dataset": active_dataset,
                "grounded": grounded,
            }

        return {
            "sql_query": safe_sql,
            "results": query_result,
            "active_dataset": active_dataset,
        }

    except HTTPException:
        raise
    except Exception as e:
        print(f"Error in ask_post: {e}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")
