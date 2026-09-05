# agents.py - Specialized agents used by the orchestrator.
#
# Each agent has one clear job and a `name`. The orchestrator (orchestrator.py) decides which
# agent handles a message; the agents themselves contain the LLM chains and the Tier 2
# guardrail/validation logic.
import os
import re
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.utilities import SQLDatabase

from db import list_user_table_names
from guardrails import guard, SQLGuardrailError
import rag

load_dotenv()

MYSQL_URI = os.getenv("MYSQL_URI")
if not MYSQL_URI:
    raise ValueError("MYSQL_URI not found in environment variables.")

ENABLE_GROUNDEDNESS_CHECK = os.getenv("ENABLE_GROUNDEDNESS_CHECK", "true").lower() != "false"
# Chat model — overridable via env. `gemini-flash-latest` resolves to a currently-available
# flash model for the project (avoids pinned-version availability issues).
LLM_MODEL = os.getenv("LLM_MODEL", "gemini-flash-latest")

# ----------------------------------------------------------------------------- prompts
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
results shown above — do not invent numbers. Keep the analysis concise but informative."""

CONCEPTUAL_PROMPT = """You are a statistics and data analysis expert. Answer the following question clearly and concisely:

**Question:** {question}

Provide a clear explanation that helps someone understand the concept."""

SQL_FIX_PROMPT = """The following MySQL query failed to execute. Fix it so it runs correctly.

**Available Schema (only your own tables):**
{schema_info}

**Dataset Context:** {active_table_context}
**Original Question:** {question}
**Broken SQL:** {sql_query}
**Database Error:** {error}

Return only the corrected, single read-only SELECT query in a ```sql code block."""

GROUNDEDNESS_PROMPT = """You are a strict fact-checker. Decide whether the proposed answer is fully supported
by the SQL results (no invented or contradicted numbers).

**Question:** {question}
**SQL Results:** {results}
**Proposed Answer:** {analysis}

Reply with exactly one line:
- "GROUNDED" if every claim in the answer is supported by the results.
- "NOT GROUNDED: <short reason>" otherwise."""

KNOWLEDGE_PROMPT = """Answer the question using ONLY the context extracted from the user's uploaded documents.
If the answer is not contained in the context, say you could not find it in the documents — do not guess.
Mention which document(s) the answer comes from.

**Context from documents:**
{context}

**Question:** {question}

Answer:"""

# ----------------------------------------------------------------------------- shared LLM/chains
try:
    llm = ChatGoogleGenerativeAI(model=LLM_MODEL, temperature=0.1, max_tokens=1500)
    judge_llm = ChatGoogleGenerativeAI(model=LLM_MODEL, temperature=0.0, max_tokens=200)
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
    groundedness_chain = LLMChain(llm=judge_llm, prompt=PromptTemplate(
        input_variables=["question", "results", "analysis"], template=GROUNDEDNESS_PROMPT))
    knowledge_chain = LLMChain(llm=llm, prompt=PromptTemplate(
        input_variables=["context", "question"], template=KNOWLEDGE_PROMPT))
except Exception as e:
    raise RuntimeError(f"Failed to initialize LLM agents: {e}")


# ----------------------------------------------------------------------------- helpers
def extract_sql_from_response(response_text):
    """Extract a SQL query from an LLM response."""
    matches = re.findall(r'```sql\s*(.*?)\s*```', response_text, re.DOTALL | re.IGNORECASE)
    if matches:
        return matches[0].strip()
    select_matches = re.findall(r'(SELECT.*?(?=\n\n|\n[A-Z]|\n\*\*|$))', response_text, re.DOTALL | re.IGNORECASE)
    if select_matches:
        return select_matches[0].strip()
    return None


def build_active_table_context(ctx):
    """Build a human-readable context string from a resolved active-table dict."""
    if not ctx["active_table"]:
        return "No active dataset. Please upload a CSV file first."
    ti = ctx["table_info"]
    context = f"Active Dataset: '{ctx['active_table']}'"
    if "original_name" in ti:
        context += f" (originally '{ti['original_name']}')"
    if ti.get("file_name"):
        context += f" from file '{ti['file_name']}'"
    if "columns" in ti:
        context += f"\nColumns in active dataset: {', '.join(ti['columns'])}"
    if "rows" in ti:
        context += f"\nTotal rows: {ti['rows']}"
    return context


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


def needs_analysis(question, sql_query):
    """Determine if results need AI analysis."""
    ql = question.lower()
    if any(p in ql for p in ['list', 'show all', 'get all', 'display all']):
        return False
    if sql_query and 'SELECT' in sql_query.upper():
        select_part = sql_query.upper().split('FROM')[0]
        if (select_part.count(',') == 0 and
                not any(fn in select_part for fn in ['COUNT', 'AVG', 'SUM', 'MIN', 'MAX', 'STDDEV'])):
            return False
    return True


# ----------------------------------------------------------------------------- agents
class ValidatorAgent:
    """Safety + quality reviewer: guards SQL before execution and fact-checks answers."""
    name = "validator"

    def validate_sql(self, sql, allowed_tables):
        """Return safe SQL to run, or raise SQLGuardrailError."""
        return guard(sql, allowed_tables)

    def check_answer(self, question, results, analysis):
        """Return (grounded: bool, reason: str). Fails open on judge error."""
        if not ENABLE_GROUNDEDNESS_CHECK:
            return True, ""
        try:
            verdict = groundedness_chain.run({
                "question": question, "results": str(results), "analysis": analysis,
            }).strip()
            if verdict.upper().startswith("GROUNDED"):
                return True, ""
            return False, verdict
        except Exception as e:
            print(f"Groundedness check failed (failing open): {e}")
            return True, ""


class GreetingAgent:
    name = "greeting"

    def answer(self, ctx):
        if ctx["active_table"]:
            name = ctx["table_info"].get("original_name", "a dataset")
            extra = f" I can see you have '{name}' loaded and ready to analyze!"
        else:
            extra = " Please upload a CSV file so I can help you analyze your data."
        return {"answer": f"Hello! I'm your data analysis assistant.{extra} Feel free to ask me questions "
                          "about your data, request summaries, or ask for specific analyses."}


class ConceptualAgent:
    name = "conceptual"

    def answer(self, question):
        try:
            return {"answer": conceptual_chain.run({"question": question}).strip()}
        except Exception as e:
            return {"answer": f"I'm having trouble accessing the AI model right now. Error: {str(e)}"}


class ColumnNamesAgent:
    name = "column_names"

    def answer(self, ctx):
        columns = ctx["table_info"].get("columns", [])
        name = ctx["table_info"].get("original_name", ctx["active_table"])
        return {"answer": f"Columns in your active dataset '{name}':\n{', '.join(columns)}"}


class KnowledgeAgent:
    """Answers questions from the user's uploaded documents (RAG), with citations."""
    name = "knowledge"

    def __init__(self, validator: ValidatorAgent):
        self.validator = validator

    def answer(self, question, ctx, current_user):
        hits = rag.query_documents(current_user, question, k=4)
        if not hits:
            return {"answer": "I couldn't find any uploaded documents to answer from. Upload a PDF or "
                              "text document first, then ask me about its contents."}

        context = "\n\n".join(f"[Source: {m.get('doc_name', 'document')}]\n{doc}" for doc, m in hits)
        try:
            answer = knowledge_chain.run({"context": context, "question": question}).strip()
        except Exception as e:
            return {"answer": f"I'm having trouble accessing the AI model right now. Error: {str(e)}"}

        # Fact-check the answer against the retrieved chunks (reuses the validator).
        grounded, reason = self.validator.check_answer(question, context, answer)
        if not grounded:
            answer += ("\n\n⚠️ Note: this may not be fully supported by your documents — please verify.")

        sources = sorted({m.get("doc_name") for _, m in hits if m.get("doc_name")})
        return {"answer": answer, "sources": sources, "grounded": grounded}


class SQLAnalystAgent:
    """Generates SQL, runs it safely (via the validator), and interprets the results."""
    name = "sql_analyst"

    def __init__(self, validator: ValidatorAgent):
        self.validator = validator

    def _run_guarded_sql(self, sql_query, allowed_tables, question, schema_text, active_context):
        """Validate + execute with one self-correction retry. Returns (safe_sql, results, error)."""
        current_sql = sql_query
        last_error = None
        for attempt in range(2):
            try:
                safe_sql = self.validator.validate_sql(current_sql, allowed_tables)
            except SQLGuardrailError as ge:
                return current_sql, None, f"__BLOCKED__:{ge}"
            try:
                return safe_sql, db.run(safe_sql), None
            except Exception as db_error:
                last_error = str(db_error)
                if attempt == 0:
                    try:
                        fixed = extract_sql_from_response(sql_fix_chain.run({
                            "schema_info": schema_text, "active_table_context": active_context,
                            "question": question, "sql_query": safe_sql, "error": last_error,
                        }))
                        if fixed:
                            current_sql = fixed
                            continue
                    except Exception as fix_error:
                        last_error = f"{last_error} (self-correction failed: {fix_error})"
                return safe_sql, None, last_error
        return current_sql, None, last_error

    def answer(self, question, ctx, current_user):
        active_context = build_active_table_context(ctx)
        allowed_tables = list_user_table_names(current_user)
        table_names, schema_info = get_schema_info(allowed_tables)
        schema_text = "".join(f"\n**Table: {t}**\n{info}\n" for t, info in schema_info.items())

        try:
            response = sql_chain.run({
                "question": question, "table_names": ", ".join(table_names),
                "schema_info": schema_text, "active_table_context": active_context,
            })
        except Exception as sql_error:
            return {"error": f"Failed to generate SQL query: {str(sql_error)}"}

        sql_query = extract_sql_from_response(response)
        if not sql_query:
            return {"answer": "Could not generate appropriate SQL query"}

        safe_sql, query_result, error = self._run_guarded_sql(
            sql_query, allowed_tables, question, schema_text, active_context)

        if error is not None:
            if error.startswith("__BLOCKED__:"):
                return {
                    "answer": "I couldn't run that request because it was blocked by the safety guardrails "
                              f"({error.split('__BLOCKED__:', 1)[1]}). I can only run read-only queries on your own datasets.",
                    "blocked": True,
                }
            return {"sql_query": safe_sql, "error": f"Query execution failed: {error}"}

        active_dataset = ctx["table_info"].get("original_name", ctx["active_table"])

        if needs_analysis(question, safe_sql):
            try:
                analysis = analysis_chain.run({
                    "question": question, "sql_query": safe_sql,
                    "results": str(query_result), "active_table_context": active_context,
                }).strip()
            except Exception as analysis_error:
                return {
                    "sql_query": safe_sql, "results": query_result,
                    "analysis": f"Query executed successfully. Analysis unavailable due to: {str(analysis_error)}",
                    "active_dataset": active_dataset,
                }

            grounded, reason = self.validator.check_answer(question, query_result, analysis)
            if not grounded:
                analysis += ("\n\n⚠️ Note: parts of this interpretation may not be fully supported "
                             "by the query results — please verify before relying on it.")
            return {
                "sql_query": safe_sql, "results": query_result, "analysis": analysis,
                "active_dataset": active_dataset, "grounded": grounded,
            }

        return {"sql_query": safe_sql, "results": query_result, "active_dataset": active_dataset}


# Singletons wired together (validator injected into the SQL analyst).
validator_agent = ValidatorAgent()
greeting_agent = GreetingAgent()
conceptual_agent = ConceptualAgent()
column_names_agent = ColumnNamesAgent()
knowledge_agent = KnowledgeAgent(validator_agent)
sql_analyst_agent = SQLAnalystAgent(validator_agent)
