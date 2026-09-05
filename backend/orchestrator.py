# orchestrator.py - Routes each message to the right agent (the "brain").
#
# Hybrid router: a cheap deterministic keyword pass handles the obvious cases for free; only
# genuinely ambiguous messages fall through to an LLM classifier. When uncertain, it biases
# toward the data (SQL) agent so normal dataset questions never dead-end in a stub.
import re
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

from db import resolve_active_table
from agents import (
    llm,
    greeting_agent, conceptual_agent, column_names_agent, knowledge_agent, sql_analyst_agent,
)

VALID_LANES = {"greeting", "conceptual", "knowledge", "column_names", "data"}

# Lanes the LLM classifier may choose (column_names is keyword-only).
ROUTER_PROMPT = """Classify the user's message into exactly one category:
- data: a question about their uploaded dataset/table (counts, averages, filtering, values, trends)
- conceptual: a general question about a statistics/data concept, not about their specific data
- knowledge: a question about the contents of an uploaded document/PDF (not tabular data)
- greeting: a greeting or small talk

If unsure, choose "data".
Message: {question}
Answer with only the category word."""

router_chain = LLMChain(llm=llm, prompt=PromptTemplate(
    input_variables=["question"], template=ROUTER_PROMPT))

_GREETINGS = ['hi', 'hello', 'hey', 'hy', 'good morning', 'good afternoon', 'good evening',
              'how are you', "what's up", 'whats up']
_DOC_KEYWORDS = ['document', 'pdf', 'the doc', 'in the document', 'according to the document',
                 'uploaded file text', 'the report says', 'the article']
_COLUMN_PHRASES = ['column names', 'list columns', 'show columns', 'what columns']
_DATA_INDICATORS = ['price', 'column', 'table', 'data', 'average', 'sum', 'count', 'min', 'max',
                    'total', 'rows', 'how many', 'group by', 'top ', 'highest', 'lowest', 'per ']
_ANALYSIS_KEYWORDS = ['analyze', 'analysis', 'insight', 'pattern', 'trend', 'summary', 'statistics', 'compare']
_CONCEPTUAL_KEYWORDS = ['what is', 'define', 'explain', 'meaning of', 'concept of', 'how does', 'what does']


def keyword_route(question):
    """Deterministic fast-path. Returns a lane, or None if the message is ambiguous."""
    ql = question.lower().strip()

    for k in _GREETINGS:
        if ql == k or ql.startswith(k + ' ') or ql.startswith(k + ','):
            return "greeting"

    if any(k in ql for k in _DOC_KEYWORDS):
        return "knowledge"

    if any(p in ql for p in _COLUMN_PHRASES):
        return "column_names"

    has_data_context = any(k in ql for k in _DATA_INDICATORS)
    if any(k in ql for k in _ANALYSIS_KEYWORDS) or has_data_context:
        return "data"

    # Conceptual only when there's no data context (guarded by the check above).
    if any(k in ql for k in _CONCEPTUAL_KEYWORDS):
        return "conceptual"

    return None  # ambiguous -> let the LLM decide


def route(question):
    """Hybrid route: keyword fast-path, then LLM fallback (defaulting to 'data')."""
    lane = keyword_route(question)
    if lane:
        return lane
    try:
        label = re.sub(r'[^a-z]', '', router_chain.run({"question": question}).strip().lower())
        if label in VALID_LANES:
            return label
    except Exception as e:
        print(f"Router LLM failed, defaulting to 'data': {e}")
    return "data"


def _tag(response, agent):
    """Attach which agent produced the response (transparency for UI/debugging)."""
    response["agent"] = agent.name
    return response


def handle(question, current_user, table_name=None):
    """Resolve context, route the question, and dispatch to the chosen agent."""
    ctx = resolve_active_table(current_user, table_name)
    lane = route(question)

    # These lanes don't require a loaded CSV dataset.
    if lane == "greeting":
        return _tag(greeting_agent.answer(ctx), greeting_agent)
    if lane == "conceptual":
        return _tag(conceptual_agent.answer(question), conceptual_agent)
    if lane == "knowledge":
        # Answers from uploaded documents (RAG) — independent of any active table.
        return _tag(knowledge_agent.answer(question, ctx, current_user), knowledge_agent)

    # Remaining lanes operate on the user's tabular data.
    if not ctx["active_table"]:
        return {
            "answer": "Please upload a CSV file first. I need a dataset to work with before I can "
                      "answer questions about your data.",
            "agent": "router",
        }

    if lane == "column_names":
        return _tag(column_names_agent.answer(ctx), column_names_agent)

    # Default: the SQL analyst (with validator guardrails + groundedness).
    return _tag(sql_analyst_agent.answer(question, ctx, current_user), sql_analyst_agent)
