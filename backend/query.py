# query.py

import os
from fastapi import APIRouter, Query, HTTPException, Body
from langchain_community.utilities import SQLDatabase
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from dotenv import load_dotenv
import re
from upload import get_current_session
from sqlalchemy import create_engine, text
from passlib.context import CryptContext

load_dotenv()

ask_question = APIRouter()

MYSQL_URI = os.getenv("MYSQL_URI")
if not MYSQL_URI:
    raise ValueError("MYSQL_URI not found in environment variables.")

# Updated SQL generation prompt with context awareness
SQL_ANALYST_PROMPT = """You are a SQL analyst. Generate SQL queries for data analysis and provide statistical insights.

**Current Active Dataset Context:**
{active_table_context}

**Available Database Schema:**
{schema_info}

**Available Tables:**
{table_names}

**Rules:**
- PRIORITIZE the active dataset (table) when generating queries unless the user specifically mentions another table
- For data analysis: Generate comprehensive SQL with clear column aliases
- For summaries: Include COUNT, AVG, MIN, MAX, STDDEV where relevant  
- Handle numeric columns that might have formatting (currency symbols, commas, etc.)
- Use backticks for column names with spaces or special characters
- Generate working MySQL queries only
- Adapt to the actual column names and data types in the schema
- When the user asks general questions without specifying a table, assume they're asking about the active dataset

**User Question:** {question}

Generate appropriate SQL query focusing on the active dataset."""

# Updated analysis prompt
ANALYSIS_PROMPT = """You are a data analyst. Interpret the following SQL query results and provide meaningful insights.

**Dataset Context:** {active_table_context}
**Original Question:** {question}
**SQL Query:** {sql_query}
**Query Results:** {results}

Provide a clear, human-readable analysis of what these results mean. Focus on:
- What the numbers tell us about the data
- Key insights and patterns
- Practical interpretation of statistics
- Any notable findings
- Context about the dataset being analyzed

Keep the analysis concise but informative."""

# Conceptual questions prompt
CONCEPTUAL_PROMPT = """You are a statistics and data analysis expert. Answer the following question clearly and concisely:

**Question:** {question}

Provide a clear explanation that helps someone understand the concept."""

try:
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.1, max_tokens=1500)
    db = SQLDatabase.from_uri(MYSQL_URI)
    
    # SQL generation chain
    sql_prompt_template = PromptTemplate(
        input_variables=["active_table_context", "schema_info", "table_names", "question"],
        template=SQL_ANALYST_PROMPT
    )
    sql_chain = LLMChain(llm=llm, prompt=sql_prompt_template)
    
    # Analysis chain
    analysis_prompt_template = PromptTemplate(
        input_variables=["active_table_context", "question", "sql_query", "results"],
        template=ANALYSIS_PROMPT
    )
    analysis_chain = LLMChain(llm=llm, prompt=analysis_prompt_template)
    
    # Conceptual chain
    conceptual_prompt_template = PromptTemplate(
        input_variables=["question"],
        template=CONCEPTUAL_PROMPT
    )
    conceptual_chain = LLMChain(llm=llm, prompt=conceptual_prompt_template)
    
except Exception as e:
    raise RuntimeError(f"Failed to initialize SQL components: {e}")

def get_active_table_context():
    """Get context about the currently active table"""
    session = get_current_session()
    if session["active_table"]:
        table_info = session["table_info"]
        context = f"Active Dataset: '{session['active_table']}'"
        if "original_name" in table_info:
            context += f" (originally '{table_info['original_name']}')"
        if "file_name" in table_info:
            context += f" from file '{table_info['file_name']}'"
        if "columns" in table_info:
            context += f"\nColumns in active dataset: {', '.join(table_info['columns'])}"
        if "rows" in table_info:
            context += f"\nTotal rows: {table_info['rows']}"
        return context
    return "No active dataset. Please upload a CSV file first."

def extract_sql_from_response(response_text):
    """Extract SQL query from response"""
    sql_pattern = r'```sql\s*(.*?)\s*```'
    matches = re.findall(sql_pattern, response_text, re.DOTALL | re.IGNORECASE)
    if matches:
        return matches[0].strip()
    
    select_pattern = r'(SELECT.*?(?=\n\n|\n[A-Z]|\n\*\*|$))'
    select_matches = re.findall(select_pattern, response_text, re.DOTALL | re.IGNORECASE)
    if select_matches:
        return select_matches[0].strip()
    
    return None

def get_schema_info():
    """Get database schema information"""
    try:
        tables = db.get_usable_table_names()
        schema_info = {}
        for table in tables:
            try:
                table_info = db.get_table_info([table])
                schema_info[table] = table_info
            except Exception as e:
                schema_info[table] = f"Error: {e}"
        return tables, schema_info
    except Exception as e:
        return [], {"error": str(e)}

def get_question_type(question):
    """Determine question type"""
    question_lower = question.lower().strip()
    
    # Greetings and casual conversation
    greeting_keywords = ['hi', 'hello', 'hey', 'hy', 'good morning', 'good afternoon', 'good evening', 'how are you', 'what\'s up', 'whats up']
    for keyword in greeting_keywords:
        if question_lower == keyword or question_lower.startswith(keyword + ' ') or question_lower.startswith(keyword + ','):
            return "greeting"
    
    # Conceptual questions - but only if not asking about specific data
    # More specific detection to avoid false positives with data questions
    data_indicators = ['price', 'column', 'table', 'data', 'average', 'sum', 'count', 'min', 'max', 'total']
    has_data_context = any(indicator in question_lower for indicator in data_indicators)
    
    if not has_data_context:
        conceptual_keywords = ['what is', 'define', 'explain', 'meaning of', 'concept of', 'how does', 'what does']
        if any(keyword in question_lower for keyword in conceptual_keywords):
            return "conceptual"
    
    # Column names - get from database dynamically
    if any(phrase in question_lower for phrase in ['column names', 'list columns', 'show columns', 'what columns']):
        return "column_names"
    
    # Check if question needs analysis (has verbs like "show", "get", "find" vs just listing)
    analysis_keywords = ['analyze', 'analysis', 'insight', 'pattern', 'trend', 'summary', 'statistics', 'compare']
    simple_list_keywords = ['list', 'show all', 'get all', 'display all']
    
    if any(keyword in question_lower for keyword in analysis_keywords):
        return "data_analysis_with_insights"
    elif any(keyword in question_lower for keyword in simple_list_keywords):
        return "simple_data_query"
    else:
        return "data_analysis_with_insights"  # Default to analysis

def needs_analysis(question, sql_query):
    """Determine if results need AI analysis"""
    question_lower = question.lower()
    
    # Skip analysis for simple listing queries
    if any(phrase in question_lower for phrase in ['list', 'show all', 'get all', 'display all']):
        return False
    
    # Skip analysis for single column selections without aggregation
    if sql_query and 'SELECT' in sql_query.upper():
        select_part = sql_query.upper().split('FROM')[0]
        # If it's a simple SELECT without aggregation functions
        if (select_part.count(',') == 0 and 
            not any(func in select_part for func in ['COUNT', 'AVG', 'SUM', 'MIN', 'MAX', 'STDDEV'])):
            return False
    
    return True

def get_table_columns():
    """Get column information for all tables"""
    try:
        table_names, schema_info = get_schema_info()
        columns_info = {}
        
        for table_name in table_names:
            try:
                # Extract column names from schema info
                schema_text = schema_info.get(table_name, "")
                if schema_text and not schema_text.startswith("Error:"):
                    # Parse schema to extract column names
                    columns = []
                    lines = schema_text.split('\n')
                    for line in lines:
                        if 'Column:' in line or '|' in line:
                            # Different formats might exist, try to extract column names
                            parts = line.split()
                            if len(parts) > 0:
                                col_name = parts[0].strip('|').strip()
                                if col_name and col_name not in ['Column:', 'CREATE', 'TABLE']:
                                    columns.append(col_name)
                    
                    if not columns:
                        # Fallback: try to get columns via DESCRIBE query
                        try:
                            describe_result = db.run(f"DESCRIBE {table_name};")
                            columns = [row[0] for row in eval(describe_result) if isinstance(row, tuple)]
                        except:
                            columns = ["Unable to determine columns"]
                    
                    columns_info[table_name] = columns
                else:
                    columns_info[table_name] = ["Error retrieving columns"]
            except Exception as e:
                columns_info[table_name] = [f"Error: {str(e)}"]
        
        return columns_info
    except Exception as e:
        return {"error": str(e)}

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def get_password_hash(password):
    return pwd_context.hash(password)

from pydantic import BaseModel

class AskRequest(BaseModel):
    message: str

@ask_question.post("/ask")
def ask_post(payload: AskRequest):
    q = payload.message
    try:
        # Get active table context
        active_context = get_active_table_context()
        session = get_current_session()
        
        # Check if there's an active table
        if not session["active_table"] and "No active dataset" in active_context:
            return {
                "answer": "Please upload a CSV file first. I need a dataset to work with before I can answer questions about your data."
            }
        
        question_type = get_question_type(q)
        
        # Handle greetings
        if question_type == "greeting":
            session_info = f" I can see you have '{session['table_info'].get('original_name', 'a dataset')}' loaded and ready to analyze!" if session["active_table"] else " Please upload a CSV file so I can help you analyze your data."
            return {
                "answer": f"Hello! I'm your data analysis assistant.{session_info} Feel free to ask me questions about your data, request summaries, or ask for specific analyses."
            }
        
        # Handle conceptual questions with Gemini
        if question_type == "conceptual":
            try:
                response = conceptual_chain.run({"question": q})
                return {"answer": response.strip()}
            except Exception as e:
                return {"answer": f"I'm having trouble accessing the AI model right now. Error: {str(e)}"}
        
        # Handle column names - show columns from active table primarily
        if question_type == "column_names":
            try:
                if session["active_table"] and "columns" in session["table_info"]:
                    columns = session["table_info"]["columns"]
                    table_name = session["table_info"].get("original_name", session["active_table"])
                    response = f"Columns in your active dataset '{table_name}':\n{', '.join(columns)}"
                    return {"answer": response}
                else:
                    # Fallback to all tables
                    columns_info = get_table_columns()
                    if "error" in columns_info:
                        return {"answer": f"Error retrieving columns: {columns_info['error']}"}
                    
                    response = "Available columns by table:\n"
                    for table, columns in columns_info.items():
                        response += f"**{table}**: {', '.join(columns)}\n"
                    return {"answer": response.strip()}
            except Exception as e:
                return {"answer": f"Error retrieving column information: {str(e)}"}
        
        # Handle data analysis
        table_names, schema_info = get_schema_info()
        schema_text = ""
        for table, info in schema_info.items():
            schema_text += f"\n**Table: {table}**\n{info}\n"
        
        # Generate SQL with active table context
        try:
            response = sql_chain.run({
                "question": q,
                "table_names": ", ".join(table_names),
                "schema_info": schema_text,
                "active_table_context": active_context
            })
            
            sql_query = extract_sql_from_response(response)
            
            if sql_query:
                try:
                    query_result = db.run(sql_query)
                    
                    # Determine if we need analysis
                    if needs_analysis(q, sql_query):
                        try:
                            analysis = analysis_chain.run({
                                "question": q,
                                "sql_query": sql_query,
                                "results": str(query_result),
                                "active_table_context": active_context
                            })
                            
                            return {
                                "sql_query": sql_query,
                                "results": query_result,
                                "analysis": analysis.strip(),
                                "active_dataset": session["table_info"].get("original_name", session["active_table"]) if session["active_table"] else None
                            }
                        except Exception as analysis_error:
                            # Fallback if analysis fails
                            return {
                                "sql_query": sql_query,
                                "results": query_result,
                                "analysis": f"Query executed successfully. Analysis unavailable due to: {str(analysis_error)}",
                                "active_dataset": session["table_info"].get("original_name", session["active_table"]) if session["active_table"] else None
                            }
                    else:
                        # Simple query, no analysis needed
                        return {
                            "sql_query": sql_query,
                            "results": query_result,
                            "active_dataset": session["table_info"].get("original_name", session["active_table"]) if session["active_table"] else None
                        }
                    
                except Exception as db_error:
                    return {
                        "sql_query": sql_query,
                        "error": f"Query execution failed: {str(db_error)}"
                    }
            else:
                return {"answer": "Could not generate appropriate SQL query"}
                
        except Exception as sql_error:
            return {"error": f"Failed to generate SQL query: {str(sql_error)}"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@ask_question.get("/list_tables/")
def list_tables():
    """List available tables and their columns with active table highlighted"""
    try:
        table_names, schema_info = get_schema_info()
        columns_info = get_table_columns()
        session = get_current_session()
        
        return {
            "tables": table_names, 
            "schema": schema_info,
            "columns": columns_info,
            "active_table": session["active_table"],
            "active_table_info": session["table_info"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {e}")

