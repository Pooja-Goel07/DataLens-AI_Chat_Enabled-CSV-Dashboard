# upload.py
import os
import re
import pandas as pd
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
import json
from datetime import datetime

load_dotenv()

upload_csv = APIRouter()

MYSQL_URI = os.getenv("MYSQL_URI")
if not MYSQL_URI:
    raise ValueError("MYSQL_URI not found in environment variables.")

# In-memory storage for the current session's table info
current_session = {
    "active_table": None,
    "table_info": {},
    "upload_time": None
}

@upload_csv.post("/upload_csv/")
async def upload(file: UploadFile = File(...), table_name: str = Form(...)):
    try:
        contents = await file.read()
        df = pd.read_csv(pd.io.common.BytesIO(contents))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error reading CSV: {e}")

    try:
        engine = create_engine(MYSQL_URI)
        # Sanitize table name
        safe_table_name = re.sub(r'\W+', '_', table_name).lower()
        df.to_sql(safe_table_name, engine, if_exists="replace", index=False)
        
        # Update session info
        current_session["active_table"] = safe_table_name
        current_session["table_info"] = {
            "original_name": table_name,
            "safe_name": safe_table_name,
            "columns": df.columns.tolist(),
            "rows": len(df),
            "file_name": file.filename,
            "data_types": df.dtypes.astype(str).to_dict(),
            "upload_time": datetime.now().isoformat()
        }
        current_session["upload_time"] = datetime.now().isoformat()
        
        # Clear any existing chat history for this table since it's a new upload
        # This will be handled by the main app when needed
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {e}")

    return {
        "message": f"Uploaded and stored in table `{safe_table_name}`",
        "table_name": safe_table_name,
        "original_name": table_name,
        "rows": len(df),
        "columns": df.columns.tolist(),
        "data_types": df.dtypes.astype(str).to_dict(),
        "note": "This table is now set as your active dataset for queries.",
        "upload_time": current_session["upload_time"],
        "report_ready": True  # Signal that report can be generated
    }

@upload_csv.get("/current_table/")
def get_current_table():
    """Get information about the currently active table"""
    if current_session["active_table"]:
        return {
            "active_table": current_session["active_table"],
            "table_info": current_session["table_info"],
            "upload_time": current_session["upload_time"],
            "is_active": True
        }
    else:
        return {
            "message": "No active table. Please upload a CSV file first.",
            "is_active": False
        }

@upload_csv.post("/set_active_table/")
def set_active_table(table_name: str = Form(...)):
    """Manually set which table should be the active one"""
    try:
        engine = create_engine(MYSQL_URI)
        # Check if table exists using text() for raw SQL
        from sqlalchemy import text
        
        with engine.connect() as conn:
            # Check if table exists
            result = conn.execute(text(f"SHOW TABLES LIKE '{table_name}'"))
            tables = result.fetchall()
            
            if tables:
                # Get column information for the table
                try:
                    columns_result = conn.execute(text(f"DESCRIBE {table_name}"))
                    columns_info = columns_result.fetchall()
                    columns = [row[0] for row in columns_info]
                    data_types = {row[0]: row[1] for row in columns_info}
                    
                    # Get row count
                    count_result = conn.execute(text(f"SELECT COUNT(*) FROM {table_name}"))
                    row_count = count_result.fetchone()[0]
                    
                    # Get sample data to infer more info
                    sample_result = conn.execute(text(f"SELECT * FROM {table_name} LIMIT 5"))
                    sample_data = sample_result.fetchall()
                    
                except Exception as col_error:
                    columns = ["Unable to determine columns"]
                    data_types = {}
                    row_count = "Unknown"
                
                # Update session
                current_session["active_table"] = table_name
                current_session["table_info"] = {
                    "original_name": table_name,
                    "safe_name": table_name,
                    "columns": columns,
                    "rows": row_count,
                    "data_types": data_types,
                    "manually_set": True,
                    "set_time": datetime.now().isoformat()
                }
                current_session["upload_time"] = datetime.now().isoformat()
                
                return {
                    "message": f"Active table set to `{table_name}`",
                    "table_name": table_name,
                    "table_info": {
                        "columns": columns,
                        "rows": row_count,
                        "data_types": data_types
                    },
                    "report_ready": True,  # Signal that report can be generated
                    "clear_chat_history": True  # Signal to clear chat history
                }
            else:
                raise HTTPException(status_code=404, detail=f"Table `{table_name}` not found")
                
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {e}")

@upload_csv.get("/table_preview/")
def get_table_preview(limit: int = 10):
    """Get a preview of the current active table data"""
    try:
        if not current_session["active_table"]:
            raise HTTPException(status_code=400, detail="No active table")
        
        engine = create_engine(MYSQL_URI)
        from sqlalchemy import text
        
        with engine.connect() as conn:
            # Get sample data
            result = conn.execute(text(f"SELECT * FROM {current_session['active_table']} LIMIT {limit}"))
            rows = result.fetchall()
            columns = result.keys()
            
            # Convert to list of dictionaries for JSON serialization
            preview_data = []
            for row in rows:
                preview_data.append(dict(zip(columns, row)))
            
            return {
                "table_name": current_session["active_table"],
                "columns": list(columns),
                "data": preview_data,
                "preview_rows": len(preview_data),
                "total_rows": current_session["table_info"].get("rows", "Unknown")
            }
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting preview: {e}")

@upload_csv.delete("/table/{table_name}")
def delete_table(table_name: str):
    """Delete a table from the database"""
    try:
        engine = create_engine(MYSQL_URI)
        from sqlalchemy import text
        
        with engine.connect() as conn:
            # Check if table exists
            result = conn.execute(text(f"SHOW TABLES LIKE '{table_name}'"))
            if not result.fetchall():
                raise HTTPException(status_code=404, detail=f"Table '{table_name}' not found")
            
            # Drop the table
            conn.execute(text(f"DROP TABLE {table_name}"))
            conn.commit()
            
            # If this was the active table, clear the session
            if current_session["active_table"] == table_name:
                current_session["active_table"] = None
                current_session["table_info"] = {}
                current_session["upload_time"] = None
            
            return {
                "message": f"Table '{table_name}' deleted successfully",
                "deleted_table": table_name,
                "was_active": current_session["active_table"] is None
            }
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting table: {e}")

# Export the session data so query.py can access it
def get_current_session():
    return current_session

def update_session(new_session_data):
    """Allow other modules to update session data"""
    current_session.update(new_session_data)