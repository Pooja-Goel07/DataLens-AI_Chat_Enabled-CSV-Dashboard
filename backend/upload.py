# upload.py - Fixed version with proper error handling
import os
import re
import pandas as pd
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Depends, Body
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
import json
from datetime import datetime
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt

load_dotenv()

upload_csv = APIRouter()
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/token")

MYSQL_URI = os.getenv("MYSQL_URI")
SECRET_KEY = os.getenv("SECRET_KEY")
ALGORITHM = "HS256"

if not MYSQL_URI:
    raise ValueError("MYSQL_URI not found in environment variables.")

if not SECRET_KEY:
    raise ValueError("SECRET_KEY not found in environment variables.")

# In-memory storage for the current session's table info
current_session = {
    "active_table": None,
    "table_info": {},
    "upload_time": None,
    "current_user": None
}

async def get_current_user(token: str = Depends(oauth2_scheme)):
    """Get current authenticated user"""
    credentials_exception = HTTPException(
        status_code=401,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    
    # Verify user exists in database
    try:
        engine = create_engine(MYSQL_URI)
        with engine.connect() as conn:
            result = conn.execute(text("SELECT username FROM users WHERE username=:username"), {"username": username})
            user = result.fetchone()
            if not user:
                raise credentials_exception
        return username
    except Exception as e:
        print(f"Database error in get_current_user: {e}")
        raise credentials_exception

@upload_csv.post("/upload_csv/")
async def upload(file: UploadFile = File(...), table_name: str = Form(...), current_user: str = Depends(get_current_user)):
    try:
        # Read and validate CSV file
        contents = await file.read()
        df = pd.read_csv(pd.io.common.BytesIO(contents))
        
        if df.empty:
            raise HTTPException(status_code=400, detail="CSV file is empty")
            
    except pd.errors.EmptyDataError:
        raise HTTPException(status_code=400, detail="CSV file is empty or invalid")
    except pd.errors.ParserError as e:
        raise HTTPException(status_code=400, detail=f"Error parsing CSV: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error reading CSV: {str(e)}")

    try:
        engine = create_engine(MYSQL_URI)
        
        # Create user-specific table name to avoid conflicts
        # Clean table name properly
        clean_table_name = re.sub(r'[^a-zA-Z0-9_]', '_', table_name).lower()
        safe_table_name = f"{current_user}_{clean_table_name}"
        
        # Ensure table name is not too long (MySQL limit is 64 characters)
        if len(safe_table_name) > 60:
            safe_table_name = safe_table_name[:60]
        
        # Store data in MySQL
        df.to_sql(safe_table_name, engine, if_exists="replace", index=False)
        
        # Record table ownership in user_tables
        with engine.connect() as conn:
            # Remove existing record if it exists
            conn.execute(text("""
                DELETE FROM user_tables 
                WHERE username = :username AND table_name = :table_name
            """), {"username": current_user, "table_name": safe_table_name})
            
            # Insert new record
            conn.execute(text("""
                INSERT INTO user_tables (username, table_name, original_name, file_name, rows_count, columns_count)
                VALUES (:username, :table_name, :original_name, :file_name, :rows_count, :columns_count)
            """), {
                "username": current_user,
                "table_name": safe_table_name,
                "original_name": table_name,
                "file_name": file.filename,
                "rows_count": len(df),
                "columns_count": len(df.columns)
            })
            conn.commit()
        
        # Update session info immediately
        current_session["active_table"] = safe_table_name
        current_session["current_user"] = current_user
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
        
        print(f"Upload successful - Active table set to: {safe_table_name} for user: {current_user}")
            
    except Exception as e:
        print(f"Database error in upload: {e}")
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

    return {
        "message": f"Uploaded and stored in table `{safe_table_name}`",
        "table_name": safe_table_name,
        "original_name": table_name,
        "rows": len(df),
        "columns": df.columns.tolist(),
        "data_types": df.dtypes.astype(str).to_dict(),
        "note": "This table is now set as your active dataset for queries.",
        "upload_time": current_session["upload_time"],
        "report_ready": True,
        "success": True
    }

@upload_csv.get("/current_table/")
async def get_current_table(current_user: str = Depends(get_current_user)):
    """Get information about the currently active table"""
    if current_session["active_table"] and current_session["current_user"] == current_user:
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

@upload_csv.post("/switch_table/")
async def switch_table(request: dict = Body(...), current_user: str = Depends(get_current_user)):
    """Switch to a different table owned by the user"""
    table_name = request.get("table_name")
    if not table_name:
        raise HTTPException(status_code=400, detail="table_name is required")
    
    try:
        engine = create_engine(MYSQL_URI)
        
        with engine.connect() as conn:
            # Verify user owns this table
            result = conn.execute(text("""
                SELECT original_name, file_name, rows_count, columns_count 
                FROM user_tables 
                WHERE username = :username AND table_name = :table_name
            """), {"username": current_user, "table_name": table_name})
            
            table_record = result.fetchone()
            if not table_record:
                raise HTTPException(status_code=404, detail=f"Table '{table_name}' not found or not owned by user")
            
            # Get column information
            columns_result = conn.execute(text(f"DESCRIBE {table_name}"))
            columns_info = columns_result.fetchall()
            columns = [row[0] for row in columns_info]
            data_types = {row[0]: row[1] for row in columns_info}
            
            # Update session
            current_session["active_table"] = table_name
            current_session["current_user"] = current_user
            current_session["table_info"] = {
                "original_name": table_record[0],
                "safe_name": table_name,
                "columns": columns,
                "rows": table_record[2],
                "file_name": table_record[1],
                "data_types": data_types,
                "switched_at": datetime.now().isoformat()
            }
            
            print(f"Switched to table: {table_name} for user: {current_user}")
            
            return {
                "message": f"Switched to table '{table_record[0]}'",
                "table_name": table_name,
                "original_name": table_record[0],
                "table_info": current_session["table_info"],
                "report_ready": True,
                "clear_chat_history": True
            }
            
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error in switch_table: {e}")
        raise HTTPException(status_code=500, detail=f"Error switching table: {str(e)}")

@upload_csv.get("/list_tables/")
async def list_user_tables(current_user: str = Depends(get_current_user)):
    """List all tables owned by the current user"""
    try:
        engine = create_engine(MYSQL_URI)
        
        with engine.connect() as conn:
            result = conn.execute(text("""
                SELECT table_name, original_name, upload_time, rows_count, columns_count
                FROM user_tables 
                WHERE username = :username
                ORDER BY upload_time DESC
            """), {"username": current_user})
            
            tables = []
            for row in result.fetchall():
                tables.append({
                    "table_name": row[0],
                    "original_name": row[1],
                    "upload_time": row[2].isoformat() if row[2] else None,
                    "rows": row[3],
                    "columns": row[4]
                })
            
            return {
                "tables": [t["table_name"] for t in tables],
                "table_details": tables,
                "active_table": current_session.get("active_table"),
                "user": current_user
            }
            
    except Exception as e:
        print(f"Error in list_user_tables: {e}")
        raise HTTPException(status_code=500, detail=f"Error listing tables: {str(e)}")

@upload_csv.delete("/table/{table_name}")
async def delete_table(table_name: str, current_user: str = Depends(get_current_user)):
    """Delete a table owned by the user"""
    try:
        engine = create_engine(MYSQL_URI)
        
        with engine.connect() as conn:
            # Verify user owns this table
            result = conn.execute(text("""
                SELECT table_name FROM user_tables 
                WHERE username = :username AND table_name = :table_name
            """), {"username": current_user, "table_name": table_name})
            
            if not result.fetchone():
                raise HTTPException(status_code=404, detail=f"Table '{table_name}' not found or not owned by user")
            
            # Drop the actual table
            conn.execute(text(f"DROP TABLE IF EXISTS {table_name}"))
            
            # Remove from user_tables
            conn.execute(text("""
                DELETE FROM user_tables 
                WHERE username = :username AND table_name = :table_name
            """), {"username": current_user, "table_name": table_name})
            
            conn.commit()
            
            # Clear session if this was the active table
            if current_session.get("active_table") == table_name:
                current_session["active_table"] = None
                current_session["table_info"] = {}
                current_session["upload_time"] = None
            
            return {
                "message": f"Table '{table_name}' deleted successfully",
                "deleted_table": table_name
            }
            
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error in delete_table: {e}")
        raise HTTPException(status_code=500, detail=f"Error deleting table: {str(e)}")

# Export functions for other modules
def get_current_session():
    return current_session

def update_session(new_session_data):
    """Allow other modules to update session data"""
    current_session.update(new_session_data)
