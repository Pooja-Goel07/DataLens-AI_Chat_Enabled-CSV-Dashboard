# upload.py - CSV upload and table lifecycle.
#
# Active-table state is no longer held in a process-global dict. It is per-user and
# persisted in the database via db.set_active_table / db.resolve_active_table.
import re
import pandas as pd
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Depends, Body
from sqlalchemy import text
from datetime import datetime

from auth import get_current_user
from db import get_engine, resolve_active_table, set_active_table, clear_active_table, get_last_active_table

upload_csv = APIRouter()


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
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error reading CSV: {str(e)}")

    try:
        engine = get_engine()

        # Create a user-specific, sanitized table name to avoid conflicts.
        clean_table_name = re.sub(r'[^a-zA-Z0-9_]', '_', table_name).lower()
        safe_table_name = f"{current_user}_{clean_table_name}"

        # Ensure table name is within MySQL's 64-character limit.
        if len(safe_table_name) > 60:
            safe_table_name = safe_table_name[:60]

        # Store data in MySQL
        df.to_sql(safe_table_name, engine, if_exists="replace", index=False)

        # Record table ownership in user_tables
        with engine.begin() as conn:
            conn.execute(text("""
                DELETE FROM user_tables
                WHERE username = :username AND table_name = :table_name
            """), {"username": current_user, "table_name": safe_table_name})

            conn.execute(text("""
                INSERT INTO user_tables (username, table_name, original_name, file_name, rows_count, columns_count)
                VALUES (:username, :table_name, :original_name, :file_name, :rows_count, :columns_count)
            """), {
                "username": current_user,
                "table_name": safe_table_name,
                "original_name": table_name,
                "file_name": file.filename,
                "rows_count": len(df),
                "columns_count": len(df.columns),
            })

        # Mark this table as the user's active dataset (persisted per-user).
        set_active_table(current_user, safe_table_name)
        upload_time = datetime.now().isoformat()

        print(f"Upload successful - Active table set to: {safe_table_name} for user: {current_user}")

    except HTTPException:
        raise
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
        "upload_time": upload_time,
        "report_ready": True,
        "success": True,
    }


@upload_csv.get("/current_table/")
async def get_current_table(current_user: str = Depends(get_current_user)):
    """Get information about the user's currently active table."""
    ctx = resolve_active_table(current_user)
    if ctx["active_table"]:
        return {
            "active_table": ctx["active_table"],
            "table_info": ctx["table_info"],
            "upload_time": ctx["upload_time"],
            "is_active": True,
        }
    return {
        "message": "No active table. Please upload a CSV file first.",
        "is_active": False,
    }


@upload_csv.post("/switch_table/")
async def switch_table(request: dict = Body(...), current_user: str = Depends(get_current_user)):
    """Switch to a different table owned by the user."""
    table_name = request.get("table_name")
    if not table_name:
        raise HTTPException(status_code=400, detail="table_name is required")

    # resolve_active_table verifies ownership (raises 403 if not owned) and builds context.
    ctx = resolve_active_table(current_user, table_name)
    set_active_table(current_user, table_name)

    print(f"Switched to table: {table_name} for user: {current_user}")

    return {
        "message": f"Switched to table '{ctx['table_info'].get('original_name', table_name)}'",
        "table_name": table_name,
        "original_name": ctx["table_info"].get("original_name", table_name),
        "table_info": ctx["table_info"],
        "report_ready": True,
        "clear_chat_history": True,
    }


@upload_csv.get("/list_tables/")
async def list_user_tables(current_user: str = Depends(get_current_user)):
    """List all tables owned by the current user."""
    try:
        engine = get_engine()
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
                    "columns": row[4],
                })

            active_table = get_last_active_table(conn, current_user)

            return {
                "tables": [t["table_name"] for t in tables],
                "table_details": tables,
                "active_table": active_table,
                "user": current_user,
            }

    except Exception as e:
        print(f"Error in list_user_tables: {e}")
        raise HTTPException(status_code=500, detail=f"Error listing tables: {str(e)}")


@upload_csv.delete("/table/{table_name}")
async def delete_table(table_name: str, current_user: str = Depends(get_current_user)):
    """Delete a table owned by the user."""
    try:
        engine = get_engine()
        with engine.begin() as conn:
            # Verify user owns this table
            result = conn.execute(text("""
                SELECT table_name FROM user_tables
                WHERE username = :username AND table_name = :table_name
            """), {"username": current_user, "table_name": table_name})

            if not result.fetchone():
                raise HTTPException(status_code=404, detail=f"Table '{table_name}' not found or not owned by user")

            # Drop the actual data table
            conn.execute(text(f"DROP TABLE IF EXISTS `{table_name}`"))

            # Remove ownership record
            conn.execute(text("""
                DELETE FROM user_tables
                WHERE username = :username AND table_name = :table_name
            """), {"username": current_user, "table_name": table_name})

        # Clear active pointer if this was the active table
        clear_active_table(current_user, table_name)

        return {
            "message": f"Table '{table_name}' deleted successfully",
            "deleted_table": table_name,
        }

    except HTTPException:
        raise
    except Exception as e:
        print(f"Error in delete_table: {e}")
        raise HTTPException(status_code=500, detail=f"Error deleting table: {str(e)}")
