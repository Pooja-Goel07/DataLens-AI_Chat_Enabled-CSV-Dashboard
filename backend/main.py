# main.py - FastAPI application entry point.
#
# Auth lives in auth.py; the shared engine and per-user active-table state live in db.py.
# Duplicate list_tables / switch_table endpoints were removed (they live in upload.py).
import os
import json
import pandas as pd
from datetime import datetime
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel
from sqlalchemy import text
from sqlmodel import SQLModel, Field, Session, select

from upload import upload_csv
from query import ask_question
from report import report_router
from csv_analysis import analyze_csv
from auth import get_current_user, get_password_hash, verify_password, create_access_token
from db import get_engine, init_db, resolve_active_table

app = FastAPI(title="CSV Analysis & Chatbot System")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files (for graphs)
os.makedirs("graphs", exist_ok=True)
app.mount("/graphs", StaticFiles(directory="graphs"), name="graphs")

# Include routers
app.include_router(upload_csv, prefix="/api")
app.include_router(ask_question, prefix="/api")
app.include_router(report_router, prefix="/api")


# --- User model & database bootstrap ---
class User(SQLModel, table=True):
    __tablename__ = "users"
    id: int | None = Field(default=None, primary_key=True)
    username: str = Field(index=True, unique=True)
    hashed_password: str


# Create the users table (SQLModel) and the metadata tables (db.init_db).
SQLModel.metadata.create_all(get_engine())
init_db()


class SignupRequest(BaseModel):
    username: str
    password: str


@app.post("/api/signup/")
async def signup(payload: SignupRequest):
    """Register a new user"""
    username = payload.username
    password = payload.password

    if not username or not password:
        raise HTTPException(status_code=400, detail="Username and password required")

    hashed_password = get_password_hash(password)

    with Session(get_engine()) as session:
        statement = select(User).where(User.username == username)
        if session.exec(statement).first():
            raise HTTPException(status_code=400, detail="Username already registered")

        new_user = User(username=username, hashed_password=hashed_password)
        session.add(new_user)
        session.commit()
        session.refresh(new_user)

    return {"message": "User registered successfully"}


@app.post("/api/token")
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    """User login endpoint"""
    with get_engine().connect() as conn:
        result = conn.execute(
            text("SELECT username, hashed_password FROM users WHERE username=:username"),
            {"username": form_data.username},
        )
        user = result.fetchone()
        if not user or not verify_password(form_data.password, user.hashed_password):
            raise HTTPException(status_code=400, detail="Incorrect username or password")

    access_token = create_access_token(data={"sub": form_data.username})
    return {"access_token": access_token, "token_type": "bearer"}


@app.post("/api/generate_report/")
async def generate_report(table_name: str = None, current_user: str = Depends(get_current_user)):
    """Generate a full report (with rendered PNG charts) for a user's table."""
    try:
        ctx = resolve_active_table(current_user, table_name)
        target_table = ctx["active_table"]

        if not target_table:
            raise HTTPException(
                status_code=400,
                detail="No active table found. Please upload a CSV file first.",
            )

        # Export table data to a DataFrame (ownership verified by resolve_active_table).
        with get_engine().connect() as conn:
            df = pd.read_sql(f"SELECT * FROM `{target_table}`", conn)

        # Create a temporary CSV for analysis
        temp_dir = "temp"
        os.makedirs(temp_dir, exist_ok=True)
        temp_file = os.path.join(temp_dir, f"{target_table}_temp.csv")
        df.to_csv(temp_file, index=False)

        try:
            summary, outliers_json, graph_paths = analyze_csv(temp_file)
        finally:
            if os.path.exists(temp_file):
                os.remove(temp_file)

        graph_urls = [f"/graphs/{os.path.basename(path)}" for path in graph_paths]

        return {
            "table_name": target_table,
            "original_name": ctx["table_info"].get("original_name"),
            "file_name": ctx["table_info"].get("file_name"),
            "summary": summary,
            "outliers": json.loads(outliers_json) if outliers_json else {},
            "graph_urls": graph_urls,
            "total_rows": len(df),
            "total_columns": len(df.columns),
            "columns": df.columns.tolist(),
            "generated_at": datetime.now().isoformat(),
        }

    except HTTPException:
        raise
    except Exception as e:
        print(f"Error in generate_report: {e}")
        raise HTTPException(status_code=500, detail=f"Report generation failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
